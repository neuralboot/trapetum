// bench.cu -- codebook-quant kernels vs cuBLAS fp16 dense (A40, sm_86)
//   1) fused_dequant_gemv (decode, M=1) vs cublasGemmEx fp16 dense
//   2) dequant_l2 (standalone, L2-cache gather, no shared staging) bandwidth
//   3) fused_tc (Tensor Core / wmma fused dequant-GEMM, prefill) vs cuBLAS
//
//   W_deq[i,j] = codebook[ indices[i,j], j ]   ;  indices [IC,OC] uint8 ;
//   codebook [K,OC] half ; Y[m,oc] = sum_ic X[m,ic]*W_deq[ic,oc]
//
// Build: nvcc -O3 -arch=sm_86 bench.cu -lcublas -o bench
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
using namespace nvcuda;

#ifndef K_CLUSTERS
#define K_CLUSTERS 256
#endif
#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// ---------- (2) standalone dequant, L2-cache gather (no shared staging) -------
// each thread: 8 cols of one row; vectorized idx load + float4 store; gather via
// __ldg straight from global -> the 2 MB codebook lives in L2, no re-staging.
__global__ void dequant_l2(const uint8_t* __restrict__ idx, const __half* __restrict__ cb,
                           __half* __restrict__ W, int IC, int OC){
    const int j0 = (blockIdx.x*blockDim.x + threadIdx.x)*8;
    if (j0+8 > OC) return;
    for (int i = blockIdx.y; i < IC; i += gridDim.y){
        const size_t base = (size_t)i*OC + j0;
        const uint32_t lo = __ldg((const uint32_t*)&idx[base]);
        const uint32_t hi = __ldg((const uint32_t*)&idx[base+4]);
        __half out[8];
        #pragma unroll
        for(int v=0;v<4;v++) out[v]   = __ldg(&cb[(size_t)((lo>>(8*v))&0xFF)*OC + j0+v]);
        #pragma unroll
        for(int v=0;v<4;v++) out[v+4] = __ldg(&cb[(size_t)((hi>>(8*v))&0xFF)*OC + j0+4+v]);
        *(float4*)&W[base] = *(const float4*)out;
    }
}

// ---------- (1) fused dequant + GEMV (decode) --------------------------------
// X in shared (reused across cols); indices streamed (uint8); codebook gathered
// from L2. M small.
#define M_MAX 8
__global__ void fused_gemv(const __half* __restrict__ X, const uint8_t* __restrict__ idx,
                           const __half* __restrict__ cb, __half* __restrict__ Y,
                           int M, int IC, int OC){
    extern __shared__ __half sX[];                 // M*IC
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    for (int t=threadIdx.x; t<M*IC; t+=blockDim.x) sX[t]=__ldg(&X[t]);
    __syncthreads();
    if (j>=OC) return;
    float acc[M_MAX];
    #pragma unroll
    for(int m=0;m<M_MAX;m++) acc[m]=0.f;
    for (int ic=0; ic<IC; ++ic){
        const uint8_t id=__ldg(&idx[(size_t)ic*OC+j]);
        const float w=__half2float(__ldg(&cb[(size_t)id*OC+j]));
        #pragma unroll
        for(int m=0;m<M_MAX;m++) if(m<M) acc[m]+=__half2float(sX[(size_t)m*IC+ic])*w;
    }
    #pragma unroll
    for(int m=0;m<M_MAX;m++) if(m<M) Y[(size_t)m*OC+j]=__float2half(acc[m]);
}

// ---------- (1b) fused GEMV, codebook tile in SHARED (fast lookup) ----------
// Each block owns TILE_COLS columns; stages those columns' K centroids into
// shared ONCE (columns partitioned across blocks -> codebook read exactly once,
// no redundancy). The per-element lookup is then a shared access, not a global
// gather -> the kernel becomes bandwidth-bound (streams uint8 indices).
#define TILE_COLS 64
__global__ void fused_gemv_v2(const __half* __restrict__ X, const uint8_t* __restrict__ idx,
                              const __half* __restrict__ cb, __half* __restrict__ Y,
                              int M, int IC, int OC){
    extern __shared__ __half sm[];
    __half* s_cb = sm;                              // K * TILE_COLS
    __half* sX   = sm + (size_t)K_CLUSTERS*TILE_COLS; // M * IC
    const int j0 = blockIdx.x*TILE_COLS;
    const int tx = threadIdx.x;                     // 0..TILE_COLS-1
    const int j  = j0 + tx;
    for (int t=tx; t<K_CLUSTERS*TILE_COLS; t+=blockDim.x){
        const int k=t/TILE_COLS, jj=j0 + (t%TILE_COLS);
        s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]);
    }
    for (int t=tx; t<M*IC; t+=blockDim.x) sX[t]=__ldg(&X[t]);
    __syncthreads();
    if (j>=OC) return;
    float acc[M_MAX];
    #pragma unroll
    for(int m=0;m<M_MAX;m++) acc[m]=0.f;
    for (int ic=0; ic<IC; ++ic){
        const uint8_t id=__ldg(&idx[(size_t)ic*OC+j]);
        const float w=__half2float(s_cb[id*TILE_COLS + tx]);   // shared lookup
        #pragma unroll
        for(int m=0;m<M_MAX;m++) if(m<M) acc[m]+=__half2float(sX[(size_t)m*IC+ic])*w;
    }
    #pragma unroll
    for(int m=0;m<M_MAX;m++) if(m<M) Y[(size_t)m*OC+j]=__float2half(acc[m]);
}

// ---------- (1c) fused GEMV, split-K + coalesced (M=1) ----------------------
// threads.x = columns (consecutive -> coalesced index reads), threads.y splits
// the IC contraction (occupancy: 4096 cols * TY rows of work), codebook in
// shared. Each thread accumulates a partial sum over its ic-stride; partials
// reduced over threadIdx.y in shared. Saturates the device.
#define TX 32
#define TY 8
__global__ void fused_gemv_splitk(const __half* __restrict__ X, const uint8_t* __restrict__ idx,
                                  const __half* __restrict__ cb, __half* __restrict__ Y,
                                  int M, int IC, int OC){
    extern __shared__ char smem[];
    __half* s_cb = (__half*)smem;                 // K*TX
    __half* sX   = s_cb + (size_t)K_CLUSTERS*TX;  // IC   (M=1)
    float*  red  = (float*)(sX + (size_t)IC);     // TY*TX
    const int tx=threadIdx.x, ty=threadIdx.y;
    const int j0=blockIdx.x*TX, j=j0+tx;
    const int tid=ty*TX+tx, nth=TX*TY;
    for (int t=tid; t<K_CLUSTERS*TX; t+=nth){ const int k=t/TX, jj=j0+(t%TX); s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]); }
    for (int t=tid; t<IC; t+=nth) sX[t]=__ldg(&X[t]);
    __syncthreads();
    float acc=0.f;
    if (j<OC){
        for (int ic=ty; ic<IC; ic+=TY){
            const uint8_t id=__ldg(&idx[(size_t)ic*OC+j]);   // coalesced over tx
            acc += __half2float(sX[ic]) * __half2float(s_cb[id*TX+tx]);
        }
    }
    red[ty*TX+tx]=acc; __syncthreads();
    if (ty==0 && j<OC){
        float s=0.f;
        #pragma unroll
        for(int y=0;y<TY;y++) s+=red[y*TX+tx];
        Y[j]=__float2half(s);
    }
}

// ---------- (1d) fused GEMV, max occupancy: split-K x32, codebook in L2 ------
// Drop the shared codebook (it capped us at ~1 block/SM). 32-way IC split gives
// 1024-thread blocks fully resident; the L2-cached codebook gather (2 MB in 6 MB
// L2) is hidden by the high warp count. Indices stay coalesced over tx.
#define TXO 32
#define TYO 32
__global__ void fused_gemv_opt(const __half* __restrict__ X, const uint8_t* __restrict__ idx,
                               const __half* __restrict__ cb, __half* __restrict__ Y,
                               int M, int IC, int OC){
    extern __shared__ char smem[];
    __half* sX = (__half*)smem;                   // IC
    float*  red= (float*)(sX + (size_t)IC);       // TYO*TXO
    const int tx=threadIdx.x, ty=threadIdx.y;
    const int j=blockIdx.x*TXO + tx;
    const int tid=ty*TXO+tx, nth=TXO*TYO;
    for (int t=tid; t<IC; t+=nth) sX[t]=__ldg(&X[t]);
    __syncthreads();
    float acc=0.f;
    if (j<OC)
        for (int ic=ty; ic<IC; ic+=TYO){
            const uint8_t id=__ldg(&idx[(size_t)ic*OC+j]);
            acc += __half2float(sX[ic]) * __half2float(__ldg(&cb[(size_t)id*OC+j]));
        }
    red[ty*TXO+tx]=acc; __syncthreads();
    if (ty==0 && j<OC){ float s=0.f;
        #pragma unroll
        for(int y=0;y<TYO;y++) s+=red[y*TXO+tx]; Y[j]=__float2half(s); }
}

// ---------- (3) fused dequant + GEMM via Tensor Cores (prefill) --------------
// One warp per 16x16 output tile (UNOPTIMIZED first cut: correctness + structure).
// K-loop: load X tile (global), dequant W tile into shared, wmma::mma_sync.
__global__ void fused_tc(const __half* __restrict__ X, const uint8_t* __restrict__ idx,
                         const __half* __restrict__ cb, __half* __restrict__ Y,
                         int M, int IC, int OC){
    const int tileM = blockIdx.y*16, tileN = blockIdx.x*16;
    if (tileM>=M || tileN>=OC) return;
    const int lane = threadIdx.x;                  // block = 32 threads (1 warp)
    __shared__ __half sW[256];
    __shared__ float  sC[256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k0=0;k0<IC;k0+=16){
        // dequant 16x16 W tile [k0.., tileN..] -> shared (row-major)
        for (int e=lane; e<256; e+=32){
            const int r=e>>4, c=e&15, gk=k0+r, gn=tileN+c;
            const uint8_t id=__ldg(&idx[(size_t)gk*OC+gn]);
            sW[e]=__ldg(&cb[(size_t)id*OC+gn]);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
        wmma::load_matrix_sync(a, &X[(size_t)tileM*IC + k0], IC);
        wmma::load_matrix_sync(b, sW, 16);
        wmma::mma_sync(acc, a, b, acc);
        __syncwarp();
    }
    wmma::store_matrix_sync(sC, acc, 16, wmma::mem_row_major);
    for (int e=lane;e<256;e+=32){ const int r=e>>4,c=e&15; Y[(size_t)(tileM+r)*OC+tileN+c]=__float2half(sC[e]); }
}

// ---------- helpers ----------------------------------------------------------
static float time_ms(int iters, void(*fn)()){
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    fn(); CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for(int i=0;i<iters;i++) fn();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b); return ms/iters;
}
static double maxabs(const std::vector<__half>&A,const std::vector<__half>&B){
    double m=0; for(size_t i=0;i<A.size();++i) m=fmax(m,fabs((double)__half2float(A[i])-(double)__half2float(B[i]))); return m;
}

// globals for the no-arg timer thunks
static uint8_t* g_idx; static __half *g_cb,*g_W,*g_X,*g_Y; static int gIC,gOC,gM;
static cublasHandle_t g_h;
void run_dequant_l2(){ dim3 bl(32); dim3 gr(gOC/256, gIC); dequant_l2<<<gr,bl>>>(g_idx,g_cb,g_W,gIC,gOC); }
void run_fused_gemv(){ dim3 bl(256); dim3 gr(gOC/256); size_t sm=(size_t)gM*gIC*sizeof(__half);
    fused_gemv<<<gr,bl,sm>>>(g_X,g_idx,g_cb,g_Y,gM,gIC,gOC); }
void run_fused_gemv_v2(){ dim3 bl(TILE_COLS); dim3 gr(gOC/TILE_COLS);
    size_t sm=((size_t)K_CLUSTERS*TILE_COLS + (size_t)gM*gIC)*sizeof(__half);
    fused_gemv_v2<<<gr,bl,sm>>>(g_X,g_idx,g_cb,g_Y,gM,gIC,gOC); }
void run_splitk(){ dim3 bl(TX,TY); dim3 gr(gOC/TX);
    size_t sm=((size_t)K_CLUSTERS*TX + (size_t)gIC)*sizeof(__half) + (size_t)TY*TX*sizeof(float);
    fused_gemv_splitk<<<gr,bl,sm>>>(g_X,g_idx,g_cb,g_Y,gM,gIC,gOC); }
void run_opt(){ dim3 bl(TXO,TYO); dim3 gr(gOC/TXO);
    size_t sm=(size_t)gIC*sizeof(__half) + (size_t)TYO*TXO*sizeof(float);
    fused_gemv_opt<<<gr,bl,sm>>>(g_X,g_idx,g_cb,g_Y,gM,gIC,gOC); }
void run_fused_tc(){ dim3 bl(32); dim3 gr(gOC/16,(gM+15)/16); fused_tc<<<gr,bl>>>(g_X,g_idx,g_cb,g_Y,gM,gIC,gOC); }
void run_cublas(){ const float al=1.f,be=0.f;
    cublasGemmEx(g_h,CUBLAS_OP_N,CUBLAS_OP_N,gOC,gM,gIC,&al,g_W,CUDA_R_16F,gOC,g_X,CUDA_R_16F,gIC,&be,g_Y,CUDA_R_16F,gOC,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); }

int main(){
    const int IC=4096, OC=4096, K=K_CLUSTERS; gIC=IC; gOC=OC;
    cublasCreate(&g_h); cublasSetMathMode(g_h, CUBLAS_TENSOR_OP_MATH);
    std::mt19937 rng(0); std::uniform_int_distribution<int> ui(0,K-1); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> h_idx((size_t)IC*OC); for(auto&v:h_idx) v=(uint8_t)ui(rng);
    std::vector<__half> h_cb((size_t)K*OC); for(auto&v:h_cb) v=__float2half(nf(rng)*0.05f);
    CK(cudaMalloc(&g_idx,h_idx.size())); CK(cudaMalloc(&g_cb,h_cb.size()*2)); CK(cudaMalloc(&g_W,(size_t)IC*OC*2));
    CK(cudaMemcpy(g_idx,h_idx.data(),h_idx.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_cb,h_cb.data(),h_cb.size()*2,cudaMemcpyHostToDevice));

    printf("A40  IC=%d OC=%d K=%d\n", IC,OC,K);

    // ---- (2) dequant_l2 bandwidth ----
    { float ms=time_ms(50,run_dequant_l2);
      double bytes=(double)IC*OC*(1+2);
      printf("\n[2] dequant_l2 (standalone): %.3f ms   %.0f GB/s effective  (vs 31.8 GB/s shared-staging)\n", ms, bytes/(ms*1e-3)/1e9); }
    // materialize dense W (for cuBLAS baselines)
    run_dequant_l2(); CK(cudaDeviceSynchronize());

    // ---- (1) decode GEMV, M=1 ----
    gM=1; CK(cudaMalloc(&g_X,(size_t)gM*IC*2)); CK(cudaMalloc(&g_Y,(size_t)gM*OC*2));
    { std::vector<__half> hx((size_t)gM*IC); for(auto&v:hx) v=__float2half(nf(rng)); CK(cudaMemcpy(g_X,hx.data(),hx.size()*2,cudaMemcpyHostToDevice)); }
    run_cublas(); CK(cudaDeviceSynchronize());
    std::vector<__half> yref((size_t)gM*OC); CK(cudaMemcpy(yref.data(),g_Y,yref.size()*2,cudaMemcpyDeviceToHost));
    run_fused_gemv(); CK(cudaDeviceSynchronize());
    std::vector<__half> yq((size_t)gM*OC); CK(cudaMemcpy(yq.data(),g_Y,yq.size()*2,cudaMemcpyDeviceToHost));
    float ms_cb=time_ms(200,run_cublas), ms_q=time_ms(200,run_fused_gemv);
    run_fused_gemv_v2(); CK(cudaDeviceSynchronize());
    std::vector<__half> yqs((size_t)gM*OC); CK(cudaMemcpy(yqs.data(),g_Y,yqs.size()*2,cudaMemcpyDeviceToHost));
    float ms_v2=time_ms(200,run_fused_gemv_v2);
    printf("\n[1] decode M=1 (W fp16=%.0f MB, idx int8=%.0f MB):\n", IC*OC*2.0/1e6, IC*OC*1.0/1e6);
    printf("    cuBLAS dense      %.4f ms   (%.0f GB/s)\n", ms_cb, IC*OC*2.0/(ms_cb*1e-3)/1e9);
    printf("    fused_gemv L2     %.4f ms   x%.2f   err=%.4g\n", ms_q,  ms_cb/ms_q,  maxabs(yref,yq));
    printf("    fused_gemv SHARED %.4f ms   x%.2f   err=%.4g   (codebook in shared)\n", ms_v2, ms_cb/ms_v2, maxabs(yref,yqs));
    run_splitk(); CK(cudaDeviceSynchronize());
    std::vector<__half> yk((size_t)gM*OC); CK(cudaMemcpy(yk.data(),g_Y,yk.size()*2,cudaMemcpyDeviceToHost));
    float ms_k=time_ms(300,run_splitk);
    printf("    fused_gemv SPLITK %.4f ms   x%.2f   err=%.4g   (split-K8 + coalesced + shared cb)\n", ms_k, ms_cb/ms_k, maxabs(yref,yk));
    run_opt(); CK(cudaDeviceSynchronize());
    std::vector<__half> yo((size_t)gM*OC); CK(cudaMemcpy(yo.data(),g_Y,yo.size()*2,cudaMemcpyDeviceToHost));
    float ms_o=time_ms(300,run_opt);
    printf("    fused_gemv OPT    %.4f ms   x%.2f   err=%.4g   (split-K32 + L2 cb, %.0f GB/s)\n",
           ms_o, ms_cb/ms_o, maxabs(yref,yo), IC*OC*1.0/(ms_o*1e-3)/1e9);
    cudaFree(g_X); cudaFree(g_Y);

    // ---- (3) prefill GEMM, M=2048, Tensor Cores ----
    gM=2048; CK(cudaMalloc(&g_X,(size_t)gM*IC*2)); CK(cudaMalloc(&g_Y,(size_t)gM*OC*2));
    { std::vector<__half> hx((size_t)gM*IC); for(auto&v:hx) v=__float2half(nf(rng)); CK(cudaMemcpy(g_X,hx.data(),hx.size()*2,cudaMemcpyHostToDevice)); }
    run_cublas(); CK(cudaDeviceSynchronize());
    std::vector<__half> yref2((size_t)gM*OC); CK(cudaMemcpy(yref2.data(),g_Y,yref2.size()*2,cudaMemcpyDeviceToHost));
    run_fused_tc(); CK(cudaDeviceSynchronize());
    std::vector<__half> yq2((size_t)gM*OC); CK(cudaMemcpy(yq2.data(),g_Y,yq2.size()*2,cudaMemcpyDeviceToHost));
    float ms_cb2=time_ms(50,run_cublas), ms_tc=time_ms(50,run_fused_tc);
    double flop=2.0*gM*IC*OC;
    printf("\n[3] prefill M=%d  cuBLAS dense %.3f ms (%.1f TFLOP/s)   fused_tc %.3f ms (%.1f TFLOP/s, UNOPTIMIZED)\n",
           gM, ms_cb2, flop/(ms_cb2*1e-3)/1e12, ms_tc, flop/(ms_tc*1e-3)/1e12);
    printf("    max|err| vs cuBLAS = %.4g  (fp16 accum)\n", maxabs(yref2,yq2));
    return 0;
}
