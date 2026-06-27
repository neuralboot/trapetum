// gemv_codebook_hopper.cu  --  Idea-4 first cut: Hopper (sm_90) fused 4-bit
// codebook-dequant GEMV using THREAD-BLOCK CLUSTERS + DISTRIBUTED SHARED MEMORY.
//
// The motivation, measured: on A40 the fused decode kernel was limited by the
// codebook being re-staged into shared memory by every split-K block of a column
// tile (redundant traffic + occupancy pressure). On Hopper a thread-block CLUSTER
// can share one block's shared memory across the whole cluster (distributed shared
// memory, DSMEM). So we stage the column tile's codebook ONCE in cluster-rank-0 and
// let every split-K block read it remotely. That removes the redundant staging the
// A40 version paid for.
//
// Scheme (4-bit, K<=16): packed[ic, j/2] holds 2 indices/byte; a thread reads
// uint32 = 8 nibbles = 8 columns; a warp reads 128 B = a full cache line over 256
// columns. Grid: x = column tiles (256 cols), y = split-K. A CLUSTER spans the
// whole y dimension of one column tile, so all split-K blocks share rank-0's
// codebook copy. Partial sums are atomic-added into a float accumulator.
//
// VERDICT (measured, H100 PCIe): compiles on sm_90/CUDA 12.4 and is CORRECT
// (relerr 3e-4), but ~100x SLOWER than cuBLAS (1.53 ms vs 0.02 ms). The cluster
// idea backfires: the codebook lookup is in the hot loop, and a REMOTE distributed-
// shared-memory read is far slower than a LOCAL shared read. The codebook is tiny
// (8 KB), so local staging was never worth "fixing" this way. DEAD END -- kept as a
// recorded negative result. The fast path is the LOCAL-shared design in v2.
//
// Build: nvcc -O3 -arch=sm_90 gemv_codebook_hopper.cu -lcublas -o gemv_hopper
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
namespace cg = cooperative_groups;

#define K   16          // 4-bit codebook
#define CPB 256         // columns per block (32 lanes * 8 nibbles)
#define TY  8           // in-block IC split
#define GS  8           // grid.y split-K  == cluster size along y (<=8 portable)
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ void cpasync16(void* s,const void* g){
    unsigned a=__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(a),"l"(g));
}
__device__ __forceinline__ void commit(){ asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void wait0(){ asm volatile("cp.async.wait_group 0;\n"); }

// each output column j gets the same per-column codebook; one cluster owns 256 cols.
// rank 0 of the cluster stages the K*256 codebook into its shared memory; every
// rank reads it through distributed shared memory.
__global__ void __cluster_dims__(1, GS, 1)
gemv_codebook_cluster(const __half* __restrict__ X, const uint8_t* __restrict__ packed,
                      const __half* __restrict__ cb, float* __restrict__ Yacc,
                      int IC, int OC){
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned yrank = cluster.block_rank();     // 0..GS-1 (the split-K index)
    extern __shared__ __half smem[];                 // K*CPB on rank 0 only
    __shared__ float red[TY*CPB];

    const int tx=threadIdx.x, ty=threadIdx.y, tid=ty*32+tx, nth=32*TY;
    const int j0 = blockIdx.x*CPB;                   // this cluster's 256 columns

    // --- rank 0 stages the codebook tile once; everyone shares it via DSMEM ---
    if (yrank == 0) {
        for (int t=tid; t<K*CPB; t+=nth){ int k=t/CPB, jj=j0+(t%CPB); smem[t]=__ldg(&cb[(size_t)k*OC+jj]); }
    }
    cluster.sync();                                  // codebook ready cluster-wide
    __half* s_cb = cluster.map_shared_rank(smem, 0); // pointer into rank-0's shared mem

    // --- this block's split-K chunk of the contraction ---
    const int per=(IC+GS-1)/GS, ic0=yrank*per, ic1=min(IC,ic0+per);
    const int jbase=j0+tx*8; const size_t OCp=OC/2;
    float acc[8]={0,0,0,0,0,0,0,0};
    for (int ic=ic0+ty; ic<ic1; ic+=TY){
        uint32_t f=__ldg((const uint32_t*)&packed[(size_t)ic*OCp + jbase/2]);  // 8 nibbles
        float x=__half2float(__ldg(&X[ic]));
        #pragma unroll
        for(int c=0;c<8;c++){ uint8_t id=(f>>(4*c))&0xF; acc[c]+=x*__half2float(s_cb[id*CPB + tx*8+c]); }
    }
    #pragma unroll
    for(int c=0;c<8;c++) red[ty*CPB+tx*8+c]=acc[c];
    __syncthreads();
    if (ty==0){
        #pragma unroll
        for(int c=0;c<8;c++){ float s=0; for(int y=0;y<TY;y++) s+=red[y*CPB+tx*8+c]; atomicAdd(&Yacc[j0+tx*8+c],s); }
    }
    cluster.sync();  // keep DSMEM alive until all ranks have finished reading s_cb
}

// reference dense dequant (for the cuBLAS baseline)
__global__ void dequant(const uint8_t* packed,const __half* cb,__half* W,int IC,int OC){
    int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y;
    if(j<OC){ uint8_t b=packed[(size_t)i*(OC/2)+j/2]; uint8_t id=(j&1)?(b>>4):(b&0xF); W[(size_t)i*OC+j]=cb[(size_t)id*OC+j]; }
}

static __half *dX,*dW,*dCb,*dYc; static uint8_t* dPk; static float* dYf;
static int M=1, IC=4096, OC=4096; static cublasHandle_t H; static size_t SMEM;
void rk(){
    cudaMemset(dYf,0,(size_t)OC*4);
    dim3 grid(OC/CPB, GS), block(32,TY);
    cudaLaunchConfig_t cfg={}; cfg.gridDim=grid; cfg.blockDim=block; cfg.dynamicSmemBytes=SMEM;
    cudaLaunchAttribute attr[1];
    attr[0].id=cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x=1; attr[0].val.clusterDim.y=GS; attr[0].val.clusterDim.z=1;
    cfg.attrs=attr; cfg.numAttrs=1;
    cudaLaunchKernelEx(&cfg, gemv_codebook_cluster, dX, dPk, dCb, dYf, IC, OC);
}
void rcb(){ const float al=1,be=0; cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,OC,M,IC,&al,dW,CUDA_R_16F,OC,dX,CUDA_R_16F,IC,&be,dYc,CUDA_R_16F,OC,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); }
float tm(int n,void(*f)()){ cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();CK(cudaDeviceSynchronize());
    cudaEventRecord(a);for(int i=0;i<n;i++)f();cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms/n;}
int main(){
    cublasCreate(&H); cublasSetMathMode(H,CUBLAS_TENSOR_OP_MATH);
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0,K-1); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> id4((size_t)IC*OC); for(auto&v:id4)v=(uint8_t)ui(r);
    std::vector<uint8_t> pk((size_t)IC*(OC/2));
    for(size_t ic=0;ic<(size_t)IC;ic++) for(int j=0;j<OC;j+=2) pk[ic*(OC/2)+j/2]=(id4[ic*OC+j]&0xF)|((id4[ic*OC+j+1]&0xF)<<4);
    std::vector<__half> hc((size_t)K*OC); for(auto&v:hc)v=__float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)M*IC); for(auto&v:hx)v=__float2half(nf(r));
    CK(cudaMalloc(&dPk,pk.size()));CK(cudaMalloc(&dCb,hc.size()*2));CK(cudaMalloc(&dX,hx.size()*2));
    CK(cudaMalloc(&dW,(size_t)IC*OC*2));CK(cudaMalloc(&dYc,(size_t)M*OC*2));CK(cudaMalloc(&dYf,(size_t)OC*4));
    CK(cudaMemcpy(dPk,pk.data(),pk.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCb,hc.data(),hc.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
    SMEM=(size_t)K*CPB*sizeof(__half);
    CK(cudaFuncSetAttribute(gemv_codebook_cluster, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
    dequant<<<dim3(OC/256,IC),256>>>(dPk,dCb,dW,IC,OC); CK(cudaDeviceSynchronize());
    rk(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> yf((size_t)OC); CK(cudaMemcpy(yf.data(),dYf,(size_t)OC*4,cudaMemcpyDeviceToHost));
    rcb(); CK(cudaDeviceSynchronize());
    std::vector<__half> yc((size_t)OC); CK(cudaMemcpy(yc.data(),dYc,(size_t)OC*2,cudaMemcpyDeviceToHost));
    double me=0,den=0; for(int j=0;j<OC;j++){double a=yf[j],b=__half2float(yc[j]); me=fmax(me,fabs(a-b)); den=fmax(den,fabs(b));}
    float tk=tm(300,rk), tc=tm(300,rcb);
    printf("Hopper cluster-DSMEM 4-bit GEMV: %.4f ms   cuBLAS %.4f ms   x%.2f   relerr=%.3g\n",
           tk, tc, tc/tk, me/den);
    return 0;
}
