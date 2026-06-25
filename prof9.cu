// prof9.cu -- Marlin-style fused codebook-dequant GEMM (prefill, Tensor Cores).
// 128x128 block tile, 8 warps (4x2), BK=32. Software-pipelined K-loop with
// cp.async double-buffering: while the Tensor Cores work on tile k, the next
// tile's X and indices are streamed global->shared asynchronously. Indices are
// cp.async'd into shared, then dequantized (codebook lookup) into shared W just
// before the mma. nvcc -O3 -arch=sm_86 prof9.cu -lcublas -o prof9
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
using namespace nvcuda;
#define BM 128
#define BN 128
#define BK 32
#define NWARP 8
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ void cpasync16(void* s,const void* g){
    unsigned a=__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(a),"l"(g));
}
__device__ __forceinline__ void commit(){ asm volatile("cp.async.commit_group;\n"); }

__global__ void dequant(const uint8_t* idx,const __half* cb,__half* W,int IC,int OC){
    int j=blockIdx.x*blockDim.x+threadIdx.x,i=blockIdx.y;
    if(j<OC) W[(size_t)i*OC+j]=cb[(size_t)idx[(size_t)i*OC+j]*OC+j];
}

__global__ void marlin_gemm(const __half* __restrict__ X,const uint8_t* __restrict__ idx,
                            const __half* __restrict__ cb,float* __restrict__ Yf,int M,int IC,int OC){
    __shared__ __half  sX[2][BM*BK];
    __shared__ uint8_t sIdx[2][BK*BN];
    __shared__ __half  sW[BK*BN];
    const int tid=threadIdx.y*32+threadIdx.x, warp=threadIdx.y;
    const int warpM=warp/2, warpN=warp%2;              // 4x2 warp grid
    const int bm0=blockIdx.y*BM, bn0=blockIdx.x*BN;
    const int nb=IC/BK;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][4];
    #pragma unroll
    for(int i=0;i<2;i++)for(int j=0;j<4;j++) wmma::fill_fragment(acc[i][j],0.f);

    auto load=[&](int kt,int buf){
        int k0=kt*BK;
        #pragma unroll
        for(int c=tid;c<BM*BK/8;c+=NWARP*32){ int e=c*8, mm=e/BK, kk=e%BK; cpasync16(&sX[buf][e], &X[(size_t)(bm0+mm)*IC+k0+kk]); }
        for(int c=tid;c<BK*BN/16;c+=NWARP*32){ int e=c*16, kk=e/BN, nn=e%BN; cpasync16(&sIdx[buf][e], &idx[(size_t)(k0+kk)*OC+bn0+nn]); }
    };
    load(0,0); commit();
    for(int kt=0;kt<nb;kt++){
        int cur=kt&1, nxt=(kt+1)&1;
        if(kt+1<nb){ load(kt+1,nxt); commit(); asm volatile("cp.async.wait_group 1;\n"); }
        else        { asm volatile("cp.async.wait_group 0;\n"); }
        __syncthreads();
        // dequant current indices -> sW
        for(int t=tid;t<BK*BN;t+=NWARP*32){ int kk=t/BN, nn=t%BN; sW[t]=__ldg(&cb[(size_t)sIdx[cur][t]*OC+bn0+nn]); }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<BK;kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b[4];
            #pragma unroll
            for(int i=0;i<2;i++) wmma::load_matrix_sync(a[i], &sX[cur][(warpM*32+i*16)*BK+kk], BK);
            #pragma unroll
            for(int j=0;j<4;j++) wmma::load_matrix_sync(b[j], &sW[kk*BN+warpN*64+j*16], BN);
            #pragma unroll
            for(int i=0;i<2;i++)for(int j=0;j<4;j++) wmma::mma_sync(acc[i][j],a[i],b[j],acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<2;i++)for(int j=0;j<4;j++){
        int rm=bm0+warpM*32+i*16, cn=bn0+warpN*64+j*16;
        wmma::store_matrix_sync(&Yf[(size_t)rm*OC+cn], acc[i][j], OC, wmma::mem_row_major);
    }
}

static __half *dX,*dW,*dCb,*dYc; static uint8_t* dIdx; static float* dYf;
static int M=2048,IC=4096,OC=4096; static cublasHandle_t H;
void rmar(){ marlin_gemm<<<dim3(OC/BN,M/BM),dim3(32,NWARP)>>>(dX,dIdx,dCb,dYf,M,IC,OC); }
void rcb(){ const float al=1,be=0; cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,OC,M,IC,&al,dW,CUDA_R_16F,OC,dX,CUDA_R_16F,IC,&be,dYc,CUDA_R_16F,OC,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); }
float tm(int n,void(*f)()){ cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();CK(cudaDeviceSynchronize());
    cudaEventRecord(a);for(int i=0;i<n;i++)f();cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms/n;}
int main(){
    cublasCreate(&H); cublasSetMathMode(H,CUBLAS_TENSOR_OP_MATH);
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0,255); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> hi((size_t)IC*OC); for(auto&v:hi)v=(uint8_t)ui(r);
    std::vector<__half> hc((size_t)256*OC); for(auto&v:hc)v=__float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)M*IC); for(auto&v:hx)v=__float2half(nf(r)*0.1f);
    CK(cudaMalloc(&dIdx,hi.size()));CK(cudaMalloc(&dCb,hc.size()*2));CK(cudaMalloc(&dX,hx.size()*2));
    CK(cudaMalloc(&dW,(size_t)IC*OC*2));CK(cudaMalloc(&dYc,(size_t)M*OC*2));CK(cudaMalloc(&dYf,(size_t)M*OC*4));
    CK(cudaMemcpy(dIdx,hi.data(),hi.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCb,hc.data(),hc.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
    dequant<<<dim3(OC/256,IC),256>>>(dIdx,dCb,dW,IC,OC); CK(cudaDeviceSynchronize());
    rmar(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); rcb(); CK(cudaDeviceSynchronize());
    std::vector<float> yf((size_t)M*OC); std::vector<__half> yc((size_t)M*OC);
    CK(cudaMemcpy(yf.data(),dYf,yf.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(yc.data(),dYc,yc.size()*2,cudaMemcpyDeviceToHost));
    double me=0,den=0; for(size_t i=0;i<yf.size();i++){double a=yf[i],b=__half2float(yc[i]);me=fmax(me,fabs(a-b));den=fmax(den,fabs(b));}
    double flop=2.0*M*IC*OC;
    float tt=tm(50,rmar), tc=tm(50,rcb);
    printf("prefill M=%d  Marlin_fused %.3f ms (%.1f TFLOP/s)  cuBLAS %.3f ms (%.1f TFLOP/s)  ratio=%.2f  relerr=%.3g\n",
           M,tt,flop/(tt*1e-3)/1e12,tc,flop/(tc*1e-3)/1e12,tc/tt,me/den);
    return 0;
}
