// prof10.cu -- Marlin + register-pipelined dequant (kills the dequant bubble).
// The codebook gather for tile k+1 is issued into REGISTERS *before* the mma of
// tile k, then stored into a double-buffered shared W *after* -> the L2 gather
// latency overlaps the Tensor-Core work instead of stalling it. cp.async still
// streams X/indices. nvcc -O3 -arch=sm_86 prof10.cu -lcublas -o prof10
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
#define NEL (BK*BN/(NWARP*32))      // dequant elements per thread = 16
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ void cpasync16(void* s,const void* g){
    unsigned a=__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(a),"l"(g));
}
__device__ __forceinline__ void commit(){ asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void wait0(){ asm volatile("cp.async.wait_group 0;\n"); }

__global__ void dequant(const uint8_t* idx,const __half* cb,__half* W,int IC,int OC){
    int j=blockIdx.x*blockDim.x+threadIdx.x,i=blockIdx.y;
    if(j<OC) W[(size_t)i*OC+j]=cb[(size_t)idx[(size_t)i*OC+j]*OC+j];
}

__global__ void marlin2(const __half* __restrict__ X,const uint8_t* __restrict__ idx,
                        const __half* __restrict__ cb,float* __restrict__ Yf,int M,int IC,int OC){
    __shared__ __half  sX[2][BM*BK];
    __shared__ uint8_t sIdx[2][BK*BN];
    __shared__ __half  sW[2][BK*BN];
    const int tid=threadIdx.y*32+threadIdx.x, warp=threadIdx.y;
    const int warpM=warp/2, warpN=warp%2;
    const int bm0=blockIdx.y*BM, bn0=blockIdx.x*BN;
    const int nb=IC/BK;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][4];
    #pragma unroll
    for(int i=0;i<2;i++)for(int j=0;j<4;j++) wmma::fill_fragment(acc[i][j],0.f);

    auto load=[&](int kt,int buf){
        int k0=kt*BK;
        #pragma unroll
        for(int c=tid;c<BM*BK/8;c+=NWARP*32){ int e=c*8,mm=e/BK,kk=e%BK; cpasync16(&sX[buf][e],&X[(size_t)(bm0+mm)*IC+k0+kk]); }
        for(int c=tid;c<BK*BN/16;c+=NWARP*32){ int e=c*16,kk=e/BN,nn=e%BN; cpasync16(&sIdx[buf][e],&idx[(size_t)(k0+kk)*OC+bn0+nn]); }
    };
    auto dq=[&](int buf,__half* dst){            // dequant sIdx[buf] -> dst
        #pragma unroll
        for(int q=0;q<NEL;q++){ int t=tid+q*NWARP*32, nn=t%BN; dst[t]=__ldg(&cb[(size_t)sIdx[buf][t]*OC+bn0+nn]); }
    };
    auto mma=[&](int buf){
        #pragma unroll
        for(int kk=0;kk<BK;kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b[4];
            #pragma unroll
            for(int i=0;i<2;i++) wmma::load_matrix_sync(a[i],&sX[buf][(warpM*32+i*16)*BK+kk],BK);
            #pragma unroll
            for(int j=0;j<4;j++) wmma::load_matrix_sync(b[j],&sW[buf][kk*BN+warpN*64+j*16],BN);
            #pragma unroll
            for(int i=0;i<2;i++)for(int j=0;j<4;j++) wmma::mma_sync(acc[i][j],a[i],b[j],acc[i][j]);
        }
    };

    // prologue: load tile0, dequant tile0 -> sW[0], prefetch-issue tile1
    load(0,0); commit(); wait0(); __syncthreads();
    dq(0, sW[0]); __syncthreads();
    if(nb>1){ load(1,1); commit(); }

    for(int kt=0;kt<nb;kt++){
        int cur=kt&1, nxt=(kt+1)&1;
        __half wreg[NEL];
        if(kt+1<nb){
            wait0(); __syncthreads();                          // tile kt+1 (X+idx) ready
            #pragma unroll
            for(int q=0;q<NEL;q++){ int t=tid+q*NWARP*32, nn=t%BN; wreg[q]=__ldg(&cb[(size_t)sIdx[nxt][t]*OC+bn0+nn]); } // gather issued
        }
        mma(cur);                                              // overlaps the wreg gather latency
        __syncthreads();                                       // all warps done reading sX[cur]/sW[cur]
        if(kt+1<nb){
            #pragma unroll
            for(int q=0;q<NEL;q++){ int t=tid+q*NWARP*32; sW[nxt][t]=wreg[q]; }   // commit dequant
            if(kt+2<nb){ load(kt+2,cur); commit(); }           // reuse cur buffer for tile kt+2
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<2;i++)for(int j=0;j<4;j++){
        int rm=bm0+warpM*32+i*16, cn=bn0+warpN*64+j*16;
        wmma::store_matrix_sync(&Yf[(size_t)rm*OC+cn],acc[i][j],OC,wmma::mem_row_major);
    }
}

static __half *dX,*dW,*dCb,*dYc; static uint8_t* dIdx; static float* dYf;
static int M=2048,IC=4096,OC=4096; static cublasHandle_t H;
void rmar(){ marlin2<<<dim3(OC/BN,M/BM),dim3(32,NWARP)>>>(dX,dIdx,dCb,dYf,M,IC,OC); }
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
    printf("prefill M=%d  Marlin2_pipelined %.3f ms (%.1f TFLOP/s)  cuBLAS %.3f ms (%.1f TFLOP/s)  ratio=%.2f  relerr=%.3g\n",
           M,tt,flop/(tt*1e-3)/1e12,tc,flop/(tc*1e-3)/1e12,tc/tt,me/den);
    return 0;
}
