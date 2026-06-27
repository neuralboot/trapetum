// prof8.cu -- fused codebook-dequant GEMM for PREFILL via Tensor Cores (wmma).
// Properly tiled: 64x64 block tile, 4 warps (2x2 of 32x32), K-loop staging X and
// the DEQUANTIZED W tile into shared, fp16 mma. Goal: approach cuBLAS fp16 (which
// is compute-bound near TC peak) while reading uint8 indices instead of fp16 W.
// nvcc -O3 -arch=sm_86 prof8.cu -lcublas -o prof8
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
#define BM 64
#define BN 64
#define BK 32
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// simple dequant to materialize dense W (for the cuBLAS baseline)
__global__ void dequant(const uint8_t* idx, const __half* cb, __half* W, int IC, int OC){
    int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y;
    if(j<OC) W[(size_t)i*OC+j]=cb[(size_t)idx[(size_t)i*OC+j]*OC+j];
}

// fused dequant -> Tensor Core GEMM. Y[M,OC]=X[M,IC]@W, W from indices+codebook.
__global__ void tc_gemm(const __half* __restrict__ X, const uint8_t* __restrict__ idx,
                        const __half* __restrict__ cb, float* __restrict__ Yf, int M, int IC, int OC){
    __shared__ __half sX[BM*BK];
    __shared__ __half sW[BK*BN];
    const int tid=threadIdx.y*32+threadIdx.x, warp=threadIdx.y;  // 4 warps
    const int warpM=warp/2, warpN=warp%2;                        // 2x2 grid of 32x32
    const int bm0=blockIdx.y*BM, bn0=blockIdx.x*BN;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][2];
    #pragma unroll
    for(int i=0;i<2;i++)for(int j=0;j<2;j++) wmma::fill_fragment(acc[i][j],0.f);
    for(int k0=0;k0<IC;k0+=BK){
        for(int t=tid;t<BM*BK;t+=128){ int mm=t/BK, kk=t%BK; sX[t]=__ldg(&X[(size_t)(bm0+mm)*IC+k0+kk]); }
        for(int t=tid;t<BK*BN;t+=128){ int kk=t/BN, nn=t%BN; uint8_t id=__ldg(&idx[(size_t)(k0+kk)*OC+bn0+nn]); sW[t]=__ldg(&cb[(size_t)id*OC+bn0+nn]); }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<BK;kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b[2];
            #pragma unroll
            for(int i=0;i<2;i++) wmma::load_matrix_sync(a[i], &sX[(warpM*32+i*16)*BK + kk], BK);
            #pragma unroll
            for(int j=0;j<2;j++) wmma::load_matrix_sync(b[j], &sW[kk*BN + warpN*32+j*16], BN);
            #pragma unroll
            for(int i=0;i<2;i++)for(int j=0;j<2;j++) wmma::mma_sync(acc[i][j],a[i],b[j],acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<2;i++)for(int j=0;j<2;j++){
        int rm=bm0+warpM*32+i*16, cn=bn0+warpN*32+j*16;
        wmma::store_matrix_sync(&Yf[(size_t)rm*OC+cn], acc[i][j], OC, wmma::mem_row_major);
    }
}

static __half *dX,*dW,*dCb,*dYc; static uint8_t* dIdx; static float* dYf;
static int M=2048,IC=4096,OC=4096; static cublasHandle_t H;
void rtc(){ tc_gemm<<<dim3(OC/BN,M/BM),dim3(32,4)>>>(dX,dIdx,dCb,dYf,M,IC,OC); }
void rcb(){ const float al=1,be=0; cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,OC,M,IC,&al,dW,CUDA_R_16F,OC,dX,CUDA_R_16F,IC,&be,dYc,CUDA_R_16F,OC,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); }
float tm(int n,void(*f)()){ cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();CK(cudaDeviceSynchronize());
    cudaEventRecord(a);for(int i=0;i<n;i++)f();cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms/n;}
int main(){
    cublasCreate(&H); cublasSetMathMode(H,CUBLAS_TENSOR_OP_MATH);
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0,255); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> hi((size_t)IC*OC); for(auto&v:hi)v=(uint8_t)ui(r);
    std::vector<__half> hc((size_t)256*OC); for(auto&v:hc)v=__float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)M*IC); for(auto&v:hx)v=__float2half(nf(r)*0.1f);
    CK(cudaMalloc(&dIdx,hi.size())); CK(cudaMalloc(&dCb,hc.size()*2)); CK(cudaMalloc(&dX,hx.size()*2));
    CK(cudaMalloc(&dW,(size_t)IC*OC*2)); CK(cudaMalloc(&dYc,(size_t)M*OC*2)); CK(cudaMalloc(&dYf,(size_t)M*OC*4));
    CK(cudaMemcpy(dIdx,hi.data(),hi.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCb,hc.data(),hc.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
    dequant<<<dim3(OC/256,IC),256>>>(dIdx,dCb,dW,IC,OC); CK(cudaDeviceSynchronize());
    rtc(); CK(cudaDeviceSynchronize()); rcb(); CK(cudaDeviceSynchronize());
    std::vector<float> yf((size_t)M*OC); std::vector<__half> yc((size_t)M*OC);
    CK(cudaMemcpy(yf.data(),dYf,yf.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(yc.data(),dYc,yc.size()*2,cudaMemcpyDeviceToHost));
    double me=0,den=0; for(size_t i=0;i<yf.size();i++){double a=yf[i],b=__half2float(yc[i]); me=fmax(me,fabs(a-b)); den=fmax(den,fabs(b));}
    double flop=2.0*M*IC*OC;
    float ttc=tm(30,rtc), tcb=tm(30,rcb);
    printf("prefill M=%d  fused_TC %.3f ms (%.1f TFLOP/s)   cuBLAS %.3f ms (%.1f TFLOP/s)   TC/cuBLAS=%.2f  relerr=%.3g\n",
           M, ttc, flop/(ttc*1e-3)/1e12, tcb, flop/(tcb*1e-3)/1e12, tcb/ttc, me/den);
    return 0;
}
