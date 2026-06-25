// gemv_codebook_hopper_v3.cu  --  H100 tuning of the 4-bit fused GEMV.
// Two knobs vs v2: (1) -DUSE_L2_CB gathers the (tiny, 128 KB) codebook straight
// from L2 instead of re-staging it in shared on every split-K block (saves the
// ~GS-redundant staging traffic; H100 has 50 MB L2). (2) X is staged once in shared.
// Sweep GS and the codebook mode to find the H100 optimum.
//
// VERDICT (measured, H100): BOTH knobs regress vs v2. L2 codebook = x0.41-0.60
// (remote gather beats nothing; shared staging wins even on H100's 50 MB L2).
// X-staging = x0.90-0.91 (more shared use -> lower occupancy -> slower than v2's
// x0.99). v2 (shared codebook, no X-staging, GS=20) stays the best. Recorded as a
// negative result: the easy levers are exhausted; beating cuBLAS on H100 needs a
// different design (atomic-free reduction, TMA/wgmma), not these tweaks.
// Build: nvcc -O3 -arch=sm_90 -DGS=16 [-DUSE_L2_CB] gemv_codebook_hopper_v3.cu -lcublas -o gh3
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define K 16
#define CPB 256
#define TY 8
#ifndef GS
#define GS 16
#endif
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__global__ void gemv4(const __half* __restrict__ X, const uint8_t* __restrict__ packed,
                      const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC){
    extern __shared__ char sm[];
    __half* sX = (__half*)sm;                       // IC  (X staged once)
#ifndef USE_L2_CB
    __half* s_cb = sX + IC;                          // K*CPB
    float*  red  = (float*)(s_cb + K*CPB);
#else
    float*  red  = (float*)(sX + IC);
#endif
    const int tx=threadIdx.x, ty=threadIdx.y, tid=ty*32+tx, nth=32*TY;
    const int j0=blockIdx.x*CPB;
    for(int t=tid;t<IC;t+=nth) sX[t]=__ldg(&X[t]);
#ifndef USE_L2_CB
    for(int t=tid;t<K*CPB;t+=nth){ int k=t/CPB, jj=j0+(t%CPB); s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]); }
#endif
    __syncthreads();
    const int per=(IC+gridDim.y-1)/gridDim.y, ic0=blockIdx.y*per, ic1=min(IC,ic0+per);
    const int jbase=j0+tx*8; const size_t OCp=OC/2;
    float acc[8]={0,0,0,0,0,0,0,0};
    for(int ic=ic0+ty; ic<ic1; ic+=TY){
        uint32_t f=__ldg((const uint32_t*)&packed[(size_t)ic*OCp + jbase/2]);
        float x=__half2float(sX[ic]);
        #pragma unroll
        for(int c=0;c<8;c++){ uint8_t id=(f>>(4*c))&0xF;
#ifdef USE_L2_CB
            acc[c]+=x*__half2float(__ldg(&cb[(size_t)id*OC + jbase+c]));
#else
            acc[c]+=x*__half2float(s_cb[id*CPB + tx*8+c]);
#endif
        }
    }
    #pragma unroll
    for(int c=0;c<8;c++) red[ty*CPB+tx*8+c]=acc[c];
    __syncthreads();
    if(ty==0){
        #pragma unroll
        for(int c=0;c<8;c++){ float s=0; for(int y=0;y<TY;y++) s+=red[y*CPB+tx*8+c]; atomicAdd(&Yacc[j0+tx*8+c],s); }
    }
}
__global__ void dequant(const uint8_t* packed,const __half* cb,__half* W,int IC,int OC){
    int j=blockIdx.x*blockDim.x+threadIdx.x, i=blockIdx.y;
    if(j<OC){ uint8_t b=packed[(size_t)i*(OC/2)+j/2]; uint8_t id=(j&1)?(b>>4):(b&0xF); W[(size_t)i*OC+j]=cb[(size_t)id*OC+j]; }
}

static __half *dX,*dW,*dCb,*dYc; static uint8_t* dPk; static float* dYf;
static int M=1, IC=4096, OC=4096; static cublasHandle_t H; static size_t SMEM;
void rk(){ cudaMemset(dYf,0,(size_t)OC*4); gemv4<<<dim3(OC/CPB,GS),dim3(32,TY),SMEM>>>(dX,dPk,dCb,dYf,IC,OC); }
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
#ifdef USE_L2_CB
    SMEM=(size_t)IC*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
#else
    SMEM=(size_t)(IC + K*CPB)*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
#endif
    CK(cudaFuncSetAttribute(gemv4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
    dequant<<<dim3(OC/256,IC),256>>>(dPk,dCb,dW,IC,OC); CK(cudaDeviceSynchronize());
    rk(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> yf((size_t)OC); CK(cudaMemcpy(yf.data(),dYf,(size_t)OC*4,cudaMemcpyDeviceToHost));
    rcb(); CK(cudaDeviceSynchronize());
    std::vector<__half> yc((size_t)OC); CK(cudaMemcpy(yc.data(),dYc,(size_t)OC*2,cudaMemcpyDeviceToHost));
    double me=0,den=0; for(int j=0;j<OC;j++){double a=yf[j],b=__half2float(yc[j]); me=fmax(me,fabs(a-b)); den=fmax(den,fabs(b));}
    float tk=tm(500,rk), tc=tm(500,rcb);
    printf("%.4f ms (%.0f GB/s)  cuBLAS %.4f ms  x%.2f  relerr=%.2g\n",
           tk, IC*OC*0.5/(tk*1e-3)/1e9, tc, tc/tk, me/den);
    return 0;
}
