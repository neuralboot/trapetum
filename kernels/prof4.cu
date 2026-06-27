// prof4.cu -- vectorized index (uint32, full cache lines) + codebook in SHARED
// (opt-in 100KB). The two fixes combined. nvcc -O3 -arch=sm_86 prof4.cu -o prof4
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define TY 8
#define CPB 128   // cols per block (32 lanes * 4)
__global__ void k5(const __half* __restrict__ X, const uint8_t* __restrict__ idx,
                   const __half* __restrict__ cb, __half* __restrict__ Y, int IC, int OC){
    extern __shared__ char sm[];
    __half* s_cb=(__half*)sm;                 // 256*CPB
    __half* sX  = s_cb + 256*CPB;             // IC
    float*  red = (float*)(sX + IC);          // TY*CPB
    const int tx=threadIdx.x, ty=threadIdx.y, tid=ty*32+tx, nth=32*TY;
    const int j0=blockIdx.x*CPB;
    for(int t=tid;t<256*CPB;t+=nth){ int k=t/CPB, jj=j0+(t%CPB); s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]); }
    for(int t=tid;t<IC;t+=nth) sX[t]=__ldg(&X[t]);
    __syncthreads();
    const int jbase=j0+tx*4;
    float acc[4]={0,0,0,0};
    for(int ic=ty; ic<IC; ic+=TY){
        uint32_t f=__ldg((const uint32_t*)&idx[(size_t)ic*OC+jbase]);
        float x=__half2float(sX[ic]);
        #pragma unroll
        for(int c=0;c<4;c++){ uint8_t id=(f>>(8*c))&0xFF; acc[c]+=x*__half2float(s_cb[id*CPB + tx*4+c]); }
    }
    #pragma unroll
    for(int c=0;c<4;c++) red[ty*CPB+tx*4+c]=acc[c];
    __syncthreads();
    if(ty==0){
        #pragma unroll
        for(int c=0;c<4;c++){ float s=0; for(int y=0;y<TY;y++) s+=red[y*CPB+tx*4+c]; Y[j0+tx*4+c]=__float2half(s); }
    }
}
static uint8_t*di; static __half*dc,*dx,*dy; static int IC=4096,OC=4096; static size_t SM;
void r5(){ k5<<<dim3(OC/CPB),dim3(32,TY),SM>>>(dx,di,dc,dy,IC,OC); }
float tm(int n, void(*f)()){ cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); f(); cudaDeviceSynchronize();
    cudaEventRecord(a); for(int i=0;i<n;i++) f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); return ms/n; }
int main(){
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0,255); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> hi((size_t)IC*OC); for(auto&v:hi)v=(uint8_t)ui(r);
    std::vector<__half> hc((size_t)256*OC); for(auto&v:hc)v=__float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)IC); for(auto&v:hx)v=__float2half(nf(r));
    cudaMalloc(&di,hi.size()); cudaMalloc(&dc,hc.size()*2); cudaMalloc(&dx,hx.size()*2); cudaMalloc(&dy,(size_t)OC*2);
    cudaMemcpy(di,hi.data(),hi.size(),cudaMemcpyHostToDevice); cudaMemcpy(dc,hc.data(),hc.size()*2,cudaMemcpyHostToDevice); cudaMemcpy(dx,hx.data(),hx.size()*2,cudaMemcpyHostToDevice);
    SM=((size_t)256*CPB + IC)*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
    cudaFuncSetAttribute(k5, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM);
    printf("shared = %zu bytes, grid = %d blocks\n", SM, OC/CPB);
    r5(); cudaError_t e=cudaGetLastError(); if(e){ printf("launch err: %s\n", cudaGetErrorString(e)); return 1; }
    cudaDeviceSynchronize();
    std::vector<__half> hy((size_t)OC); cudaMemcpy(hy.data(),dy,(size_t)OC*2,cudaMemcpyDeviceToHost);
    double maxerr=0;
    for(int j=0;j<OC;j++){ float ref=0; for(int ic=0;ic<IC;ic++) ref+=__half2float(hx[ic])*__half2float(hc[(size_t)hi[(size_t)ic*OC+j]*OC+j]); maxerr=fmax(maxerr,fabs(ref-__half2float(hy[j]))); }
    double idxMB=(double)IC*OC/1e6;
    float t=tm(500,r5);
    printf("k5 vec-idx + shared-cb: %.4f ms   %.0f GB/s   max|err|=%.4g   x%.2f vs cuBLAS(0.0612)\n",
           t, idxMB/1e3/(t*1e-3), maxerr, 0.0612/t);
    return 0;
}
