// prof7.cu -- 4-bit packed codebook GEMV (K=16). Indices packed 2/byte -> the
// dominant index traffic HALVES (8.5 MB vs 17 MB). A thread reads uint32 = 8
// nibbles = 8 columns; a warp reads 128 B = full line covering 256 columns.
// nvcc -O3 -arch=sm_86 -DGS=8 prof7.cu -o prof7
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define K 16
#ifndef TY
#define TY 8
#endif
#ifndef GS
#define GS 8
#endif
#define CPB 256   // 32 lanes * 8 cols (uint32 = 8 nibbles)
__global__ void k7(const __half* __restrict__ X, const uint8_t* __restrict__ packed,
                   const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC){
    extern __shared__ char sm[];
    __half* s_cb=(__half*)sm; float* red=(float*)(s_cb + K*CPB);
    const int tx=threadIdx.x, ty=threadIdx.y, tid=ty*32+tx, nth=32*TY;
    const int j0=blockIdx.x*CPB;
    for(int t=tid;t<K*CPB;t+=nth){ int k=t/CPB, jj=j0+(t%CPB); s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]); }
    __syncthreads();
    const int per=(IC+gridDim.y-1)/gridDim.y, ic0=blockIdx.y*per, ic1=min(IC,ic0+per);
    const int jbase=j0+tx*8;                       // 8 cols
    const size_t OCp=OC/2;                          // packed row stride (bytes)
    float acc[8]={0,0,0,0,0,0,0,0};
    for(int ic=ic0+ty; ic<ic1; ic+=TY){
        uint32_t f=__ldg((const uint32_t*)&packed[(size_t)ic*OCp + jbase/2]); // 8 nibbles
        float x=__half2float(__ldg(&X[ic]));
        #pragma unroll
        for(int c=0;c<8;c++){ uint8_t id=(f>>(4*c))&0xF; acc[c]+=x*__half2float(s_cb[id*CPB + tx*8+c]); }
    }
    #pragma unroll
    for(int c=0;c<8;c++) red[ty*CPB+tx*8+c]=acc[c];
    __syncthreads();
    if(ty==0){
        #pragma unroll
        for(int c=0;c<8;c++){ float s=0; for(int y=0;y<TY;y++) s+=red[y*CPB+tx*8+c]; atomicAdd(&Yacc[j0+tx*8+c],s);} }
}
static uint8_t*dp; static __half*dc,*dx; static float*dy; static int IC=4096,OC=4096; static size_t SM;
void r7(){ cudaMemset(dy,0,(size_t)OC*4); k7<<<dim3(OC/CPB,GS),dim3(32,TY),SM>>>(dx,dp,dc,dy,IC,OC); }
float tm(int n,void(*f)()){ cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);f();cudaDeviceSynchronize();
    cudaEventRecord(a);for(int i=0;i<n;i++)f();cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms/n;}
int main(){
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0,K-1); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> id4((size_t)IC*OC); for(auto&v:id4)v=(uint8_t)ui(r);        // 4-bit ids
    std::vector<uint8_t> pk((size_t)IC*(OC/2));                                       // packed 2/byte
    for(size_t ic=0;ic<(size_t)IC;ic++) for(int j=0;j<OC;j+=2) pk[ic*(OC/2)+j/2]=(id4[ic*OC+j]&0xF)|((id4[ic*OC+j+1]&0xF)<<4);
    std::vector<__half> hc((size_t)K*OC); for(auto&v:hc)v=__float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)IC); for(auto&v:hx)v=__float2half(nf(r));
    cudaMalloc(&dp,pk.size());cudaMalloc(&dc,hc.size()*2);cudaMalloc(&dx,hx.size()*2);cudaMalloc(&dy,(size_t)OC*4);
    cudaMemcpy(dp,pk.data(),pk.size(),cudaMemcpyHostToDevice);cudaMemcpy(dc,hc.data(),hc.size()*2,cudaMemcpyHostToDevice);cudaMemcpy(dx,hx.data(),hx.size()*2,cudaMemcpyHostToDevice);
    SM=(size_t)K*CPB*sizeof(__half)+(size_t)TY*CPB*sizeof(float);
    cudaFuncSetAttribute(k7,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SM);
    r7(); if(cudaGetLastError()){printf("GS=%d LAUNCH_ERR sh=%zu\n",GS,SM);return 0;}
    cudaDeviceSynchronize();
    std::vector<float> hy((size_t)OC);cudaMemcpy(hy.data(),dy,(size_t)OC*4,cudaMemcpyDeviceToHost);
    double me=0; for(int j=0;j<OC;j++){float ref=0;for(int ic=0;ic<IC;ic++)ref+=__half2float(hx[ic])*__half2float(hc[(size_t)id4[(size_t)ic*OC+j]*OC+j]);me=fmax(me,fabs(ref-hy[j]));}
    double idxMB=(double)IC*OC/2/1e6;  // packed bytes
    float t=tm(500,r7);
    printf("4-bit K=16 GS=%d : %.4f ms  %4.0f GB/s  x%.2f vs cuBLAS  err=%.3g  (idx=%.1fMB, sh=%zuKB)\n",
           GS,t,idxMB/1e3/(t*1e-3),0.0612/t,me,idxMB,SM/1024);
    return 0;
}
