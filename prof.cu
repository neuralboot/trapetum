// prof.cu -- minimal harness to profile ONLY fused_gemv_splitk under ncu.
// nvcc -O3 -arch=sm_86 prof.cu -o prof ; ncu --kernel-name fused_gemv_splitk ./prof
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <random>
#ifndef K_CLUSTERS
#define K_CLUSTERS 256
#endif
#define TX 32
#define TY 8
__global__ void fused_gemv_splitk(const __half* __restrict__ X, const uint8_t* __restrict__ idx,
                                  const __half* __restrict__ cb, __half* __restrict__ Y,
                                  int M, int IC, int OC){
    extern __shared__ __half sm[];
    __half* s_cb = sm;
    __half* sX   = sm + (size_t)K_CLUSTERS*TX;
    const int tx=threadIdx.x, ty=threadIdx.y;
    const int j0=blockIdx.x*TX, j=j0+tx;
    const int tid=ty*TX+tx, nth=TX*TY;
    for (int t=tid; t<K_CLUSTERS*TX; t+=nth){ const int k=t/TX, jj=j0+(t%TX); s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]); }
    for (int t=tid; t<IC; t+=nth) sX[t]=__ldg(&X[t]);
    __syncthreads();
    float acc=0.f;
    if (j<OC) for (int ic=ty; ic<IC; ic+=TY){
        const uint8_t id=__ldg(&idx[(size_t)ic*OC+j]);
        acc += __half2float(sX[ic]) * __half2float(s_cb[id*TX+tx]);
    }
    __shared__ float red[TY*TX];
    red[ty*TX+tx]=acc; __syncthreads();
    if (ty==0 && j<OC){ float s=0.f;
        #pragma unroll
        for(int y=0;y<TY;y++) s+=red[y*TX+tx]; Y[j]=__float2half(s); }
}
int main(){
    const int IC=4096,OC=4096,K=K_CLUSTERS,M=1;
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0,K-1); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> hi((size_t)IC*OC); for(auto&v:hi)v=(uint8_t)ui(r);
    std::vector<__half> hc((size_t)K*OC); for(auto&v:hc)v=__float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)M*IC); for(auto&v:hx)v=__float2half(nf(r));
    uint8_t*di; __half*dc,*dx,*dy;
    cudaMalloc(&di,hi.size()); cudaMalloc(&dc,hc.size()*2); cudaMalloc(&dx,hx.size()*2); cudaMalloc(&dy,(size_t)M*OC*2);
    cudaMemcpy(di,hi.data(),hi.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dc,hc.data(),hc.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dx,hx.data(),hx.size()*2,cudaMemcpyHostToDevice);
    dim3 bl(TX,TY), gr(OC/TX);
    size_t smem=((size_t)K_CLUSTERS*TX + (size_t)IC)*sizeof(__half);
    for(int i=0;i<3;i++) fused_gemv_splitk<<<gr,bl,smem>>>(dx,di,dc,dy,M,IC,OC);
    cudaDeviceSynchronize();
    return 0;
}
