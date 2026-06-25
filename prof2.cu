// prof2.cu -- ablation profiling (no ncu counters needed): isolate each cost of
// the split-K GEMV by timing kernels that add one operation at a time.
//   k0 = read indices only (pure index streaming, 16 MB)
//   k1 = + codebook lookup (shared)
//   k2 = + X read & FMA (full split-K)
// nvcc -O3 -arch=sm_86 prof2.cu -o prof2
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#define K_CLUSTERS 256
#define TX 32
#define TY 8

// k0: index read only -> reduction. No codebook, no X.
__global__ void k0(const uint8_t* __restrict__ idx, __half* __restrict__ Y, int IC, int OC){
    const int tx=threadIdx.x, ty=threadIdx.y; const int j=blockIdx.x*TX+tx;
    float acc=0.f;
    if (j<OC) for(int ic=ty; ic<IC; ic+=TY) acc += (float)__ldg(&idx[(size_t)ic*OC+j]);
    __shared__ float red[TY*TX]; red[ty*TX+tx]=acc; __syncthreads();
    if(ty==0&&j<OC){ float s=0; for(int y=0;y<TY;y++) s+=red[y*TX+tx]; Y[j]=__float2half(s); }
}
// k1: index + codebook shared lookup (no X)
__global__ void k1(const uint8_t* __restrict__ idx, const __half* __restrict__ cb, __half* __restrict__ Y, int IC, int OC){
    extern __shared__ __half s_cb[]; const int tx=threadIdx.x, ty=threadIdx.y;
    const int j0=blockIdx.x*TX, j=j0+tx, tid=ty*TX+tx, nth=TX*TY;
    for(int t=tid;t<K_CLUSTERS*TX;t+=nth){ int k=t/TX, jj=j0+(t%TX); s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]); }
    __syncthreads();
    float acc=0.f;
    if(j<OC) for(int ic=ty; ic<IC; ic+=TY){ uint8_t id=__ldg(&idx[(size_t)ic*OC+j]); acc += __half2float(s_cb[id*TX+tx]); }
    __shared__ float red[TY*TX]; red[ty*TX+tx]=acc; __syncthreads();
    if(ty==0&&j<OC){ float s=0; for(int y=0;y<TY;y++) s+=red[y*TX+tx]; Y[j]=__float2half(s); }
}
// k2: full split-K
__global__ void k2(const __half* __restrict__ X, const uint8_t* __restrict__ idx, const __half* __restrict__ cb, __half* __restrict__ Y, int IC, int OC){
    extern __shared__ __half sm[]; __half* s_cb=sm; __half* sX=sm+(size_t)K_CLUSTERS*TX;
    const int tx=threadIdx.x, ty=threadIdx.y; const int j0=blockIdx.x*TX, j=j0+tx, tid=ty*TX+tx, nth=TX*TY;
    for(int t=tid;t<K_CLUSTERS*TX;t+=nth){ int k=t/TX, jj=j0+(t%TX); s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]); }
    for(int t=tid;t<IC;t+=nth) sX[t]=__ldg(&X[t]);
    __syncthreads();
    float acc=0.f;
    if(j<OC) for(int ic=ty; ic<IC; ic+=TY){ uint8_t id=__ldg(&idx[(size_t)ic*OC+j]); acc += __half2float(sX[ic])*__half2float(s_cb[id*TX+tx]); }
    __shared__ float red[TY*TX]; red[ty*TX+tx]=acc; __syncthreads();
    if(ty==0&&j<OC){ float s=0; for(int y=0;y<TY;y++) s+=red[y*TX+tx]; Y[j]=__float2half(s); }
}

static uint8_t*di; static __half*dc,*dx,*dy; static int IC=4096,OC=4096;
void r0(){ k0<<<dim3(OC/TX),dim3(TX,TY)>>>(di,dy,IC,OC); }
void r1(){ k1<<<dim3(OC/TX),dim3(TX,TY),K_CLUSTERS*TX*2>>>(di,dc,dy,IC,OC); }
void r2(){ k2<<<dim3(OC/TX),dim3(TX,TY),(K_CLUSTERS*TX+IC)*2>>>(dx,di,dc,dy,IC,OC); }
float tm(int n, void(*f)()){ cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); f(); cudaDeviceSynchronize();
    cudaEventRecord(a); for(int i=0;i<n;i++) f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); return ms/n; }
int main(){
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0,255); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> hi((size_t)IC*OC); for(auto&v:hi)v=(uint8_t)ui(r);
    std::vector<__half> hc((size_t)K_CLUSTERS*OC); for(auto&v:hc)v=__float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)IC); for(auto&v:hx)v=__float2half(nf(r));
    cudaMalloc(&di,hi.size()); cudaMalloc(&dc,hc.size()*2); cudaMalloc(&dx,hx.size()*2); cudaMalloc(&dy,(size_t)OC*2);
    cudaMemcpy(di,hi.data(),hi.size(),cudaMemcpyHostToDevice); cudaMemcpy(dc,hc.data(),hc.size()*2,cudaMemcpyHostToDevice); cudaMemcpy(dx,hx.data(),hx.size()*2,cudaMemcpyHostToDevice);
    double idxMB=(double)IC*OC/1e6;
    float t0=tm(500,r0), t1=tm(500,r1), t2=tm(500,r2);
    printf("ablation (idx=%.0f MB):\n", idxMB);
    printf("  k0 idx-only       %.4f ms   %.0f GB/s\n", t0, idxMB/1e3/(t0*1e-3));
    printf("  k1 +codebook LUT  %.4f ms   (+%.4f vs k0)\n", t1, t1-t0);
    printf("  k2 full split-K   %.4f ms   (+%.4f vs k1)\n", t2, t2-t1);
    return 0;
}
