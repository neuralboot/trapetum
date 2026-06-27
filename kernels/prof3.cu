// prof3.cu -- vectorized-index split-K GEMV (the fix from the ablation).
// Each thread reads 4 contiguous indices as one uint32 -> a warp reads 128 B =
// a full cache line. grid.y splits IC across blocks for SM coverage; partials
// atomic-added into a float accumulator. Codebook gathered from L2.
// nvcc -O3 -arch=sm_86 prof3.cu -o prof3
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define TY 8
#define GSPLIT 8           // IC split across blocks (occupancy / SM coverage)
#define COLS_PER_BLOCK 128 // 32 lanes * 4 cols

// vectorized index-only (measures the improved index-read bandwidth)
__global__ void k0v(const uint8_t* __restrict__ idx, float* __restrict__ Yacc, int IC, int OC){
    const int tx=threadIdx.x, ty=threadIdx.y;
    const int jbase=blockIdx.x*COLS_PER_BLOCK + tx*4;
    const int per=(IC+GSPLIT-1)/GSPLIT, ic0=blockIdx.y*per, ic1=min(IC,ic0+per);
    float acc[4]={0,0,0,0};
    for(int ic=ic0+ty; ic<ic1; ic+=TY){
        uint32_t f=__ldg((const uint32_t*)&idx[(size_t)ic*OC+jbase]);
        #pragma unroll
        for(int c=0;c<4;c++) acc[c]+=(float)((f>>(8*c))&0xFF);
    }
    __shared__ float red[TY][COLS_PER_BLOCK];
    #pragma unroll
    for(int c=0;c<4;c++) red[ty][tx*4+c]=acc[c];
    __syncthreads();
    if(ty==0){
        #pragma unroll
        for(int c=0;c<4;c++){ float s=0; for(int y=0;y<TY;y++) s+=red[y][tx*4+c]; atomicAdd(&Yacc[blockIdx.x*COLS_PER_BLOCK+tx*4+c],s);} }
}
// full vectorized split-K GEMV
__global__ void k4(const __half* __restrict__ X, const uint8_t* __restrict__ idx,
                   const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC){
    const int tx=threadIdx.x, ty=threadIdx.y;
    const int jbase=blockIdx.x*COLS_PER_BLOCK + tx*4;
    const int per=(IC+GSPLIT-1)/GSPLIT, ic0=blockIdx.y*per, ic1=min(IC,ic0+per);
    float acc[4]={0,0,0,0};
    for(int ic=ic0+ty; ic<ic1; ic+=TY){
        uint32_t f=__ldg((const uint32_t*)&idx[(size_t)ic*OC+jbase]);
        float x=__half2float(__ldg(&X[ic]));
        #pragma unroll
        for(int c=0;c<4;c++){ uint8_t id=(f>>(8*c))&0xFF; acc[c]+=x*__half2float(__ldg(&cb[(size_t)id*OC+jbase+c])); }
    }
    __shared__ float red[TY][COLS_PER_BLOCK];
    #pragma unroll
    for(int c=0;c<4;c++) red[ty][tx*4+c]=acc[c];
    __syncthreads();
    if(ty==0){
        #pragma unroll
        for(int c=0;c<4;c++){ float s=0; for(int y=0;y<TY;y++) s+=red[y][tx*4+c]; atomicAdd(&Yacc[blockIdx.x*COLS_PER_BLOCK+tx*4+c],s);} }
}

static uint8_t*di; static __half*dc,*dx; static float*dy; static int IC=4096,OC=4096;
dim3 blk(){ return dim3(32,TY); }
dim3 grd(){ return dim3(OC/COLS_PER_BLOCK, GSPLIT); }
void rv(){ cudaMemset(dy,0,(size_t)OC*4); k0v<<<grd(),blk()>>>(di,dy,IC,OC); }
void r4(){ cudaMemset(dy,0,(size_t)OC*4); k4<<<grd(),blk()>>>(dx,di,dc,dy,IC,OC); }
float tm(int n, void(*f)()){ cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); f(); cudaDeviceSynchronize();
    cudaEventRecord(a); for(int i=0;i<n;i++) f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); return ms/n; }
int main(){
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0,255); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> hi((size_t)IC*OC); for(auto&v:hi)v=(uint8_t)ui(r);
    std::vector<__half> hc((size_t)256*OC); for(auto&v:hc)v=__float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)IC); for(auto&v:hx)v=__float2half(nf(r));
    cudaMalloc(&di,hi.size()); cudaMalloc(&dc,hc.size()*2); cudaMalloc(&dx,hx.size()*2); cudaMalloc(&dy,(size_t)OC*4);
    cudaMemcpy(di,hi.data(),hi.size(),cudaMemcpyHostToDevice); cudaMemcpy(dc,hc.data(),hc.size()*2,cudaMemcpyHostToDevice); cudaMemcpy(dx,hx.data(),hx.size()*2,cudaMemcpyHostToDevice);
    // correctness of k4 vs CPU
    r4(); cudaDeviceSynchronize();
    std::vector<float> hy((size_t)OC); cudaMemcpy(hy.data(),dy,(size_t)OC*4,cudaMemcpyDeviceToHost);
    double maxerr=0;
    for(int j=0;j<OC;j++){ float ref=0; for(int ic=0;ic<IC;ic++) ref+=__half2float(hx[ic])*__half2float(hc[(size_t)hi[(size_t)ic*OC+j]*OC+j]); maxerr=fmax(maxerr,fabs(ref-hy[j])); }
    double idxMB=(double)IC*OC/1e6;
    float t0=tm(500,rv), t4=tm(500,r4);
    printf("VECTORIZED (uint32, full cache lines), idx=%.0f MB:\n", idxMB);
    printf("  k0v idx-only      %.4f ms   %.0f GB/s   (was 288 GB/s scalar)\n", t0, idxMB/1e3/(t0*1e-3));
    printf("  k4 full GEMV      %.4f ms   %.0f GB/s   max|err|=%.4g\n", t4, idxMB/1e3/(t4*1e-3), maxerr);
    printf("  cuBLAS dense ref  0.0612 ms   ->  k4 is x%.2f vs cuBLAS\n", 0.0612/t4);
    return 0;
}
