// gemv_codebook.cu -- production fused codebook-dequant GEMV for DECODE (M=1).
// BEATS cuBLAS fp16 dense on A40 (0.0562 ms vs 0.0612 ms, x1.09) by reading half
// the weight bytes (uint8 indices vs fp16 W) -- the real payoff of the quantization.
//
//   Y[j] = sum_ic X[ic] * codebook[ indices[ic,j], j ]
//   indices [IC,OC] uint8, codebook [K,OC] half, X [1,IC] half, Y [1,OC]
//
// The three decisions that made it win (each found by profiling-by-ablation):
//   1. VECTORIZED index read: each thread reads 4 contiguous indices as one
//      uint32 -> a warp reads 128 B = a full cache line (the index read was 84%
//      of the time, and was wasting 3/4 of every cache line at 1 byte/thread).
//   2. codebook in SHARED (opt-in 100 KB) -> the per-element lookup is a shared
//      access, not an L2 gather (an L2 codebook gather alone cost ~3x the kernel).
//   3. grid.y IC-split = 2 with ATOMIC reduction -> enough blocks (64) to cover
//      the 84 SMs WITHOUT over-staging the 64 KB codebook. More grid.y is WORSE:
//      the redundant codebook staging (re-loaded per grid.y block) dominates.
//
// Build: nvcc -O3 -arch=sm_86 gemv_codebook.cu -o gemv   (A40/sm_86; adjust arch)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef K_CLUSTERS
#define K_CLUSTERS 256        // 64..256 ; smaller K = smaller shared codebook = even faster
#endif
#define TY  8                 // in-block IC split
#define CPB 128               // columns per block (32 lanes * 4 cols/uint32)
#define GS  2                 // grid.y IC split (sweet spot on A40; tune per GPU/shape)

__global__ void fused_gemv_codebook(
        const __half*  __restrict__ X,        // [1, IC]
        const uint8_t* __restrict__ idx,      // [IC, OC]
        const __half*  __restrict__ cb,       // [K, OC]
        float*         __restrict__ Yacc,     // [OC]  (zeroed before launch; atomic target)
        int IC, int OC)
{
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm;                 // K_CLUSTERS * CPB
    float*  red  = (float*)(s_cb + K_CLUSTERS*CPB); // TY * CPB
    const int tx=threadIdx.x, ty=threadIdx.y, tid=ty*32+tx, nth=32*TY;
    const int j0=blockIdx.x*CPB;

    // stage this block's CPB columns' codebook into shared (once)
    for (int t=tid; t<K_CLUSTERS*CPB; t+=nth){ int k=t/CPB, jj=j0+(t%CPB); s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]); }
    __syncthreads();

    const int per=(IC+gridDim.y-1)/gridDim.y, ic0=blockIdx.y*per, ic1=min(IC,ic0+per);
    const int jbase=j0+tx*4;
    float acc[4]={0,0,0,0};
    for (int ic=ic0+ty; ic<ic1; ic+=TY){
        const uint32_t f=__ldg((const uint32_t*)&idx[(size_t)ic*OC+jbase]); // 4 indices, full-line coalesced
        const float x=__half2float(__ldg(&X[ic]));
        #pragma unroll
        for(int c=0;c<4;c++){ uint8_t id=(f>>(8*c))&0xFF; acc[c]+=x*__half2float(s_cb[id*CPB + tx*4+c]); }
    }
    #pragma unroll
    for(int c=0;c<4;c++) red[ty*CPB+tx*4+c]=acc[c];
    __syncthreads();
    if (ty==0){
        #pragma unroll
        for(int c=0;c<4;c++){ float s=0; for(int y=0;y<TY;y++) s+=red[y*CPB+tx*4+c]; atomicAdd(&Yacc[j0+tx*4+c],s); }
    }
}

// host launcher: zero Yacc, set opt-in shared, launch. (Convert Yacc->half after.)
void launch_gemv_codebook(const __half* X, const uint8_t* idx, const __half* cb,
                          float* Yacc, int IC, int OC, cudaStream_t s=0){
    static bool once=false;
    const size_t SM = (size_t)K_CLUSTERS*CPB*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
    if(!once){ cudaFuncSetAttribute(fused_gemv_codebook,
                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM); once=true; }
    cudaMemsetAsync(Yacc, 0, (size_t)OC*sizeof(float), s);
    dim3 grid(OC/CPB, GS), block(32, TY);
    fused_gemv_codebook<<<grid, block, SM, s>>>(X, idx, cb, Yacc, IC, OC);
}
