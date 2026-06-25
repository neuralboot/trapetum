// gemv_codebook_4bit.cu -- 4-bit (K<=16) fused codebook-dequant GEMV, decode (M=1).
// x2.34 vs cuBLAS fp16 dense on A40 (0.0261 ms vs 0.0612 ms): 4-bit packing halves
// the dominant index traffic (8.4 MB vs 17 MB for uint8) -- bits-per-index is the lever.
//
//   Y[j] = sum_ic X[ic] * codebook[ idx4[ic,j], j ]   ; idx4 in [0,16), packed 2/byte
//   packed[ic, j/2] = idx4[ic,2j] | (idx4[ic,2j+1] << 4)   ; codebook [16, OC] half
//
// Design (same family as gemv_codebook.cu, taken further by the bit width):
//   - a thread reads uint32 = 8 nibbles = 8 cols ; a warp reads 128 B = full cache line
//     spanning 256 cols (CPB) -> peak-efficiency index streaming.
//   - K=16 codebook for 256 cols = 8 KB shared (tiny) -> high occupancy, cheap re-stage.
//   - grid.y IC-split (GS) + atomic float reduction covers all SMs. Sweet spot GS~20-24.
// Build: nvcc -O3 -arch=sm_86 gemv_codebook_4bit.cu -o gemv4
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#define K   16
#define TY  8
#define GS  20      // grid.y IC split (A40 sweet spot; tune per GPU/shape)
#define CPB 256     // 32 lanes * 8 cols (uint32 = 8 nibbles)

__global__ void fused_gemv_codebook_4bit(
        const __half* __restrict__ X, const uint8_t* __restrict__ packed,
        const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC){
    extern __shared__ char sm[];
    __half* s_cb=(__half*)sm; float* red=(float*)(s_cb + K*CPB);
    const int tx=threadIdx.x, ty=threadIdx.y, tid=ty*32+tx, nth=32*TY;
    const int j0=blockIdx.x*CPB;
    for(int t=tid;t<K*CPB;t+=nth){ int k=t/CPB, jj=j0+(t%CPB); s_cb[t]=__ldg(&cb[(size_t)k*OC+jj]); }
    __syncthreads();
    const int per=(IC+gridDim.y-1)/gridDim.y, ic0=blockIdx.y*per, ic1=min(IC,ic0+per);
    const int jbase=j0+tx*8; const size_t OCp=OC/2;
    float acc[8]={0,0,0,0,0,0,0,0};
    for(int ic=ic0+ty; ic<ic1; ic+=TY){
        uint32_t f=__ldg((const uint32_t*)&packed[(size_t)ic*OCp + jbase/2]); // 8 nibbles, full line
        float x=__half2float(__ldg(&X[ic]));
        #pragma unroll
        for(int c=0;c<8;c++){ uint8_t id=(f>>(4*c))&0xF; acc[c]+=x*__half2float(s_cb[id*CPB + tx*8+c]); }
    }
    #pragma unroll
    for(int c=0;c<8;c++) red[ty*CPB+tx*8+c]=acc[c];
    __syncthreads();
    if(ty==0){
        #pragma unroll
        for(int c=0;c<8;c++){ float s=0; for(int y=0;y<TY;y++) s+=red[y*CPB+tx*8+c]; atomicAdd(&Yacc[j0+tx*8+c],s); }
    }
}

void launch_gemv_codebook_4bit(const __half* X, const uint8_t* packed, const __half* cb,
                               float* Yacc, int IC, int OC, cudaStream_t s=0){
    static bool once=false; const size_t SM=(size_t)K*CPB*sizeof(__half)+(size_t)TY*CPB*sizeof(float);
    if(!once){ cudaFuncSetAttribute(fused_gemv_codebook_4bit,
                 cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SM); once=true; }
    cudaMemsetAsync(Yacc,0,(size_t)OC*sizeof(float),s);
    fused_gemv_codebook_4bit<<<dim3(OC/CPB,GS),dim3(32,TY),SM,s>>>(X,packed,cb,Yacc,IC,OC);
}
