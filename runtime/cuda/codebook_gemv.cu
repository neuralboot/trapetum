// C-ABI wrapper around the fused 4-bit codebook decode GEMV, for the Rust runtime.
// A QLinear holds the quantized weights (packed 4-bit indices + per-output codebook)
// resident on the GPU; forward() uploads only the activation vector and runs the kernel.
// Host side speaks f32 + u8; half conversion and all CUDA memory live here.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

#define K 16
#define CPB 256
#define TY 8
#define GS 20   // split-K groups (optimum for 4-bit)

__global__ void __launch_bounds__(32*TY)
gemv4(const __half* __restrict__ X, const unsigned char* __restrict__ packed,
      const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm; float* red = (float*)(s_cb + K*CPB);
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32+tx, nth = 32*TY;
    int j0 = blockIdx.x*CPB;
    for (int t = tid; t < K*CPB/2; t += nth) {
        int idx = t*2, k = idx/CPB, jj = j0 + (idx%CPB);
        *reinterpret_cast<__half2*>(&s_cb[idx]) = *reinterpret_cast<const __half2*>(&cb[(size_t)k*OC+jj]);
    }
    __syncthreads();
    int per = (IC+gridDim.y-1)/gridDim.y, ic0 = blockIdx.y*per, ic1 = min(IC, ic0+per);
    int jbase = j0 + tx*8; size_t OCp = OC/2;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int ic = ic0+ty; ic < ic1; ic += TY) {
        unsigned f = __ldg((const unsigned*)&packed[(size_t)ic*OCp + jbase/2]);
        float xx = __half2float(__ldg(&X[ic]));
        #pragma unroll
        for (int c = 0; c < 8; c++) { unsigned char id = (f>>(4*c))&0xF; acc[c] += xx*__half2float(s_cb[id*CPB+tx*8+c]); }
    }
    #pragma unroll
    for (int c = 0; c < 8; c++) red[ty*CPB+tx*8+c] = acc[c];
    __syncthreads();
    if (ty == 0) {
        #pragma unroll
        for (int c = 0; c < 8; c++) { float s = 0; for (int y = 0; y < TY; y++) s += red[y*CPB+tx*8+c]; atomicAdd(&Yacc[j0+tx*8+c], s); }
    }
}

struct QLinear {
    unsigned char* d_packed;  // (IC, OC/2)
    __half* d_cb;             // (K, OC)
    __half* d_x;              // (IC,)  device activation buffer
    float*  d_y;              // (OC,)  device output buffer
    int IC, OC;
};

extern "C" {

// packed: (IC*OC/2) bytes ; cb_f32: (K*OC) floats. Uploads weights to the GPU once.
void* qlinear_create(const unsigned char* packed, const float* cb_f32, int IC, int OC) {
    QLinear* q = (QLinear*)malloc(sizeof(QLinear));
    q->IC = IC; q->OC = OC;
    size_t np = (size_t)IC * (OC/2);
    cudaMalloc(&q->d_packed, np);
    cudaMemcpy(q->d_packed, packed, np, cudaMemcpyHostToDevice);
    size_t ncb = (size_t)K * OC;
    __half* cb_h = (__half*)malloc(ncb*sizeof(__half));
    for (size_t i = 0; i < ncb; i++) cb_h[i] = __float2half(cb_f32[i]);
    cudaMalloc(&q->d_cb, ncb*sizeof(__half));
    cudaMemcpy(q->d_cb, cb_h, ncb*sizeof(__half), cudaMemcpyHostToDevice);
    free(cb_h);
    cudaMalloc(&q->d_x, (size_t)IC*sizeof(__half));
    cudaMalloc(&q->d_y, (size_t)OC*sizeof(float));
    return q;
}

// x: (IC,) f32 in, y: (OC,) f32 out.
void qlinear_forward(void* handle, const float* x, float* y) {
    QLinear* q = (QLinear*)handle;
    __half* xh = (__half*)malloc((size_t)q->IC*sizeof(__half));
    for (int i = 0; i < q->IC; i++) xh[i] = __float2half(x[i]);
    cudaMemcpy(q->d_x, xh, (size_t)q->IC*sizeof(__half), cudaMemcpyHostToDevice);
    free(xh);
    cudaMemset(q->d_y, 0, (size_t)q->OC*sizeof(float));
    size_t smem = (size_t)K*CPB*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
    dim3 grid(q->OC/CPB, GS), block(32, TY);
    gemv4<<<grid, block, smem>>>(q->d_x, q->d_packed, q->d_cb, q->d_y, q->IC, q->OC);
    cudaMemcpy(y, q->d_y, (size_t)q->OC*sizeof(float), cudaMemcpyDeviceToHost);
}

void qlinear_free(void* handle) {
    QLinear* q = (QLinear*)handle;
    cudaFree(q->d_packed); cudaFree(q->d_cb); cudaFree(q->d_x); cudaFree(q->d_y);
    free(q);
}

} // extern "C"
