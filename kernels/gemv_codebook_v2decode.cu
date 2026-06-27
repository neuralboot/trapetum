// gemv_codebook_v2decode.cu  --  Decode v2: ATOMIC-FREE 4-bit codebook GEMV.
//
// The H100 parity (x0.99) was pinned by the split-K atomic reduction + redundant
// codebook staging (measured). v2 removes the atomic with a two-pass reduction:
//   pass 1 (gemv4_partial): each split-K block writes its OWN disjoint slice of a
//           [GS, OC] partial buffer. No atomics, no contention.
//   pass 2 (reduce_partials): sum the GS partials per column into Y. Cheap.
// Plus __launch_bounds__ (helps occupancy) and a half2-vectorized codebook stage.
//
// Benchmarks cuBLAS fp16, the OLD atomic kernel, and the NEW two-pass kernel on the
// current GPU, verified for correctness. Build per arch:
//   nvcc -O3 -arch=sm_89 -DGS=20 gemv_codebook_v2decode.cu -lcublas -o gd2
//
// VERDICT (measured, H100 PCIe, GS=20): the two-pass is EXACTLY even with the atomic
// version (both x0.84 vs cuBLAS, 1.00x relative). The atomic reduction was NOT the
// bottleneck. __launch_bounds__ and the vectorized codebook stage also did not move
// it. The H100 parity gap is fundamental (inner-loop / occupancy bound; cuBLAS fp16
// GEMV is very well tuned), not the reduction. Recorded negative result: this
// redesign does not help. Closing the H100 gap needs a different path (byte_perm
// nibble unpack for inner-loop ILP, or a TMA / wgmma rewrite), not atomic-free.
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
#define GS 20
#endif
#define NTH (32*TY)
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// shared codebook staging, half2-vectorized (cb is contiguous in j)
__device__ __forceinline__ void stage_cb(__half* s_cb, const __half* cb, int j0, int OC, int tid) {
    for (int t = tid; t < K * CPB / 2; t += NTH) {
        int idx = t * 2, k = idx / CPB, jj = j0 + (idx % CPB);
        *reinterpret_cast<__half2*>(&s_cb[idx]) =
            *reinterpret_cast<const __half2*>(&cb[(size_t)k * OC + jj]);
    }
}

// OLD: split-K with atomic reduction (the baseline to beat)
__global__ void __launch_bounds__(NTH)
gemv4_atomic(const __half* __restrict__ X, const uint8_t* __restrict__ packed,
             const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm; float* red = (float*)(s_cb + K * CPB);
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 32 + tx;
    int j0 = blockIdx.x * CPB;
    stage_cb(s_cb, cb, j0, OC, tid); __syncthreads();
    int per = (IC + gridDim.y - 1) / gridDim.y, ic0 = blockIdx.y * per, ic1 = min(IC, ic0 + per);
    int jbase = j0 + tx * 8; size_t OCp = OC / 2;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int ic = ic0 + ty; ic < ic1; ic += TY) {
        uint32_t f = __ldg((const uint32_t*)&packed[(size_t)ic * OCp + jbase / 2]);
        float xx = __half2float(__ldg(&X[ic]));
        #pragma unroll
        for (int c = 0; c < 8; c++) { uint8_t id = (f >> (4 * c)) & 0xF; acc[c] += xx * __half2float(s_cb[id * CPB + tx * 8 + c]); }
    }
    #pragma unroll
    for (int c = 0; c < 8; c++) red[ty * CPB + tx * 8 + c] = acc[c];
    __syncthreads();
    if (ty == 0) {
        #pragma unroll
        for (int c = 0; c < 8; c++) { float s = 0; for (int y = 0; y < TY; y++) s += red[y * CPB + tx * 8 + c]; atomicAdd(&Yacc[j0 + tx * 8 + c], s); }
    }
}

// NEW pass 1: same compute, but write a disjoint partial slice. No atomic.
__global__ void __launch_bounds__(NTH)
gemv4_partial(const __half* __restrict__ X, const uint8_t* __restrict__ packed,
              const __half* __restrict__ cb, float* __restrict__ partials, int IC, int OC) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm; float* red = (float*)(s_cb + K * CPB);
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 32 + tx;
    int j0 = blockIdx.x * CPB;
    stage_cb(s_cb, cb, j0, OC, tid); __syncthreads();
    int per = (IC + gridDim.y - 1) / gridDim.y, ic0 = blockIdx.y * per, ic1 = min(IC, ic0 + per);
    int jbase = j0 + tx * 8; size_t OCp = OC / 2;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int ic = ic0 + ty; ic < ic1; ic += TY) {
        uint32_t f = __ldg((const uint32_t*)&packed[(size_t)ic * OCp + jbase / 2]);
        float xx = __half2float(__ldg(&X[ic]));
        #pragma unroll
        for (int c = 0; c < 8; c++) { uint8_t id = (f >> (4 * c)) & 0xF; acc[c] += xx * __half2float(s_cb[id * CPB + tx * 8 + c]); }
    }
    #pragma unroll
    for (int c = 0; c < 8; c++) red[ty * CPB + tx * 8 + c] = acc[c];
    __syncthreads();
    if (ty == 0) {
        #pragma unroll
        for (int c = 0; c < 8; c++) { float s = 0; for (int y = 0; y < TY; y++) s += red[y * CPB + tx * 8 + c];
            partials[(size_t)blockIdx.y * OC + j0 + tx * 8 + c] = s; } // unique slice
    }
}
// NEW pass 2: reduce GS partials per column into Y. Memory-bound, tiny.
__global__ void __launch_bounds__(256)
reduce_partials(const float* __restrict__ partials, float* __restrict__ Y, int OC, int gs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < OC) { float s = 0; for (int g = 0; g < gs; g++) s += partials[(size_t)g * OC + j]; Y[j] = s; }
}
__global__ void dequant(const uint8_t* packed, const __half* cb, __half* W, int IC, int OC) {
    int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (j < OC) { uint8_t b = packed[(size_t)i * (OC / 2) + j / 2]; uint8_t id = (j & 1) ? (b >> 4) : (b & 0xF); W[(size_t)i * OC + j] = cb[(size_t)id * OC + j]; }
}

static __half *dX, *dW, *dCb, *dYc; static uint8_t* dPk; static float *dYf, *dPart;
static int M = 1, IC = 4096, OC = 4096; static cublasHandle_t H; static size_t SMEM;
void run_atomic() { cudaMemset(dYf, 0, (size_t)OC * 4); gemv4_atomic<<<dim3(OC/CPB, GS), dim3(32,TY), SMEM>>>(dX, dPk, dCb, dYf, IC, OC); }
void run_twopass() {
    gemv4_partial<<<dim3(OC/CPB, GS), dim3(32,TY), SMEM>>>(dX, dPk, dCb, dPart, IC, OC);
    reduce_partials<<<(OC+255)/256, 256>>>(dPart, dYf, OC, GS);
}
void run_cublas() { const float a=1, b=0; cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, OC, M, IC, &a, dW, CUDA_R_16F, OC, dX, CUDA_R_16F, IC, &b, dYc, CUDA_R_16F, OC, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); }
float tm(int n, void(*f)()) { cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); f(); CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for (int i = 0; i < n; i++) f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms / n; }

int main() {
    cublasCreate(&H); cublasSetMathMode(H, CUBLAS_TENSOR_OP_MATH);
    std::mt19937 r(0); std::uniform_int_distribution<int> ui(0, K-1); std::normal_distribution<float> nf(0,1);
    std::vector<uint8_t> id4((size_t)IC*OC); for (auto& v : id4) v = (uint8_t)ui(r);
    std::vector<uint8_t> pk((size_t)IC*(OC/2));
    for (size_t ic = 0; ic < (size_t)IC; ic++) for (int j = 0; j < OC; j += 2) pk[ic*(OC/2)+j/2] = (id4[ic*OC+j]&0xF) | ((id4[ic*OC+j+1]&0xF)<<4);
    std::vector<__half> hc((size_t)K*OC); for (auto& v : hc) v = __float2half(nf(r)*0.05f);
    std::vector<__half> hx((size_t)M*IC); for (auto& v : hx) v = __float2half(nf(r));
    CK(cudaMalloc(&dPk, pk.size())); CK(cudaMalloc(&dCb, hc.size()*2)); CK(cudaMalloc(&dX, hx.size()*2));
    CK(cudaMalloc(&dW, (size_t)IC*OC*2)); CK(cudaMalloc(&dYc, (size_t)M*OC*2)); CK(cudaMalloc(&dYf, (size_t)OC*4));
    CK(cudaMalloc(&dPart, (size_t)GS*OC*4));
    CK(cudaMemcpy(dPk, pk.data(), pk.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCb, hc.data(), hc.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, hx.data(), hx.size()*2, cudaMemcpyHostToDevice));
    SMEM = (size_t)K*CPB*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
    CK(cudaFuncSetAttribute(gemv4_atomic, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
    CK(cudaFuncSetAttribute(gemv4_partial, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
    dequant<<<dim3(OC/256, IC), 256>>>(dPk, dCb, dW, IC, OC); CK(cudaDeviceSynchronize());
    // correctness: two-pass vs cuBLAS
    run_twopass(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> yf((size_t)OC); CK(cudaMemcpy(yf.data(), dYf, (size_t)OC*4, cudaMemcpyDeviceToHost));
    run_cublas(); CK(cudaDeviceSynchronize());
    std::vector<__half> yc((size_t)OC); CK(cudaMemcpy(yc.data(), dYc, (size_t)OC*2, cudaMemcpyDeviceToHost));
    double me = 0, den = 0; for (int j = 0; j < OC; j++) { double a = yf[j], b = __half2float(yc[j]); me = fmax(me, fabs(a-b)); den = fmax(den, fabs(b)); }
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    float tc = tm(500, run_cublas), ta = tm(500, run_atomic), tt = tm(500, run_twopass);
    printf("%s  GS=%d  (rel err two-pass vs cuBLAS = %.2g)\n", prop.name, GS, me/den);
    printf("  cuBLAS fp16 GEMV     : %.4f ms   (x1.00)\n", tc);
    printf("  OLD atomic split-K   : %.4f ms   (x%.2f vs cuBLAS)\n", ta, tc/ta);
    printf("  NEW two-pass (no atom): %.4f ms   (x%.2f vs cuBLAS,  %.2fx vs old)\n", tt, tc/tt, ta/tt);
    return 0;
}
