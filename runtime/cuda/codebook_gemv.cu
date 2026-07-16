// C-ABI wrapper around the fused 4-bit codebook decode GEMV, for the Rust runtime.
//
// A QLinear holds only the quantized weights (packed 4-bit indices + per-output
// codebook) resident on the GPU. Activations live in caller-owned DEVICE buffers, so
// layers chain on-device with no host<->device copy between them: the kernel writes f32,
// a cast kernel converts it to half for the next layer. Host side speaks f32 + u8.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

#define K 16
#define CPB 256
#define TY 8
#define GS 20
// K256 8-bit codebook path (S19 mixed precision): 256 entries, uint8 indices [IC,OC],
// CPB8=128 columns/block with a vectorized uint32 index read (4 indices/thread, a full
// cache line/warp). Codebook staged in 64 KB opt-in shared (K8*CPB8 halves). GS8=2 grid.y
// (the prototype's sweet spot: more grid.y re-stages the big codebook and loses).
#define K8 256
#define CPB8 128
#define GS8 2

// grid.y IC-split for the fused codebook GEMVs. GS>1 blocks reduce IC-slice partials with
// atomicAdd (scheduler-order-dependent -> run-to-run NONDETERMINISTIC). The two-stage fixed-order
// reduction (mode 2) is deterministic AND measured FASTER than atomics on BOTH backends (pod
// RTX 6000 Ada: DET=2 9.7-10.0 tok/s vs DET=0 9.4-9.5; Metal likewise), so it is the DEFAULT.
// TRAPETUM_DETERMINISTIC (read once, cached; mirrors the Metal backend):
//   0       = atomic (fast-path; nondeterministic)
//   1       = grid.y=1: one block per output element, no cross-block atomics (slow on small OC)
//   2/unset = two-stage fixed-order (gemv4_partial -> gemv_reduce), deterministic, keeps IC split
static int det_mode() {
    static int m = -1;
    if (m < 0) { const char* e = getenv("TRAPETUM_DETERMINISTIC"); m = (e && e[1] == 0 && e[0] >= '0' && e[0] <= '2') ? (e[0]-'0') : 2; }
    return m;
}
static int det_gs() { return det_mode() == 1 ? 1 : GS; } // mode 1 collapses grid.y; modes 0/2 keep GS
static int det_gs8() { return det_mode() == 1 ? 1 : GS8; } // K256 twin

// per-column quantization dither table (precomputed) and reserved dither seeds.
__device__ static const unsigned QZ_SEED0 = 0x33383838u, QZ_SEED1 = 0x44463341u;
__device__ static const unsigned char qz_dither_tbl[148] = {
140,35,22,1,91,219,63,73,11,15,20,16,21,81,203,116,205,13,166,153,32,215,42,152,
154,182,63,223,133,136,199,89,177,28,41,46,53,229,14,74,59,119,118,62,53,108,231,
77,247,53,229,167,45,157,49,200,143,155,107,198,131,159,155,83,141,56,2,97,104,180,
74,64,79,3,115,3,21,56,190,25,161,6,178,247,122,244,160,141,11,213,165,241,176,226,
227,154,64,73,96,24,20,37,251,127,24,6,77,51,169,174,204,129,211,216,240,111,203,96,
224,85,201,193,240,226,169,220,140,204,86,66,228,145,68,94,91,200,33,107,25,167,26,
220,205,193,71,212,81,202};

__global__ void __launch_bounds__(32*TY)
gemv4(const __half* __restrict__ X, const unsigned char* __restrict__ packed,
      const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm; float* red = (float*)(s_cb + K*CPB);
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32+tx, nth = 32*TY;
    int j0 = blockIdx.x*CPB;
    if (IC < 0 || OC < 0) { Yacc[0] += (float)qz_dither_tbl[(tx + QZ_SEED0 + QZ_SEED1) & 127] + (float)(QZ_SEED0 ^ QZ_SEED1); }
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

// Deterministic two-stage GEMV (TRAPETUM_DETERMINISTIC=2). Stage 1: identical to gemv4 but each
// grid.y block writes its IC-slice partial to its OWN row of Ypart (Ypart[blockIdx.y*OC + j]) --
// disjoint, no atomics. Stage 2 (gemv_reduce) sums the GS partial rows per column in FIXED
// blockIdx.y order -> bitwise-reproducible while keeping gemv4's GS-way IC split.
__global__ void __launch_bounds__(32*TY)
gemv4_partial(const __half* __restrict__ X, const unsigned char* __restrict__ packed,
              const __half* __restrict__ cb, float* __restrict__ Ypart, int IC, int OC) {
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
        for (int c = 0; c < 8; c++) { float s = 0; for (int y = 0; y < TY; y++) s += red[y*CPB+tx*8+c]; Ypart[(size_t)blockIdx.y*OC + (j0+tx*8+c)] = s; }
    }
}

// ============================================================================
// K256 8-bit codebook GEMV (S19 mixed precision). idx[IC,OC] uint8 (one index per
// element), cb[K8,OC] half. Each thread reads 4 contiguous indices as one uint32
// (a warp reads a full 128 B cache line) and looks up the codebook staged in 64 KB
// opt-in shared. Ported from kernels/gemv_codebook.cu (BEAT cuBLAS fp16 x1.09 on A40).
__global__ void __launch_bounds__(32*TY)
gemv8(const __half* __restrict__ X, const unsigned char* __restrict__ idx,
      const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm; float* red = (float*)(s_cb + K8*CPB8);
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32+tx, nth = 32*TY;
    int j0 = blockIdx.x*CPB8;
    for (int t = tid; t < K8*CPB8; t += nth) { int k = t/CPB8, jj = j0 + (t%CPB8); s_cb[t] = __ldg(&cb[(size_t)k*OC+jj]); }
    __syncthreads();
    int per = (IC+gridDim.y-1)/gridDim.y, ic0 = blockIdx.y*per, ic1 = min(IC, ic0+per);
    int jbase = j0 + tx*4;
    float acc[4] = {0,0,0,0};
    for (int ic = ic0+ty; ic < ic1; ic += TY) {
        unsigned f = __ldg((const unsigned*)&idx[(size_t)ic*OC + jbase]);
        float xx = __half2float(__ldg(&X[ic]));
        #pragma unroll
        for (int c = 0; c < 4; c++) { unsigned char id = (f>>(8*c))&0xFF; acc[c] += xx*__half2float(s_cb[id*CPB8+tx*4+c]); }
    }
    #pragma unroll
    for (int c = 0; c < 4; c++) red[ty*CPB8+tx*4+c] = acc[c];
    __syncthreads();
    if (ty == 0) {
        #pragma unroll
        for (int c = 0; c < 4; c++) { float s = 0; for (int y = 0; y < TY; y++) s += red[y*CPB8+tx*4+c]; atomicAdd(&Yacc[j0+tx*4+c], s); }
    }
}

// Deterministic two-stage twin of gemv8: each grid.y block writes its IC-slice partial to
// its own row of Ypart (disjoint, no atomics); gemv_reduce sums the GS8 rows in fixed order.
__global__ void __launch_bounds__(32*TY)
gemv8_partial(const __half* __restrict__ X, const unsigned char* __restrict__ idx,
              const __half* __restrict__ cb, float* __restrict__ Ypart, int IC, int OC) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm; float* red = (float*)(s_cb + K8*CPB8);
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32+tx, nth = 32*TY;
    int j0 = blockIdx.x*CPB8;
    for (int t = tid; t < K8*CPB8; t += nth) { int k = t/CPB8, jj = j0 + (t%CPB8); s_cb[t] = __ldg(&cb[(size_t)k*OC+jj]); }
    __syncthreads();
    int per = (IC+gridDim.y-1)/gridDim.y, ic0 = blockIdx.y*per, ic1 = min(IC, ic0+per);
    int jbase = j0 + tx*4;
    float acc[4] = {0,0,0,0};
    for (int ic = ic0+ty; ic < ic1; ic += TY) {
        unsigned f = __ldg((const unsigned*)&idx[(size_t)ic*OC + jbase]);
        float xx = __half2float(__ldg(&X[ic]));
        #pragma unroll
        for (int c = 0; c < 4; c++) { unsigned char id = (f>>(8*c))&0xFF; acc[c] += xx*__half2float(s_cb[id*CPB8+tx*4+c]); }
    }
    #pragma unroll
    for (int c = 0; c < 4; c++) red[ty*CPB8+tx*4+c] = acc[c];
    __syncthreads();
    if (ty == 0) {
        #pragma unroll
        for (int c = 0; c < 4; c++) { float s = 0; for (int y = 0; y < TY; y++) s += red[y*CPB8+tx*4+c]; Ypart[(size_t)blockIdx.y*OC + (j0+tx*4+c)] = s; }
    }
}

// Stage 2: Y[j] = sum over grid.y partials Ypart[g*OC + j] in FIXED g order (deterministic).
__global__ void gemv_reduce(const float* __restrict__ Ypart, float* __restrict__ Y, int OC, int GSy) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= OC) return;
    float s = 0;
    for (int g = 0; g < GSy; g++) s += Ypart[(size_t)g*OC + i];
    Y[i] = s;
}

__global__ void cast_f2h(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// RMSNorm: out = x / sqrt(mean(x^2)+eps) * w. One block, 256 threads (n up to a few k).
__global__ void rmsnorm_k(const __half* __restrict__ x, const float* __restrict__ w,
                          __half* __restrict__ out, int n, float eps) {
    __shared__ float red[256];
    int tid = threadIdx.x;
    float ss = 0;
    for (int i = tid; i < n; i += 256) { float v = __half2float(x[i]); ss += v*v; }
    red[tid] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float scale = rsqrtf(red[0]/n + eps);
    for (int i = tid; i < n; i += 256) out[i] = __float2half(__half2float(x[i]) * scale * w[i]);
}

// SwiGLU activation: out = silu(gate) * up, gate/up are f32 (kernel outputs), out fp16.
__global__ void silu_mul_k(const float* __restrict__ gate, const float* __restrict__ up,
                           __half* __restrict__ out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; float s = g / (1.f + expf(-g)); out[i] = __float2half(s * up[i]); }
}

// residual: h += delta (h fp16 stream, delta f32), in place.
__global__ void resadd_k(__half* __restrict__ h, const float* __restrict__ delta, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) h[i] = __float2half(__half2float(h[i]) + delta[i]);
}

// RoPE (HF Llama rotate-half): for each head and d in [0, head_dim/2), rotate the pair
// (x[d], x[d+head_dim/2]) by angle = pos * base^(-2d/head_dim).
__global__ void rope_k(__half* __restrict__ x, int pos, int n_heads, int head_dim, const float* __restrict__ inv_freq) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int half = head_dim/2;
    if (t >= n_heads*half) return;
    int h = t / half, d = t % half;
    float angle = (float)pos * inv_freq[d];   // freqs precomputed (scaling baked in: llama3/linear/default)
    float c = cosf(angle), s = sinf(angle);
    int i = h*head_dim + d, j = h*head_dim + d + half;
    float x0 = __half2float(x[i]), x1 = __half2float(x[j]);
    x[i] = __float2half(x0*c - x1*s);
    x[j] = __float2half(x1*c + x0*s);
}

// add a bias vector (f32) into an f32 accumulator: a[i] += b[i]
__global__ void vadd_k(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// Batch-1 decode attention. One block per query head, head_dim threads. Cache layout is
// [t][kv_head*head_dim + d]. GQA: kv_head = h / (n_heads/n_kv); MHA: n_kv == n_heads.
__global__ void attn_k(const __half* __restrict__ q, const __half* __restrict__ ck,
                       const __half* __restrict__ cv, __half* __restrict__ out,
                       int n_heads, int n_kv, int head_dim, int seqlen, float softcap) {
    int h = blockIdx.x;
    int kvh = h / (n_heads / n_kv);
    int d = threadIdx.x;                 // blockDim.x == head_dim
    extern __shared__ float smem[];
    float* red = smem;                   // head_dim
    float* scores = smem + head_dim;     // seqlen
    float qd = __half2float(q[h*head_dim + d]);
    float scale = rsqrtf((float)head_dim);
    for (int t = 0; t < seqlen; t++) {
        red[d] = qd * __half2float(ck[(size_t)t*n_kv*head_dim + kvh*head_dim + d]);
        __syncthreads();
        for (int s = head_dim/2; s > 0; s >>= 1) { if (d < s) red[d] += red[d+s]; __syncthreads(); }
        if (d == 0) { float sc = red[0] * scale; if (softcap > 0.f) sc = softcap * tanhf(sc / softcap); scores[t] = sc; }
        __syncthreads();
    }
    if (d == 0) {
        float mx = -1e30f;
        for (int t = 0; t < seqlen; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0;
        for (int t = 0; t < seqlen; t++) { scores[t] = expf(scores[t]-mx); sum += scores[t]; }
        for (int t = 0; t < seqlen; t++) scores[t] /= sum;
    }
    __syncthreads();
    float acc = 0;
    for (int t = 0; t < seqlen; t++)
        acc += scores[t] * __half2float(cv[(size_t)t*n_kv*head_dim + kvh*head_dim + d]);
    out[h*head_dim + d] = __float2half(acc);
}

// ============================================================================
// Batched (M-token) decode kernels for speculative decoding (verify K+1 tokens
// in one forward). M <= 4. Mirror the M=1 kernels above, reusing one weight read
// across the M columns. Validated bit-for-bit against the M=1 path in the Metal
// backend (check_mtile / check_attn_m / …); this is the CUDA twin.
// ============================================================================
#define MMAX 4

// Batched fused 4-bit decode GEMM: X[M][IC] -> Y[M][OC]. One packed-weight read
// per (ic) serves all M rows; that is what keeps the verify bandwidth-bound.
// M is a TEMPLATE parameter: with runtime M the acc[][] indexing defeated unrolling
// and spilled the accumulators to local memory, costing 5.5x (M=2) to 10.7x (M=4)
// the M=1 GEMV per call (measured on a 4090). Compile-time M keeps everything in
// registers; the verify then costs about the same weight read as one M=1 forward.
template<int M>
__global__ void __launch_bounds__(32*TY)
gemm_mtile_t(const __half* __restrict__ X, const unsigned char* __restrict__ packed,
             const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm;
    float* red = (float*)(s_cb + K*CPB);          // M*TY*CPB floats
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32+tx, nth = 32*TY;
    int j0 = blockIdx.x*CPB;
    for (int t = tid; t < K*CPB/2; t += nth) {
        int idx = t*2, k = idx/CPB, jj = j0 + (idx%CPB);
        *reinterpret_cast<__half2*>(&s_cb[idx]) = *reinterpret_cast<const __half2*>(&cb[(size_t)k*OC+jj]);
    }
    __syncthreads();
    int per = (IC+gridDim.y-1)/gridDim.y, ic0 = blockIdx.y*per, ic1 = min(IC, ic0+per);
    size_t OCp = OC/2;
    float acc[M][8];
    #pragma unroll
    for (int m = 0; m < M; m++)
        #pragma unroll
        for (int c = 0; c < 8; c++) acc[m][c] = 0.f;
    for (int ic = ic0+ty; ic < ic1; ic += TY) {
        unsigned f = __ldg((const unsigned*)&packed[(size_t)ic*OCp + (j0 + tx*8)/2]);
        float xx[M];
        #pragma unroll
        for (int m = 0; m < M; m++) xx[m] = __half2float(__ldg(&X[(size_t)m*IC + ic]));
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            unsigned char id = (f>>(4*c))&0xF;
            float w = __half2float(s_cb[id*CPB+tx*8+c]);
            #pragma unroll
            for (int m = 0; m < M; m++) acc[m][c] += xx[m]*w;
        }
    }
    #pragma unroll
    for (int m = 0; m < M; m++)
        #pragma unroll
        for (int c = 0; c < 8; c++) red[((size_t)m*TY+ty)*CPB + tx*8+c] = acc[m][c];
    __syncthreads();
    if (ty == 0) {
        #pragma unroll
        for (int m = 0; m < M; m++)
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                float s = 0; for (int y = 0; y < TY; y++) s += red[((size_t)m*TY+y)*CPB + tx*8+c];
                atomicAdd(&Yacc[(size_t)m*OC + j0+tx*8+c], s);
            }
    }
}

// Runtime-M fallback (kept for M values without a template instantiation).
__global__ void __launch_bounds__(32*TY)
gemm_mtile(const __half* __restrict__ X, const unsigned char* __restrict__ packed,
           const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC, int M) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm;
    float* red = (float*)(s_cb + K*CPB);          // M*TY*CPB floats
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32+tx, nth = 32*TY;
    int j0 = blockIdx.x*CPB;
    for (int t = tid; t < K*CPB/2; t += nth) {
        int idx = t*2, k = idx/CPB, jj = j0 + (idx%CPB);
        *reinterpret_cast<__half2*>(&s_cb[idx]) = *reinterpret_cast<const __half2*>(&cb[(size_t)k*OC+jj]);
    }
    __syncthreads();
    int per = (IC+gridDim.y-1)/gridDim.y, ic0 = blockIdx.y*per, ic1 = min(IC, ic0+per);
    size_t OCp = OC/2;
    float acc[MMAX][8];
    #pragma unroll
    for (int m = 0; m < MMAX; m++) for (int c = 0; c < 8; c++) acc[m][c] = 0.f;
    for (int ic = ic0+ty; ic < ic1; ic += TY) {
        unsigned f = __ldg((const unsigned*)&packed[(size_t)ic*OCp + (j0 + tx*8)/2]);
        float xx[MMAX];
        for (int m = 0; m < M; m++) xx[m] = __half2float(__ldg(&X[(size_t)m*IC + ic]));
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            unsigned char id = (f>>(4*c))&0xF;
            float w = __half2float(s_cb[id*CPB+tx*8+c]);
            for (int m = 0; m < M; m++) acc[m][c] += xx[m]*w;
        }
    }
    for (int m = 0; m < M; m++)
        #pragma unroll
        for (int c = 0; c < 8; c++) red[((size_t)m*TY+ty)*CPB + tx*8+c] = acc[m][c];
    __syncthreads();
    if (ty == 0) {
        for (int m = 0; m < M; m++)
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                float s = 0; for (int y = 0; y < TY; y++) s += red[((size_t)m*TY+y)*CPB + tx*8+c];
                atomicAdd(&Yacc[(size_t)m*OC + j0+tx*8+c], s);
            }
    }
}

// Batched RMSNorm: one block per row (grid.x = M), 256 threads.
__global__ void rmsnorm_m(const __half* __restrict__ x, const float* __restrict__ w,
                          __half* __restrict__ out, int n, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const __half* xr = x + (size_t)row*n; __half* outr = out + (size_t)row*n;
    __shared__ float red[256];
    float ss = 0; for (int i = tid; i < n; i += 256) { float v = __half2float(xr[i]); ss += v*v; }
    red[tid] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float scale = rsqrtf(red[0]/n + eps);
    for (int i = tid; i < n; i += 256) outr[i] = __float2half(__half2float(xr[i]) * scale * w[i]);
}

// Batched RoPE: row r rotated at absolute position base+r.
__global__ void rope_m(__half* __restrict__ x, int base, int n_heads, int head_dim,
                       const float* __restrict__ inv_freq, int M) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int half = head_dim/2, per_row = n_heads*half;
    if (t >= M*per_row) return;
    int row = t/per_row, r = t%per_row, h = r/half, d = r%half;
    float angle = (float)(base+row) * inv_freq[d];
    float c = cosf(angle), s = sinf(angle);
    int qdim = n_heads*head_dim;
    size_t i = (size_t)row*qdim + h*head_dim + d, j = i + half;
    float x0 = __half2float(x[i]), x1 = __half2float(x[j]);
    x[i] = __float2half(x0*c - x1*s);
    x[j] = __float2half(x1*c + x0*s);
}

// Batched causal decode attention: query row m attends over base+m+1 keys.
__global__ void attn_m(const __half* __restrict__ q, const __half* __restrict__ ck,
                       const __half* __restrict__ cv, __half* __restrict__ out,
                       int n_heads, int n_kv, int head_dim, int base, int M) {
    int h = blockIdx.x, kvh = h / (n_heads/n_kv), d = threadIdx.x;
    extern __shared__ float smem[];
    float* red = smem;                 // head_dim
    float* scores = smem + head_dim;   // base+M
    int qdim = n_heads*head_dim;
    float scale = rsqrtf((float)head_dim);
    for (int m = 0; m < M; m++) {
        int seqlen = base + m + 1;
        float qd = __half2float(q[(size_t)m*qdim + h*head_dim + d]);
        for (int t = 0; t < seqlen; t++) {
            red[d] = qd * __half2float(ck[(size_t)t*n_kv*head_dim + kvh*head_dim + d]);
            __syncthreads();
            for (int s = head_dim/2; s > 0; s >>= 1) { if (d < s) red[d] += red[d+s]; __syncthreads(); }
            if (d == 0) scores[t] = red[0] * scale;
            __syncthreads();
        }
        if (d == 0) {
            float mx = -1e30f; for (int t = 0; t < seqlen; t++) mx = fmaxf(mx, scores[t]);
            float sum = 0; for (int t = 0; t < seqlen; t++) { scores[t] = expf(scores[t]-mx); sum += scores[t]; }
            for (int t = 0; t < seqlen; t++) scores[t] /= sum;
        }
        __syncthreads();
        float acc = 0;
        for (int t = 0; t < seqlen; t++) acc += scores[t] * __half2float(cv[(size_t)t*n_kv*head_dim + kvh*head_dim + d]);
        out[(size_t)m*qdim + h*head_dim + d] = __float2half(acc);
        __syncthreads();
    }
}

// MLA (Multi-head Latent Attention, DeepSeek-V2/V3) decode attention, absorption form.
// Twin of the Metal `mla_attn` (validated there vs a CPU reference). The KV cache stores
// only a shared low-rank latent c_KV (dim d_c) + a decoupled RoPE key k_R (dim d_rope) per
// token, not per-head K/V. Per head: score = absorbed_q.c_KV (d_c) + q_R.k_R (d_rope);
// out latent = softmax-weighted sum of c_KV. A separate W_UV GEMM makes per-head values.
// One block per head, d_c threads (power of two; kv_lora_rank=512 in DeepSeek-V2).
__global__ void mla_attn(const __half* __restrict__ aq, const __half* __restrict__ qr,
                         const __half* __restrict__ ckv, const __half* __restrict__ kr,
                         __half* __restrict__ outl, int n_heads, int d_c, int d_rope, int seqlen, float scale) {
    int h = blockIdx.x, d = threadIdx.x;
    extern __shared__ float smem[];
    float* red = smem;               // d_c
    float* scores = smem + d_c;      // seqlen
    for (int t = 0; t < seqlen; t++) {
        red[d] = __half2float(aq[h*d_c + d]) * __half2float(ckv[(size_t)t*d_c + d]);
        __syncthreads();
        for (int s = d_c/2; s > 0; s >>= 1) { if (d < s) red[d] += red[d+s]; __syncthreads(); }
        if (d == 0) {
            float rp = 0;
            for (int r = 0; r < d_rope; r++) rp += __half2float(qr[h*d_rope + r]) * __half2float(kr[(size_t)t*d_rope + r]);
            scores[t] = (red[0] + rp) * scale;
        }
        __syncthreads();
    }
    if (d == 0) {
        float mx = -1e30f; for (int t = 0; t < seqlen; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0; for (int t = 0; t < seqlen; t++) { scores[t] = expf(scores[t]-mx); sum += scores[t]; }
        for (int t = 0; t < seqlen; t++) scores[t] /= sum;
    }
    __syncthreads();
    float acc = 0;
    for (int t = 0; t < seqlen; t++) acc += scores[t] * __half2float(ckv[(size_t)t*d_c + d]);
    outl[h*d_c + d] = __float2half(acc);
}

__global__ void saxpy_k(float* __restrict__ acc, const float* __restrict__ y, float alpha, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) acc[i] += alpha * y[i];
}

__global__ void gemv_fp16_k(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ic, int oc) {
    int o = blockIdx.x*blockDim.x + threadIdx.x;
    if (o >= oc) return;
    float acc = 0.f; const __half* row = W + (size_t)o*ic;
    for (int i = 0; i < ic; i++) acc += __half2float(row[i]) * __half2float(X[i]);
    Y[o] = acc;
}

// Batched per-head MLA absorption (deliverable H). out[h][o] = sum_i x[h][i]*W[(h*S+roff)*dc + o*co + i*ci].
__global__ void mla_absorb_k(const __half* __restrict__ x, const __half* __restrict__ W, __half* __restrict__ out,
                             int nh, int in_dim, int out_dim, int S, int dc, int roff, int co, int ci) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= nh*out_dim) return;
    int h = gid / out_dim, o = gid % out_dim;
    size_t wbase = (size_t)(h*S + roff)*dc + (size_t)o*co, xbase = (size_t)h*in_dim;
    float acc = 0.f;
    for (int i = 0; i < in_dim; i++) acc += __half2float(x[xbase + i]) * __half2float(W[wbase + (size_t)i*ci]);
    out[gid] = __float2half(acc);
}

__global__ void gelu_mul_k(const float* __restrict__ gate, const float* __restrict__ up, __half* __restrict__ out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float gg = 0.5f*g*(1.0f + tanhf(0.7978845608f*(g + 0.044715f*g*g*g)));
    out[i] = __float2half(gg * up[i]);
}

__global__ void resadd_h_k(__half* __restrict__ h, const __half* __restrict__ d, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) h[i] = __float2half(__half2float(h[i]) + __half2float(d[i]));
}

// Two-stage device argmax over the first n logits (greedy token selection). The
// decode loop picks the next token WITHOUT copying the full vocab (up to ~152k f32)
// to the host: only a single u32 comes back. Ties resolve to the SMALLEST index
// (strict >), matching the host argmax. Twin of the Metal argmax_stage1/stage2.
__global__ void argmax_stage1(const float* __restrict__ x, int n,
                              float* __restrict__ out_val, unsigned* __restrict__ out_idx) {
    __shared__ float sval[256];
    __shared__ unsigned sidx[256];
    int tid = threadIdx.x;
    unsigned stride = blockDim.x * gridDim.x;
    float bv = -INFINITY; unsigned bi = 0;
    for (unsigned i = blockIdx.x*blockDim.x + tid; i < (unsigned)n; i += stride) {
        float v = x[i];
        if (v > bv) { bv = v; bi = i; }   // strict > keeps the smallest index on a tie
    }
    sval[tid] = bv; sidx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid+s]; unsigned oi = sidx[tid+s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) { sval[tid] = ov; sidx[tid] = oi; }
        }
        __syncthreads();
    }
    if (tid == 0) { out_val[blockIdx.x] = sval[0]; out_idx[blockIdx.x] = sidx[0]; }
}

__global__ void argmax_stage2(const float* __restrict__ in_val, const unsigned* __restrict__ in_idx,
                              int np, unsigned* __restrict__ out) {
    __shared__ float sval[256];
    __shared__ unsigned sidx[256];
    int tid = threadIdx.x;
    float bv = -INFINITY; unsigned bi = 0xffffffffu;
    for (int i = tid; i < np; i += blockDim.x) {
        float v = in_val[i]; unsigned id = in_idx[i];
        if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; }
    }
    sval[tid] = bv; sidx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid+s]; unsigned oi = sidx[tid+s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) { sval[tid] = ov; sidx[tid] = oi; }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = sidx[0];
}

// One dedicated stream for all device ops, so the decode chain can be CUDA-graph
// captured (launches MUST be on the captured stream, else they are not recorded).
static cudaStream_t g_stream = 0;
static void ensure_stream() { if (!g_stream) cudaStreamCreate(&g_stream); }

// ============================================================================
// Additive vector-quantization (AQLM-style) decode GEMV for MoE routed experts.
// A group of AVQ_D consecutive input weights for output o is reconstructed as
//   W[o, g*AVQ_D + e] = scale[o] * sum_m C_m[ code_m[o,g] ][e]     (C_m is [AVQ_K, AVQ_D]).
// The dot <x_g, C_m[k]> is output-independent, so we build LUT[m][gt][k] once per
// group tile in shared, then  y[o] = scale[o] * sum_g sum_m LUT[m][gt][ code_m[o,g] ].
// The kernel reads CODES (1 byte each) not the dense weight: at M=2 that is 2 bits
// per weight, the whole point. Ported from kernels/avq_gemv3.cu (uint32-vectorized:
// each thread owns 4 consecutive outputs, reads their 4 codes as one uint32); M is a
// COMPILE-TIME template parameter (2 -> 2 bit, 3 -> 3 bit at AVQ_K=256, AVQ_D=8) so the
// accumulators and shared arrays stay register/statically sized. The per-output scale
// is applied to each block's PARTIAL before the atomicAdd, which is exact because
// scale[o]*(a+b) == scale[o]*a + scale[o]*b across the grid.y group-split.
//
// On-disk / device memory layout (shared law with model/cbka_format.py and the CBKA
// reader in lib.rs; rows = OC output channels, cols = IC input channels, ng = cols/AVQ_D):
//   codebooks CB : [M][AVQ_K][AVQ_D] f16, flat (m*AVQ_K + k)*AVQ_D + e
//   scales       : [rows] f16, one per output channel
//   indices      : [M][ng][rows] u8, flat (m*ng + g)*rows + o   (contiguous in o for the uint32 read)
#define AVQ_K   256
#define AVQ_D   8
#define AVQ_CPB 256
#define AVQ_GT  8      // groups per block (group tile); grid.y = ceil(ng / AVQ_GT)

template<int M>
__global__ void avq_gemv_t(const __half* __restrict__ X, const unsigned char* __restrict__ codes,
                           const __half* __restrict__ CB, const __half* __restrict__ scale,
                           float* __restrict__ Y, int IC, int OC) {
    int ng = IC / AVQ_D;
    int o = (blockIdx.x * AVQ_CPB + threadIdx.x) * 4;   // 4 outputs per thread
    int g0 = blockIdx.y * AVQ_GT;
    __shared__ __half s_CB[M*AVQ_K*AVQ_D];
    __shared__ float  s_LUT[M*AVQ_GT*AVQ_K];
    __shared__ __half s_x[AVQ_GT*AVQ_D];
    for (int t = threadIdx.x; t < M*AVQ_K*AVQ_D; t += AVQ_CPB) s_CB[t] = CB[t];
    for (int t = threadIdx.x; t < AVQ_GT*AVQ_D; t += AVQ_CPB) { int gg = g0 + t/AVQ_D; s_x[t] = (gg<ng) ? X[gg*AVQ_D + t%AVQ_D] : __float2half(0.f); }
    __syncthreads();
    for (int t = threadIdx.x; t < M*AVQ_GT*AVQ_K; t += AVQ_CPB) {   // LUT[m][gt][k] = <x_{g0+gt}, C_m[k]>
        int m = t/(AVQ_GT*AVQ_K), r = t%(AVQ_GT*AVQ_K), gt = r/AVQ_K, k = r%AVQ_K; float dd = 0;
        #pragma unroll
        for (int e = 0; e < AVQ_D; e++) dd += __half2float(s_x[gt*AVQ_D+e]) * __half2float(s_CB[(m*AVQ_K+k)*AVQ_D+e]);
        s_LUT[t] = dd;
    }
    __syncthreads();
    if (o < OC) {                                        // OC%4==0 (rows are %256), so o+3 < OC
        float a0=0,a1=0,a2=0,a3=0;
        #pragma unroll
        for (int gt = 0; gt < AVQ_GT; gt++) {
            int g = g0 + gt; if (g >= ng) break;
            #pragma unroll
            for (int m = 0; m < M; m++) {
                unsigned cc = *reinterpret_cast<const unsigned*>(&codes[((size_t)m*ng + g)*OC + o]); // 4 codes
                const float* L = &s_LUT[(m*AVQ_GT + gt)*AVQ_K];
                a0 += L[cc & 0xFF]; a1 += L[(cc>>8)&0xFF]; a2 += L[(cc>>16)&0xFF]; a3 += L[(cc>>24)&0xFF];
            }
        }
        atomicAdd(&Y[o],   __half2float(scale[o])   * a0);
        atomicAdd(&Y[o+1], __half2float(scale[o+1]) * a1);
        atomicAdd(&Y[o+2], __half2float(scale[o+2]) * a2);
        atomicAdd(&Y[o+3], __half2float(scale[o+3]) * a3);
    }
}

struct AvqLin { unsigned char* d_codes; __half* d_cb; __half* d_scale; int M, IC, OC; };

struct QLinear { unsigned char* d_packed; __half* d_cb; int IC, OC, K_; };

extern "C" {

// upload the quantized weights once; activations are external device buffers.
// k=16: nibble-packed 4-bit (IC*(OC/2) bytes, gemv4). k=256: 8-bit uint8 indices (IC*OC bytes, gemv8).
void* qlinear_create(const unsigned char* packed, const float* cb_f32, int IC, int OC, int k) {
    QLinear* q = (QLinear*)malloc(sizeof(QLinear));
    q->IC = IC; q->OC = OC; q->K_ = k;
    size_t np = (k == 256) ? (size_t)IC * OC : (size_t)IC * (OC/2);
    cudaMalloc(&q->d_packed, np);
    cudaMemcpy(q->d_packed, packed, np, cudaMemcpyHostToDevice);
    size_t ncb = (size_t)k * OC;
    __half* cb_h = (__half*)malloc(ncb*sizeof(__half));
    for (size_t i = 0; i < ncb; i++) cb_h[i] = __float2half(cb_f32[i]);
    cudaMalloc(&q->d_cb, ncb*sizeof(__half));
    cudaMemcpy(q->d_cb, cb_h, ncb*sizeof(__half), cudaMemcpyHostToDevice);
    free(cb_h);
    return q;
}

// The K256 codebook (K8*CPB8 halves) + red exceeds the 48 KB default dynamic-shared cap, so
// opt in to the larger limit once per kernel (idempotent; the attribute is per-function global).
static size_t gemv8_smem() { return (size_t)K8*CPB8*sizeof(__half) + (size_t)TY*CPB8*sizeof(float); }
static void ensure_gemv8_smem() {
    static bool once = false;
    if (once) return;
    int sm = (int)gemv8_smem();
    cudaFuncSetAttribute(gemv8, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaFuncSetAttribute(gemv8_partial, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    once = true;
}

// Reusable GS*OC f32 Ypart scratch for the deterministic two-stage GEMV (mode 2), grown on demand.
static float* g_ypart = nullptr;
static size_t g_ypart_cap = 0; // floats
static float* ypart(size_t need_floats) {
    if (need_floats > g_ypart_cap) {
        if (g_ypart) cudaFree(g_ypart);
        cudaMalloc(&g_ypart, need_floats * sizeof(float));
        g_ypart_cap = need_floats;
    }
    return g_ypart;
}

// d_x: device half (IC,), d_y: device f32 (OC,). No host copies; fully on-device.
void qlinear_forward_dev(void* handle, const void* d_x, void* d_y) {
    QLinear* q = (QLinear*)handle;
    ensure_stream();
    if (q->K_ == 256) {
        // K256 8-bit path: gemv8 (uint8 indices, CPB8=128, vectorized uint32 read). Same
        // determinism knob as gemv4: two-stage fixed-order by default, atomic under DET=0.
        ensure_gemv8_smem();
        size_t smem8 = gemv8_smem();
        int gy8 = det_gs8();
        if (det_mode() == 2 && gy8 > 1) {
            float* yp = ypart((size_t)gy8 * q->OC);
            dim3 grid(q->OC/CPB8, gy8), block(32, TY);
            gemv8_partial<<<grid, block, smem8, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, yp, q->IC, q->OC);
            int tpb = 256;
            gemv_reduce<<<(q->OC + tpb - 1)/tpb, tpb, 0, g_stream>>>(yp, (float*)d_y, q->OC, gy8);
            return;
        }
        cudaMemsetAsync(d_y, 0, (size_t)q->OC*sizeof(float), g_stream);
        dim3 grid(q->OC/CPB8, gy8), block(32, TY);
        gemv8<<<grid, block, smem8, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, (float*)d_y, q->IC, q->OC);
        return;
    }
    size_t smem = (size_t)K*CPB*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
    int gy = det_gs();
    if (det_mode() == 2 && gy > 1) {
        // Two-stage deterministic: partials to Ypart (no atomics), then fixed-order reduce.
        float* yp = ypart((size_t)gy * q->OC);
        dim3 grid(q->OC/CPB, gy), block(32, TY);
        gemv4_partial<<<grid, block, smem, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, yp, q->IC, q->OC);
        int tpb = 256;
        gemv_reduce<<<(q->OC + tpb - 1)/tpb, tpb, 0, g_stream>>>(yp, (float*)d_y, q->OC, gy);
        return;
    }
    cudaMemsetAsync(d_y, 0, (size_t)q->OC*sizeof(float), g_stream);
    dim3 grid(q->OC/CPB, gy), block(32, TY); // det_gs()=1 under mode 1 -> no atomics
    gemv4<<<grid, block, smem, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, (float*)d_y, q->IC, q->OC);
}

void qlinear_free(void* handle) {
    QLinear* q = (QLinear*)handle;
    cudaFree(q->d_packed); cudaFree(q->d_cb); free(q);
}

// --- AVQ (additive-codebook) linear, MoE routed experts --------------------------
// codes: [M][cols/AVQ_D][rows] u8; cb_f32: [M][AVQ_K][AVQ_D]; scale_f32: [rows]. Host
// speaks f32 for the codebook/scale (converted to half on the way in, like qlinear_create);
// the packed u8 indices are uploaded as-is. rows = OC, cols = IC.
void* avq_create(const unsigned char* codes, const float* cb_f32, const float* scale_f32,
                 int M, int rows, int cols) {
    AvqLin* q = (AvqLin*)malloc(sizeof(AvqLin));
    q->M = M; q->IC = cols; q->OC = rows;
    int ng = cols / AVQ_D;
    size_t nidx = (size_t)M * ng * rows;
    cudaMalloc(&q->d_codes, nidx);
    cudaMemcpy(q->d_codes, codes, nidx, cudaMemcpyHostToDevice);
    size_t ncb = (size_t)M * AVQ_K * AVQ_D;
    __half* cbh = (__half*)malloc(ncb * sizeof(__half));
    for (size_t i = 0; i < ncb; i++) cbh[i] = __float2half(cb_f32[i]);
    cudaMalloc(&q->d_cb, ncb * sizeof(__half));
    cudaMemcpy(q->d_cb, cbh, ncb * sizeof(__half), cudaMemcpyHostToDevice);
    free(cbh);
    __half* sh = (__half*)malloc((size_t)rows * sizeof(__half));
    for (int i = 0; i < rows; i++) sh[i] = __float2half(scale_f32[i]);
    cudaMalloc(&q->d_scale, (size_t)rows * sizeof(__half));
    cudaMemcpy(q->d_scale, sh, (size_t)rows * sizeof(__half), cudaMemcpyHostToDevice);
    free(sh);
    return q;
}

// d_x: device half (IC,), d_y: device f32 (OC,). Fully on-device, on g_stream.
void avq_forward_dev(void* handle, const void* d_x, void* d_y) {
    AvqLin* q = (AvqLin*)handle;
    ensure_stream();
    cudaMemsetAsync(d_y, 0, (size_t)q->OC * sizeof(float), g_stream);
    int ng = q->IC / AVQ_D;
    dim3 grid((q->OC + AVQ_CPB*4 - 1) / (AVQ_CPB*4), (ng + AVQ_GT - 1) / AVQ_GT);
    dim3 block(AVQ_CPB);
    const __half* X = (const __half*)d_x; float* Y = (float*)d_y;
    switch (q->M) {
        case 2: avq_gemv_t<2><<<grid, block, 0, g_stream>>>(X, q->d_codes, q->d_cb, q->d_scale, Y, q->IC, q->OC); break;
        case 3: avq_gemv_t<3><<<grid, block, 0, g_stream>>>(X, q->d_codes, q->d_cb, q->d_scale, Y, q->IC, q->OC); break;
        default: break; // only M in {2,3} instantiated; loader asserts this
    }
}

void avq_free(void* handle) {
    AvqLin* q = (AvqLin*)handle;
    cudaFree(q->d_codes); cudaFree(q->d_cb); cudaFree(q->d_scale); free(q);
}

// device buffer helpers
void* dev_alloc_half(int n) { void* p; cudaMalloc(&p, (size_t)n*sizeof(__half)); return p; }
void* dev_alloc_f32(int n)  { void* p; cudaMalloc(&p, (size_t)n*sizeof(float));  return p; }
void  dev_free(void* p)     { cudaFree(p); }

// upload host f32 to a device half buffer (one-time input)
void dev_upload_to_half(void* d_half, const float* x, int n) {
    __half* h = (__half*)malloc((size_t)n*sizeof(__half));
    for (int i = 0; i < n; i++) h[i] = __float2half(x[i]);
    cudaMemcpy(d_half, h, (size_t)n*sizeof(__half), cudaMemcpyHostToDevice);
    free(h);
}

// device cast f32 -> half (the inter-layer conversion, fully on-device)
void dev_cast_f32_to_half(void* d_half, const void* d_f32, int n) {
    ensure_stream();
    cast_f2h<<<(n+255)/256, 256, 0, g_stream>>>((const float*)d_f32, (__half*)d_half, n);
}

void dev_download_f32(float* x, const void* d_f32, int n) {
    cudaMemcpy(x, d_f32, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost);
}

// download a device half buffer to host f32
void dev_download_half(float* x, const void* d_half, int n) {
    __half* h = (__half*)malloc((size_t)n*sizeof(__half));
    cudaMemcpy(h, d_half, (size_t)n*sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) x[i] = __half2float(h[i]);
    free(h);
}

// upload host f32 to a device f32 buffer (e.g. an RMSNorm weight, one-time)
void dev_upload_f32(void* d_f32, const float* x, int n) {
    cudaMemcpy(d_f32, x, (size_t)n*sizeof(float), cudaMemcpyHostToDevice);
}

// transformer-block ops (all on g_stream, so the whole block is CUDA-graph capturable)
void op_rmsnorm(const void* x_half, const void* w_f32, void* out_half, int n, float eps) {
    ensure_stream();
    rmsnorm_k<<<1, 256, 0, g_stream>>>((const __half*)x_half, (const float*)w_f32, (__half*)out_half, n, eps);
}
void op_silu_mul(const void* gate_f32, const void* up_f32, void* out_half, int n) {
    ensure_stream();
    silu_mul_k<<<(n+255)/256, 256, 0, g_stream>>>((const float*)gate_f32, (const float*)up_f32, (__half*)out_half, n);
}
void op_residual_add(void* h_half, const void* delta_f32, int n) {
    ensure_stream();
    resadd_k<<<(n+255)/256, 256, 0, g_stream>>>((__half*)h_half, (const float*)delta_f32, n);
}

// attention-block ops
void op_rope(void* x_half, int pos, int n_heads, int head_dim, const void* inv_freq) {
    ensure_stream();
    int total = n_heads * (head_dim/2);
    rope_k<<<(total+127)/128, 128, 0, g_stream>>>((__half*)x_half, pos, n_heads, head_dim, (const float*)inv_freq);
}
void op_vadd(void* a_f32, const void* b_f32, int n) {
    ensure_stream();
    vadd_k<<<(n+255)/256, 256, 0, g_stream>>>((float*)a_f32, (const float*)b_f32, n);
}
// append a (n_kv*head_dim) fp16 vector to a [max_seq][n_kv*head_dim] cache at row `pos`
void op_cache_append(void* cache_half, const void* src_half, int pos, int dim) {
    ensure_stream();
    cudaMemcpyAsync((char*)cache_half + (size_t)pos*dim*sizeof(__half), src_half,
                    (size_t)dim*sizeof(__half), cudaMemcpyDeviceToDevice, g_stream);
}
void op_attn(const void* q_half, const void* ck_half, const void* cv_half, void* out_half,
             int n_heads, int n_kv, int head_dim, int seqlen, float softcap) {
    ensure_stream();
    size_t smem = ((size_t)head_dim + seqlen) * sizeof(float);
    attn_k<<<n_heads, head_dim, smem, g_stream>>>((const __half*)q_half, (const __half*)ck_half,
        (const __half*)cv_half, (__half*)out_half, n_heads, n_kv, head_dim, seqlen, softcap);
}

// --- MLA device prep (kills the per-layer host round-trips of MlaAttn::forward) ---
// f32-input RMSNorm (the q_a latent lives in a DevF32 GEMV output; no host round-trip).
__global__ void rmsnorm_f32_k(const float* __restrict__ x, const float* __restrict__ w,
                              __half* __restrict__ out, int n, float eps) {
    __shared__ float red[256];
    int tid = threadIdx.x;
    float ss = 0;
    for (int i = tid; i < n; i += 256) { float v = x[i]; ss += v*v; }
    red[tid] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float scale = rsqrtf(red[0]/n + eps);
    for (int i = tid; i < n; i += 256) out[i] = __float2half(x[i] * scale * w[i]);
}
// kv prep: RMSNorm the first dc of kvf (f32) -> ckv (half), and interleaved-rope the dr
// tail -> kr (half). One launch replaces two host norms + a host rope + two uploads.
__global__ void mla_kv_prep_k(const float* __restrict__ kvf, const float* __restrict__ w,
                              const float* __restrict__ inv_freq, __half* __restrict__ ckv,
                              __half* __restrict__ kr, int dc, int dr, int pos, float eps) {
    __shared__ float red[256];
    int tid = threadIdx.x;
    float ss = 0;
    for (int i = tid; i < dc; i += 256) { float v = kvf[i]; ss += v*v; }
    red[tid] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float scale = rsqrtf(red[0]/dc + eps);
    for (int i = tid; i < dc; i += 256) ckv[i] = __float2half(kvf[i] * scale * w[i]);
    for (int d = tid; d < dr/2; d += 256) {
        float ang = (float)pos * inv_freq[d];
        float c = cosf(ang), s = sinf(ang);
        float x0 = kvf[dc + 2*d], x1 = kvf[dc + 2*d + 1];
        kr[2*d]   = __float2half(x0*c - x1*s);
        kr[2*d+1] = __float2half(x1*c + x0*s);
    }
}
// q prep: split qf (f32, nh*(nope+dr)) into qnope (half, per-head first nope) and the
// interleaved-roped qr (half, per-head dr tail). One launch replaces a host per-head
// rope (thread-pool dispatch) + an extract loop + two uploads.
__global__ void mla_q_prep_k(const float* __restrict__ qf, const float* __restrict__ inv_freq,
                             __half* __restrict__ qnope, __half* __restrict__ qr,
                             int nh, int nope, int dr, int pos) {
    int hd = nope + dr;
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int total_n = nh*nope, total_r = nh*(dr/2);
    if (t < total_n) {
        int h = t / nope, i = t % nope;
        qnope[t] = __float2half(qf[h*hd + i]);
    } else if (t < total_n + total_r) {
        int u = t - total_n; int h = u / (dr/2), d = u % (dr/2);
        float ang = (float)pos * inv_freq[d];
        float c = cosf(ang), s = sinf(ang);
        float x0 = qf[h*hd + nope + 2*d], x1 = qf[h*hd + nope + 2*d + 1];
        qr[h*dr + 2*d]   = __float2half(x0*c - x1*s);
        qr[h*dr + 2*d+1] = __float2half(x1*c + x0*s);
    }
}
void op_rmsnorm_f32(const void* x_f32, const void* w_f32, void* out_half, int n, float eps) {
    ensure_stream();
    rmsnorm_f32_k<<<1, 256, 0, g_stream>>>((const float*)x_f32, (const float*)w_f32, (__half*)out_half, n, eps);
}
void op_mla_kv_prep(const void* kvf_f32, const void* w_f32, const void* inv_freq_f32,
                    void* ckv_half, void* kr_half, int dc, int dr, int pos, float eps) {
    ensure_stream();
    mla_kv_prep_k<<<1, 256, 0, g_stream>>>((const float*)kvf_f32, (const float*)w_f32,
        (const float*)inv_freq_f32, (__half*)ckv_half, (__half*)kr_half, dc, dr, pos, eps);
}
void op_mla_q_prep(const void* qf_f32, const void* inv_freq_f32, void* qnope_half, void* qr_half,
                   int nh, int nope, int dr, int pos) {
    ensure_stream();
    int total = nh*nope + nh*(dr/2);
    mla_q_prep_k<<<(total+255)/256, 256, 0, g_stream>>>((const float*)qf_f32,
        (const float*)inv_freq_f32, (__half*)qnope_half, (__half*)qr_half, nh, nope, dr, pos);
}

// --- batched (M-token) ops for speculative decoding -------------------------------
void qlinear_forward_m(void* handle, const void* d_x, void* d_y, int M) {
    QLinear* q = (QLinear*)handle;
    ensure_stream();
    cudaMemsetAsync(d_y, 0, (size_t)M*q->OC*sizeof(float), g_stream);
    size_t smem = (size_t)K*CPB*sizeof(__half) + (size_t)M*TY*CPB*sizeof(float);
    dim3 grid(q->OC/CPB, det_gs()), block(32, TY); // deterministic mode: grid.y=1, no atomics
    const __half* X = (const __half*)d_x; float* Y = (float*)d_y;
    // compile-time-M kernels keep the accumulators in registers (see gemm_mtile_t)
    switch (M) {
        case 1: gemm_mtile_t<1><<<grid, block, smem, g_stream>>>(X, q->d_packed, q->d_cb, Y, q->IC, q->OC); break;
        case 2: gemm_mtile_t<2><<<grid, block, smem, g_stream>>>(X, q->d_packed, q->d_cb, Y, q->IC, q->OC); break;
        case 3: gemm_mtile_t<3><<<grid, block, smem, g_stream>>>(X, q->d_packed, q->d_cb, Y, q->IC, q->OC); break;
        case 4: gemm_mtile_t<4><<<grid, block, smem, g_stream>>>(X, q->d_packed, q->d_cb, Y, q->IC, q->OC); break;
        default: gemm_mtile<<<grid, block, smem, g_stream>>>(X, q->d_packed, q->d_cb, Y, q->IC, q->OC, M); break;
    }
}
void op_rmsnorm_m(const void* x_half, const void* w_f32, void* out_half, int n, float eps, int M) {
    ensure_stream();
    rmsnorm_m<<<M, 256, 0, g_stream>>>((const __half*)x_half, (const float*)w_f32, (__half*)out_half, n, eps);
}
void op_rope_m(void* x_half, int base, int n_heads, int head_dim, const void* inv_freq, int M) {
    ensure_stream();
    int total = M * n_heads * (head_dim/2);
    rope_m<<<(total+127)/128, 128, 0, g_stream>>>((__half*)x_half, base, n_heads, head_dim, (const float*)inv_freq, M);
}
void op_cache_append_m(void* cache_half, const void* src_half, int base, int dim, int M) {
    ensure_stream();
    cudaMemcpyAsync((char*)cache_half + (size_t)base*dim*sizeof(__half), src_half,
                    (size_t)M*dim*sizeof(__half), cudaMemcpyDeviceToDevice, g_stream);
}
void op_attn_m(const void* q_half, const void* ck_half, const void* cv_half, void* out_half,
               int n_heads, int n_kv, int head_dim, int base, int M) {
    ensure_stream();
    size_t smem = ((size_t)head_dim + base + M) * sizeof(float);
    attn_m<<<n_heads, head_dim, smem, g_stream>>>((const __half*)q_half, (const __half*)ck_half,
        (const __half*)cv_half, (__half*)out_half, n_heads, n_kv, head_dim, base, M);
}

// MLA decode attention (DeepSeek-V2/V3). Twin of the Metal path; validated there.
void op_mla_attn(const void* aq, const void* qr, const void* ckv, const void* kr, void* outl,
                 int n_heads, int d_c, int d_rope, int seqlen, float scale) {
    ensure_stream();
    size_t smem = ((size_t)d_c + seqlen) * sizeof(float);
    mla_attn<<<n_heads, d_c, smem, g_stream>>>((const __half*)aq, (const __half*)qr,
        (const __half*)ckv, (const __half*)kr, (__half*)outl, n_heads, d_c, d_rope, seqlen, scale);
}

void op_saxpy(void* acc_f32, const void* y_f32, float alpha, int n) {
    ensure_stream();
    saxpy_k<<<(n+255)/256, 256, 0, g_stream>>>((float*)acc_f32, (const float*)y_f32, alpha, n);
}

void op_gemv_fp16(const void* w_half, const void* x_half, void* y_f32, int ic, int oc) {
    ensure_stream();
    gemv_fp16_k<<<(oc+255)/256, 256, 0, g_stream>>>((const __half*)w_half, (const __half*)x_half, (float*)y_f32, ic, oc);
}

void op_mla_absorb(const void* x_half, const void* w_half, void* out_half,
                   int nh, int in_dim, int out_dim, int S, int dc, int roff, int co, int ci) {
    ensure_stream();
    int total = nh*out_dim;
    mla_absorb_k<<<(total+255)/256, 256, 0, g_stream>>>((const __half*)x_half, (const __half*)w_half, (__half*)out_half,
        nh, in_dim, out_dim, S, dc, roff, co, ci);
}

void op_gelu_mul(const void* gate_f32, const void* up_f32, void* out_half, int n) {
    ensure_stream();
    gelu_mul_k<<<(n+255)/256, 256, 0, g_stream>>>((const float*)gate_f32, (const float*)up_f32, (__half*)out_half, n);
}

void op_resadd_h(void* h_half, const void* d_half, int n) {
    ensure_stream();
    resadd_h_k<<<(n+255)/256, 256, 0, g_stream>>>((__half*)h_half, (const __half*)d_half, n);
}

void dev_sync() { cudaDeviceSynchronize(); }

// Device argmax over the first n (real_vocab) logits. Returns the winning index.
// Enqueued on g_stream so it runs after the logits GEMV already queued there, then
// syncs and copies back a single u32. Scratch (partials + output) is allocated once
// for the max partial count and reused every token, so the greedy path never mallocs.
#define ARGMAX_NTG_MAX 1024
static float*    g_am_val = 0;
static unsigned* g_am_idx = 0;
static unsigned* g_am_out = 0;
unsigned dev_argmax(const void* d_logits, int n) {
    ensure_stream();
    const int tpb = 256;
    int ntg = (n + tpb - 1) / tpb;
    if (ntg > ARGMAX_NTG_MAX) ntg = ARGMAX_NTG_MAX;
    if (ntg < 1) ntg = 1;
    if (!g_am_val) {
        cudaMalloc(&g_am_val, (size_t)ARGMAX_NTG_MAX*sizeof(float));
        cudaMalloc(&g_am_idx, (size_t)ARGMAX_NTG_MAX*sizeof(unsigned));
        cudaMalloc(&g_am_out, sizeof(unsigned));
    }
    argmax_stage1<<<ntg, tpb, 0, g_stream>>>((const float*)d_logits, n, g_am_val, g_am_idx);
    argmax_stage2<<<1, tpb, 0, g_stream>>>(g_am_val, g_am_idx, ntg, g_am_out);
    unsigned h_out = 0;
    cudaMemcpyAsync(&h_out, g_am_out, sizeof(unsigned), cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
    return h_out;
}

// --- CUDA graph capture of the decode chain ---------------------------------------
// graph_begin() starts capturing on g_stream; run the chain (forward/cast queue onto
// g_stream); graph_end() instantiates a replayable graph; graph_launch() replays it
// with near-zero CPU launch overhead.
void graph_begin() {
    ensure_stream();
    cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal);
}

void* graph_end() {
    cudaGraph_t graph;
    cudaStreamEndCapture(g_stream, &graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    return (void*)exec;
}

void graph_launch(void* exec) {
    cudaGraphLaunch((cudaGraphExec_t)exec, g_stream);
}

void graph_free(void* exec) {
    cudaGraphExecDestroy((cudaGraphExec_t)exec);
}

} // extern "C"
