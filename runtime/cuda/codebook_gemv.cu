// C-ABI wrapper around the fused 4-bit codebook decode GEMV, for the Rust runtime.
//
// A QLinear holds only the quantized weights (packed 4-bit indices + per-output
// codebook) resident on the GPU. Activations live in caller-owned DEVICE buffers, so
// layers chain on-device with no host<->device copy between them: the kernel writes f32,
// a cast kernel converts it to half for the next layer. Host side speaks f32 + u8.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Set once a CUDA graph has been captured: from then on, scratch buffers that a captured
// graph may reference are never freed (see ypart), only superseded.
static int g_graphs_captured = 0;
// Activation sparsity (TEAL-style, opt-in): set by set_sparse_rows() when a model loads activation
// thresholds; the 4-bit GEMVs then skip the index rows of zero inputs. Chosen at launch, so it
// must be set before any CUDA-graph capture.
static int g_sparse_rows = 0;

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
// SKIP (activation sparsity): read the input first and skip the row's indices when it is zero. A
// zero input adds exactly nothing, so the result is the same bits as the dense loop, which keeps
// its original load order when SKIP is false.
template <bool SKIP>
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
        unsigned f;
        float xx;
        if (SKIP) {
            xx = __half2float(__ldg(&X[ic]));
            if (xx == 0.f) continue;              // warp-uniform: the 32 lanes share the row
            f = __ldg((const unsigned*)&packed[(size_t)ic*OCp + jbase/2]);
        } else {
            f = __ldg((const unsigned*)&packed[(size_t)ic*OCp + jbase/2]);
            xx = __half2float(__ldg(&X[ic]));
        }
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

// ---- Fused decode GEMVs (bitwise identical to the separate kernels they replace) ------------
// Up to three K=16 projections that read the same input (q, k, v; or gate, up) run as ONE
// gemv4_partial launch over their concatenated output columns: a block finds its projection
// from its first column (every projection's OC is a multiple of CPB, so no block straddles
// two), then runs exactly gemv4_partial's code on that projection's indices and codebook. Each
// column therefore gets the same IC split, the same loop order and the same block reduction
// as before. The fused reduce kernels then sum the GS partial rows in the same fixed order as
// gemv_reduce and apply, in the same float operations, what the separate kernels did next:
// bias add and cast to half (q, k, v), SiLU(gate) * up (MLP), or the residual add (o, down).
// The projections are scalar __restrict__ arguments on purpose: a first version read them from
// a by-value segment table indexed at run time and was SLOWER than the separate kernels (same
// 40 registers, but +1.8 ms per token on an A40 and no gain on an RTX 4090, 2026-11-05); this
// form took the 4-bit decode from 6.36 to 5.28 ms per token on the 4090 with identical bits.
// An unused third projection repeats the second one with oc2 = 0 and is never selected.
template <bool SKIP>
__global__ void __launch_bounds__(32*TY)
gemv4_partial_seg(const __half* __restrict__ X,
                  const unsigned char* __restrict__ p0, const unsigned char* __restrict__ p1,
                  const unsigned char* __restrict__ p2,
                  const __half* __restrict__ c0, const __half* __restrict__ c1, const __half* __restrict__ c2,
                  int oc0, int oc1, int oc2, float* __restrict__ Ypart, int IC, int OCt) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm; float* red = (float*)(s_cb + K*CPB);
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32+tx, nth = 32*TY;
    int jg = blockIdx.x*CPB;                       // first column, concatenated space
    int s = jg < oc0 ? 0 : (jg < oc0 + oc1 ? 1 : 2);
    const unsigned char* packed = s == 0 ? p0 : (s == 1 ? p1 : p2);
    const __half* cb = s == 0 ? c0 : (s == 1 ? c1 : c2);
    int OC = s == 0 ? oc0 : (s == 1 ? oc1 : oc2);
    int j0 = jg - (s == 0 ? 0 : (s == 1 ? oc0 : oc0 + oc1));   // first column inside projection s
    for (int t = tid; t < K*CPB/2; t += nth) {
        int idx = t*2, k = idx/CPB, jj = j0 + (idx%CPB);
        *reinterpret_cast<__half2*>(&s_cb[idx]) = *reinterpret_cast<const __half2*>(&cb[(size_t)k*OC+jj]);
    }
    __syncthreads();
    int per = (IC+gridDim.y-1)/gridDim.y, ic0 = blockIdx.y*per, ic1 = min(IC, ic0+per);
    int jbase = j0 + tx*8; size_t OCp = OC/2;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int ic = ic0+ty; ic < ic1; ic += TY) {
        unsigned f;
        float xx;
        if (SKIP) {
            xx = __half2float(__ldg(&X[ic]));
            if (xx == 0.f) continue;              // warp-uniform: the 32 lanes share the row
            f = __ldg((const unsigned*)&packed[(size_t)ic*OCp + jbase/2]);
        } else {
            f = __ldg((const unsigned*)&packed[(size_t)ic*OCp + jbase/2]);
            xx = __half2float(__ldg(&X[ic]));
        }
        #pragma unroll
        for (int c = 0; c < 8; c++) { unsigned char id = (f>>(4*c))&0xF; acc[c] += xx*__half2float(s_cb[id*CPB+tx*8+c]); }
    }
    #pragma unroll
    for (int c = 0; c < 8; c++) red[ty*CPB+tx*8+c] = acc[c];
    __syncthreads();
    if (ty == 0) {
        #pragma unroll
        for (int c = 0; c < 8; c++) { float s2 = 0; for (int y = 0; y < TY; y++) s2 += red[y*CPB+tx*8+c]; Ypart[(size_t)blockIdx.y*OCt + (jg+tx*8+c)] = s2; }
    }
}
// q, k, v: reduce, add the optional bias (null pointer = none), cast to half, into three outputs.
__global__ void gemv_reduce_half3(const float* __restrict__ Ypart,
                                  __half* __restrict__ q, __half* __restrict__ k, __half* __restrict__ v,
                                  const float* __restrict__ bq, const float* __restrict__ bk,
                                  const float* __restrict__ bv, int oq, int ok, int OCt, int GSy) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= OCt) return;
    float s = 0;
    for (int g = 0; g < GSy; g++) s += Ypart[(size_t)g*OCt + i];
    if (i < oq) {
        if (bq) s += bq[i];                        // vadd_k: a[i] += b[i]
        q[i] = __float2half(s);                    // cast_f2h
    } else if (i < oq + ok) {
        int li = i - oq;
        if (bk) s += bk[li];
        k[li] = __float2half(s);
    } else {
        int li = i - oq - ok;
        if (bv) s += bv[li];
        v[li] = __float2half(s);
    }
}
// gate, up (two projections of n columns each): reduce both, then silu_mul_k's arithmetic.
__global__ void gemv_reduce_silu(const float* __restrict__ Ypart, __half* __restrict__ out, int n, int GSy, float thr) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t OCt = (size_t)2*n;
    float g = 0, u = 0;
    for (int y = 0; y < GSy; y++) g += Ypart[(size_t)y*OCt + i];
    for (int y = 0; y < GSy; y++) u += Ypart[(size_t)y*OCt + n + i];
    float s = g / (1.f + expf(-g));
    __half o = __float2half(s * u);
    if (thr > 0.f && fabsf(__half2float(o)) < thr) o = __float2half(0.f);   // activation sparsity
    out[i] = o;
}
// o and down: reduce, then resadd_k's arithmetic into the fp16 residual stream.
__global__ void gemv_reduce_resadd(const float* __restrict__ Ypart, __half* __restrict__ h, int OC, int GSy) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= OC) return;
    float s = 0;
    for (int g = 0; g < GSy; g++) s += Ypart[(size_t)g*OC + i];
    h[i] = __float2half(__half2float(h[i]) + s);
}

__global__ void cast_f2h(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// RMSNorm: out = x / sqrt(mean(x^2)+eps) * w. One block, 256 threads (n up to a few k).
// thr > 0 (activation sparsity): outputs whose fp16 value is below thr in magnitude become 0.
__global__ void rmsnorm_k(const __half* __restrict__ x, const float* __restrict__ w,
                          __half* __restrict__ out, int n, float eps, float thr) {
    __shared__ float red[256];
    int tid = threadIdx.x;
    float ss = 0;
    for (int i = tid; i < n; i += 256) { float v = __half2float(x[i]); ss += v*v; }
    red[tid] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float scale = rsqrtf(red[0]/n + eps);
    for (int i = tid; i < n; i += 256) {
        __half o = __float2half(__half2float(x[i]) * scale * w[i]);
        if (thr > 0.f && fabsf(__half2float(o)) < thr) o = __float2half(0.f);
        out[i] = o;
    }
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
// Shared by rope_k (position as a kernel argument) and rope_dk (position read from device
// memory, for the CUDA-graph decode), so both paths compute bit-identical results.
// One rotated pair (head h, d in [0, head_dim/2)): the arithmetic shared by rope_body and the
// fused rope_append kernel, so both write the same bits. `dst` (may equal x) receives the result.
__device__ __forceinline__ void rope_pair(const __half* x, __half* dst, int h, int d, int pos, int head_dim,
                                          const float* __restrict__ inv_freq) {
    int half = head_dim/2;
    float angle = (float)pos * inv_freq[d];   // freqs precomputed (scaling baked in: llama3/linear/default)
    float c = cosf(angle), s = sinf(angle);
    int i = h*head_dim + d, j = h*head_dim + d + half;
    float x0 = __half2float(x[i]), x1 = __half2float(x[j]);
    dst[i] = __float2half(x0*c - x1*s);
    dst[j] = __float2half(x1*c + x0*s);
}
__device__ __forceinline__ void rope_body(__half* __restrict__ x, int pos, int n_heads, int head_dim,
                                          const float* __restrict__ inv_freq) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int half = head_dim/2;
    if (t >= n_heads*half) return;
    rope_pair(x, x, t / half, t % half, pos, head_dim, inv_freq);
}
// Fused attention prologue, one launch instead of four: RoPE on q (in place) and on k (written
// to k and to the cache row), and the v row copied to its cache row. The position is `pos`, or
// d_tp[1] when d_tp is not null (CUDA-graph decode). Same arithmetic as rope_k + cache append.
__global__ void rope_append_k(__half* __restrict__ q, __half* __restrict__ k, const __half* __restrict__ v,
                              __half* __restrict__ cache_k, __half* __restrict__ cache_v, int pos_arg,
                              const int* __restrict__ d_tp, int n_heads, int n_kv, int head_dim,
                              const float* __restrict__ inv_freq) {
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int half = head_dim/2;
    int pos = d_tp ? d_tp[1] : pos_arg;
    int nq = n_heads*half, nk = n_kv*half, kvdim = n_kv*head_dim;
    if (t < nq) {
        rope_pair(q, q, t / half, t % half, pos, head_dim, inv_freq);
    } else if (t < nq + nk) {
        int u = t - nq, h = u / half, d = u % half;
        rope_pair(k, k, h, d, pos, head_dim, inv_freq);
        __half* row = cache_k + (size_t)pos*kvdim;
        int i = h*head_dim + d, j = i + half;
        row[i] = k[i];                            // the thread's own two writes: no race
        row[j] = k[j];
    } else if (t < nq + nk + kvdim) {
        int i = t - nq - nk;
        cache_v[(size_t)pos*kvdim + i] = v[i];
    }
}
__global__ void rope_k(__half* __restrict__ x, int pos, int n_heads, int head_dim, const float* __restrict__ inv_freq) {
    rope_body(x, pos, n_heads, head_dim, inv_freq);
}
// d_tp[0] = token id, d_tp[1] = position (written by the host before each graph launch).
__global__ void rope_dk(__half* __restrict__ x, const int* __restrict__ d_tp, int n_heads, int head_dim,
                        const float* __restrict__ inv_freq) {
    rope_body(x, d_tp[1], n_heads, head_dim, inv_freq);
}

// add a bias vector (f32) into an f32 accumulator: a[i] += b[i]
__global__ void vadd_k(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// Batch-1 decode attention. One block per query head, head_dim threads. Cache layout is
// [t][kv_head*head_dim + d]. GQA: kv_head = h / (n_heads/n_kv); MHA: n_kv == n_heads.
// The body is shared by attn_k (seqlen as an argument) and attn_dk (seqlen = position+1
// read from device memory, for the CUDA-graph decode): identical arithmetic on both paths.
__device__ __forceinline__ void attn_body(const __half* __restrict__ q, const __half* __restrict__ ck,
                                          const __half* __restrict__ cv, __half* __restrict__ out,
                                          int n_heads, int n_kv, int head_dim, int seqlen, float softcap);
__global__ void attn_k(const __half* __restrict__ q, const __half* __restrict__ ck,
                       const __half* __restrict__ cv, __half* __restrict__ out,
                       int n_heads, int n_kv, int head_dim, int seqlen, float softcap) {
    attn_body(q, ck, cv, out, n_heads, n_kv, head_dim, seqlen, softcap);
}
__global__ void attn_dk(const __half* __restrict__ q, const __half* __restrict__ ck,
                        const __half* __restrict__ cv, __half* __restrict__ out,
                        int n_heads, int n_kv, int head_dim, const int* __restrict__ d_tp, float softcap) {
    attn_body(q, ck, cv, out, n_heads, n_kv, head_dim, d_tp[1] + 1, softcap);
}
// Graph-decode helper: the KV-cache append at row d_tp[1].
__global__ void cache_append_dk(__half* __restrict__ cache, const __half* __restrict__ src,
                                const int* __restrict__ d_tp, int dim) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < dim) cache[(size_t)d_tp[1]*dim + i] = src[i];
}
__device__ __forceinline__ void attn_body(const __half* __restrict__ q, const __half* __restrict__ ck,
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

// ---- Split-KV batch-1 decode attention (the Flash-Decoding scheme) --------------------------
// attn_body walks the cache one position at a time in a single block per head (one dependent
// global load and two barriers per position), so its cost grows by about 0.65 us per cached
// position and layer. Here the cache is cut into chunks of ATTN_CHUNK positions: one block per
// (head, chunk) runs an online softmax with one warp per position stride and writes a partial
// (running max m, sum l, unnormalized output o), then attn_merge_k combines the chunks of a head.
// No atomics and fixed reduction orders, so runs are bitwise reproducible; the summation order
// differs from attn_body, so outputs move by fp32 rounding. TRAPETUM_ATTN=seq selects attn_body.
#define ATTN_CHUNK 64
#define ATTN_WARPS 4
static __host__ __device__ __forceinline__ int attn_chunks(int seqlen) { return (seqlen + ATTN_CHUNK - 1) / ATTN_CHUNK; }

// part layout: [n_heads][stride_chunks][2 + HD] floats = (m, l, o[HD]) per (head, chunk).
// The seqlen is seqlen_arg on the eager path, d_tp[1] + 1 on the CUDA-graph path (whose grid
// covers max_seq, so chunks past the current length exit at once).
template <int PER>   // PER = head_dim / 32 values per lane
__global__ void attn_split_k(const __half* __restrict__ q, const __half* __restrict__ ck,
                             const __half* __restrict__ cv, float* __restrict__ part,
                             int n_heads, int n_kv, int seqlen_arg, const int* __restrict__ d_tp,
                             float softcap, int stride_chunks) {
    const int HD = PER * 32;
    int seqlen = d_tp ? d_tp[1] + 1 : seqlen_arg;
    int h = blockIdx.x, c = blockIdx.y;
    int t0 = c * ATTN_CHUNK;
    if (t0 >= seqlen) return;                    // block-uniform: no barrier is skipped by part of a block
    int t1 = min(t0 + ATTN_CHUNK, seqlen);
    int kvh = h / (n_heads / n_kv);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float scale = rsqrtf((float)HD);
    float qv[PER], o[PER];
    #pragma unroll
    for (int i = 0; i < PER; i += 2) {
        float2 f = __half22float2(*(const __half2*)(q + (size_t)h*HD + lane*PER + i));
        qv[i] = f.x; qv[i + 1] = f.y;
        o[i] = 0.f; o[i + 1] = 0.f;
    }
    // finite start, as attn_body: expf(-1e30 - s) is exactly 0 for any finite score, a -inf score
    // gets weight 0 and a NaN score propagates, instead of the NaN that -inf - -inf would give
    float m = -1e30f, l = 0.f;
    const size_t stride = (size_t)n_kv * HD;
    const __half* kb = ck + (size_t)kvh*HD + lane*PER;
    const __half* vb = cv + (size_t)kvh*HD + lane*PER;
    for (int t = t0 + warp; t < t1; t += ATTN_WARPS) {
        // issue the V row load with the K row, before the reduction and the softcap branch,
        // so that both DRAM round trips overlap
        float vv[PER];
        #pragma unroll
        for (int i = 0; i < PER; i += 2) {
            float2 f = __half22float2(*(const __half2*)(vb + (size_t)t*stride + i));
            vv[i] = f.x; vv[i + 1] = f.y;
        }
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < PER; i += 2) {
            float2 f = __half22float2(*(const __half2*)(kb + (size_t)t*stride + i));
            s += qv[i] * f.x + qv[i + 1] * f.y;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;
        if (softcap > 0.f) s = softcap * tanhf(s / softcap);
        float mn = fmaxf(m, s);
        float corr = expf(m - mn);               // 0 at the warp's first position (m = -1e30)
        float p = expf(s - mn);
        l = l * corr + p;
        #pragma unroll
        for (int i = 0; i < PER; i++) o[i] = o[i] * corr + p * vv[i];
        m = mn;
    }
    // combine the warps of the block in a fixed order (a warp with no position keeps m = -1e30,
    // so its weight expf(-1e30 - M) is 0)
    __shared__ float sm_m[ATTN_WARPS], sm_l[ATTN_WARPS];
    __shared__ float sm_o[ATTN_WARPS][PER * 32];
    if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
    #pragma unroll
    for (int i = 0; i < PER; i++) sm_o[warp][lane*PER + i] = o[i];
    __syncthreads();
    float M = -INFINITY;
    #pragma unroll
    for (int w = 0; w < ATTN_WARPS; w++) M = fmaxf(M, sm_m[w]);
    float* dst = part + ((size_t)h * stride_chunks + c) * (HD + 2);
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float od = 0.f;
        #pragma unroll
        for (int w = 0; w < ATTN_WARPS; w++) if (sm_m[w] > -INFINITY) od += sm_o[w][d] * expf(sm_m[w] - M);
        dst[2 + d] = od;
    }
    if (threadIdx.x == 0) {
        float L = 0.f;
        #pragma unroll
        for (int w = 0; w < ATTN_WARPS; w++) if (sm_m[w] > -INFINITY) L += sm_l[w] * expf(sm_m[w] - M);
        dst[0] = M;
        dst[1] = L;
    }
}

// One block per head, one thread per output dimension: chunks in a fixed order (0, 1, 2, ...).
__global__ void attn_merge_k(const float* __restrict__ part, __half* __restrict__ out, int head_dim,
                             int seqlen_arg, const int* __restrict__ d_tp, int stride_chunks) {
    int seqlen = d_tp ? d_tp[1] + 1 : seqlen_arg;
    int nch = attn_chunks(seqlen);
    int h = blockIdx.x;
    const size_t rec = (size_t)head_dim + 2;
    const float* base = part + (size_t)h * stride_chunks * rec;
    float M = -INFINITY;
    for (int c = 0; c < nch; c++) M = fmaxf(M, base[c * rec]);
    float L = 0.f;
    for (int c = 0; c < nch; c++) L += base[c * rec + 1] * expf(base[c * rec] - M);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float od = 0.f;
        for (int c = 0; c < nch; c++) od += base[c * rec + 2 + d] * expf(base[c * rec] - M);
        out[(size_t)h * head_dim + d] = __float2half(od / L);
    }
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

// Warp-per-row fp16 GEMV (H-inc5). The thread-per-row kernel above is catastrophic for the
// skinny hidden-dim projections (q_a: oc=1536 -> 6 blocks on a 128-SM part, uncoalesced
// 22KB-strided reads, scalar loop): measured 768 us for a 22 MB read whose byte bound is
// ~25 us. Here each WARP owns one output row: lanes read consecutive float4s (8 halves,
// fully coalesced), then a shuffle reduction. Requires ic % 8 == 0 (all model dims are);
// the wrapper falls back to the scalar kernel otherwise.
__global__ void gemv_fp16_warp_k(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int ic, int oc) {
    int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= oc) return;
    const float4* row4 = (const float4*)(W + (size_t)warp*ic);
    const float4* x4 = (const float4*)X;
    int n8 = ic >> 3;                                   // 8 halves per float4
    float acc = 0.f;
    for (int i = lane; i < n8; i += 32) {
        float4 wv = row4[i], xv = x4[i];
        const __half2* wh = (const __half2*)&wv;
        const __half2* xh = (const __half2*)&xv;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 wf = __half22float2(wh[j]), xf = __half22float2(xh[j]);
            acc += wf.x*xf.x + wf.y*xf.y;
        }
    }
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) Y[warp] = acc;
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
    if (k == 256) {
        // gemv8 stages a K8 x CPB8 half codebook plus the TY x CPB8 f32 reduction in shared
        // memory (gemv8_smem(): 69,632 B per block), above what Turing (sm_75: 65,536 B) and
        // older GPUs allow. Refuse at load time: a launch over the limit fails silently and the
        // layer would return stale or zero outputs. The caller asserts a non-null handle.
        size_t need = (size_t)K8*CPB8*sizeof(__half) + (size_t)TY*CPB8*sizeof(float);
        int dev = 0, optin = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if ((size_t)optin < need) {
            fprintf(stderr, "trapetum: this model has an 8-bit (K=256) layer, whose kernel needs %zu B of shared "
                            "memory per block; this GPU allows %d B. Re-export the model with a 4-bit LM head "
                            "(model/export_runtime.py --lm-k 16).\n", need, optin);
            return nullptr;
        }
    }
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
    cudaError_t e1 = cudaFuncSetAttribute(gemv8, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaError_t e2 = cudaFuncSetAttribute(gemv8_partial, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        // qlinear_create already refuses GPUs below the limit; never decode with a kernel that cannot launch
        fprintf(stderr, "trapetum: cannot give gemv8 %d B of shared memory: %s\n", sm,
                cudaGetErrorString(e1 != cudaSuccess ? e1 : e2));
        abort();
    }
    once = true;
}

// Reusable GS*OC f32 Ypart scratch for the deterministic two-stage GEMV (mode 2), grown on demand.
static float* g_ypart = nullptr;
static size_t g_ypart_cap = 0; // floats
static float* ypart(size_t need_floats) {
    if (need_floats > g_ypart_cap) {
        // A captured graph keeps the old pointer in its kernel nodes: never free it once a
        // graph exists (the superseded buffer is simply retired, a bounded one-off cost).
        if (g_ypart && !g_graphs_captured) cudaFree(g_ypart);
        g_ypart = nullptr;
        cudaError_t e = cudaMalloc(&g_ypart, need_floats * sizeof(float));
        if (e != cudaSuccess) {
            fprintf(stderr, "trapetum: cudaMalloc(ypart, %zu floats) failed: %s\n", need_floats, cudaGetErrorString(e));
            abort();
        }
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
        if (g_sparse_rows) gemv4_partial<true><<<grid, block, smem, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, yp, q->IC, q->OC);
        else gemv4_partial<false><<<grid, block, smem, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, yp, q->IC, q->OC);
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
static void* dev_alloc_checked(size_t bytes, const char* what) {
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "trapetum: cudaMalloc(%zu B, %s) failed: %s\n", bytes, what, cudaGetErrorString(e));
        abort();
    }
    return p;
}
void* dev_alloc_half(int n) { return dev_alloc_checked((size_t)n*sizeof(__half), "half buffer"); }
void* dev_alloc_f32(int n)  { return dev_alloc_checked((size_t)n*sizeof(float), "f32 buffer"); }
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
    rmsnorm_k<<<1, 256, 0, g_stream>>>((const __half*)x_half, (const float*)w_f32, (__half*)out_half, n, eps, 0.f);
}
// RMSNorm whose small outputs (|fp16 value| < thr) become 0 (activation sparsity); thr 0 = op_rmsnorm.
void op_rmsnorm_thr(const void* x_half, const void* w_f32, void* out_half, int n, float eps, float thr) {
    ensure_stream();
    rmsnorm_k<<<1, 256, 0, g_stream>>>((const __half*)x_half, (const float*)w_f32, (__half*)out_half, n, eps, thr);
}
// In place: fp16 values below thr in magnitude become 0 (the o_proj input under activation sparsity).
__global__ void sparsify_half_k(__half* __restrict__ x, int n, float thr) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && fabsf(__half2float(x[i])) < thr) x[i] = __float2half(0.f);
}
void op_sparsify_half(void* x_half, int n, float thr) {
    if (thr <= 0.f) return;
    ensure_stream();
    sparsify_half_k<<<(n+255)/256, 256, 0, g_stream>>>((__half*)x_half, n, thr);
}
// Turn the row-skipping 4-bit GEMVs on (activation thresholds loaded) or off.
void set_sparse_rows(int on) { g_sparse_rows = on ? 1 : 0; }
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
// --- split-KV attention host side: one partial buffer shared by every attention call (calls
// are ordered on g_stream, so a layer's merge finishes before the next layer's split starts).
static float* g_attn_part = nullptr;
static size_t g_attn_part_cap = 0;   // floats
static int attn_mode() {             // 1: split-KV (default), 0: sequential attn_body (TRAPETUM_ATTN=seq)
    static int mode = -1;
    if (mode < 0) {
        const char* e = getenv("TRAPETUM_ATTN");
        mode = (e && e[0] == 's' && e[1] == 'e' && e[2] == 'q' && e[3] == 0) ? 0 : 1;
    }
    return mode;
}
static bool attn_split_ok(int head_dim) {
    return attn_mode() == 1 && (head_dim == 64 || head_dim == 128 || head_dim == 256);
}
// Make the buffer hold `need` floats, allocating `want` (>= need) when it grows. Never allocates
// during a stream capture (not allowed there) and never frees a buffer a captured graph may use.
static bool attn_part_fit(size_t need, size_t want) {
    if (need <= g_attn_part_cap) return true;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(g_stream, &st);
    if (st != cudaStreamCaptureStatusNone) return false;
    if (g_attn_part && !g_graphs_captured) cudaFree(g_attn_part);
    g_attn_part = nullptr;
    g_attn_part_cap = 0;
    if (cudaMalloc(&g_attn_part, want * sizeof(float)) != cudaSuccess) {
        g_attn_part = nullptr;
        cudaGetLastError();             // clear the (non-sticky) allocation error; callers fall back
        return false;
    }
    g_attn_part_cap = want;
    return true;
}
static void attn_split_launch(const void* q, const void* ck, const void* cv, void* out, int n_heads, int n_kv,
                              int head_dim, int seqlen_arg, const int* d_tp, float softcap, int grid_chunks) {
    dim3 grid(n_heads, grid_chunks), block(32 * ATTN_WARPS);
    const __half* qh = (const __half*)q;
    const __half* kh = (const __half*)ck;
    const __half* vh = (const __half*)cv;
    if (head_dim == 64)
        attn_split_k<2><<<grid, block, 0, g_stream>>>(qh, kh, vh, g_attn_part, n_heads, n_kv, seqlen_arg, d_tp, softcap, grid_chunks);
    else if (head_dim == 128)
        attn_split_k<4><<<grid, block, 0, g_stream>>>(qh, kh, vh, g_attn_part, n_heads, n_kv, seqlen_arg, d_tp, softcap, grid_chunks);
    else
        attn_split_k<8><<<grid, block, 0, g_stream>>>(qh, kh, vh, g_attn_part, n_heads, n_kv, seqlen_arg, d_tp, softcap, grid_chunks);
    attn_merge_k<<<n_heads, head_dim, 0, g_stream>>>(g_attn_part, (__half*)out, head_dim, seqlen_arg, d_tp, grid_chunks);
}
// Reserve the partial buffer for a block of max_seq positions, at model construction, so that
// neither the eager path nor a captured graph ever has to allocate while decoding.
void attn_reserve(int n_heads, int head_dim, int max_seq) {
    ensure_stream();
    if (!attn_split_ok(head_dim)) return;
    size_t need = (size_t)n_heads * attn_chunks(max_seq) * ((size_t)head_dim + 2);
    attn_part_fit(need, need);
}

void op_attn(const void* q_half, const void* ck_half, const void* cv_half, void* out_half,
             int n_heads, int n_kv, int head_dim, int seqlen, float softcap) {
    ensure_stream();
    if (attn_split_ok(head_dim)) {
        int nch = attn_chunks(seqlen);
        size_t need = (size_t)n_heads * nch * ((size_t)head_dim + 2);
        size_t want = (size_t)n_heads * attn_chunks(((seqlen + 1023) / 1024) * 1024) * ((size_t)head_dim + 2);
        if (attn_part_fit(need, want)) {
            attn_split_launch(q_half, ck_half, cv_half, out_half, n_heads, n_kv, head_dim, seqlen, nullptr, softcap, nch);
            return;
        }
    }
    size_t smem = ((size_t)head_dim + seqlen) * sizeof(float);
    if (smem > 48 * 1024) {              // the sequential kernel keeps every score in shared memory
        fprintf(stderr, "trapetum: sequential attention needs %zu B of shared memory for %d positions (limit 49152 B); "
                        "use the default split-KV kernel or a shorter context\n", smem, seqlen);
        abort();
    }
    attn_k<<<n_heads, head_dim, smem, g_stream>>>((const __half*)q_half, (const __half*)ck_half,
        (const __half*)cv_half, (__half*)out_half, n_heads, n_kv, head_dim, seqlen, softcap);
}

// --- CUDA-graph decode: the position comes from device memory (d_tp = [token, pos]), so one
// captured graph replays every decode step. Same kernels/arithmetic as the M=1 path. (The
// embedding row is uploaded into the residual buffer before each launch, as on the eager path.)
void op_rope_dk(void* x_half, const void* d_tp, int n_heads, int head_dim, const void* inv_freq) {
    ensure_stream();
    int total = n_heads * (head_dim/2);
    rope_dk<<<(total+127)/128, 128, 0, g_stream>>>((__half*)x_half, (const int*)d_tp, n_heads, head_dim,
                                                   (const float*)inv_freq);
}
void op_cache_append_dk(void* cache_half, const void* src_half, const void* d_tp, int dim) {
    ensure_stream();
    cache_append_dk<<<(dim+255)/256, 256, 0, g_stream>>>((__half*)cache_half, (const __half*)src_half,
                                                          (const int*)d_tp, dim);
}
// Shared memory is sized for max_seq (fixed at capture); the kernel reads seqlen = pos+1.
void op_attn_dk(const void* q_half, const void* ck_half, const void* cv_half, void* out_half,
                int n_heads, int n_kv, int head_dim, const void* d_tp, int max_seq, float softcap) {
    ensure_stream();
    if (attn_split_ok(head_dim)) {
        int nch = attn_chunks(max_seq);  // the captured grid covers max_seq; idle chunks exit at once
        size_t need = (size_t)n_heads * nch * ((size_t)head_dim + 2);
        if (attn_part_fit(need, need)) {
            attn_split_launch(q_half, ck_half, cv_half, out_half, n_heads, n_kv, head_dim, 0, (const int*)d_tp, softcap, nch);
            return;
        }
        static bool warned = false;      // only reachable if attn_reserve was skipped before a capture
        if (!warned) {
            fprintf(stderr, "trapetum: split-KV attention buffer not reserved before capture; using the sequential kernel\n");
            warned = true;
        }
    }
    size_t smem = ((size_t)head_dim + max_seq) * sizeof(float);
    if (smem > 48 * 1024) {              // the sequential kernel keeps every score in shared memory
        fprintf(stderr, "trapetum: sequential attention needs %zu B of shared memory for max_seq %d (limit 49152 B); "
                        "use the default split-KV kernel or a shorter context\n", smem, max_seq);
        abort();
    }
    attn_dk<<<n_heads, head_dim, smem, g_stream>>>((const __half*)q_half, (const __half*)ck_half,
        (const __half*)cv_half, (__half*)out_half, n_heads, n_kv, head_dim, (const int*)d_tp, softcap);
}
// Stream-ordered upload of a few ints (pageable source: staged before the call returns).
void dev_upload_i32(void* d, const int* x, int n) {
    ensure_stream();
    cudaMemcpyAsync(d, x, (size_t)n*sizeof(int), cudaMemcpyHostToDevice, g_stream);
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
    if (ic % 8 == 0) {
        // warp-per-row, coalesced float4 loads (H-inc5); 8 warps per 256-thread block.
        gemv_fp16_warp_k<<<(oc+7)/8, 256, 0, g_stream>>>((const __half*)w_half, (const __half*)x_half, (float*)y_f32, ic, oc);
    } else {
        gemv_fp16_k<<<(oc+255)/256, 256, 0, g_stream>>>((const __half*)w_half, (const __half*)x_half, (float*)y_f32, ic, oc);
    }
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
// Enqueue-only half (graph-capturable once the scratch exists; the eager dev_argmax that
// precedes a capture allocates it), and the read-back half (D2H + sync, never captured).
void dev_argmax_enqueue(const void* d_logits, int n) {
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
}
unsigned dev_argmax_read() {
    ensure_stream();
    unsigned h_out = 0;
    cudaMemcpyAsync(&h_out, g_am_out, sizeof(unsigned), cudaMemcpyDeviceToHost, g_stream);
    cudaError_t e = cudaStreamSynchronize(g_stream);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) {  // never hand back a stale token from a failed step
        fprintf(stderr, "trapetum: decode step failed: %s\n", cudaGetErrorString(e));
        abort();
    }
    return h_out;
}
unsigned dev_argmax(const void* d_logits, int n) {
    dev_argmax_enqueue(d_logits, n);
    return dev_argmax_read();
}

// --- CUDA graph capture of the decode chain ---------------------------------------
// graph_begin() starts capturing on g_stream; run the chain (forward/cast queue onto
// g_stream); graph_end() instantiates a replayable graph; graph_launch() replays it
// with near-zero CPU launch overhead.
void graph_begin() {
    ensure_stream();
    cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal);
}

// Returns nullptr (the Rust side asserts) if the capture was invalidated or cannot be
// instantiated, instead of an uninitialized handle that would replay nothing.
void* graph_end() {
    cudaGraph_t graph = nullptr;
    cudaError_t e = cudaStreamEndCapture(g_stream, &graph);
    if (e != cudaSuccess || graph == nullptr) {
        fprintf(stderr, "trapetum: cudaStreamEndCapture failed: %s\n", cudaGetErrorString(e));
        if (graph) cudaGraphDestroy(graph);
        return nullptr;
    }
    cudaGraphExec_t exec = nullptr;
    e = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess || exec == nullptr) {
        fprintf(stderr, "trapetum: cudaGraphInstantiate failed: %s\n", cudaGetErrorString(e));
        return nullptr;
    }
    g_graphs_captured = 1;
    return (void*)exec;
}

void graph_launch(void* exec) {
    cudaError_t e = cudaGraphLaunch((cudaGraphExec_t)exec, g_stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "trapetum: cudaGraphLaunch failed: %s\n", cudaGetErrorString(e));
        abort();
    }
}

void graph_free(void* exec) {
    cudaGraphExecDestroy((cudaGraphExec_t)exec);
}

// ---- Fused decode entry points (Llama blocks with scalar 4-bit projections) -----------------
// Each one runs the fused kernels above when fusion applies (default two-stage deterministic
// mode, K=16 projections sharing the input width, its bit set in TRAPETUM_FUSE) and otherwise the
// exact sequence of separate kernels the Rust blocks used to call, so the outputs are the same
// bits either way. The f32 buffers are only scratch for that fallback.
// TRAPETUM_FUSE is a bit mask: 1 RoPE + KV-cache append, 2 residual add in the o and down
// reduces, 4 q/k/v in one GEMV, 8 gate/up in one GEMV; 0 = none (the separate kernels), unset =
// 15 = all. Measured on an RTX 4090, 2026-11-05, Llama-2-7B 4-bit, CUDA graph, mean ms per token
// over the first 128 positions, two rounds: none 6.36, 1 6.26, 1+2 6.21, all 5.28 (bitwise
// identical prompt logits for every setting; 16/16 greedy).
static int fuse_mask() {
    static int f = -1;
    if (f < 0) {
        const char* e = getenv("TRAPETUM_FUSE");
        f = 15;
        if (e && e[0] >= '0' && e[0] <= '9') { f = atoi(e) & 15; }
    }
    return f;
}
static bool fuse_ok(QLinear* const* ps, int n, int bit) {
    if (!(fuse_mask() & bit) || det_mode() != 2 || det_gs() <= 1) return false;
    for (int i = 0; i < n; i++)
        if (ps[i]->K_ != 16 || ps[i]->IC != ps[0]->IC || ps[i]->OC % CPB != 0) return false;
    return true;
}
static void launch_partial_seg(QLinear* const* ps, int n, const void* d_x, float* yp, int OCt, int gy) {
    size_t smem = (size_t)K*CPB*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
    dim3 grid(OCt/CPB, gy), block(32, TY);
    QLinear* a = ps[0];
    QLinear* b = n > 1 ? ps[1] : ps[0];
    QLinear* c = n > 2 ? ps[2] : b;              // unused third projection: oc 0, never selected
    int o0 = a->OC, o1 = n > 1 ? b->OC : 0, o2 = n > 2 ? c->OC : 0;
    if (g_sparse_rows)
        gemv4_partial_seg<true><<<grid, block, smem, g_stream>>>((const __half*)d_x, a->d_packed, b->d_packed, c->d_packed,
                                                                 a->d_cb, b->d_cb, c->d_cb, o0, o1, o2, yp, a->IC, OCt);
    else
        gemv4_partial_seg<false><<<grid, block, smem, g_stream>>>((const __half*)d_x, a->d_packed, b->d_packed, c->d_packed,
                                                                  a->d_cb, b->d_cb, c->d_cb, o0, o1, o2, yp, a->IC, OCt);
}

// q, k, v of an attention block: one GEMV launch and one reduce (bias add, cast to half).
void qlinear3_forward_half(void* hq, void* hk, void* hv, const void* d_x,
                           void* qf, void* kf, void* vf, void* qh, void* kh, void* vh,
                           const void* bq, const void* bk, const void* bv) {
    ensure_stream();
    QLinear* ps[3] = {(QLinear*)hq, (QLinear*)hk, (QLinear*)hv};
    if (fuse_ok(ps, 3, 4)) {
        int gy = det_gs();
        int OCt = ps[0]->OC + ps[1]->OC + ps[2]->OC;
        float* yp = ypart((size_t)gy * OCt);
        launch_partial_seg(ps, 3, d_x, yp, OCt, gy);
        int tpb = 256;
        gemv_reduce_half3<<<(OCt + tpb - 1)/tpb, tpb, 0, g_stream>>>(yp, (__half*)qh, (__half*)kh, (__half*)vh,
            (const float*)bq, (const float*)bk, (const float*)bv, ps[0]->OC, ps[1]->OC, OCt, gy);
        return;
    }
    void* fs[3] = {qf, kf, vf};
    void* hs[3] = {qh, kh, vh};
    const void* bs[3] = {bq, bk, bv};
    for (int i = 0; i < 3; i++) {                   // the previous per-projection sequence
        qlinear_forward_dev(ps[i], d_x, fs[i]);
        if (bs[i]) op_vadd(fs[i], bs[i], ps[i]->OC);
        dev_cast_f32_to_half(hs[i], fs[i], ps[i]->OC);
    }
}

// gate, up of an MLP block: one GEMV launch, then one kernel for both reductions and SiLU * up.
// thr > 0: the SiLU output (the down_proj input) is thresholded too (activation sparsity).
void qlinear2_forward_silu(void* hg, void* hu, const void* d_x, void* gf, void* uf, void* act_h, float thr) {
    ensure_stream();
    QLinear* ps[2] = {(QLinear*)hg, (QLinear*)hu};
    if (fuse_ok(ps, 2, 8) && ps[0]->OC == ps[1]->OC) {
        int gy = det_gs();
        int n = ps[0]->OC, OCt = 2*n;
        float* yp = ypart((size_t)gy * OCt);
        launch_partial_seg(ps, 2, d_x, yp, OCt, gy);
        int tpb = 256;
        gemv_reduce_silu<<<(n + tpb - 1)/tpb, tpb, 0, g_stream>>>(yp, (__half*)act_h, n, gy, thr);
        return;
    }
    qlinear_forward_dev(ps[0], d_x, gf);
    qlinear_forward_dev(ps[1], d_x, uf);
    op_silu_mul(gf, uf, act_h, ps[0]->OC);
    op_sparsify_half(act_h, ps[0]->OC, thr);
}

// o and down projections: GEMV, then the residual add folded into the reduce.
void qlinear_forward_resadd(void* handle, const void* d_x, void* yf, void* h_half) {
    ensure_stream();
    QLinear* ps[1] = {(QLinear*)handle};
    if (fuse_ok(ps, 1, 2)) {
        QLinear* q = ps[0];
        int gy = det_gs();
        float* yp = ypart((size_t)gy * q->OC);
        size_t smem = (size_t)K*CPB*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
        dim3 grid(q->OC/CPB, gy), block(32, TY);
        if (g_sparse_rows) gemv4_partial<true><<<grid, block, smem, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, yp, q->IC, q->OC);
        else gemv4_partial<false><<<grid, block, smem, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, yp, q->IC, q->OC);
        int tpb = 256;
        gemv_reduce_resadd<<<(q->OC + tpb - 1)/tpb, tpb, 0, g_stream>>>(yp, (__half*)h_half, q->OC, gy);
        return;
    }
    qlinear_forward_dev(ps[0], d_x, yf);
    op_residual_add(h_half, yf, ps[0]->OC);
}

// RoPE on q and k, k and v appended to the KV cache: one launch instead of four. The position is
// `pos`, or d_tp[1] when d_tp is not null (CUDA-graph decode).
void op_rope_append(void* q_half, void* k_half, const void* v_half, void* cache_k, void* cache_v,
                    int pos, const void* d_tp, int n_heads, int n_kv, int head_dim, const void* inv_freq) {
    ensure_stream();
    int kvdim = n_kv*head_dim;
    if (!(fuse_mask() & 1)) {
        if (d_tp) {
            op_rope_dk(q_half, d_tp, n_heads, head_dim, inv_freq);
            op_rope_dk(k_half, d_tp, n_kv, head_dim, inv_freq);
            op_cache_append_dk(cache_k, k_half, d_tp, kvdim);
            op_cache_append_dk(cache_v, v_half, d_tp, kvdim);
        } else {
            op_rope(q_half, pos, n_heads, head_dim, inv_freq);
            op_rope(k_half, pos, n_kv, head_dim, inv_freq);
            op_cache_append(cache_k, k_half, pos, kvdim);
            op_cache_append(cache_v, v_half, pos, kvdim);
        }
        return;
    }
    int total = n_heads*(head_dim/2) + n_kv*(head_dim/2) + kvdim;
    rope_append_k<<<(total + 255)/256, 256, 0, g_stream>>>((__half*)q_half, (__half*)k_half, (const __half*)v_half,
        (__half*)cache_k, (__half*)cache_v, pos, (const int*)d_tp, n_heads, n_kv, head_dim, (const float*)inv_freq);
}

} // extern "C"
