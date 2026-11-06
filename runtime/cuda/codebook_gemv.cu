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
// The body is shared with the batch-invariant verify kernel (attn_split_v), which calls it once
// per query row with that row's q, partial buffer and length: the same code, hence the same bits.
template <int PER>   // PER = head_dim / 32 values per lane
__device__ __forceinline__ void attn_split_body(const __half* __restrict__ q, const __half* __restrict__ ck,
                                                const __half* __restrict__ cv, float* __restrict__ part,
                                                int n_heads, int n_kv, int seqlen, float softcap, int stride_chunks) {
    const int HD = PER * 32;
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
template <int PER>
__global__ void attn_split_k(const __half* __restrict__ q, const __half* __restrict__ ck,
                             const __half* __restrict__ cv, float* __restrict__ part,
                             int n_heads, int n_kv, int seqlen_arg, const int* __restrict__ d_tp,
                             float softcap, int stride_chunks) {
    int seqlen = d_tp ? d_tp[1] + 1 : seqlen_arg;
    attn_split_body<PER>(q, ck, cv, part, n_heads, n_kv, seqlen, softcap, stride_chunks);
}

// One block per head, one thread per output dimension: chunks in a fixed order (0, 1, 2, ...).
// The body is shared with the verify kernel attn_merge_v, as for the split kernel.
__device__ __forceinline__ void attn_merge_body(const float* __restrict__ part, __half* __restrict__ out, int head_dim,
                                                int seqlen, int stride_chunks) {
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
__global__ void attn_merge_k(const float* __restrict__ part, __half* __restrict__ out, int head_dim,
                             int seqlen_arg, const int* __restrict__ d_tp, int stride_chunks) {
    int seqlen = d_tp ? d_tp[1] + 1 : seqlen_arg;
    attn_merge_body(part, out, head_dim, seqlen, stride_chunks);
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

// ============================================================================
// Batch-invariant verify (speculative decoding): M tokens at positions base .. base+M-1 in one
// forward whose logits are, row by row, the SAME BITS as M sequential decode steps on the default
// deterministic path (two-stage GEMVs, split-KV attention). Each reduction keeps its decode order:
//  - GEMV: a column's sum depends only on the IC split (gridDim.y slices of ceil(IC/gridDim.y)
//    rows), the TY interleave of rows inside a slice (same ty, same FMA chain), the sum of the TY
//    partials from 0 in y order, and the sum of the slices from 0 in slice order. The verify
//    kernels keep all four; they only give each thread 4 columns instead of 8 when M > 4, so
//    that 16 rows of accumulators fit in registers. One weight read serves the M rows.
//  - RMSNorm, RoPE and the KV append, the attention chunks and their merge, the reduce
//    epilogues (bias and cast, SiLU times up, residual add): the decode kernels' arithmetic
//    with one row per block (grid y or z), at position base + row.
// M is 1, 2, 4, 8 or 16 (VMAX). Scalar 4-bit (K=16) body projections; the head may be K=16 or 256.
// ============================================================================
#define VMAX 16

template <int M, int CPT>   // CPT columns per thread: 8 (M <= 4) or 4
__global__ void __launch_bounds__(32*TY)
gemv4_partial_v(const __half* __restrict__ X,                   // [M][IC]
                const unsigned char* __restrict__ p0, const unsigned char* __restrict__ p1,
                const unsigned char* __restrict__ p2,
                const __half* __restrict__ c0, const __half* __restrict__ c1, const __half* __restrict__ c2,
                int oc0, int oc1, int oc2, float* __restrict__ Ypart, int IC, int OCt) {   // Ypart [gy][M][OCt]
    constexpr int CPBV = 32 * CPT;                  // columns per block
    constexpr int MG = M < TY ? M : TY;             // rows reduced per pass (M is a multiple of MG)
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm;                     // [K][CPBV]
    float* red = (float*)(s_cb + K*CPBV);           // [MG][TY][CPBV]
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32 + tx, nth = 32*TY;
    int jg = blockIdx.x*CPBV;                       // first column, concatenated space
    int s = jg < oc0 ? 0 : (jg < oc0 + oc1 ? 1 : 2);
    const unsigned char* packed = s == 0 ? p0 : (s == 1 ? p1 : p2);
    const __half* cb = s == 0 ? c0 : (s == 1 ? c1 : c2);
    int OC = s == 0 ? oc0 : (s == 1 ? oc1 : oc2);
    int j0 = jg - (s == 0 ? 0 : (s == 1 ? oc0 : oc0 + oc1));
    for (int t = tid; t < K*CPBV/2; t += nth) {
        int idx = t*2, k = idx/CPBV, jj = j0 + (idx%CPBV);
        *reinterpret_cast<__half2*>(&s_cb[idx]) = *reinterpret_cast<const __half2*>(&cb[(size_t)k*OC + jj]);
    }
    __syncthreads();
    int per = (IC + gridDim.y - 1)/gridDim.y, ic0 = blockIdx.y*per, ic1 = min(IC, ic0 + per);
    int jbase = j0 + tx*CPT; size_t OCp = OC/2;
    float acc[M][CPT];
    #pragma unroll
    for (int m = 0; m < M; m++)
        #pragma unroll
        for (int c = 0; c < CPT; c++) acc[m][c] = 0.f;
    for (int ic = ic0 + ty; ic < ic1; ic += TY) {
        unsigned f;
        if (CPT == 8) f = __ldg((const unsigned*)&packed[(size_t)ic*OCp + jbase/2]);
        else f = __ldg((const unsigned short*)&packed[(size_t)ic*OCp + jbase/2]);
        float xx[M];
        #pragma unroll
        for (int m = 0; m < M; m++) xx[m] = __half2float(__ldg(&X[(size_t)m*IC + ic]));
        #pragma unroll
        for (int c = 0; c < CPT; c++) {
            unsigned char id = (f >> (4*c)) & 0xF;
            float w = __half2float(s_cb[id*CPBV + tx*CPT + c]);
            #pragma unroll
            for (int m = 0; m < M; m++) acc[m][c] += xx[m]*w;   // gemv4_partial: acc[c] += xx*w
        }
    }
    // each row's TY partials, summed from 0 in y order as in gemv4_partial, MG rows per pass
    #pragma unroll
    for (int mb = 0; mb < M; mb += MG) {
        #pragma unroll
        for (int mm = 0; mm < MG; mm++)
            #pragma unroll
            for (int c = 0; c < CPT; c++) red[((size_t)mm*TY + ty)*CPBV + tx*CPT + c] = acc[mb + mm][c];
        __syncthreads();
        if (ty < MG) {
            #pragma unroll
            for (int c = 0; c < CPT; c++) {
                float s2 = 0;
                for (int y = 0; y < TY; y++) s2 += red[((size_t)ty*TY + y)*CPBV + tx*CPT + c];
                Ypart[((size_t)blockIdx.y*M + mb + ty)*OCt + (jg + tx*CPT + c)] = s2;
            }
        }
        __syncthreads();
    }
}

// 8-bit (K=256) twin for the LM head: gemv8_partial's tiling (4 columns per thread, CPB8), the
// reduction space reusing the codebook's once every thread has left the main loop.
template <int M>
__global__ void __launch_bounds__(32*TY)
gemv8_partial_v(const __half* __restrict__ X, const unsigned char* __restrict__ idx,
                const __half* __restrict__ cb, float* __restrict__ Ypart, int IC, int OC) {   // Ypart [gy][M][OC]
    constexpr int MG = M < TY ? M : TY;
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm;                     // [K8][CPB8]
    float* red = (float*)sm;                        // [MG][TY][CPB8], after the main loop
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32 + tx, nth = 32*TY;
    int j0 = blockIdx.x*CPB8;
    for (int t = tid; t < K8*CPB8; t += nth) { int k = t/CPB8, jj = j0 + (t%CPB8); s_cb[t] = __ldg(&cb[(size_t)k*OC + jj]); }
    __syncthreads();
    int per = (IC + gridDim.y - 1)/gridDim.y, ic0 = blockIdx.y*per, ic1 = min(IC, ic0 + per);
    int jbase = j0 + tx*4;
    float acc[M][4];
    #pragma unroll
    for (int m = 0; m < M; m++)
        #pragma unroll
        for (int c = 0; c < 4; c++) acc[m][c] = 0.f;
    for (int ic = ic0 + ty; ic < ic1; ic += TY) {
        unsigned f = __ldg((const unsigned*)&idx[(size_t)ic*OC + jbase]);
        float xx[M];
        #pragma unroll
        for (int m = 0; m < M; m++) xx[m] = __half2float(__ldg(&X[(size_t)m*IC + ic]));
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            unsigned char id = (f >> (8*c)) & 0xFF;
            float w = __half2float(s_cb[id*CPB8 + tx*4 + c]);
            #pragma unroll
            for (int m = 0; m < M; m++) acc[m][c] += xx[m]*w;   // gemv8_partial: acc[c] += xx*w
        }
    }
    __syncthreads();                                 // the codebook space becomes the reduction space
    #pragma unroll
    for (int mb = 0; mb < M; mb += MG) {
        #pragma unroll
        for (int mm = 0; mm < MG; mm++)
            #pragma unroll
            for (int c = 0; c < 4; c++) red[((size_t)mm*TY + ty)*CPB8 + tx*4 + c] = acc[mb + mm][c];
        __syncthreads();
        if (ty < MG) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float s2 = 0;
                for (int y = 0; y < TY; y++) s2 += red[((size_t)ty*TY + y)*CPB8 + tx*4 + c];
                Ypart[((size_t)blockIdx.y*M + mb + ty)*OC + (j0 + tx*4 + c)] = s2;
            }
        }
        __syncthreads();
    }
}

// Reduces over the slices, row m = blockIdx.y, with the decode reduces' order and epilogues.
__global__ void gemv_reduce_v(const float* __restrict__ Ypart, float* __restrict__ Y, int OC, int GSy, int M) {
    int i = blockIdx.x*blockDim.x + threadIdx.x, m = blockIdx.y;
    if (i >= OC) return;
    float s = 0;
    for (int g = 0; g < GSy; g++) s += Ypart[((size_t)g*M + m)*OC + i];
    Y[(size_t)m*OC + i] = s;
}
__global__ void gemv_reduce_half3_v(const float* __restrict__ Ypart,
                                    __half* __restrict__ q, __half* __restrict__ k, __half* __restrict__ v,
                                    const float* __restrict__ bq, const float* __restrict__ bk,
                                    const float* __restrict__ bv, int oq, int ok, int OCt, int GSy, int M) {
    int i = blockIdx.x*blockDim.x + threadIdx.x, m = blockIdx.y;
    if (i >= OCt) return;
    float s = 0;
    for (int g = 0; g < GSy; g++) s += Ypart[((size_t)g*M + m)*OCt + i];
    int ov = OCt - oq - ok;
    if (i < oq) {
        if (bq) s += bq[i];
        q[(size_t)m*oq + i] = __float2half(s);
    } else if (i < oq + ok) {
        int li = i - oq;
        if (bk) s += bk[li];
        k[(size_t)m*ok + li] = __float2half(s);
    } else {
        int li = i - oq - ok;
        if (bv) s += bv[li];
        v[(size_t)m*ov + li] = __float2half(s);
    }
}
__global__ void gemv_reduce_silu_v(const float* __restrict__ Ypart, __half* __restrict__ out, int n, int GSy, int M) {
    int i = blockIdx.x*blockDim.x + threadIdx.x, m = blockIdx.y;
    if (i >= n) return;
    size_t OCt = (size_t)2*n;
    float g = 0, u = 0;
    for (int y = 0; y < GSy; y++) g += Ypart[((size_t)y*M + m)*OCt + i];
    for (int y = 0; y < GSy; y++) u += Ypart[((size_t)y*M + m)*OCt + n + i];
    float s = g / (1.f + expf(-g));
    out[(size_t)m*n + i] = __float2half(s * u);
}
__global__ void gemv_reduce_resadd_v(const float* __restrict__ Ypart, __half* __restrict__ h, int OC, int GSy, int M) {
    int i = blockIdx.x*blockDim.x + threadIdx.x, m = blockIdx.y;
    if (i >= OC) return;
    float s = 0;
    for (int g = 0; g < GSy; g++) s += Ypart[((size_t)g*M + m)*OC + i];
    __half* hr = h + (size_t)m*OC;
    hr[i] = __float2half(__half2float(hr[i]) + s);
}


// ============================================================================
// Tensor-core 4-bit codebook GEMV (sm_80 and newer), shared by the decode (M = 1) and the
// batch-invariant verify (M up to 16). mma.sync m16n8k16, f16 inputs, f32 accumulation.
// Weights are the A operand (16 output columns x 16 input channels), read straight from the
// [IC][OC/2] nibble bytes; the M token rows are the B operand (16 channels x 8 tokens per mma),
// token n in B column n & 7 of mma n >> 3; tokens >= M are zero columns and are never stored.
// What fixes the bits of an output: GS_TC slices of the k16-steps, TC_KC chains per slice
// (chain c takes steps c, c+8, ...), each chain one mma accumulator chain in increasing step
// order, the chains summed in order 0..7, the slices summed in order 0..3 by the *_v reduce
// kernels, the k-slot permutation (slots 2t,2t+1,2t+8,2t+9 = channels 4t..4t+3 of the step) and
// the column mapping (tile b of a 64-column group: A row g = column 8g+2b, row g+8 = 8g+2b+1).
// None of these depends on M, on fusion, on the GPU, or on the scheduling knobs TC_CG, TC_U,
// TC_WLOAD and the launch bounds, which can be tuned freely without changing a single bit.
// ============================================================================
#define TC_COLS 64      // columns per column group (one K-warp set)
#define TC_KC 8         // accumulation chains (K-warps) per slice: fixes the bits
#define GS_TC 4         // IC slices (grid.y): fixes the bits
#ifndef TC_CG
#define TC_CG 1         // column groups per CTA (1 or 2): scheduling only
#endif
#ifndef TC_U
#define TC_U 4          // weight ring depth in k16-steps (2..6): scheduling only
#endif
#ifndef TC_WLOAD
#define TC_WLOAD 1      // weight load form (0..3, see tc_ldw): scheduling only
#endif

__device__ __forceinline__ void tc_mma(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
// Weights: streamed once per token, never reused by the SM: no L1 allocation; 1 = 128-B L2 fill.
__device__ __forceinline__ unsigned tc_ldw(const unsigned char* p, unsigned long long pol) {
    unsigned v;
#if TC_WLOAD == 0
    asm volatile("ld.global.nc.L1::no_allocate.b32 %0, [%1];" : "=r"(v) : "l"(p));
#elif TC_WLOAD == 1
    asm volatile("ld.global.nc.L1::no_allocate.L2::128B.b32 %0, [%1];" : "=r"(v) : "l"(p));
#elif TC_WLOAD == 2
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.b32 %0, [%1];" : "=r"(v) : "l"(p));
#else
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::128B.b32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(pol));
#endif
    return v;
}
__device__ __forceinline__ unsigned long long tc_policy() {
    unsigned long long pol = 0;
#if TC_WLOAD == 3 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
#endif
    return pol;
}
// Activations: 4 halves (8 bytes) through the read-only L1 path.
__device__ __forceinline__ uint2 tc_ldx(const __half* p) {
    uint2 v;
    asm volatile("ld.global.nc.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned tc_pack(unsigned lo, unsigned hi) {   // = __byte_perm(lo, hi, 0x5410)
    unsigned r;
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(r) : "r"(lo), "r"(hi));
    return r;
}

// One k16-step of one warp: the 4 index words of rows 16s+4t+j (j = 0..3) -> 4 tiles x NH mma.
template <int NH>
__device__ __forceinline__ void tc_step(const unsigned (&w)[4], const uint2 (&xb)[NH], const unsigned char* lut,
                                        const unsigned (&base)[4], float (&acc)[4][NH][4]) {
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        unsigned ev[4], od[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // nibble of column 8g+2b (low, A row g) and of 8g+2b+1 (high, A row g+8), moved to bits 8..11
            unsigned xe = b == 0 ? (w[j] << 8) : (b == 1 ? w[j] : (w[j] >> (8 * b - 8)));
            unsigned xo = b == 0 ? (w[j] << 4) : (w[j] >> (8 * b - 4));
            ev[j] = *(const unsigned short*)(lut + ((xe & 0xF00u) | base[b]));
            od[j] = *(const unsigned short*)(lut + ((xo & 0xF00u) | base[b]) + 2);
        }
        const unsigned a0 = tc_pack(ev[0], ev[1]);   // A[g][slots 2t, 2t+1]     = channels 4t, 4t+1
        const unsigned a1 = tc_pack(od[0], od[1]);   // A[g+8][slots 2t, 2t+1]
        const unsigned a2 = tc_pack(ev[2], ev[3]);   // A[g][slots 2t+8, 2t+9]   = channels 4t+2, 4t+3
        const unsigned a3 = tc_pack(od[2], od[3]);   // A[g+8][slots 2t+8, 2t+9]
        #pragma unroll
        for (int h = 0; h < NH; h++) tc_mma(acc[b][h], a0, a1, a2, a3, xb[h].x, xb[h].y);
    }
}

// Ypart [GS_TC][M][OCt] (partial of slice y, row m, concatenated column j). NH = 1 for M <= 8, 2 for M <= 16.
// Up to three projections sharing IC (q, k, v or gate, up) run as one launch over their concatenated
// columns; a CTA finds its projection from its first column (every OC is a multiple of 256).
template <int NH, int CG, int U>
__global__ void __launch_bounds__(CG * TC_KC * 32, CG == 1 ? (NH == 1 ? 3 : 2) : 1)
gemv4_tc(const __half* __restrict__ X, int M,
         const unsigned char* __restrict__ p0, const unsigned char* __restrict__ p1,
         const unsigned char* __restrict__ p2,
         const __half* __restrict__ c0, const __half* __restrict__ c1, const __half* __restrict__ c2,
         int oc0, int oc1, int oc2, float* __restrict__ Ypart, int IC, int OCt) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) unsigned tc_lut[CG * 2048];       // 4 rotated codebook copies per group, 8 KB
    extern __shared__ __align__(16) float4 tc_red[];           // [CG][TC_KC][NH][4 tiles][32 lanes]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int cg = CG == 1 ? 0 : (warp >> 3), kw = warp & 7, g = lane >> 2, t = lane & 3;
    const int jb = blockIdx.x * (TC_COLS * CG);                 // CTA's first column, concatenated space
    const int sg = jb < oc0 ? 0 : (jb < oc0 + oc1 ? 1 : 2);
    const unsigned char* packed = sg == 0 ? p0 : (sg == 1 ? p1 : p2);
    const __half* cb = sg == 0 ? c0 : (sg == 1 ? c1 : c2);
    const int OC = sg == 0 ? oc0 : (sg == 1 ? oc1 : oc2);
    const int jp = jb - (sg == 0 ? 0 : (sg == 1 ? oc0 : oc0 + oc1));   // CTA's first column inside projection sg
    const int j0 = jp + cg * TC_COLS;                           // this warp's column group
    const size_t OCp = (size_t)(OC >> 1);
    const int NS = IC >> 4, y = blockIdx.y;                     // k16-steps; slice y = [s0, s1)
    const int s0 = (y * NS) / GS_TC, s1 = ((y + 1) * NS) / GS_TC;
    const int n = s1 - s0 > kw ? (s1 - s0 - kw + TC_KC - 1) / TC_KC : 0;   // steps of chain kw
    const unsigned char* wp = packed + (size_t)(16 * (s0 + kw) + 4 * t) * OCp + (j0 >> 1) + 4 * g;
    const size_t wstep = (size_t)(16 * TC_KC) * OCp;            // a chain advances 128 rows per step
    const unsigned long long pol = tc_policy();
    // 1) the weight ring first (DRAM latency), 2) step 0 activations, 3) the codebook tables
    unsigned ring[U][4];
    #pragma unroll
    for (int u = 0; u < U; u++)
        #pragma unroll
        for (int j = 0; j < 4; j++) ring[u][j] = u < n ? tc_ldw(wp + (size_t)u * wstep + (size_t)j * OCp, pol) : 0u;
    const __half* xp = X + (size_t)g * IC + 16 * (s0 + kw) + 4 * t;   // token g (+8h), channels 16s+4t..+3
    uint2 xb[NH];
    #pragma unroll
    for (int h = 0; h < NH; h++) {
        xb[h] = make_uint2(0u, 0u);
        if (8 * h + g < M && n > 0) xb[h] = tc_ldx(xp + (size_t)(8 * h) * IC);
    }
    {   // thread r of group cgs: entry k = r >> 4, column pairs jj, jj + 1 (columns 2jj .. 2jj+3)
        const int cgs = tid >> 8, r = tid & 255, k = r >> 4, jj = 2 * (r & 15);
        const uint2 pr = *(const uint2*)(cb + (size_t)k * OC + jp + cgs * TC_COLS + 2 * jj);
        unsigned* L = tc_lut + cgs * 2048;
        #pragma unroll
        for (int u = 0; u < 4; u++) {   // word (u>>1)*1024 + k*64 + (u&1)*32 + 4*(j>>2) + ((j&3)+u)&3
            const int wb = (u >> 1) * 1024 + k * 64 + (u & 1) * 32 + 4 * (jj >> 2);
            L[wb + (((jj & 3) + u) & 3)] = pr.x;
            L[wb + ((((jj + 1) & 3) + u) & 3)] = pr.y;
        }
    }
    unsigned base[4];   // byte offset of the lane's word for tile b, entry 0, inside tc_lut (bits 2..7, 12, 13)
    #pragma unroll
    for (int b = 0; b < 4; b++) base[b] = cg * 8192 + (t >> 1) * 4096 + (t & 1) * 128 + 16 * g + 4 * ((b + t) & 3);
    const unsigned char* lut = (const unsigned char*)tc_lut;
    float acc[4][NH][4];
    #pragma unroll
    for (int b = 0; b < 4; b++)
        #pragma unroll
        for (int h = 0; h < NH; h++) acc[b][h][0] = acc[b][h][1] = acc[b][h][2] = acc[b][h][3] = 0.f;
    __syncthreads();
    // chain kw: steps s0 + kw + 8i, i = 0 .. n-1, in increasing order
    for (int i0 = 0; i0 < n; i0 += U) {
        #pragma unroll
        for (int u = 0; u < U; u++) {
            const int i = i0 + u;
            if (i < n) {
                uint2 xn[NH];                                   // activations of step i + 1
                #pragma unroll
                for (int h = 0; h < NH; h++) {
                    xn[h] = make_uint2(0u, 0u);
                    if (8 * h + g < M && i + 1 < n) xn[h] = tc_ldx(xp + (size_t)(8 * h) * IC + (size_t)(i + 1) * (16 * TC_KC));
                }
                tc_step<NH>(ring[u], xb, lut, base, acc);
                if (i + U < n) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) ring[u][j] = tc_ldw(wp + (size_t)(i + U) * wstep + (size_t)j * OCp, pol);
                }
                #pragma unroll
                for (int h = 0; h < NH; h++) xb[h] = xn[h];
            }
        }
    }
    // chain partials to shared memory: unit ((cg*8 + kw)*NH + h)*128 + b*32 + (lane ^ b)
    #pragma unroll
    for (int h = 0; h < NH; h++)
        #pragma unroll
        for (int b = 0; b < 4; b++)
            tc_red[((cg * TC_KC + kw) * NH + h) * 128 + b * 32 + (lane ^ b)] =
                make_float4(acc[b][h][0], acc[b][h][1], acc[b][h][2], acc[b][h][3]);
    __syncthreads();
    {   // thread (cgs, h, tt, gg, b): columns 8gg+2b, +1 of group cgs, tokens 8h+2tt, +1; chains summed 0..7
        const int cgs = tid >> 8, r = tid & 255;
        const int h = r >> 7, tt = (r >> 5) & 3, q = r & 31, gg = q >> 2, b = q & 3;
        const int tok = 8 * h + 2 * tt;
        if (h < NH && tok < M) {
            const int unit = (cgs * TC_KC * NH + h) * 128 + b * 32 + ((4 * gg + tt) ^ b);
            float4 v = tc_red[unit];
            #pragma unroll
            for (int c = 1; c < TC_KC; c++) {
                const float4 e = tc_red[unit + c * NH * 128];
                v.x += e.x; v.y += e.y; v.z += e.z; v.w += e.w;
            }
            float* yr = Ypart + ((size_t)y * M + tok) * OCt + jb + cgs * TC_COLS + 8 * gg + 2 * b;
            *(float2*)yr = make_float2(v.x, v.z);                          // token tok: d0, d2
            if (tok + 1 < M) *(float2*)(yr + OCt) = make_float2(v.y, v.w); // token tok+1: d1, d3
        }
    }
#else
    __trap();   // sm_70/75 fatbin entries: never launched (tc_enabled checks the device and the binary)
#endif
}

// rmsnorm_k (thr 0) on row blockIdx.x.
__global__ void rmsnorm_v(const __half* __restrict__ x, const float* __restrict__ w,
                          __half* __restrict__ out, int n, float eps) {
    __shared__ float red[256];
    int tid = threadIdx.x;
    const __half* xr = x + (size_t)blockIdx.x*n;
    __half* orow = out + (size_t)blockIdx.x*n;
    float ss = 0;
    for (int i = tid; i < n; i += 256) { float v = __half2float(xr[i]); ss += v*v; }
    red[tid] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float scale = rsqrtf(red[0]/n + eps);
    for (int i = tid; i < n; i += 256) orow[i] = __float2half(__half2float(xr[i]) * scale * w[i]);
}

// rope_append_k on row m = blockIdx.y at position base + m (base = d_tp[1] on the graph path).
__global__ void rope_append_v(__half* __restrict__ q, __half* __restrict__ k, const __half* __restrict__ v,
                              __half* __restrict__ cache_k, __half* __restrict__ cache_v, int base_arg,
                              const int* __restrict__ d_tp, int n_heads, int n_kv, int head_dim,
                              const float* __restrict__ inv_freq) {
    int t = blockIdx.x*blockDim.x + threadIdx.x, m = blockIdx.y;
    int half = head_dim/2;
    int pos = (d_tp ? d_tp[1] : base_arg) + m;
    int nq = n_heads*half, nk = n_kv*half, kvdim = n_kv*head_dim, qdim = n_heads*head_dim;
    __half* qr = q + (size_t)m*qdim;
    __half* kr = k + (size_t)m*kvdim;
    const __half* vr = v + (size_t)m*kvdim;
    if (t < nq) {
        rope_pair(qr, qr, t / half, t % half, pos, head_dim, inv_freq);
    } else if (t < nq + nk) {
        int u = t - nq, h = u / half, d = u % half;
        rope_pair(kr, kr, h, d, pos, head_dim, inv_freq);
        __half* row = cache_k + (size_t)pos*kvdim;
        int i = h*head_dim + d, j = i + half;
        row[i] = kr[i];
        row[j] = kr[j];
    } else if (t < nq + nk + kvdim) {
        int i = t - nq - nk;
        cache_v[(size_t)pos*kvdim + i] = vr[i];
    }
}

// Split-KV attention for query row m = blockIdx.z (position base + m, length base + m + 1): the
// decode bodies on that row's q, partial records and output.
template <int PER>
__global__ void attn_split_v(const __half* __restrict__ q, const __half* __restrict__ ck,
                             const __half* __restrict__ cv, float* __restrict__ part,
                             int n_heads, int n_kv, int base_arg, const int* __restrict__ d_tp,
                             float softcap, int stride_chunks) {
    const int HD = PER * 32;
    int m = blockIdx.z;
    int seqlen = (d_tp ? d_tp[1] : base_arg) + m + 1;
    attn_split_body<PER>(q + (size_t)m*n_heads*HD, ck, cv, part + (size_t)m*n_heads*stride_chunks*(HD + 2),
                         n_heads, n_kv, seqlen, softcap, stride_chunks);
}
__global__ void attn_merge_v(const float* __restrict__ part, __half* __restrict__ out, int n_heads, int head_dim,
                             int base_arg, const int* __restrict__ d_tp, int stride_chunks) {
    int m = blockIdx.y;
    int seqlen = (d_tp ? d_tp[1] : base_arg) + m + 1;
    attn_merge_body(part + (size_t)m*n_heads*stride_chunks*((size_t)head_dim + 2), out + (size_t)m*n_heads*head_dim,
                    head_dim, seqlen, stride_chunks);
}

// Row-wise argmax (row m = blockIdx.y in stage 1, blockIdx.x in stage 2), argmax_stage1/2's rules:
// the largest value, the smallest index on a tie.
__global__ void argmax_v_stage1(const float* __restrict__ x, int n, int ld,
                                float* __restrict__ out_val, unsigned* __restrict__ out_idx) {
    __shared__ float sval[256];
    __shared__ unsigned sidx[256];
    int tid = threadIdx.x;
    const float* xr = x + (size_t)blockIdx.y*ld;
    unsigned stride = blockDim.x * gridDim.x;
    float bv = -INFINITY; unsigned bi = 0;
    for (unsigned i = blockIdx.x*blockDim.x + tid; i < (unsigned)n; i += stride) {
        float v = xr[i];
        if (v > bv) { bv = v; bi = i; }
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
    if (tid == 0) { out_val[blockIdx.y*gridDim.x + blockIdx.x] = sval[0]; out_idx[blockIdx.y*gridDim.x + blockIdx.x] = sidx[0]; }
}
__global__ void argmax_v_stage2(const float* __restrict__ in_val, const unsigned* __restrict__ in_idx,
                                int np, unsigned* __restrict__ out) {
    __shared__ float sval[256];
    __shared__ unsigned sidx[256];
    int tid = threadIdx.x, m = blockIdx.x;
    float bv = -INFINITY; unsigned bi = 0xffffffffu;
    for (int i = tid; i < np; i += blockDim.x) {
        float v = in_val[m*np + i]; unsigned id = in_idx[m*np + i];
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
    if (tid == 0) out[m] = sidx[0];
}

struct AvqLin { unsigned char* d_codes; __half* d_cb; __half* d_scale; int M, IC, OC; };

struct QLinear { unsigned char* d_packed; __half* d_cb; int IC, OC, K_; };

// Host helpers of the verify path (templates: outside the extern "C" block).
static void verify_need_t(bool ok, const char* what) {
    if (!ok) { fprintf(stderr, "trapetum: batch-invariant verify: %s\n", what); abort(); }
}
template <int M>
static void launch_partial_v_m(const __half* X, QLinear* a, QLinear* b, QLinear* c, int o0, int o1, int o2,
                               float* yp, int OCt, int gy) {
    constexpr int CPT = M <= 4 ? 8 : 4;
    constexpr int CPBV = 32 * CPT;
    constexpr int MG = M < TY ? M : TY;
    size_t smem = (size_t)K*CPBV*sizeof(__half) + (size_t)MG*TY*CPBV*sizeof(float);   // at most 40 KB
    dim3 grid(OCt/CPBV, gy), block(32, TY);
    gemv4_partial_v<M, CPT><<<grid, block, smem, g_stream>>>(X, a->d_packed, b->d_packed, c->d_packed,
                                                             a->d_cb, b->d_cb, c->d_cb, o0, o1, o2, yp, a->IC, OCt);
}
// One verify GEMV over up to three projections' concatenated columns (each OC a multiple of CPB).
static void launch_partial_v(int M, QLinear* const* ps, int n, const void* d_x, float* yp, int OCt, int gy) {
    QLinear* a = ps[0];
    QLinear* b = n > 1 ? ps[1] : ps[0];
    QLinear* c = n > 2 ? ps[2] : b;              // unused third projection: oc 0, never selected
    int o0 = a->OC, o1 = n > 1 ? b->OC : 0, o2 = n > 2 ? c->OC : 0;
    const __half* X = (const __half*)d_x;
    switch (M) {
        case 1: launch_partial_v_m<1>(X, a, b, c, o0, o1, o2, yp, OCt, gy); break;
        case 2: launch_partial_v_m<2>(X, a, b, c, o0, o1, o2, yp, OCt, gy); break;
        case 4: launch_partial_v_m<4>(X, a, b, c, o0, o1, o2, yp, OCt, gy); break;
        case 8: launch_partial_v_m<8>(X, a, b, c, o0, o1, o2, yp, OCt, gy); break;
        case 16: launch_partial_v_m<16>(X, a, b, c, o0, o1, o2, yp, OCt, gy); break;
        default: verify_need_t(false, "M must be 1, 2, 4, 8 or 16");
    }
}
template <int M>
static void launch_gemv8_v_m(const __half* X, QLinear* q, float* yp, int gy) {
    size_t smem = (size_t)K8*CPB8*sizeof(__half);   // the codebook; the reduction (32 KB) reuses it
    static bool once = false;                        // one per instantiation; set on the eager first call
    if (!once) {
        cudaError_t e = cudaFuncSetAttribute(gemv8_partial_v<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        verify_need_t(e == cudaSuccess, "cannot give the 8-bit verify GEMV its shared memory");
        once = true;
    }
    dim3 grid(q->OC/CPB8, gy), block(32, TY);
    gemv8_partial_v<M><<<grid, block, smem, g_stream>>>(X, q->d_packed, q->d_cb, yp, q->IC, q->OC);
}


// ---- Tensor-core GEMV, host side ----------------------------------------------------------------
// Decided once, at the first qlinear_create (load time, never inside a capture): TRAPETUM_TC=0 turns
// it off; it needs a compute capability 8.x+ device AND real sm_80+ code in the binary (a CUDA_ARCH=sm_75
// build JIT-run on an sm_89 board reports binaryVersion 89 but ptxVersion 75, and its body is the trap).
template <int NH>
static bool tc_prepare_kernel() {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, gemv4_tc<NH, TC_CG, TC_U>) != cudaSuccess) { cudaGetLastError(); return false; }
    if (fa.binaryVersion < 80 || fa.ptxVersion < 80) return false;
    size_t dyn = (size_t)TC_CG * TC_KC * NH * 128 * sizeof(float4);
    if (fa.sharedSizeBytes + dyn > 48 * 1024 &&
        cudaFuncSetAttribute(gemv4_tc<NH, TC_CG, TC_U>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaFuncSetAttribute(gemv4_tc<NH, TC_CG, TC_U>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaGetLastError();               // the carveout is a hint; ignore a refusal
    return true;
}
static int tc_compute() {
    const char* e = getenv("TRAPETUM_TC");
    if (e && e[0] == '0' && e[1] == 0) return 0;
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    if (major < 8 || !tc_prepare_kernel<1>() || !tc_prepare_kernel<2>()) return 0;
    return 1;
}
// A function-local static is initialized exactly once, thread-safely (C++11): a second thread that
// loads a model at the same time waits for the final answer instead of reading a provisional one,
// so every GEMV of the process takes the same path.
static int tc_enabled() {
    static const int v = tc_compute();
    return v;
}
// The dispatch rule, the same for the decode and the verify: a group of K=16 projections sharing IC
// (q/k/v, gate/up, o, down, a 4-bit head) takes the tensor-core path iff this holds.
static bool tc_on() { return tc_enabled() && det_mode() == 2 && !g_sparse_rows; }
static bool tc_group_ok(QLinear* const* ps, int n) {
    if (!tc_on()) return false;
    for (int i = 0; i < n; i++)
        if (ps[i]->K_ != 16 || ps[i]->IC != ps[0]->IC || ps[i]->IC % 16 != 0 || ps[i]->OC % (TC_COLS * TC_CG) != 0)
            return false;
    return true;
}
// Partials of M rows (x [M][IC], 16-byte aligned) into yp [GS_TC][M][OCt]; the caller reduces with a *_v kernel.
static void launch_tc(int M, QLinear* const* ps, int n, const void* d_x, float* yp, int OCt) {
    verify_need_t(M >= 1 && M <= VMAX, "tensor-core GEMV: M must be 1 to 16");
    verify_need_t(((size_t)d_x & 15) == 0, "tensor-core GEMV: the input must be 16-byte aligned");
    QLinear* a = ps[0];
    QLinear* b = n > 1 ? ps[1] : a;
    QLinear* c = n > 2 ? ps[2] : b;              // unused third projection: oc 0, never selected
    int o0 = a->OC, o1 = n > 1 ? b->OC : 0, o2 = n > 2 ? c->OC : 0;
    dim3 grid(OCt / (TC_COLS * TC_CG), GS_TC), block(32 * TC_KC * TC_CG);
    const __half* X = (const __half*)d_x;
    if (M <= 8)
        gemv4_tc<1, TC_CG, TC_U><<<grid, block, (size_t)TC_CG * TC_KC * 1 * 128 * sizeof(float4), g_stream>>>(
            X, M, a->d_packed, b->d_packed, c->d_packed, a->d_cb, b->d_cb, c->d_cb, o0, o1, o2, yp, a->IC, OCt);
    else
        gemv4_tc<2, TC_CG, TC_U><<<grid, block, (size_t)TC_CG * TC_KC * 2 * 128 * sizeof(float4), g_stream>>>(
            X, M, a->d_packed, b->d_packed, c->d_packed, a->d_cb, b->d_cb, c->d_cb, o0, o1, o2, yp, a->IC, OCt);
}

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
    (void)tc_enabled();   // decided at load, never inside a capture
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
    QLinear* ps1[1] = {q};
    if (tc_group_ok(ps1, 1)) {                       // tensor-core path (sm_80+, mode 2, no sparsity)
        float* yp = ypart((size_t)GS_TC * q->OC);
        launch_tc(1, ps1, 1, d_x, yp, q->OC);
        gemv_reduce_v<<<dim3((q->OC + 255)/256, 1), 256, 0, g_stream>>>(yp, (float*)d_y, q->OC, GS_TC, 1);
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
    if ((fuse_mask() & 4) && tc_group_ok(ps, 3)) {
        int OCt = ps[0]->OC + ps[1]->OC + ps[2]->OC;
        float* yp = ypart((size_t)GS_TC * OCt);
        launch_tc(1, ps, 3, d_x, yp, OCt);
        gemv_reduce_half3_v<<<dim3((OCt + 255)/256, 1), 256, 0, g_stream>>>(yp, (__half*)qh, (__half*)kh, (__half*)vh,
            (const float*)bq, (const float*)bk, (const float*)bv, ps[0]->OC, ps[1]->OC, OCt, GS_TC, 1);
        return;
    }
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
    if ((fuse_mask() & 8) && thr == 0.f && ps[0]->OC == ps[1]->OC && tc_group_ok(ps, 2)) {
        int n = ps[0]->OC, OCt = 2*n;
        float* yp = ypart((size_t)GS_TC * OCt);
        launch_tc(1, ps, 2, d_x, yp, OCt);
        gemv_reduce_silu_v<<<dim3((n + 255)/256, 1), 256, 0, g_stream>>>(yp, (__half*)act_h, n, GS_TC, 1);
        return;
    }
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
    if ((fuse_mask() & 2) && tc_group_ok(ps, 1)) {
        QLinear* q = ps[0];
        float* yp = ypart((size_t)GS_TC * q->OC);
        launch_tc(1, ps, 1, d_x, yp, q->OC);
        gemv_reduce_resadd_v<<<dim3((q->OC + 255)/256, 1), 256, 0, g_stream>>>(yp, (__half*)h_half, q->OC, GS_TC, 1);
        return;
    }
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

// ---- Batch-invariant verify, host side (see the kernels above gemv4_partial_v) ---------------
// It reproduces the default deterministic decode path only: two-stage GEMVs (mode 2, more than
// one IC slice) and the split-KV attention.
int verify_supported(void) {
    return det_mode() == 2 && det_gs() > 1 && det_gs8() > 1 && attn_mode() == 1;
}
static void verify_need(bool ok, const char* what) {
    if (!ok) { fprintf(stderr, "trapetum: batch-invariant verify: %s\n", what); abort(); }
}
static void verify_check_projs(QLinear* const* ps, int n) {
    verify_need(verify_supported(), "needs the default deterministic decode path (TRAPETUM_DETERMINISTIC=2, split-KV attention)");
    for (int i = 0; i < n; i++)
        verify_need(ps[i]->K_ == 16 && ps[i]->IC == ps[0]->IC && ps[i]->OC % CPB == 0,
                    "needs 4-bit (K=16) projections that share their input width");
}
// q, k, v of M rows (x [M][IC]): one verify GEMV, then the reduce (bias, cast to half) into
// q [M][oq], k [M][ok], v [M][ov].
void qlinear3_forward_half_v(void* hq, void* hk, void* hv, const void* d_x, void* qh, void* kh, void* vh,
                             const void* bq, const void* bk, const void* bv, int M) {
    ensure_stream();
    QLinear* ps[3] = {(QLinear*)hq, (QLinear*)hk, (QLinear*)hv};
    int gy = det_gs();
    if (tc_group_ok(ps, 3)) {                        // same rule as the decode: the same kernel and bits
        verify_need(verify_supported(), "needs the default deterministic decode path");
        gy = GS_TC;
        int OCt = ps[0]->OC + ps[1]->OC + ps[2]->OC;
        float* yp = ypart((size_t)gy * M * OCt);
        launch_tc(M, ps, 3, d_x, yp, OCt);
        dim3 grid((OCt + 255)/256, M);
        gemv_reduce_half3_v<<<grid, 256, 0, g_stream>>>(yp, (__half*)qh, (__half*)kh, (__half*)vh,
            (const float*)bq, (const float*)bk, (const float*)bv, ps[0]->OC, ps[1]->OC, OCt, gy, M);
        return;
    }
    verify_check_projs(ps, 3);
    int OCt = ps[0]->OC + ps[1]->OC + ps[2]->OC;
    float* yp = ypart((size_t)gy * M * OCt);
    launch_partial_v(M, ps, 3, d_x, yp, OCt, gy);
    dim3 grid((OCt + 255)/256, M);
    gemv_reduce_half3_v<<<grid, 256, 0, g_stream>>>(yp, (__half*)qh, (__half*)kh, (__half*)vh,
        (const float*)bq, (const float*)bk, (const float*)bv, ps[0]->OC, ps[1]->OC, OCt, gy, M);
}
// gate, up of M rows: one verify GEMV, then both reductions and SiLU(gate) * up into act [M][n].
void qlinear2_forward_silu_v(void* hg, void* hu, const void* d_x, void* act_h, int M) {
    ensure_stream();
    QLinear* ps[2] = {(QLinear*)hg, (QLinear*)hu};
    verify_need(ps[0]->OC == ps[1]->OC, "gate and up must have the same width");
    if (tc_group_ok(ps, 2)) {
        verify_need(verify_supported(), "needs the default deterministic decode path");
        int n = ps[0]->OC, OCt = 2*n;
        float* yp = ypart((size_t)GS_TC * M * OCt);
        launch_tc(M, ps, 2, d_x, yp, OCt);
        dim3 grid((n + 255)/256, M);
        gemv_reduce_silu_v<<<grid, 256, 0, g_stream>>>(yp, (__half*)act_h, n, GS_TC, M);
        return;
    }
    verify_check_projs(ps, 2);
    int gy = det_gs();
    int n = ps[0]->OC, OCt = 2*n;
    float* yp = ypart((size_t)gy * M * OCt);
    launch_partial_v(M, ps, 2, d_x, yp, OCt, gy);
    dim3 grid((n + 255)/256, M);
    gemv_reduce_silu_v<<<grid, 256, 0, g_stream>>>(yp, (__half*)act_h, n, gy, M);
}
// o or down of M rows, the residual add into h [M][OC] folded into the reduce.
void qlinear_forward_resadd_v(void* handle, const void* d_x, void* h_half, int M) {
    ensure_stream();
    QLinear* ps[1] = {(QLinear*)handle};
    QLinear* q = ps[0];
    if (tc_group_ok(ps, 1)) {
        verify_need(verify_supported(), "needs the default deterministic decode path");
        float* yp = ypart((size_t)GS_TC * M * q->OC);
        launch_tc(M, ps, 1, d_x, yp, q->OC);
        dim3 grid((q->OC + 255)/256, M);
        gemv_reduce_resadd_v<<<grid, 256, 0, g_stream>>>(yp, (__half*)h_half, q->OC, GS_TC, M);
        return;
    }
    verify_check_projs(ps, 1);
    int gy = det_gs();
    float* yp = ypart((size_t)gy * M * q->OC);
    launch_partial_v(M, ps, 1, d_x, yp, q->OC, gy);
    dim3 grid((q->OC + 255)/256, M);
    gemv_reduce_resadd_v<<<grid, 256, 0, g_stream>>>(yp, (__half*)h_half, q->OC, gy, M);
}
// y [M][OC] f32 = x [M][IC] W^T for a 4-bit or 8-bit projection (the LM head).
void qlinear_forward_v(void* handle, const void* d_x, void* d_y, int M) {
    ensure_stream();
    QLinear* q = (QLinear*)handle;
    if (q->K_ == 256) {
        verify_need(verify_supported(), "needs the default deterministic decode path");
        int gy = det_gs8();
        float* yp = ypart((size_t)gy * M * q->OC);
        const __half* X = (const __half*)d_x;
        switch (M) {
            case 1: launch_gemv8_v_m<1>(X, q, yp, gy); break;
            case 2: launch_gemv8_v_m<2>(X, q, yp, gy); break;
            case 4: launch_gemv8_v_m<4>(X, q, yp, gy); break;
            case 8: launch_gemv8_v_m<8>(X, q, yp, gy); break;
            case 16: launch_gemv8_v_m<16>(X, q, yp, gy); break;
            default: verify_need(false, "M must be 1, 2, 4, 8 or 16");
        }
        dim3 grid((q->OC + 255)/256, M);
        gemv_reduce_v<<<grid, 256, 0, g_stream>>>(yp, (float*)d_y, q->OC, gy, M);
        return;
    }
    QLinear* ps[1] = {q};
    if (tc_group_ok(ps, 1)) {
        verify_need(verify_supported(), "needs the default deterministic decode path");
        float* yp = ypart((size_t)GS_TC * M * q->OC);
        launch_tc(M, ps, 1, d_x, yp, q->OC);
        dim3 grid((q->OC + 255)/256, M);
        gemv_reduce_v<<<grid, 256, 0, g_stream>>>(yp, (float*)d_y, q->OC, GS_TC, M);
        return;
    }
    verify_check_projs(ps, 1);
    int gy = det_gs();
    float* yp = ypart((size_t)gy * M * q->OC);
    launch_partial_v(M, ps, 1, d_x, yp, q->OC, gy);
    dim3 grid((q->OC + 255)/256, M);
    gemv_reduce_v<<<grid, 256, 0, g_stream>>>(yp, (float*)d_y, q->OC, gy, M);
}
void op_rmsnorm_v(const void* x_half, const void* w_f32, void* out_half, int n, float eps, int M) {
    ensure_stream();
    rmsnorm_v<<<M, 256, 0, g_stream>>>((const __half*)x_half, (const float*)w_f32, (__half*)out_half, n, eps);
}
// RoPE on q and k of M rows and their KV-cache rows base .. base+M-1 (base = d_tp[1] when d_tp is set).
void op_rope_append_v(void* q_half, void* k_half, const void* v_half, void* cache_k, void* cache_v,
                      int base, const void* d_tp, int n_heads, int n_kv, int head_dim, const void* inv_freq, int M) {
    ensure_stream();
    int total = n_heads*(head_dim/2) + n_kv*(head_dim/2) + n_kv*head_dim;
    dim3 grid((total + 255)/256, M);
    rope_append_v<<<grid, 256, 0, g_stream>>>((__half*)q_half, (__half*)k_half, (const __half*)v_half,
        (__half*)cache_k, (__half*)cache_v, base, (const int*)d_tp, n_heads, n_kv, head_dim, (const float*)inv_freq);
}
// The verify path's attention partial records: M rows of n_heads x chunks(max_seq), reserved on
// the eager first call (never during a capture), never freed once a graph may hold it.
static float* g_attn_part_v = nullptr;
static size_t g_attn_part_v_cap = 0;
void attn_reserve_v(int n_heads, int head_dim, int max_seq, int M) {
    ensure_stream();
    size_t need = (size_t)M * n_heads * attn_chunks(max_seq) * ((size_t)head_dim + 2);
    if (need <= g_attn_part_v_cap) return;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(g_stream, &st);
    verify_need(st == cudaStreamCaptureStatusNone, "attention buffer reserved during a capture");
    if (g_attn_part_v && !g_graphs_captured) cudaFree(g_attn_part_v);
    g_attn_part_v = nullptr;
    verify_need(cudaMalloc(&g_attn_part_v, need * sizeof(float)) == cudaSuccess, "cannot allocate the attention buffer");
    g_attn_part_v_cap = need;
}
// Attention of M query rows (row m at position base + m over base + m + 1 cached positions).
void op_attn_v(const void* q_half, const void* ck_half, const void* cv_half, void* out_half,
               int n_heads, int n_kv, int head_dim, int base, const void* d_tp, int max_seq, float softcap, int M) {
    ensure_stream();
    verify_need(attn_split_ok(head_dim), "needs the split-KV attention (head_dim 64, 128 or 256)");
    int nch = d_tp ? attn_chunks(max_seq) : attn_chunks(base + M);   // graph: the grid covers max_seq
    size_t need = (size_t)M * n_heads * nch * ((size_t)head_dim + 2);
    verify_need(need <= g_attn_part_v_cap, "attention buffer not reserved (attn_reserve_v)");
    dim3 grid(n_heads, nch, M), block(32 * ATTN_WARPS);
    const __half* qh = (const __half*)q_half;
    const __half* kh = (const __half*)ck_half;
    const __half* vh = (const __half*)cv_half;
    const int* tp = (const int*)d_tp;
    if (head_dim == 64)
        attn_split_v<2><<<grid, block, 0, g_stream>>>(qh, kh, vh, g_attn_part_v, n_heads, n_kv, base, tp, softcap, nch);
    else if (head_dim == 128)
        attn_split_v<4><<<grid, block, 0, g_stream>>>(qh, kh, vh, g_attn_part_v, n_heads, n_kv, base, tp, softcap, nch);
    else
        attn_split_v<8><<<grid, block, 0, g_stream>>>(qh, kh, vh, g_attn_part_v, n_heads, n_kv, base, tp, softcap, nch);
    dim3 gm(n_heads, M);
    attn_merge_v<<<gm, head_dim, 0, g_stream>>>(g_attn_part_v, (__half*)out_half, n_heads, head_dim, base, tp, nch);
}
// Row-wise device argmax over the first n logits of M rows (row stride ld): enqueue, then read.
static float*    g_amv_val = 0;
static unsigned* g_amv_idx = 0;
static unsigned* g_amv_out = 0;
void dev_argmax_v_enqueue(const void* d_logits, int n, int ld, int M) {
    ensure_stream();
    const int tpb = 256;
    int ntg = (n + tpb - 1) / tpb;
    if (ntg > ARGMAX_NTG_MAX) ntg = ARGMAX_NTG_MAX;
    if (ntg < 1) ntg = 1;
    if (!g_amv_val) {
        cudaMalloc(&g_amv_val, (size_t)VMAX*ARGMAX_NTG_MAX*sizeof(float));
        cudaMalloc(&g_amv_idx, (size_t)VMAX*ARGMAX_NTG_MAX*sizeof(unsigned));
        cudaMalloc(&g_amv_out, (size_t)VMAX*sizeof(unsigned));
    }
    dim3 g1(ntg, M);
    argmax_v_stage1<<<g1, tpb, 0, g_stream>>>((const float*)d_logits, n, ld, g_amv_val, g_amv_idx);
    argmax_v_stage2<<<M, tpb, 0, g_stream>>>(g_amv_val, g_amv_idx, ntg, g_amv_out);
}
void dev_argmax_v_read(unsigned* out, int M) {
    ensure_stream();
    cudaMemcpyAsync(out, g_amv_out, (size_t)M*sizeof(unsigned), cudaMemcpyDeviceToHost, g_stream);
    cudaError_t e = cudaStreamSynchronize(g_stream);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "trapetum: verify step failed: %s\n", cudaGetErrorString(e));
        abort();
    }
}

// 1 when the 4-bit GEMVs take the tensor-core path (for the tools' mode line).
int gemv_tc_active(void) { return tc_on() ? 1 : 0; }

// Test helper: fill the GEMV partials, both attention partial buffers and both argmax scratches with
// 0xFF bytes (NaN), so that a read of a slot the checked call did not write shows up as a mismatch.
// Never during a capture.
void dev_debug_scramble(void) {
    ensure_stream();
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(g_stream, &st);
    if (st != cudaStreamCaptureStatusNone) return;
    if (g_ypart) cudaMemsetAsync(g_ypart, 0xFF, g_ypart_cap * sizeof(float), g_stream);
    if (g_attn_part) cudaMemsetAsync(g_attn_part, 0xFF, g_attn_part_cap * sizeof(float), g_stream);
    if (g_attn_part_v) cudaMemsetAsync(g_attn_part_v, 0xFF, g_attn_part_v_cap * sizeof(float), g_stream);
    if (g_am_val) {
        cudaMemsetAsync(g_am_val, 0xFF, (size_t)ARGMAX_NTG_MAX * sizeof(float), g_stream);
        cudaMemsetAsync(g_am_idx, 0xFF, (size_t)ARGMAX_NTG_MAX * sizeof(unsigned), g_stream);
        cudaMemsetAsync(g_am_out, 0xFF, sizeof(unsigned), g_stream);
    }
    if (g_amv_val) {
        cudaMemsetAsync(g_amv_val, 0xFF, (size_t)VMAX * ARGMAX_NTG_MAX * sizeof(float), g_stream);
        cudaMemsetAsync(g_amv_idx, 0xFF, (size_t)VMAX * ARGMAX_NTG_MAX * sizeof(unsigned), g_stream);
        cudaMemsetAsync(g_amv_out, 0xFF, (size_t)VMAX * sizeof(unsigned), g_stream);
    }
    cudaStreamSynchronize(g_stream);
}

} // extern "C"
