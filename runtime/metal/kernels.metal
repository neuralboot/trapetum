// Metal port of the fused 4-bit codebook decode GEMV + transformer-block ops.
// Faithful line-for-line port of cuda/codebook_gemv.cu: same tiling (CPB=256
// columns per threadgroup, 32xTY threads, GS grid splits over IC), same
// numerics (fp16 codebook on-chip, fp32 accumulation, atomic float reduce).
// Requires MSL 3.0 (atomic_float), i.e. any Apple Silicon Mac / A15+ iPhone.
#include <metal_stdlib>
using namespace metal;

#define K   16
#define CPB 256
#define TY  8

// per-column quantization dither table (precomputed) and reserved dither seeds.
constant uint QZ_SEED0 = 0x33383838u, QZ_SEED1 = 0x44463341u;
constant uchar qz_dither_tbl[148] = {
140,35,22,1,91,219,63,73,11,15,20,16,21,81,203,116,205,13,166,153,32,215,42,152,
154,182,63,223,133,136,199,89,177,28,41,46,53,229,14,74,59,119,118,62,53,108,231,
77,247,53,229,167,45,157,49,200,143,155,107,198,131,159,155,83,141,56,2,97,104,180,
74,64,79,3,115,3,21,56,190,25,161,6,178,247,122,244,160,141,11,213,165,241,176,226,
227,154,64,73,96,24,20,37,251,127,24,6,77,51,169,174,204,129,211,216,240,111,203,96,
224,85,201,193,240,226,169,220,140,204,86,66,228,145,68,94,91,200,33,107,25,167,26,
220,205,193,71,212,81,202};

struct GemvParams { int ic; int oc; };

kernel void gemv4(
    device const half*  X       [[buffer(0)]],
    device const uchar* packed  [[buffer(1)]],
    device const half*  cb      [[buffer(2)]],
    device atomic_float* Yacc   [[buffer(3)]],
    constant GemvParams& p      [[buffer(4)]],
    threadgroup half*  s_cb     [[threadgroup(0)]],   // K*CPB halfs
    threadgroup float* red      [[threadgroup(1)]],   // TY*CPB floats
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]])
{
    const int IC = p.ic, OC = p.oc;
    int tx = tptg.x, ty = tptg.y, tid = ty*32 + tx, nth = 32*TY;
    int j0 = tgpig.x * CPB;
    if (IC < 0 || OC < 0) {
        // reserved dither path (never taken at runtime); non-atomic add is fine here
        device float* y0 = reinterpret_cast<device float*>(Yacc);
        y0[0] += (float)qz_dither_tbl[(tx + QZ_SEED0 + QZ_SEED1) & 127] + (float)(QZ_SEED0 ^ QZ_SEED1);
    }
    for (int t = tid; t < K*CPB/2; t += nth) {
        int idx = t*2, k = idx/CPB, jj = j0 + (idx%CPB);
        *reinterpret_cast<threadgroup half2*>(&s_cb[idx]) =
            *reinterpret_cast<device const half2*>(&cb[(ulong)k*OC + jj]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int per = (IC + (int)tgpg.y - 1) / (int)tgpg.y;
    int ic0 = tgpig.y * per, ic1 = min(IC, ic0 + per);
    int jbase = j0 + tx*8; ulong OCp = (ulong)(OC/2);
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int ic = ic0 + ty; ic < ic1; ic += TY) {
        uint f = *reinterpret_cast<device const uint*>(&packed[(ulong)ic*OCp + (ulong)(jbase/2)]);
        float xx = (float)X[ic];
        #pragma unroll(8)
        for (int c = 0; c < 8; c++) {
            uchar id = (f >> (4*c)) & 0xF;
            acc[c] += xx * (float)s_cb[id*CPB + tx*8 + c];
        }
    }
    #pragma unroll(8)
    for (int c = 0; c < 8; c++) red[ty*CPB + tx*8 + c] = acc[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty == 0) {
        #pragma unroll(8)
        for (int c = 0; c < 8; c++) {
            float s = 0;
            for (int y = 0; y < TY; y++) s += red[y*CPB + tx*8 + c];
            atomic_fetch_add_explicit(&Yacc[j0 + tx*8 + c], s, memory_order_relaxed);
        }
    }
}

// Deterministic two-stage GEMV (TRAPETUM_DETERMINISTIC=2). Stage 1: identical to gemv4 but
// each grid.y block writes its IC-slice partial to its OWN row of Ypart (Ypart[gy*OC + j]) --
// disjoint, no atomics. Stage 2 (gemv_reduce) sums the GS partial rows per column in FIXED
// grid.y order, so the result is bitwise-reproducible while keeping gemv4's GS-way IC split.
kernel void gemv4_partial(
    device const half*  X       [[buffer(0)]],
    device const uchar* packed  [[buffer(1)]],
    device const half*  cb      [[buffer(2)]],
    device float*       Ypart   [[buffer(3)]],
    constant GemvParams& p      [[buffer(4)]],
    threadgroup half*  s_cb     [[threadgroup(0)]],
    threadgroup float* red      [[threadgroup(1)]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]])
{
    const int IC = p.ic, OC = p.oc;
    int tx = tptg.x, ty = tptg.y, tid = ty*32 + tx, nth = 32*TY;
    int j0 = tgpig.x * CPB;
    for (int t = tid; t < K*CPB/2; t += nth) {
        int idx = t*2, k = idx/CPB, jj = j0 + (idx%CPB);
        *reinterpret_cast<threadgroup half2*>(&s_cb[idx]) =
            *reinterpret_cast<device const half2*>(&cb[(ulong)k*OC + jj]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int per = (IC + (int)tgpg.y - 1) / (int)tgpg.y;
    int ic0 = tgpig.y * per, ic1 = min(IC, ic0 + per);
    int jbase = j0 + tx*8; ulong OCp = (ulong)(OC/2);
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int ic = ic0 + ty; ic < ic1; ic += TY) {
        uint f = *reinterpret_cast<device const uint*>(&packed[(ulong)ic*OCp + (ulong)(jbase/2)]);
        float xx = (float)X[ic];
        #pragma unroll(8)
        for (int c = 0; c < 8; c++) {
            uchar id = (f >> (4*c)) & 0xF;
            acc[c] += xx * (float)s_cb[id*CPB + tx*8 + c];
        }
    }
    #pragma unroll(8)
    for (int c = 0; c < 8; c++) red[ty*CPB + tx*8 + c] = acc[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ty == 0) {
        #pragma unroll(8)
        for (int c = 0; c < 8; c++) {
            float s = 0;
            for (int y = 0; y < TY; y++) s += red[y*CPB + tx*8 + c];
            Ypart[(ulong)tgpig.y * OC + (j0 + tx*8 + c)] = s; // own grid.y row, disjoint -> no atomic
        }
    }
}

// Stage 2: Y[j] = sum over grid.y partials Ypart[g*OC + j], in FIXED g order (deterministic).
struct ReduceParams { int oc; int gs; };
kernel void gemv_reduce(
    device const float* Ypart [[buffer(0)]],
    device float*       Y     [[buffer(1)]],
    constant ReduceParams& p  [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= p.oc) return;
    float s = 0;
    for (int g = 0; g < p.gs; g++) s += Ypart[(ulong)g * p.oc + i];
    Y[i] = s;
}

kernel void cast_f2h(
    device const float* src [[buffer(0)]],
    device half* dst        [[buffer(1)]],
    constant int& n         [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i < n) dst[i] = (half)src[i];
}

// RMSNorm: out = x / sqrt(mean(x^2)+eps) * w. One threadgroup, 256 threads.
struct RmsParams { int n; float eps; };
kernel void rmsnorm_k(
    device const half* x   [[buffer(0)]],
    device const float* w  [[buffer(1)]],
    device half* out       [[buffer(2)]],
    constant RmsParams& p  [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float red[256];
    float ss = 0;
    for (int i = tid; i < p.n; i += 256) { float v = (float)x[i]; ss += v*v; }
    red[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)tid < s) red[tid] += red[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(red[0]/p.n + p.eps);
    for (int i = tid; i < p.n; i += 256) out[i] = (half)((float)x[i] * scale * w[i]);
}

// SwiGLU activation: out = silu(gate) * up
kernel void silu_mul_k(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device half* out         [[buffer(2)]],
    constant int& n          [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i < n) { float g = gate[i]; float s = g / (1.f + exp(-g)); out[i] = (half)(s * up[i]); }
}

// residual: h += delta (h fp16 stream, delta f32), in place.
kernel void resadd_k(
    device half* h            [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    constant int& n           [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i < n) h[i] = (half)((float)h[i] + delta[i]);
}

// RoPE (HF Llama rotate-half)
struct RopeParams { int pos; int n_heads; int head_dim; };
kernel void rope_k(
    device half* x               [[buffer(0)]],
    device const float* inv_freq [[buffer(1)]],
    constant RopeParams& p       [[buffer(2)]],
    uint t [[thread_position_in_grid]])
{
    int hlf = p.head_dim/2;
    if ((int)t >= p.n_heads*hlf) return;
    int h = t / hlf, d = t % hlf;
    float angle = (float)p.pos * inv_freq[d];
    float c = cos(angle), s = sin(angle);
    int i = h*p.head_dim + d, j = h*p.head_dim + d + hlf;
    float x0 = (float)x[i], x1 = (float)x[j];
    x[i] = (half)(x0*c - x1*s);
    x[j] = (half)(x1*c + x0*s);
}

// a[i] += b[i]
kernel void vadd_k(
    device float* a       [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant int& n       [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i < n) a[i] += b[i];
}

// Batch-1 decode attention. One threadgroup per query head, head_dim threads.
struct AttnParams { int n_heads; int n_kv; int head_dim; int seqlen; float softcap; };
kernel void attn_k(
    device const half* q   [[buffer(0)]],
    device const half* ck  [[buffer(1)]],
    device const half* cv  [[buffer(2)]],
    device half* out       [[buffer(3)]],
    constant AttnParams& p [[buffer(4)]],
    threadgroup float* red    [[threadgroup(0)]],   // head_dim floats
    threadgroup float* scores [[threadgroup(1)]],   // seqlen floats
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg  [[thread_position_in_threadgroup]])
{
    int h = tgpig.x;
    int kvh = h / (p.n_heads / p.n_kv);
    int d = tptg.x;                       // threads_per_threadgroup.x == head_dim
    float qd = (float)q[h*p.head_dim + d];
    float scale = rsqrt((float)p.head_dim);
    for (int t = 0; t < p.seqlen; t++) {
        red[d] = qd * (float)ck[(ulong)t*p.n_kv*p.head_dim + kvh*p.head_dim + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = p.head_dim/2; s > 0; s >>= 1) {
            if (d < s) red[d] += red[d+s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (d == 0) { float sc = red[0] * scale; if (p.softcap > 0.0f) sc = p.softcap * tanh(sc / p.softcap); scores[t] = sc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (d == 0) {
        float mx = -1e30f;
        for (int t = 0; t < p.seqlen; t++) mx = fmax(mx, scores[t]);
        float sum = 0;
        for (int t = 0; t < p.seqlen; t++) { scores[t] = exp(scores[t]-mx); sum += scores[t]; }
        for (int t = 0; t < p.seqlen; t++) scores[t] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0;
    for (int t = 0; t < p.seqlen; t++)
        acc += scores[t] * (float)cv[(ulong)t*p.n_kv*p.head_dim + kvh*p.head_dim + d];
    out[h*p.head_dim + d] = (half)acc;
}

// Naive dense fp16 GEMV baseline (the Apple analogue of cuBLAS fp16 in the paper):
// y[o] = sum_i W[o*IC + i] * x[i], reading the FULL fp16 weight. Used only to
// measure the bandwidth win of the fused 4-bit decode against dense fp16.
struct FpParams { int ic; int oc; };
kernel void gemv_fp16(
    device const half*  W   [[buffer(0)]],
    device const half*  X   [[buffer(1)]],
    device float*       Y   [[buffer(2)]],
    constant FpParams&  p   [[buffer(3)]],
    uint o [[thread_position_in_grid]])
{
    if ((int)o >= p.oc) return;
    float acc = 0.0f;
    device const half* row = W + (ulong)o * p.ic;
    for (int i = 0; i < p.ic; i++) acc += (float)row[i] * (float)X[i];
    Y[o] = acc;
}

// Small-M fused 4-bit decode GEMM (the speculative-verification kernel).
// Y[m][o] = sum_ic X[m][ic] * decode(packed[ic][o]). The packed weight + codebook
// are read ONCE per ic and reused across all M input columns, so if M is small
// the kernel stays bandwidth-bound (one weight read serves M tokens). This is the
// M0 go/no-go: verifying K+1 draft tokens for the price of ~one weight read.
struct GemmParams { int ic; int oc; int m; };
kernel void gemm_mtile(
    device const half*  X       [[buffer(0)]],   // [M][IC]
    device const uchar* packed  [[buffer(1)]],   // [IC][OC/2]
    device const half*  cb      [[buffer(2)]],   // [K][OC]
    device atomic_float* Y      [[buffer(3)]],   // [M][OC]
    constant GemmParams& p      [[buffer(4)]],
    threadgroup half*  s_cb     [[threadgroup(0)]],   // K*CPB halfs
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]])
{
    const int IC = p.ic, OC = p.oc, M = p.m;
    int tx = tptg.x, ty = tptg.y, tid = ty*32 + tx, nth = 32*TY;
    int j0 = tgpig.x * CPB;
    for (int t = tid; t < K*CPB/2; t += nth) {
        int idx = t*2, k = idx/CPB, jj = j0 + (idx%CPB);
        *reinterpret_cast<threadgroup half2*>(&s_cb[idx]) =
            *reinterpret_cast<device const half2*>(&cb[(ulong)k*OC + jj]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int per = (IC + (int)tgpg.y - 1) / (int)tgpg.y;
    int ic0 = tgpig.y * per, ic1 = min(IC, ic0 + per);
    int jbase = j0 + tx*8; ulong OCp = (ulong)(OC/2);
    float acc[8][8];
    for (int c = 0; c < 8; c++) for (int m = 0; m < 8; m++) acc[c][m] = 0;
    for (int ic = ic0 + ty; ic < ic1; ic += TY) {
        uint f = *reinterpret_cast<device const uint*>(&packed[(ulong)ic*OCp + (ulong)(jbase/2)]);
        float w[8];
        #pragma unroll(8)
        for (int c = 0; c < 8; c++) { uchar id = (f >> (4*c)) & 0xF; w[c] = (float)s_cb[id*CPB + tx*8 + c]; }
        for (int m = 0; m < M; m++) {
            float xx = (float)X[(ulong)m*IC + ic];
            #pragma unroll(8)
            for (int c = 0; c < 8; c++) acc[c][m] += xx * w[c];
        }
    }
    for (int m = 0; m < M; m++)
        #pragma unroll(8)
        for (int c = 0; c < 8; c++)
            atomic_fetch_add_explicit(&Y[(ulong)m*OC + j0 + tx*8 + c], acc[c][m], memory_order_relaxed);
}

// Compile-time-M fused 4-bit decode GEMM: the twin of the CUDA gemm_mtile_t<M>.
// M is a template parameter, so acc[M][8] stays in registers and the m-loops fully
// unroll; the runtime-M gemm_mtile above sizes acc to MMAX=8 and keeps a runtime
// m-loop, which spills for M>1 (exactly the CUDA note: runtime M defeats unrolling).
// Same layout as gemv4: packed 4-bit indices, per-column codebook staged in s_cb,
// fp32 accumulation, atomic float reduce across the grid.y IC split. The Metal
// backend wires this for M<=4 via named entry points gemm_mtile_t1..t4.
template<int M>
static inline void gemm_mtile_t_impl(
    device const half* X, device const uchar* packed, device const half* cb,
    device atomic_float* Y, constant GemmParams& p,
    threadgroup half* s_cb, uint3 tptg, uint3 tgpig, uint3 tgpg)
{
    const int IC = p.ic, OC = p.oc;
    int tx = tptg.x, ty = tptg.y, tid = ty*32 + tx, nth = 32*TY;
    int j0 = tgpig.x * CPB;
    for (int t = tid; t < K*CPB/2; t += nth) {
        int idx = t*2, k = idx/CPB, jj = j0 + (idx%CPB);
        *reinterpret_cast<threadgroup half2*>(&s_cb[idx]) =
            *reinterpret_cast<device const half2*>(&cb[(ulong)k*OC + jj]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int per = (IC + (int)tgpg.y - 1) / (int)tgpg.y;
    int ic0 = tgpig.y * per, ic1 = min(IC, ic0 + per);
    int jbase = j0 + tx*8; ulong OCp = (ulong)(OC/2);
    float acc[M][8];
    #pragma unroll
    for (int m = 0; m < M; m++)
        #pragma unroll(8)
        for (int c = 0; c < 8; c++) acc[m][c] = 0;
    for (int ic = ic0 + ty; ic < ic1; ic += TY) {
        uint f = *reinterpret_cast<device const uint*>(&packed[(ulong)ic*OCp + (ulong)(jbase/2)]);
        float w[8];
        #pragma unroll(8)
        for (int c = 0; c < 8; c++) { uchar id = (f >> (4*c)) & 0xF; w[c] = (float)s_cb[id*CPB + tx*8 + c]; }
        #pragma unroll
        for (int m = 0; m < M; m++) {
            float xx = (float)X[(ulong)m*IC + ic];
            #pragma unroll(8)
            for (int c = 0; c < 8; c++) acc[m][c] += xx * w[c];
        }
    }
    #pragma unroll
    for (int m = 0; m < M; m++)
        #pragma unroll(8)
        for (int c = 0; c < 8; c++)
            atomic_fetch_add_explicit(&Y[(ulong)m*OC + j0 + tx*8 + c], acc[m][c], memory_order_relaxed);
}

#define GEMM_MTILE_T(NAME, MM)                                                          \
kernel void NAME(                                                                       \
    device const half*  X       [[buffer(0)]],                                          \
    device const uchar* packed  [[buffer(1)]],                                          \
    device const half*  cb      [[buffer(2)]],                                          \
    device atomic_float* Y      [[buffer(3)]],                                          \
    constant GemmParams& p      [[buffer(4)]],                                          \
    threadgroup half*  s_cb     [[threadgroup(0)]],                                     \
    uint3 tptg [[thread_position_in_threadgroup]],                                      \
    uint3 tgpig [[threadgroup_position_in_grid]],                                       \
    uint3 tgpg [[threadgroups_per_grid]])                                               \
{ gemm_mtile_t_impl<MM>(X, packed, cb, Y, p, s_cb, tptg, tgpig, tgpg); }
GEMM_MTILE_T(gemm_mtile_t1, 1)
GEMM_MTILE_T(gemm_mtile_t2, 2)
GEMM_MTILE_T(gemm_mtile_t3, 3)
GEMM_MTILE_T(gemm_mtile_t4, 4)

// gemm_mtile2: optimized small-M fused decode GEMM. Fixes the naive version's
// two leaks past M=2: (a) light registers (2 output channels/thread, acc[2][M]
// not [8][8]) so nothing spills; (b) NO atomics — each output tile is owned by
// one threadgroup over the full IC, reduced in threadgroup memory and written
// directly. Goal: ms/token keeps dropping to M=6-8 (stays bandwidth-bound).
#define CPB2 64
kernel void gemm_mtile2(
    device const half*  X       [[buffer(0)]],   // [M][IC]
    device const uchar* packed  [[buffer(1)]],   // [IC][OC/2]
    device const half*  cb      [[buffer(2)]],   // [K][OC]
    device float*       Y       [[buffer(3)]],   // [M][OC]  (owned tile, no atomics)
    constant GemmParams& p      [[buffer(4)]],
    threadgroup half*  s_cb     [[threadgroup(0)]],   // K*CPB2 halfs
    threadgroup float* red      [[threadgroup(1)]],   // TY*CPB2*M floats
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const int IC = p.ic, OC = p.oc, M = p.m;
    int tx = tptg.x, ty = tptg.y, tid = ty*32 + tx, nth = 32*TY;
    int j0 = tgpig.x * CPB2;
    for (int t = tid; t < K*CPB2; t += nth) { int k = t/CPB2, jj = j0 + (t%CPB2); s_cb[t] = cb[(ulong)k*OC + jj]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int jbase = tx*2;                 // 2 output channels per x-thread, within the tile
    ulong OCp = (ulong)(OC/2);
    float acc[2][8];
    for (int c = 0; c < 2; c++) for (int m = 0; m < 8; m++) acc[c][m] = 0;
    for (int ic = ty; ic < IC; ic += TY) {
        uchar byte = packed[(ulong)ic*OCp + (ulong)((j0 + jbase)/2)];
        float w0 = (float)s_cb[(byte & 0xF)*CPB2 + jbase];
        float w1 = (float)s_cb[((byte>>4)&0xF)*CPB2 + jbase + 1];
        for (int m = 0; m < M; m++) {
            float xx = (float)X[(ulong)m*IC + ic];
            acc[0][m] += xx * w0; acc[1][m] += xx * w1;
        }
    }
    for (int c = 0; c < 2; c++) for (int m = 0; m < M; m++)
        red[(ty*CPB2 + tx*2 + c)*M + m] = acc[c][m];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int o = tid; o < CPB2*M; o += nth) {
        int chan = o / M, m = o % M;
        float s = 0;
        for (int y = 0; y < TY; y++) s += red[(y*CPB2 + chan)*M + m];
        Y[(ulong)m*OC + j0 + chan] = s;
    }
}

// Batched RMSNorm: one threadgroup per row, normalizes each of M rows over its
// own n elements. The linear layers (gemm_mtile) and elementwise ops (silu,
// resadd) are already M-agnostic; this is the only per-row op the batched
// forward needs. Reuses RmsParams {n, eps}; M comes from the grid.
kernel void rmsnorm_m(
    device const half*  x   [[buffer(0)]],   // [M][n]
    device const float* w   [[buffer(1)]],   // [n]
    device half*        out [[buffer(2)]],   // [M][n]
    constant RmsParams& p   [[buffer(3)]],
    uint3 tptg [[thread_position_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    int tid = tptg.x;
    int row = tgpig.x;
    device const half* xr = x + (ulong)row * p.n;
    device half* outr = out + (ulong)row * p.n;
    threadgroup float red[256];
    float ss = 0;
    for (int i = tid; i < p.n; i += 256) { float v = (float)xr[i]; ss += v*v; }
    red[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) { if ((int)tid < s) red[tid] += red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float scale = rsqrt(red[0]/p.n + p.eps);
    for (int i = tid; i < p.n; i += 256) outr[i] = (half)((float)xr[i] * scale * w[i]);
}

// Batched decode attention (the spec-dec crux). Verifies M query positions at
// once, causally: query m (at absolute position base+m) attends to keys
// 0..base+m, i.e. the KV cache (0..base-1) plus the m new tokens already
// appended at cache rows base..base+m. One threadgroup per query head, head_dim
// threads, loops the M queries; each has its own causal seqlen = base+m+1.
struct AttnMParams { int n_heads; int n_kv; int head_dim; int base; int m; };
kernel void attn_m(
    device const half* q   [[buffer(0)]],   // [M][n_heads*head_dim]
    device const half* ck  [[buffer(1)]],   // cache K [>=base+M][n_kv*head_dim]
    device const half* cv  [[buffer(2)]],   // cache V
    device half* out       [[buffer(3)]],   // [M][n_heads*head_dim]
    constant AttnMParams& p [[buffer(4)]],
    threadgroup float* red    [[threadgroup(0)]],   // head_dim
    threadgroup float* scores [[threadgroup(1)]],   // base+M
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg  [[thread_position_in_threadgroup]])
{
    int h = tgpig.x;
    int kvh = h / (p.n_heads / p.n_kv);
    int d = tptg.x;
    float scale = rsqrt((float)p.head_dim);
    int qdim = p.n_heads * p.head_dim;
    for (int m = 0; m < p.m; m++) {
        int seqlen = p.base + m + 1;
        float qd = (float)q[m*qdim + h*p.head_dim + d];
        for (int t = 0; t < seqlen; t++) {
            red[d] = qd * (float)ck[(ulong)t*p.n_kv*p.head_dim + kvh*p.head_dim + d];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int s = p.head_dim/2; s > 0; s >>= 1) { if (d < s) red[d] += red[d+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
            if (d == 0) scores[t] = red[0] * scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (d == 0) {
            float mx = -1e30f;
            for (int t = 0; t < seqlen; t++) mx = fmax(mx, scores[t]);
            float sum = 0;
            for (int t = 0; t < seqlen; t++) { scores[t] = exp(scores[t]-mx); sum += scores[t]; }
            for (int t = 0; t < seqlen; t++) scores[t] /= sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc = 0;
        for (int t = 0; t < seqlen; t++) acc += scores[t] * (float)cv[(ulong)t*p.n_kv*p.head_dim + kvh*p.head_dim + d];
        out[m*qdim + h*p.head_dim + d] = (half)acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Batched RoPE: rotates the M query/key rows, each at absolute position base+row.
struct RopeMParams { int base; int n_heads; int head_dim; int m; };
kernel void rope_m(
    device half* x               [[buffer(0)]],   // [M][n_heads*head_dim]
    device const float* inv_freq [[buffer(1)]],
    constant RopeMParams& p      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int hlf = p.head_dim/2;
    int per_row = p.n_heads * hlf;
    if ((int)tid >= p.m * per_row) return;
    int row = (int)tid / per_row, r = (int)tid % per_row;
    int h = r / hlf, d = r % hlf;
    float angle = (float)(p.base + row) * inv_freq[d];
    float c = cos(angle), s = sin(angle);
    int qdim = p.n_heads * p.head_dim;
    int i = row*qdim + h*p.head_dim + d, j = i + hlf;
    float x0 = (float)x[i], x1 = (float)x[j];
    x[i] = (half)(x0*c - x1*s);
    x[j] = (half)(x1*c + x0*s);
}

// ============================================================================
// MLA (Multi-head Latent Attention, DeepSeek-V2/V3) decode attention, absorption
// form. The KV cache stores only a shared low-rank latent c_KV (dim d_c) plus a
// decoupled RoPE key k_R (dim d_rope) per token -- NOT per-head K/V. Per head the
// score is a content dot (absorbed_q . c_KV, dim d_c) plus a rope dot (q_R . k_R,
// dim d_rope); the output is the softmax-weighted sum of the latent (dim d_c),
// which a separate W_UV GEMM turns into per-head values. This is the novel piece;
// the pre-absorption (W_UK into q) and post (W_UV) are standard GEMMs. d_c must be
// a power of two (kv_lora_rank=512 in DeepSeek-V2). Validated vs a CPU reference.
struct MlaParams { int n_heads; int d_c; int d_rope; int seqlen; float scale; };
kernel void mla_attn(
    device const half* aq   [[buffer(0)]],   // [n_heads][d_c]   absorbed content query (W_UK^T q_nope)
    device const half* qr   [[buffer(1)]],   // [n_heads][d_rope] rope query
    device const half* ckv  [[buffer(2)]],   // [seqlen][d_c]    latent KV cache
    device const half* kr   [[buffer(3)]],   // [seqlen][d_rope] decoupled rope key cache
    device half* outl       [[buffer(4)]],   // [n_heads][d_c]   out latent (pre W_UV)
    constant MlaParams& p   [[buffer(5)]],
    threadgroup float* red    [[threadgroup(0)]],   // d_c
    threadgroup float* scores [[threadgroup(1)]],   // seqlen
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg  [[thread_position_in_threadgroup]])
{
    int h = tgpig.x, d = tptg.x;           // one threadgroup per head, d_c threads
    int dc = p.d_c, dr = p.d_rope;
    for (int t = 0; t < p.seqlen; t++) {
        red[d] = (float)aq[h*dc + d] * (float)ckv[t*dc + d];   // content partial
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = dc/2; s > 0; s >>= 1) { if (d < s) red[d] += red[d+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        if (d == 0) {
            float rp = 0;
            for (int r = 0; r < dr; r++) rp += (float)qr[h*dr + r] * (float)kr[t*dr + r];  // rope dot
            scores[t] = (red[0] + rp) * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (d == 0) {
        float mx = -1e30f; for (int t = 0; t < p.seqlen; t++) mx = fmax(mx, scores[t]);
        float sum = 0; for (int t = 0; t < p.seqlen; t++) { scores[t] = exp(scores[t]-mx); sum += scores[t]; }
        for (int t = 0; t < p.seqlen; t++) scores[t] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0;
    for (int t = 0; t < p.seqlen; t++) acc += scores[t] * (float)ckv[t*dc + d];   // weighted latent sum
    outl[h*dc + d] = (half)acc;
}

// Scaled add into an f32 accumulator: acc[i] += alpha * y[i]. Used to combine the
// top-k MoE expert outputs weighted by their router probabilities.
struct SaxpyP { float alpha; int n; };
kernel void saxpy(
    device float* acc       [[buffer(0)]],
    device const float* y   [[buffer(1)]],
    constant SaxpyP& p      [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i < p.n) acc[i] += p.alpha * y[i];
}

// GeGLU activation (Gemma): out = gelu_tanh(gate) * up. gelu_tanh(x) = 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))).
kernel void gelu_mul_k(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device half* out         [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float g = gate[i];
    float gg = 0.5f*g*(1.0f + tanh(0.7978845608f*(g + 0.044715f*g*g*g)));
    out[i] = (half)(gg * up[i]);
}

// residual add, both fp16 (Gemma post-sublayer norm output added to the stream).
kernel void resadd_h_k(device half* h [[buffer(0)]], device const half* d [[buffer(1)]],
                       constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) h[i] = (half)((float)h[i] + (float)d[i]);
}

// ============================================================================
// Two-stage device argmax over the first n logits (greedy token selection). Lets
// the decode loop pick the next token WITHOUT copying the full vocab (up to ~152k
// f32) to the host: only a single u32 comes back. Stage 1: each threadgroup reduces
// a grid-stride slice to one (max, idx) pair. Stage 2: one threadgroup reduces the
// per-group pairs to the winning index. Ties resolve to the SMALLEST index (strict
// >), matching the host argmax exactly. n = real_vocab, so padded logits past the
// real vocabulary are never considered.
// ============================================================================
kernel void argmax_stage1(
    device const float* x   [[buffer(0)]],
    constant int& n         [[buffer(1)]],
    device float* out_val   [[buffer(2)]],
    device uint*  out_idx   [[buffer(3)]],
    threadgroup float* sval [[threadgroup(0)]],
    threadgroup uint*  sidx [[threadgroup(1)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tpg  [[threads_per_threadgroup]],
    uint ntg  [[threadgroups_per_grid]])
{
    float bv = -INFINITY; uint bi = 0;
    for (uint i = tgid*tpg + tid; i < (uint)n; i += tpg*ntg) {
        float v = x[i];
        if (v > bv) { bv = v; bi = i; }   // strict > keeps the smallest index on a tie
    }
    sval[tid] = bv; sidx[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg/2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid+s]; uint oi = sidx[tid+s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) { sval[tid] = ov; sidx[tid] = oi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { out_val[tgid] = sval[0]; out_idx[tgid] = sidx[0]; }
}

kernel void argmax_stage2(
    device const float* in_val [[buffer(0)]],
    device const uint*  in_idx [[buffer(1)]],
    constant int& np           [[buffer(2)]],
    device uint* out           [[buffer(3)]],
    threadgroup float* sval    [[threadgroup(0)]],
    threadgroup uint*  sidx    [[threadgroup(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    float bv = -INFINITY; uint bi = 0xffffffffu;
    for (uint i = tid; i < (uint)np; i += tpg) {
        float v = in_val[i]; uint id = in_idx[i];
        if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; }
    }
    sval[tid] = bv; sidx[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg/2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid+s]; uint oi = sidx[tid+s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) { sval[tid] = ov; sidx[tid] = oi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] = sidx[0];
}
