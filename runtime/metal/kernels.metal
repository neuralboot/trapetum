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
struct AttnParams { int n_heads; int n_kv; int head_dim; int seqlen; };
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
        if (d == 0) scores[t] = red[0] * scale;
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
