// C-ABI wrapper around the fused 4-bit codebook decode GEMV, for the Rust runtime.
//
// A QLinear holds only the quantized weights (packed 4-bit indices + per-output
// codebook) resident on the GPU. Activations live in caller-owned DEVICE buffers, so
// layers chain on-device with no host<->device copy between them: the kernel writes f32,
// a cast kernel converts it to half for the next layer. Host side speaks f32 + u8.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define K 16
#define CPB 256
#define TY 8
#define GS 20

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
                       int n_heads, int n_kv, int head_dim, int seqlen) {
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
        if (d == 0) scores[t] = red[0] * scale;
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
// per (ic) serves all M columns; that is what keeps the verify bandwidth-bound.
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

// One dedicated stream for all device ops, so the decode chain can be CUDA-graph
// captured (launches MUST be on the captured stream, else they are not recorded).
static cudaStream_t g_stream = 0;
static void ensure_stream() { if (!g_stream) cudaStreamCreate(&g_stream); }

struct QLinear { unsigned char* d_packed; __half* d_cb; int IC, OC; };

extern "C" {

// upload the quantized weights once; activations are external device buffers
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
    return q;
}

// d_x: device half (IC,), d_y: device f32 (OC,). No host copies; fully on-device.
void qlinear_forward_dev(void* handle, const void* d_x, void* d_y) {
    QLinear* q = (QLinear*)handle;
    ensure_stream();
    cudaMemsetAsync(d_y, 0, (size_t)q->OC*sizeof(float), g_stream);
    size_t smem = (size_t)K*CPB*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
    dim3 grid(q->OC/CPB, GS), block(32, TY);
    gemv4<<<grid, block, smem, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, (float*)d_y, q->IC, q->OC);
}

void qlinear_free(void* handle) {
    QLinear* q = (QLinear*)handle;
    cudaFree(q->d_packed); cudaFree(q->d_cb); free(q);
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
             int n_heads, int n_kv, int head_dim, int seqlen) {
    ensure_stream();
    size_t smem = ((size_t)head_dim + seqlen) * sizeof(float);
    attn_k<<<n_heads, head_dim, smem, g_stream>>>((const __half*)q_half, (const __half*)ck_half,
        (const __half*)cv_half, (__half*)out_half, n_heads, n_kv, head_dim, seqlen);
}

// --- batched (M-token) ops for speculative decoding -------------------------------
void qlinear_forward_m(void* handle, const void* d_x, void* d_y, int M) {
    QLinear* q = (QLinear*)handle;
    ensure_stream();
    cudaMemsetAsync(d_y, 0, (size_t)M*q->OC*sizeof(float), g_stream);
    size_t smem = (size_t)K*CPB*sizeof(__half) + (size_t)M*TY*CPB*sizeof(float);
    dim3 grid(q->OC/CPB, GS), block(32, TY);
    gemm_mtile<<<grid, block, smem, g_stream>>>((const __half*)d_x, q->d_packed, q->d_cb, (float*)d_y, q->IC, q->OC, M);
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

void dev_sync() { cudaDeviceSynchronize(); }

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
