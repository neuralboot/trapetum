// codebook_quant.cu
// ---------------------------------------------------------------------------
// Weight-clustering (K-means) quantization kernels.
//
//   W_deq[i, j] = codebook[ indices[i, j], j ]
//
//   indices  : [IC, OC]  uint8  (cluster id 0..K-1)   row-major
//   codebook : [K,  OC]  __half (per-output-channel centroids)  row-major
//   W_deq    : [IC, OC]  __half
//
// IMPORTANT dtype note: with K in [128, 256] you MUST use *unsigned* 8-bit
// (uint8_t). Signed int8 only reaches 127. Use uint16_t for K > 256.
//
// IMPORTANT layout note: the codebook is given as [K, OC] (your spec). For the
// shared-memory staging below, [OC, K] (centroids of a column contiguous) gives
// nicer coalesced loads. We keep [K, OC] as specified and stage a column tile
// into shared memory once, amortized over the IC rows.
//
// Build:  nvcc -O3 -arch=sm_80 codebook_quant.cu -o codebook_quant
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>

#ifndef K_CLUSTERS
#define K_CLUSTERS 256          // typically 64..256
#endif

// ===========================================================================
// ETAPE 1 -- standalone dequant (TESTING / FALLBACK only; prefer fusion).
// ===========================================================================
//
// Tile of output columns staged into shared memory. Shared budget for the
// codebook tile = K_CLUSTERS * TILE_OC * sizeof(half). Tune TILE_OC so this
// stays under your shared-mem budget (48 KB default):
//   K=256, TILE_OC=64  -> 32 KB   (ok)
//   K=64,  TILE_OC=128 -> 16 KB   (ok)
#define VEC            8                       // cols per thread (8 half = 1 float4 store)
#define THREADS_X      (TILE_OC / VEC)
#define TILE_OC        64
#define ROWS_PER_BLOCK 16

__global__ void dequant_kernel_vec(
        const uint8_t* __restrict__ indices,    // [IC, OC]
        const __half*  __restrict__ codebook,   // [K,  OC]
        __half*        __restrict__ W_deq,       // [IC, OC]
        int IC, int OC)
{
    // Codebook column tile: s_cb[k][col_local].  ~32 KB for K=256, TILE_OC=64.
    __shared__ __half s_cb[K_CLUSTERS][TILE_OC];

    const int oc0      = blockIdx.x * TILE_OC;
    const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;

    // --- Cooperatively stage codebook[:, oc0..oc0+TILE_OC) into shared mem.
    //     Strided global reads (stride OC across k), done ONCE per block and
    //     amortized over all IC rows this block processes.  __ldg = read-only
    //     cache. ----------------------------------------------------------------
    for (int t = tid; t < K_CLUSTERS * TILE_OC; t += nthreads) {
        const int k  = t / TILE_OC;
        const int jl = t % TILE_OC;
        const int j  = oc0 + jl;
        s_cb[k][jl] = (j < OC) ? __ldg(&codebook[(size_t)k * OC + j])
                               : __float2half(0.f);
    }
    __syncthreads();

    const int col_local = threadIdx.x * VEC;    // first of VEC cols for this thread
    const int j0        = oc0 + col_local;
    if (j0 + VEC > OC) return;                   // assumes OC % TILE_OC == 0

    // Walk down the assigned columns over the row tile.
    for (int i = blockIdx.y * ROWS_PER_BLOCK + threadIdx.y;
             i < IC;
             i += gridDim.y * ROWS_PER_BLOCK)
    {
        const size_t base = (size_t)i * OC + j0;

        // --- Vectorized index load: 8 uint8 indices in two 32-bit loads
        //     (coalesced across threads; consecutive threads -> consecutive j). --
        const uint32_t idx_lo = __ldg(reinterpret_cast<const uint32_t*>(&indices[base]));
        const uint32_t idx_hi = __ldg(reinterpret_cast<const uint32_t*>(&indices[base + 4]));

        // --- 8 gathers from shared memory (no global gather). ------------------
        __half out[VEC];
        #pragma unroll
        for (int v = 0; v < 4; ++v)
            out[v]     = s_cb[(idx_lo >> (8 * v)) & 0xFF][col_local + v];
        #pragma unroll
        for (int v = 0; v < 4; ++v)
            out[v + 4] = s_cb[(idx_hi >> (8 * v)) & 0xFF][col_local + 4 + v];

        // --- Vectorized 128-bit store (8 half = float4). Aligned because j0 % 8
        //     == 0 and OC % 8 == 0. ----------------------------------------------
        *reinterpret_cast<float4*>(&W_deq[base]) =
            *reinterpret_cast<const float4*>(out);
    }
}

void launch_dequant(const uint8_t* d_idx, const __half* d_cb,
                    __half* d_W, int IC, int OC, cudaStream_t s = 0)
{
    dim3 block(THREADS_X, ROWS_PER_BLOCK);
    dim3 grid(OC / TILE_OC, /*row tiles*/ 256);   // gridDim.y tunable / grid-stride
    dequant_kernel_vec<<<grid, block, 0, s>>>(d_idx, d_cb, d_W, IC, OC);
}

// ===========================================================================
// ETAPE 2 -- FUSED dequant + GEMV for DECODE (batch M small, memory-bound).
//   Y[m, j] = sum_ic X[m, ic] * codebook[ indices[ic, j], j ]
// W is never materialized. Codebook tile in shared; indices streamed as uint8
// (2x less traffic than fp16 weights). X staged in shared (reused across cols).
// ===========================================================================
#define GEMV_TILE_OC 128
#define M_MAX        8                 // max batch rows handled per block

__global__ void fused_dequant_gemv(
        const __half*  __restrict__ X,        // [M, IC]
        const uint8_t* __restrict__ indices,  // [IC, OC]
        const __half*  __restrict__ codebook, // [K, OC]
        __half*        __restrict__ Y,        // [M, OC]
        int M, int IC, int OC)
{
    extern __shared__ char smem[];
    __half* s_cb = reinterpret_cast<__half*>(smem);            // [K * GEMV_TILE_OC]
    __half* s_X  = s_cb + (size_t)K_CLUSTERS * GEMV_TILE_OC;    // [M * IC]

    const int oc0 = blockIdx.x * GEMV_TILE_OC;
    const int tx  = threadIdx.x;                                // 0..GEMV_TILE_OC-1
    const int j   = oc0 + tx;

    // Stage codebook tile [K, oc0..oc0+TILE) into shared.
    for (int t = tx; t < K_CLUSTERS * GEMV_TILE_OC; t += blockDim.x) {
        const int k = t / GEMV_TILE_OC, jl = t % GEMV_TILE_OC, jj = oc0 + jl;
        s_cb[t] = (jj < OC) ? __ldg(&codebook[(size_t)k * OC + jj]) : __float2half(0.f);
    }
    // Stage X (all M rows) into shared once; reused across every column.
    for (int t = tx; t < M * IC; t += blockDim.x)
        s_X[t] = __ldg(&X[t]);
    __syncthreads();

    if (j >= OC) return;

    float acc[M_MAX];
    #pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[m] = 0.f;

    // Contraction over input channels. indices read is coalesced across threads
    // (consecutive j -> consecutive bytes); codebook gather hits shared mem.
    for (int ic = 0; ic < IC; ++ic) {
        const uint8_t idx = __ldg(&indices[(size_t)ic * OC + j]);
        const float   w   = __half2float(s_cb[idx * GEMV_TILE_OC + tx]);
        #pragma unroll
        for (int m = 0; m < M_MAX; ++m)
            if (m < M) acc[m] += __half2float(s_X[(size_t)m * IC + ic]) * w;
    }
    #pragma unroll
    for (int m = 0; m < M_MAX; ++m)
        if (m < M) Y[(size_t)m * OC + j] = __float2half(acc[m]);
}

void launch_fused_gemv(const __half* d_X, const uint8_t* d_idx,
                       const __half* d_cb, __half* d_Y,
                       int M, int IC, int OC, cudaStream_t s = 0)
{
    const dim3 block(GEMV_TILE_OC);
    const dim3 grid(OC / GEMV_TILE_OC);
    const size_t smem = ((size_t)K_CLUSTERS * GEMV_TILE_OC + (size_t)M * IC) * sizeof(__half);
    // For smem > 48 KB, opt in once:
    // cudaFuncSetAttribute(fused_dequant_gemv,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fused_dequant_gemv<<<grid, block, smem, s>>>(d_X, d_idx, d_cb, d_Y, M, IC, OC);
}

// ===========================================================================
// CPU reference (correctness check) + minimal driver.
// ===========================================================================
static void cpu_dequant(const std::vector<uint8_t>& idx,
                        const std::vector<float>& cb, std::vector<float>& W,
                        int IC, int OC) {
    for (int i = 0; i < IC; ++i)
        for (int j = 0; j < OC; ++j)
            W[(size_t)i*OC + j] = cb[(size_t)idx[(size_t)i*OC + j]*OC + j];
}

int main() {
    const int IC = 4096, OC = 4096, K = K_CLUSTERS;
    std::mt19937 rng(0);
    std::uniform_int_distribution<int>  ui(0, K - 1);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);

    std::vector<uint8_t> h_idx((size_t)IC*OC);
    std::vector<float>   h_cb_f((size_t)K*OC), h_Wref((size_t)IC*OC);
    for (auto& v : h_idx)  v = (uint8_t)ui(rng);
    for (auto& v : h_cb_f) v = uf(rng);
    cpu_dequant(h_idx, h_cb_f, h_Wref, IC, OC);

    std::vector<__half> h_cb((size_t)K*OC);
    for (size_t i = 0; i < h_cb.size(); ++i) h_cb[i] = __float2half(h_cb_f[i]);

    uint8_t* d_idx; __half *d_cb, *d_W;
    cudaMalloc(&d_idx, h_idx.size());
    cudaMalloc(&d_cb,  h_cb.size()  * sizeof(__half));
    cudaMalloc(&d_W,   (size_t)IC*OC* sizeof(__half));
    cudaMemcpy(d_idx, h_idx.data(), h_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cb,  h_cb.data(),  h_cb.size()*sizeof(__half), cudaMemcpyHostToDevice);

    launch_dequant(d_idx, d_cb, d_W, IC, OC);
    cudaDeviceSynchronize();

    std::vector<__half> h_W((size_t)IC*OC);
    cudaMemcpy(h_W.data(), d_W, h_W.size()*sizeof(__half), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (size_t i = 0; i < h_W.size(); ++i)
        max_err = std::max(max_err, (double)fabsf(__half2float(h_W[i]) - h_Wref[i]));
    printf("dequant max abs err vs fp16(codebook) = %.4g\n", max_err);  // ~fp16 rounding only

    // --- timing (effective bandwidth) ---
    const int iters = 50;
    cudaEvent_t ev0, ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    launch_dequant(d_idx, d_cb, d_W, IC, OC); cudaDeviceSynchronize();   // warmup
    cudaEventRecord(ev0);
    for (int it = 0; it < iters; ++it) launch_dequant(d_idx, d_cb, d_W, IC, OC);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms = 0; cudaEventElapsedTime(&ms, ev0, ev1); ms /= iters;
    double bytes = (double)IC * OC * (1 /*idx read*/ + 2 /*half write*/);
    printf("dequant: %.3f ms/launch   %.1f GB/s effective   (K=%d, %dx%d)\n",
           ms, bytes / (ms * 1e-3) / 1e9, K, IC, OC);

    cudaFree(d_idx); cudaFree(d_cb); cudaFree(d_W);
    return 0;
}
