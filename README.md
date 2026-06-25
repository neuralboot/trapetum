# Fused Codebook-Quant GEMV / GEMM (CUDA)

Hand-written CUDA kernels for **K-means (codebook) weight quantization** of LLM
weights, with **on-the-fly dequantization fused into the matmul** — the weight
matrix `W` is never materialized in global memory.

```
W_deq[i, j] = codebook[ indices[i, j], j ]          # per-output-channel codebooks
Y[m, j]     = sum_i  X[m, i] * W_deq[i, j]
```

- `indices` — `[IC, OC]`, one cluster id per weight (`uint8`, or packed `4-bit`)
- `codebook` — `[K, OC]`, `fp16` (the small per-column "catalogue", `K` ∈ 16…256)
- `X` — activations `fp16`, `Y` — output `fp16`

All numbers below were **measured on a real NVIDIA A40** (`sm_86`, CUDA 11.8),
matrices `4096 × 4096`, and verified for correctness against cuBLAS fp16.

---

## Why this exists

Storing each weight as a small cluster index instead of `fp16` shrinks the model
2–4×. The catch is that you must turn indices back into weights to compute. Doing
that as a separate pass (materialize `W`, then GEMM) wastes the bandwidth you just
saved. These kernels fuse the lookup into the matmul so the dequantization is free
of extra global traffic.

The project also answers a practical question with measurements rather than
intuition: **when does weight quantization actually make inference faster?**

---

## Results (A40, vs cuBLAS fp16 dense)

### Decode — GEMV, batch = 1 (memory-bound → quantization *wins*)

| Kernel | scheme | time | vs cuBLAS |
|---|---|---|---|
| `gemv_codebook.cu` | uint8, K = 256 | 0.056 ms | **×1.09** |
| `gemv_codebook.cu` | uint8, K = 64 | 0.039 ms | **×1.57** |
| `gemv_codebook_4bit.cu` | 4-bit, K = 16 | 0.026 ms | **×2.34** |
| cuBLAS fp16 GEMV | dense | 0.061 ms | 1.00 |

Decode is memory-bound, so reading fewer weight bytes directly buys speed. The
dominant cost is streaming the index matrix; the lever is **bits per index**
(uint8 → 4-bit halves it).

### Standalone dequant

| Kernel | bandwidth |
|---|---|
| naive (redundant shared staging) | 31.8 GB/s |
| `dequant_l2` (L2-cached gather) | **213 GB/s** (×6.7) |

### Prefill — GEMM, M = 2048 (compute-bound → quantization is *memory-only*)

| Kernel | TFLOP/s | vs cuBLAS |
|---|---|---|
| naive wmma (1 warp / tile) | 2.3 | ×0.02 |
| tiled fused (`prof8`) | 9.5 | ×0.09 |
| Marlin: cp.async + double-buffer (`prof9`) | 12.3 | ×0.12 |
| + register-pipelined dequant (`prof10`) | **22.4** | ×0.21 |
| raw `mma.sync` (`prof11`) | 22.0 | ×0.21 |
| cuBLAS fp16 | 107 | 1.00 |

Prefill is compute-bound: cuBLAS already runs near the Tensor-Core peak, so a
fused-dequant kernel can at best *match* it — quantization buys memory, not speed.
The biggest single win here was killing the **dequant bubble** by issuing the
codebook gather into registers *before* the mma and overlapping its latency with
the Tensor-Core work (`prof9 → prof10`, +80%). Swapping the wmma API for raw
`mma.sync` (`prof11`) made no difference, which proves the mma path was **not** the
bottleneck.

---

## The takeaway

Decode and prefill have **opposite economics**:

- **Decode** (memory-bound) → quantization is a real speedup (up to **×2.34**).
- **Prefill** (compute-bound) → quantization is a memory saving only; use cuBLAS.

So in a serving stack: quantize and use these kernels for the per-token generation
loop, and keep a dense path (cuBLAS) for prefill.

---

## File guide

**Production kernels**
- `gemv_codebook_4bit.cu` — decode GEMV, 4-bit packed indices (K ≤ 16). Fastest, **×2.34**.
- `gemv_codebook.cu` — decode GEMV, uint8 indices (K ≤ 256). **×1.09–1.57**.
- `codebook_quant.cu` — step 1: clean on-the-fly dequant kernel + first fused GEMV.

**Benchmark harness**
- `bench.cu` — decode benchmarks: every GEMV variant + the L2 dequant, vs cuBLAS.

**Profiling / optimization trail** (the measured path from naive to fast)
- `prof2.cu` — ablation: index-read vs +codebook vs +activation (locates the bottleneck).
- `prof3.cu` — vectorized index reads (full cache lines).
- `prof4.cu` — vectorized + shared codebook.
- `prof5.cu` — + grid.y split-K with atomic reduction.
- `prof6.cu` — K × grid.y sweep (smaller K shifts the optimum).
- `prof7.cu` — 4-bit packed indices.
- `prof8.cu` — tiled fused Tensor-Core GEMM (prefill).
- `prof9.cu` — Marlin-style: cp.async double-buffering, 128×128 tiles.
- `prof10.cu` — + register-pipelined dequant (best prefill).
- `prof11.cu` — raw `mma.sync` with hand-built m16n8k16 fragments (layout-exact).

---

## Build & run

```bash
# decode kernels (no cuBLAS needed for the standalone kernels)
nvcc -O3 -arch=sm_86 bench.cu        -lcublas -o bench   && ./bench

# 4-bit decode kernel
nvcc -O3 -arch=sm_86 gemv_codebook_4bit.cu -o gemv4

# prefill Tensor-Core kernels (need cuBLAS for the baseline)
nvcc -O3 -arch=sm_86 prof10.cu       -lcublas -o prof10  && ./prof10
```

Adjust `-arch` to your GPU (`sm_86` = A40/A10/RTX 30xx; `sm_80` = A100;
`sm_89` = L4/RTX 40xx; `sm_90` = H100). The `mma.sync` and `cp.async` paths
require `sm_80+`.

---

## Method note

Everything was benchmarked on real hardware (A40 via RunPod), not estimated.
Nsight Compute hardware counters are blocked inside the cloud container
(`ERR_NVGPUCTRPERM`), so the bottleneck at each step was found by **ablation
profiling** — timing kernels that add one operation at a time. Several intuitive
hypotheses (the codebook gather, raw occupancy, the wmma API) were ruled out by
measurement rather than assumed.

## Caveats

- Codebook quantization reduces model quality; `K` and bit-width trade accuracy
  for size/speed (4-bit is more aggressive than uint8).
- Kernels are tuned for `4096 × 4096` on A40; tile sizes / `grid.y` want retuning
  per GPU and shape.
- The prefill kernels are correct and ~10× over the naive baseline, but not yet at
  cuBLAS parity — closing that needs `ldmatrix` + shared-memory swizzling, for a
  parity (not speedup) result.
