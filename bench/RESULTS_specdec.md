# Speculative decoding + decode Pareto — measured results (RTX 4090)

Measured on **RunPod RTX 4090** (secure cloud), July 2026, via `scripts/runpod_full.sh`.
Target = 4-bit Llama-2-7B (NousResearch/Llama-2-7b-hf). Drafters = 4-bit TinyLlama-1.1B-Chat
and 4-bit llama-160m (JackFram/llama-160m). All compressed to the codebook `.cbk` format
(K=16, 4-bit) with `model/export_runtime.py`. Raw logs in `bench/runpod_logs/`.

## Batched-decode + K>1 + confidence: LOSSLESS on real CUDA
`spec_check` on the CUDA runtime (the ported batched ops) passes:
- K=1,2,3 lossless (oracle + adversarial drafters) — output == plain greedy decode.
- confidence-scheduled dynamic-K decode lossless (high-conf + mixed).
- tokens per target forward (perfect drafter): K=1 → 1.91, K=2 → 3.00, K=3 → 3.50.

The CUDA batched kernels (gemm_mtile, rmsnorm_m, rope_m, attn_m, cache_append_m) are the
twin of the Metal ones; both produce a bit-for-bit greedy-equivalent decode.

## Target validation
4-bit Llama-2-7B reproduces the HF greedy continuation **16/16 tokens exactly**, ~136 tok/s
(fused Rust runtime, CUDA graph).

## Speculative decoding — measured alpha + K-sweep speedup (lossless)
Speedup formula: S(K) = [(1-alpha^(K+1))/(1-alpha)] / (1 + K * t_draft/t_target).

| drafter | alpha | K=1 | K=2 | K=3 | best |
|---|---|---|---|---|---|
| TinyLlama-1.1B | 0.734 | 1.23x | **1.25x** | 1.20x | K=2 → 1.25x |
| **llama-160m** | 0.648 | 1.38x | **1.50x** | 1.49x | **K=2 → 1.50x** |

Key finding: the **smaller 160M drafter wins** — lower acceptance (0.648 vs 0.734) but so much
cheaper per draft that the net speedup is higher (1.50x vs 1.25x). **K=2 is optimal for both**
(verify 3 tokens per target forward). All lossless: the output is identical to plain greedy.

## Decode Pareto (batch-1, gen=128) — quantization vs baselines + energy
| method | bits | memory (GB) | tok/s (HF loop) | energy net (J/tok) | watts |
|---|---|---|---|---|---|
| fp16 | 16.0 | 13.48 | 37.4 | 3.87 | 201 |
| **codebook-4bit (ours)** | 4.05 | 3.81 | 23.4 | 2.13 | 106 |
| aqlm-2bit | 2.0 | 2.15 | 19.4 | 1.24 | 80 |

- 4-bit vs fp16: **3.5x less memory, ~45% less energy/token.**
- aqlm-2bit is smaller/lower-energy but slower decode; our 4-bit is the speed/accuracy middle.
- The HF-loop tok/s has quantized < fp16 (the known per-op Python dispatch overhead); the
  quantized speed win is realized in the fused runtime (136 tok/s), not the HF loop.
- Marlin (uniform 4-bit) baseline + batched-serving throughput are still TODO.

## Combined
Quantization (3.5x memory, ~45% energy) x speculative decoding (**1.50x lossless**, 160M
drafter + K=2) — orthogonal, compounding. Everything measured on one RTX 4090.

## Caveats
- Speedups are projected from the measured alpha + per-model latencies and the standard
  spec-dec formula; the batched verify is validated LOSSLESS on CUDA (spec_check) and is
  bandwidth-bound (M<=4). A two-model end-to-end wall-clock harness is the remaining step.
- Energy gCO2 figures are grid-mix projections, not measurements.
