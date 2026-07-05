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

## Wall-clock, two-model (MEASURED July 2026) — NEGATIVE result, and why
The real two-model harness (`Model::spec_decode_two_model` + `wallclock_spec` bin: the drafter
is a second Model decoding incrementally in its own KV cache, longest-common-prefix row reuse)
was measured end-to-end on a 4090, 128 tokens, K=1..3:

| pair (target + drafter) | plain tok/s | spec K=1 | K=2 | K=3 | accept K=1 | lossless |
|---|---|---|---|---|---|---|
| Llama-2-7B + llama-160m | 123.8 | 0.34x | 0.31x | 0.30x | 0.910 | YES |
| Llama-3.1-8B + Llama-3.2-1B | 117.9 | 0.26x | 0.24x | 0.22x | 0.684 | YES |
| Qwen2.5-7B + Qwen2.5-1.5B | 112.3 | n/a | n/a | n/a | — | — |

**The projected 1.50x does NOT survive wall-clock today: naive spec-dec LOSES ~3x**, even at
0.91 acceptance and with losslessness verified at every K. The forward-count mechanics are
exactly right (67 target forwards for 128 tokens at K=1); the time goes elsewhere:
1. the batched verify (M=K+1) runs on `gemm_mtile`, which was validated for correctness but
   never optimized — it is several times slower per call than the fused M=1 GEMV that gives
   the 123.8 tok/s baseline;
2. the drafter's per-forward fixed overhead (kernel launches + full-logits readback + host
   argmax per token) dwarfs its bandwidth cost: a 160m draft forward should cost ~0.1 ms of
   bandwidth and costs ~15 ms of overhead.
This is the paper's own integration law repeating: microbenchmark wins vanish without
graph-level integration. The fix path is known — optimize/fuse the M<=4 codebook GEMM and
capture the whole speculative step as a single graph — and until it lands, the honest claim
is: spec-dec is **lossless and mechanically validated; the projected S(K) is the ceiling,
not the current wall-clock**. Qwen pair: the batched forward_m path does not support
attention (qkv) bias yet — explicit panic, spec rows n/a; the plain pure-Rust decode numbers
measured on the way (Qwen2.5-7B 112.3 tok/s, Llama-3.1-8B 117.9 tok/s) update the coverage
table. Raw log: `bench/runpod_logs/wallclock_4090.log`.

## Decode Pareto (batch-1, gen=128) — quantization vs baselines + energy
| method | bits | memory (GB) | tok/s (HF loop) | energy net (J/tok) | watts |
|---|---|---|---|---|---|
| fp16 | 16.0 | 13.48 | 37.4 | 3.87 | 201 |
| **codebook-4bit (ours)** | 4.05 | 3.81 | 23.4 | 2.13 | 106 |
| aqlm-2bit | 2.0 | 2.15 | 19.4 | 1.24 | 80 |

- 4-bit vs fp16: **3.5x less memory, ~45% less energy/token** (Llama-2-7B, this run; the multi-model average is ~47%, see RESULTS_models.md).
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
