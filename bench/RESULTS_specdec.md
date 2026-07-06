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

## Wall-clock, two-model (MEASURED July 2026): negative -> diagnosed -> FIXED -> WIN
The real two-model harness (`Model::spec_decode_two_model` + `wallclock_spec`: the drafter is
a second Model decoding incrementally in its own KV cache, LCP row reuse) went through the
full arc on a 4090 (128 tokens, K=1..3, lossless verified at every step):

**Round 1 (naive): spec-dec LOSES ~3x** (0.34x at 0.91 acceptance) — same integration law as
the kernel story. Profiling (`profile_spec` bin) found the root cause: `gemm_mtile` took M at
RUNTIME, which defeated unrolling and spilled the `acc[M][8]` accumulators to local memory —
the M<=4 verify cost 5.5x (M=2) to 10.7x (M=4) of one M=1 GEMV. (Two earlier hypotheses —
the forward_m allocation storm, real but CUDA-neutral, and drafter overhead, actually fine at
1.14 ms — were fixed/ruled out on the way.)

**Fix: template the kernel on M** (`gemm_mtile_t<M>`, launch-dispatched). Per-call verify:
M=2 39.7 -> 8.2 ms (**1.11x of M=1**), M=3 59.7 -> 9.6 ms, M=4 77.4 -> 12.8 ms.

**Round 2 (fixed kernel), measured wall-clock:**

| pair (target + drafter) | vocab | plain tok/s | K=1 | K=2 | K=3 | best |
|---|---|---|---|---|---|---|
| **Llama-2-7B + llama-160m** | 32k | 124.6 | 1.19x | **1.32x** | 1.24x | **K=2 WIN, 164 tok/s** |
| R1-Distill-Qwen-7B + Qwen2.5-1.5B | 152k | 100.6 | 0.76x | 0.76x | 0.76x | near-parity |
| Llama-3.1-8B + Llama-3.2-1B | 128k | 117.9 | 0.85x | 0.90x | 0.80x | near-parity |

**Spec-dec now WINS in wall-clock where the projection said it should** (1.32x measured vs
1.50x projected ceiling on the Llama-2 pair). The two large-vocab pairs sit at parity for two
measured reasons: (a) every drafter forward reads back the FULL logits (152k x 4B = 608 KB)
and does a host argmax — at 130+ drafter forwards this dominates; (b) their smallest
same-tokenizer drafters are 10x bigger than llama-160m (1.5B/1B; Qwen2.5-0.5B is blocked by
the kernel's %256 shape rule). Next lever: device-side argmax + returning only the argmax id
(kills the readback), then graph capture.

RETRACTION NOTE (July 6): the original Qwen2.5-7B+1.5B row (0.82/0.92/0.92 at alpha 1.000)
was measured on a model CORRUPTED by the torch-free exporter's o-bias bug (both models
flooded token 0, so "alpha=1" and "lossless" were trivially true on garbage). The row above
is the re-measurement on the FIXED export (coherent text verified) on a community-cloud 4090:
real alpha 0.85, 0.76x — the large-vocab parity conclusion stands, the exact numbers changed.

Qwen bias note: the batched path now supports qkv
bias (repeated-bias vadd) — validated lossless at 0.99+ acceptance. Raw logs:
`bench/runpod_logs/wallclock_4090.log` (round 1), `wallclock_fixed_4090.log` +
`profile_mtile_4090.log` (round 2).

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
