# Speculative decoding K=1 + decode Pareto — measured results

Measured on **RunPod RTX 4090** (secure cloud), July 2026, via `scripts/runpod_specdec.sh`.
Target = 4-bit Llama-2-7B (NousResearch/Llama-2-7b-hf), drafter = 4-bit TinyLlama-1.1B-Chat.
Both compressed to the codebook `.cbk` format (K=16, 4-bit) with `model/export_runtime.py`.

## Target validation (fused Rust runtime, CUDA graph)
- 4-bit Llama-2-7B reproduces HF greedy continuation **16/16 tokens exactly**
- decode throughput: **136.3 tok/s** (7.34 ms/token)

## Speculative decoding K=1 (lossless), alpha measured by teacher-forcing
| metric | value |
|---|---|
| acceptance alpha | **0.734** (94/128) |
| target latency (7B) | 8.21 ms/token (121.8 tok/s) |
| drafter latency (1.1B) | 3.25 ms/token (307.9 tok/s) |
| draft/target cost ratio | 0.396 |
| **projected speedup** | **1.24x** = (1+alpha)/(1+ratio) |
| tokens per target forward | 1+alpha = 1.73 |

Lossless by construction: the emitted sequence is token-for-token identical to plain greedy
decode. The projected speedup follows from measured alpha + latencies; the M=2 verify is
bandwidth-bound (M0 bench: ms/token flat to M=4), so it costs ~one target forward.

## Decode Pareto (4-bit codebook vs fp16), batch-1, gen=128
| method | bits | memory (GB) | tok/s (HF loop) | energy net (J/tok) | watts |
|---|---|---|---|---|---|
| fp16 | 16.0 | 13.48 | 38.5 | 3.99 | 217 |
| **codebook-4bit** | 4.05 | **3.81** | 26.4 | **2.11** | **119** |
| aqlm-2bit | 2.0 | — | — | — | — (aqlm pkg not installed) |

Wins: **3.5x less memory, ~47% less energy per token.** The raw HF-loop tok/s has 4-bit < fp16
(the known per-op Python dispatch overhead — the paper's "naive integration loses" finding);
the 4-bit speed win is realized in the fused runtime (136 tok/s above), not the HF loop.

## Combined story
Quantization (memory x energy) x speculative decoding (1.24x lossless speed) — the two levers
are orthogonal and compound. Bigger speculative speedup lever: a smaller drafter (~160M) lowers
the 0.40 cost ratio if alpha holds.

## Caveats
- 1.24x is projected from measured alpha + latencies. The end-to-end wall-clock spec run needs
  the 5 batched decode ops ported to CUDA (they exist and are validated in the Metal backend).
- aqlm-2bit / Marlin baselines were skipped this run (backends not installed).
