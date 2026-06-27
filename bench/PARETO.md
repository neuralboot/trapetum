# Pareto: real Llama-2-7B, one machine (RTX 4090)

Speed, memory, accuracy and **energy** for quantization schemes on the same model and
machine, batch-1 decode. Wikitext-2 PPL over 30k tokens (ctx 2048); energy from `pynvml`
power.draw. Reproduce with `python bench/pareto.py` (Python rows) and the Rust runtime's
`generate <model.cbk> ... <bench_tokens>` (Rust row).

| method | bits | mem (GB) | Wikitext PPL | decode tok/s | J/token | gCO2/1k-tok (FR / US) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 (cuBLAS) | 16.0 | 13.48 | 5.28 | 49.9 | 5.57 | 0.077 / 0.62 |
| codebook-4bit (ours, Python op) | 4.05 | 3.81 | 5.92 | 30.8 | 4.57 | 0.063 / 0.51 |
| aqlm-2bit (official kernel) | 2.0 | 2.15 | 7.01 | 23.9 | 4.86 | 0.068 / 0.54 |
| **codebook-4bit (ours, Rust runtime)** | 4.05 | 4.73* | 5.92 | 81 / 135** | **2.58** | **0.036 / 0.29** |

\* 4.73 GB measured peak over a 512-token generation (3.5 GB weights + KV cache + activations).
\*\* 81 tok/s averaged over a 512-token generation (context grows to ~540); 135 tok/s at short
context. The drop is the attention O(seqlen) cost.

## Honest reading

- **Energy is the headline.** The pure-Rust runtime runs the *same* 4-bit weights at
  **2.58 J/token — 2.2x less than fp16 (5.57)** and 1.8x less than the Python-wrapped kernel
  (4.57). Quantized decode is bandwidth-bound, so it draws less power; removing the Python
  tax then lifts tok/s, and J/token = power/tok/s collapses.
- **Iso-context caveat (important):** the Rust 2.58 J/token is measured over a 512-token
  generation (context up to ~540, 81 tok/s); the Python rows were over 128 tokens (shorter
  context, cheaper attention). So the Rust number is measured at a *harder* regime and still
  wins — it is conservative. A strictly iso-context sweep is the next refinement.
- **CO2 is a derived, secondary axis, not a measurement.** gCO2/1k-tok = J/token x grid
  intensity; we do *not* know RunPod's grid mix, so these are projections at France-like
  (~50 gCO2/kWh) and US-like (~400) intensities. J/token is the metric we actually measure
  and control.
- **Speed:** our 4-bit *via a Python op* is slower than fp16 (30.8 vs 49.9 tok/s) — Python
  per-op dispatch (224 calls/token), not the kernel. The *same weights* in Rust run at
  135 tok/s short-context (2.7x fp16). The Python→Rust gap *is* the overhead.
- **Accuracy/memory** is a clean curve: fp16 (5.28, 13.5 GB) → ours 4-bit (5.92, 3.8 GB) →
  AQLM 2-bit (7.01, 2.1 GB). Nothing beats dense on accuracy.

## Caveats / TODO (not faked)

- Batch-1 decode only. Batched throughput needs a GEMM path (our kernel is a decode GEMV);
  the Tensor-Core prefill experiments (`prof10/11.cu`) were *negative* (slower than cuBLAS),
  so this is an open problem, not a quick reuse.
- Missing columns (honest): strictly iso-context J/token across all rows, avq-2bit decoded
  through our kernel at op-level (vs `aqlm::code2x8_matmat`), Marlin uniform 4-bit (GPTQ/vLLM).
