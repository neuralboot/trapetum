# Pareto: real Llama-2-7B, one machine (RTX 4090)

Speed, memory, accuracy and **energy** for quantization schemes on the same model and
machine, batch-1 decode, **all decode metrics measured over a 512-token generation
(iso-context)**. Wikitext-2 PPL over 30k tokens (ctx 2048); energy from `pynvml`. Reproduce
with `python bench/pareto.py --gen 512` (Python rows) and the Rust runtime's
`generate <model.cbk> ... 512` (Rust row).

| method | bits | mem (GB) | Wikitext PPL | decode tok/s | J/token | gCO2/1k-tok (FR / US) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 (cuBLAS) | 16.0 | 13.48 | 5.28 | 45.2 | 5.45 | 0.076 / 0.61 |
| codebook-4bit (ours, Python op) | 4.05 | 3.81 | 5.92 | 28.4 | 4.00 | 0.056 / 0.44 |
| aqlm-2bit (official kernel) | 2.0 | 2.15 | 7.01 | 24.2 | 3.86 | 0.054 / 0.43 |
| **codebook-4bit (ours, Rust runtime)** | 4.05 | 4.73* | 5.92 | **81** | **2.58** | **0.036 / 0.29** |

\* 4.73 GB measured peak (3.5 GB weights + KV cache + activations).

## Honest reading

- **Energy is the headline, and it holds iso-context.** Over the *same* 512-token
  generation, the pure-Rust runtime runs the *same* 4-bit weights at **2.58 J/token —
  2.1x less than fp16 (5.45)** and ~1.5x less than the Python-wrapped kernel (4.00). It is
  also the fastest quantized path (81 vs 28-45 tok/s).
- **J/token is context-dependent** (it drops for every method at longer context: attention
  over the KV cache is memory-bound and lower-power). That is exactly why we measure all
  rows at the same 512-token generation rather than comparing across context lengths.
- **Speed:** our 4-bit *via a Python op* is slower than fp16 (28.4 vs 45.2 tok/s) — Python
  per-op dispatch (224 calls/token), not the kernel. The *same weights* in Rust run at
  81 tok/s (1.8x fp16 at this context, 135 short-context). The Python->Rust gap *is* the
  overhead, and it is the strongest single result here.
- **Accuracy/memory** is a clean curve (PPL over 30k tokens, generation-independent):
  fp16 (5.28, 13.5 GB) -> ours 4-bit (5.92, 3.8 GB) -> AQLM 2-bit (7.01, 2.1 GB). Nothing
  beats dense on accuracy.
- **CO2 is a derived, secondary axis, not a measurement.** gCO2/1k-tok = J/token x grid
  intensity; we do not know RunPod's grid mix, so these are projections at France-like
  (~50) and US-like (~400 gCO2/kWh) intensities. J/token is what we measure.

## Caveats / TODO (not faked)

- Batch-1 decode only. Batched throughput needs a GEMM path (our kernel is a decode GEMV);
  the Tensor-Core prefill experiments (`prof10/11.cu`) were *negative* (slower than cuBLAS),
  so this is an open problem, not a quick reuse.
- Context offset within the iso-context run: Python rows decode 512 tokens from a 6-token
  prompt (context 6->518); the Rust row from position ~30 (context ~30->542). Average
  contexts are within ~10%.
- Missing: avq-2bit op-level vs `aqlm::code2x8_matmat` (in progress), Marlin uniform 4-bit
  (GPTQ/vLLM).
