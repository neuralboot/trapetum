# Pareto: real Llama-2-7B, one machine (RTX 4090)

Speed, memory, accuracy and **energy** for quantization schemes on the same model and
machine, batch-1 decode. Wikitext-2 PPL over 30k tokens (ctx 2048); energy from `pynvml`
power.draw sampled during the decode loop. Reproduce with `python bench/pareto.py`.

| method | bits | mem (GB) | Wikitext PPL | decode tok/s | J/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 (cuBLAS) | 16.0 | 13.48 | 5.28 | 49.9 | 5.57 |
| codebook-4bit (ours, Python op) | 4.05 | 3.81 | 5.92 | 30.8 | 4.57 |
| aqlm-2bit (official kernel) | 2.0 | 2.15 | 7.01 | 23.9 | 4.86 |
| **codebook-4bit (ours, Rust runtime)** | 4.05 | 3.5 | 5.92 | **135.0** | TODO |

## Honest reading

- **Accuracy/memory is a clean curve**: fp16 (5.28 PPL, 13.5 GB) → ours 4-bit (5.92, 3.8 GB,
  3.5x smaller) → AQLM 2-bit (7.01, 2.1 GB, 6.3x smaller). Lower bits buy memory at a real
  PPL cost; nothing here beats dense on accuracy.
- **Our 4-bit *via a Python custom op* is slower than fp16 (30.8 vs 49.9 tok/s).** This is
  not the kernel; it is Python per-op dispatch (224 custom-op calls per token: 7 linears x
  32 layers). The *same quantized weights* in the pure-Rust runtime run at **135 tok/s** —
  2.7x fp16 and 4.4x the Python number. The gap *is* the Python overhead. This is why the
  Rust-runtime row exists, and it is the strongest single result here.
- **Energy**: the quantized decode draws less power, so even the slow Python path has the
  lowest J/token (4.57 vs fp16 5.57). At 135 tok/s the Rust runtime's J/token would fall
  several-fold further; measuring it (power-sampled) is the next step.

## Caveats / TODO (not faked)

- Batch-1 decode only. Batched throughput needs a GEMM path (our kernel is a decode GEMV);
  the Tensor-Core prefill experiments (`prof10/11.cu`) were *negative* (slower than cuBLAS),
  so this is an open problem, not a quick reuse.
- `codebook-4bit (Python op)` is our kernel wrapped per-linear in PyTorch; the fair
  deployable number is the Rust-runtime row.
- Missing columns (honest): Rust-runtime J/token (power-sampled), avq-2bit decoded through
  our kernel (needs a torch binding), Marlin uniform 4-bit (GPTQ/vLLM).
