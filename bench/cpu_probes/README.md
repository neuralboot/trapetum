# CPU probes: the K=16 codebook is CPU-native (tbl/pshufb isomorphism)

Measured on Apple M4 (10 cores 4P+6E, ~120 GB/s, 32 GB), 2026-07-11.
Build all: `clang -O3 -march=armv8.4-a+dotprod -o <name> <name>.c -lpthread -lm`

| Probe | What it measures | Result (best clean window) |
|---|---|---|
| `s12_tbl_gemv.c` | Micro-kernel: fused `vqtbl1q_s8` decode + SDOT int8 GEMV, packed 4-bit | 6.6 / 26.3 / 53.1 GB/s packed (1/4/8 threads), linear thread scaling = memory-bound |
| `s13m2_worksteal.c` | FULL DeepSeek expert (gate+up 2048x7168, SiLU, int8 requant, down 7168x2048, per-row scales), 58 layers x 8 random experts, work-stealing chunks, spin barriers | 47.0 GB/s sustained 20 tokens, 217 ms/token expert-side, 4.6 tok/s. Loss vs micro-kernel: 11.5% |
| `s15_full_token.c` | COMPLETE 671B decode token: MLA attention (real dims), 4K-ctx KV pass, router, 3 dense layers, shared+routed experts (full structure), lm_head = 18.46 GB/token | 361 ms/token = 2.77 tok/s FULL MODEL pure CPU, 51.1 GB/s (96% of micro-kernel). Breakdown ms: MoE 187.5, attention 138.5, dense 15.1, KV 11.5, lm_head 8.7 |

Context: these probes back the "inversion" program (ship the 14 KB activation to
RAM-resident experts instead of shipping 10 GB of expert weights to the GPU per
token). K=16 scalar codebook decode is one `tbl`/`pshufb` instruction per 32
weights, so the CPU expert path is memory-bound, never decode-bound.
Hybrid attribution: KTransformers pioneered CPU-experts/GPU-attention on DeepSeek;
our edge is the format (decode cost ~0 on CPU) + one artifact across CUDA/Metal/CPU.

Measurement gotchas (macOS), learned the hard way:
- Never fill benchmark buffers with memset: uniform pages get eaten by the macOS
  memory compressor under pressure and every read pays decompression. Fill with
  xorshift (incompressible).
- Keep the footprint under free RAM (ring-buffer slices >> LLC size) or you
  benchmark the SSD swap path (~4-5 GB/s) instead of DDR.
- Static 1-expert-per-thread splits lose ~2x on asymmetric P/E cores (every
  barrier waits for the slowest E-core); dynamic row-chunk work-stealing fixes it.
- threads == cores collapses with spin barriers (preemption); 8 threads is the
  sweet spot on a 10-core M4.
- Run best-of-N short passes if the host is loaded; interference only subtracts.

Next: S14 = real hybrid path (CPU experts + GPU attention) in the Rust runtime
on V2-Lite, measuring tok/s AND greedy output fidelity.
