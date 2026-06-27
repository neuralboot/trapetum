# Trapetum (Rust runtime)

A minimal **Rust** inference runtime that hosts the fused 4-bit codebook decode CUDA
kernel, with **no Python in the loop**. This is the bootstrap of a deployable, single
binary inference path (the way `llama.cpp` and `candle` are adopted), built on the same
kernel measured in the paper.

A [`QuantLinear`](src/lib.rs) uploads codebook-quantized weights to the GPU once and runs
a batch-1 decode GEMV through the kernel. The weight matrix is never materialized; only
the 4-bit codes are read.

```rust
use trapetum::QuantLinear;

// packed: (ic, oc/2) 4-bit indices; codebook: (K=16, oc) f32
let layer = QuantLinear::new(&packed, &codebook, ic, oc);
let y = layer.forward(&x);   // y = x W^T, decoded on the GPU
```

## Build and run (needs an NVIDIA GPU + CUDA toolkit)

```bash
cd runtime
CUDA_ARCH=sm_89 cargo run --release --bin demo    # sm_86 A40, sm_89 RTX40, sm_90 H100
```

`build.rs` compiles `cuda/codebook_gemv.cu` with `nvcc` into a static library and links
the CUDA runtime; no `bindgen` or CUDA Rust crate is required. The demo chains **two**
quantized layers **on the GPU** (the activation never leaves the device between them; a
cast kernel converts each layer's f32 output to fp16 for the next), **verifies against a
CPU reconstruction** that emulates the GPU's fp16 rounding, and times the chain.

## What this prototype establishes

- Rust loads quantized weights and calls the CUDA kernel directly (C-ABI FFI), correct to
  the kernel's tolerance (rel err ~3e-4), weights resident on the device.
- Activations stay on the device across layers: chaining two layers with no host<->device
  copy between them runs at ~0.015 ms/layer, about 3x faster per layer than copying host
  to device and back each call. On-device residency is the right architecture.
- No Python, no PyTorch: a step toward a `pip`-free single binary that runs a quantized
  model fast on consumer GPUs.

## Honest scope (it is a bootstrap)

- Scalar 4-bit codebook, two square layers; no transformer yet (attention, RMSNorm,
  rotary, KV cache), no real weight loading. The kernel launches are not yet captured in
  a CUDA graph (each forward still pays Rust/launch dispatch).

## CUDA graph and the Python-overhead finding

The decode chain is also captured as a single CUDA graph and replayed (correct to the
eager result, rel err identical). At two layers the graph is only ~1.09x over eager, and
that is the point: the paper's 2.0x end-to-end lever was about removing **Python**
per-token overhead, and a Rust eager loop has none (each chain is a few FFI calls). So in
Rust the graph is a refinement, not a transformation; its benefit is the accumulated
launch dispatch, which grows with model depth (a full 224-layer-per-token model issues
far more launches than two). The Rust runtime is, by construction, the demonstration that
the overhead was Python.

## Transformer block

A complete Llama-style gated **MLP block** runs in the runtime (`cargo run --bin mlp`):
RMSNorm, gate and up codebook GEMVs, SwiGLU (`silu(gate)*up`), down codebook GEMV, and a
residual add, all on-device and captured as one CUDA graph. At the Llama-2 7B dims
(hidden 4096, inter 11008) it is correct to rel err 4.9e-4 against a CPU reference (which
emulates the fp16 rounding at every step) and runs in ~0.088 ms. The new kernels
(`rmsnorm_k`, `silu_mul_k`, `resadd_k`) join the codebook GEMV, which is used three times.

The **attention block** also runs (`cargo run --bin attn`): RMSNorm, q/k/v codebook
projections, RoPE (HF rotate-half), a growing KV cache, batch-1 attention (QK^T, softmax,
AV) and the o-projection, plus a residual. Decoding 6 tokens at the Llama-2 7B MHA dims
(32 heads, head_dim 128) it is correct to rel err 7e-4 against a CPU reference that
replicates RoPE, softmax and the fp16 rounding, and one step runs in ~0.079 ms. New
kernels: `rope_k`, `attn_k` (one block per head). A full decoder layer is the composition
of the two verified sub-blocks (`Layer = AttnBlock + MlpBlock`, ~0.17 ms/token).

## Real model, end-to-end, pure Rust

A real **Llama-2-7B**, quantized to a 4-bit codebook by `model/export_runtime.py` (which
writes a `.cbk` file and a HuggingFace reference), is loaded and run by the runtime with
**no Python at runtime** (`cargo run --bin generate model.cbk prompt.bin ref.bin cont.bin`).
On an RTX 4090, decoding from `"The capital of France is"`:

| metric | value |
| --- | --- |
| logits rel err vs HF (worst over prompt) | 7.9e-3 |
| top-1 agreement with HF | 6/6 |
| greedy continuation reproduces HF | **16/16 tokens** |
| decode throughput | 7.4 ms/token (**135 tok/s**) |

The runtime reproduces HuggingFace's greedy generation exactly, token for token, from the
same quantized weights, in pure Rust on the device. The `.cbk` is 3.5 GB (vs 13 GB fp16).

## Roadmap

1. Activations on the device, layers chained, no per-call copies. **(done)**
2. Capture the decode chain as a CUDA graph. **(done)**
3. The MLP block (RMSNorm + SwiGLU + residual, codebook GEMVs). **(done)**
4. The attention block (RoPE, KV cache, batch-1 attention) and a full layer = attention +
   MLP. **(done)**
5. Load real weights and run a real Llama-2-7B end-to-end in pure Rust. **(done:** matches
   HF greedy generation 16/16, 135 tok/s on a 4090.**)**
6. Next: the additive-VQ kernel in the model path (2-bit AQLM weights), and an honest
   speed/memory/PPL/energy Pareto vs Marlin and AQLM at equal effective bits.
