# codebook-runtime (Rust)

A minimal **Rust** inference runtime that hosts the fused 4-bit codebook decode CUDA
kernel, with **no Python in the loop**. This is the bootstrap of a deployable, single
binary inference path (the way `llama.cpp` and `candle` are adopted), built on the same
kernel measured in the paper.

A [`QuantLinear`](src/lib.rs) uploads codebook-quantized weights to the GPU once and runs
a batch-1 decode GEMV through the kernel. The weight matrix is never materialized; only
the 4-bit codes are read.

```rust
use codebook_runtime::QuantLinear;

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

## Roadmap

1. Activations on the device, layers chained, no per-call copies. **(done)**
2. Capture the decode chain as a CUDA graph. **(done)**
3. The MLP block (RMSNorm + SwiGLU + residual, codebook GEMVs). **(done)**
4. The attention block: RoPE, the KV cache, and batch-1 attention (QK^T, softmax, AV),
   then a full layer = attention + MLP.
5. Load real weights (safetensors, quantized by the Python `model/` scripts), and the
   additive-VQ kernel, to measure end-to-end tokens/s on a real model in pure Rust.
