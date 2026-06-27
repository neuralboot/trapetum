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
the CUDA runtime; no `bindgen` or CUDA Rust crate is required. The demo builds a synthetic
quantized layer, decodes it on the GPU, **verifies the output against a CPU reconstruction**,
and times the forward.

## What this prototype establishes

- Rust loads quantized weights and calls the CUDA kernel directly (C-ABI FFI), correct to
  the kernel's numerical tolerance, with the weights resident on the device.
- No Python, no PyTorch: a step toward a `pip`-free, single binary that runs a quantized
  model fast on consumer GPUs.

## Honest scope (it is a bootstrap)

- One layer, scalar 4-bit codebook. The forward copies the activation host to device and
  the output back each call; a real runtime keeps activations on device across layers.
- No transformer yet (attention, norms, KV cache), no real weight loading.

## Roadmap

1. Keep activations on the device and chain layers (no per-call copies).
2. Capture the decode step as a CUDA graph (the paper's 2.0x end-to-end lever).
3. Swap in the additive vector-quantization kernel (`avq_gemv`) for AQLM-accuracy weights.
4. Load real weights (safetensors) and wire a full transformer block.
