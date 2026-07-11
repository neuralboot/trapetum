//! Minimal Rust runtime over the fused 4-bit codebook decode CUDA kernel.
//!
//! A [`QuantLinear`] holds codebook-quantized weights resident on the GPU. Activations
//! live in caller-owned device buffers ([`DevHalf`], [`DevF32`]), so layers chain
//! **on-device** with no host<->device copy between them: the kernel writes f32 and
//! [`DevHalf::copy_cast_from`] converts it to half for the next layer. No Python.
use half::f16;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::os::raw::c_void;
use std::sync::Arc;

// ---- GPU backend seam -------------------------------------------------------
// The entire runtime talks to the GPU through these 24 C-ABI entry points and
// nothing else. `cuda` (default) binds them to the nvcc-compiled fused kernel.
// `metal` provides same-signature stubs so the crate builds and links on Apple
// platforms today; the Metal kernels replace the stubs in the next work package.
#[cfg(feature = "cuda")]
mod backend {
    use std::os::raw::c_void;
    extern "C" {
        pub fn qlinear_create(packed: *const u8, cb: *const f32, ic: i32, oc: i32) -> *mut c_void;
        pub fn qlinear_forward_dev(h: *mut c_void, d_x: *const c_void, d_y: *mut c_void);
        pub fn qlinear_free(h: *mut c_void);
        // AVQ (additive-codebook) linear for MoE routed experts (M in {2,3}); see AvqLinear.
        pub fn avq_create(codes: *const u8, cb: *const f32, scale: *const f32, m: i32, rows: i32, cols: i32) -> *mut c_void;
        pub fn avq_forward_dev(h: *mut c_void, d_x: *const c_void, d_y: *mut c_void);
        pub fn avq_free(h: *mut c_void);
        pub fn dev_alloc_half(n: i32) -> *mut c_void;
        pub fn dev_alloc_f32(n: i32) -> *mut c_void;
        pub fn dev_free(p: *mut c_void);
        pub fn dev_upload_to_half(d_half: *mut c_void, x: *const f32, n: i32);
        pub fn dev_cast_f32_to_half(d_half: *mut c_void, d_f32: *const c_void, n: i32);
        pub fn dev_download_f32(x: *mut f32, d_f32: *const c_void, n: i32);
        pub fn dev_download_half(x: *mut f32, d_half: *const c_void, n: i32);
        pub fn dev_argmax(logits: *const c_void, n: i32) -> u32;
        pub fn dev_sync();
        pub fn graph_begin();
        pub fn graph_end() -> *mut c_void;
        pub fn graph_launch(exec: *mut c_void);
        pub fn graph_free(exec: *mut c_void);
        pub fn dev_upload_f32(d_f32: *mut c_void, x: *const f32, n: i32);
        pub fn op_rmsnorm(x_half: *const c_void, w_f32: *const c_void, out_half: *mut c_void, n: i32, eps: f32);
        pub fn op_silu_mul(gate_f32: *const c_void, up_f32: *const c_void, out_half: *mut c_void, n: i32);
        pub fn op_residual_add(h_half: *mut c_void, delta_f32: *const c_void, n: i32);
        pub fn op_rope(x_half: *mut c_void, pos: i32, n_heads: i32, head_dim: i32, inv_freq: *const c_void);
        pub fn op_vadd(a_f32: *mut c_void, b_f32: *const c_void, n: i32);
        pub fn op_cache_append(cache_half: *mut c_void, src_half: *const c_void, pos: i32, dim: i32);
        pub fn op_attn(
            q_half: *const c_void,
            ck_half: *const c_void,
            cv_half: *const c_void,
            out_half: *mut c_void,
            n_heads: i32,
            n_kv: i32,
            head_dim: i32,
            seqlen: i32,
            softcap: f32,
        );
        pub fn op_resadd_h(h_half: *mut c_void, d_half: *const c_void, n: i32);
        // batched (M-token) ops for speculative decoding
        pub fn qlinear_forward_m(handle: *mut c_void, d_x: *const c_void, d_y: *mut c_void, m: i32);
        pub fn op_rmsnorm_m(x_half: *const c_void, w_f32: *const c_void, out_half: *mut c_void, n: i32, eps: f32, m: i32);
        pub fn op_rope_m(x_half: *mut c_void, base: i32, n_heads: i32, head_dim: i32, inv_freq: *const c_void, m: i32);
        pub fn op_cache_append_m(cache_half: *mut c_void, src_half: *const c_void, base: i32, dim: i32, m: i32);
        pub fn op_saxpy(acc_f32: *mut c_void, y_f32: *const c_void, alpha: f32, n: i32);
pub fn op_gelu_mul(gate_f32: *const c_void, up_f32: *const c_void, out_half: *mut c_void, n: i32);
                pub fn op_gemv_fp16(w_half: *const c_void, x_half: *const c_void, y_f32: *mut c_void, ic: i32, oc: i32);
        pub fn op_mla_attn(q_half: *const c_void, qr_half: *const c_void, ckv_half: *const c_void, kr_half: *const c_void, out_half: *mut c_void, n_heads: i32, d_c: i32, d_rope: i32, seqlen: i32, scale: f32);
        pub fn op_attn_m(
            q_half: *const c_void,
            ck_half: *const c_void,
            cv_half: *const c_void,
            out_half: *mut c_void,
            n_heads: i32,
            n_kv: i32,
            head_dim: i32,
            base: i32,
            m: i32,
        );
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[path = "backend_metal.rs"]
mod backend;

#[cfg(not(any(feature = "cuda", feature = "metal")))]
compile_error!("enable a GPU backend feature: `cuda` (default) or `metal`");

use backend::*;

/// Number of codebook entries (4-bit indices).
pub const K: usize = 16;

/// A device fp16 activation buffer (kernel input).
pub struct DevHalf {
    ptr: *mut c_void,
    n: usize,
}

impl DevHalf {
    pub fn zeros(n: usize) -> Self {
        Self { ptr: unsafe { dev_alloc_half(n as i32) }, n }
    }
    /// Upload a host f32 slice, converting to fp16 on the way in.
    pub fn from_host(x: &[f32]) -> Self {
        let b = Self::zeros(x.len());
        unsafe { dev_upload_to_half(b.ptr, x.as_ptr(), x.len() as i32) };
        b
    }
    /// Re-upload into the existing buffer (keeps the device pointer, e.g. for a graph).
    pub fn upload(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.n);
        unsafe { dev_upload_to_half(self.ptr, x.as_ptr(), self.n as i32) };
    }
    /// Device-side cast: fill this fp16 buffer from a device f32 buffer (no host copy).
    /// This is the inter-layer conversion when chaining.
    pub fn copy_cast_from(&mut self, src: &DevF32) {
        assert_eq!(self.n, src.n, "length mismatch in copy_cast_from");
        unsafe { dev_cast_f32_to_half(self.ptr, src.ptr, self.n as i32) };
    }
    /// Download the fp16 buffer to host f32.
    pub fn to_host(&self) -> Vec<f32> {
        let mut x = vec![0f32; self.n];
        unsafe { dev_download_half(x.as_mut_ptr(), self.ptr, self.n as i32) };
        x
    }
}

impl Drop for DevHalf {
    fn drop(&mut self) {
        unsafe { dev_free(self.ptr) };
    }
}

/// A device f32 activation buffer (kernel output).
pub struct DevF32 {
    ptr: *mut c_void,
    n: usize,
}

impl DevF32 {
    pub fn zeros(n: usize) -> Self {
        Self { ptr: unsafe { dev_alloc_f32(n as i32) }, n }
    }
    /// Upload a host f32 slice (e.g. an RMSNorm weight) to the device.
    pub fn from_host(x: &[f32]) -> Self {
        let b = Self::zeros(x.len());
        unsafe { dev_upload_f32(b.ptr, x.as_ptr(), x.len() as i32) };
        b
    }
    /// Copy the buffer back to host.
    pub fn to_host(&self) -> Vec<f32> {
        let mut y = vec![0f32; self.n];
        unsafe { dev_download_f32(y.as_mut_ptr(), self.ptr, self.n as i32) };
        y
    }
    /// Device-side argmax over the first `real_vocab` entries (greedy token). Avoids
    /// copying the whole buffer to the host; only the winning index comes back. Ties
    /// resolve to the smallest index, matching the host [`argmax`].
    pub fn argmax_device(&self, real_vocab: usize) -> u32 {
        debug_assert!(real_vocab <= self.n, "real_vocab must not exceed the buffer length");
        unsafe { dev_argmax(self.ptr, real_vocab as i32) }
    }
    pub fn len(&self) -> usize {
        self.n
    }
}

impl Drop for DevF32 {
    fn drop(&mut self) {
        unsafe { dev_free(self.ptr) };
    }
}

/// A codebook-quantized linear layer whose weights live on the GPU.
pub struct QuantLinear {
    handle: *mut c_void,
    ic: usize,
    oc: usize,
}

impl QuantLinear {
    /// - `packed`: `(ic, oc/2)` bytes, two 4-bit indices per byte (low nibble first).
    /// - `codebook`: `(K, oc)` f32, one per-output-channel table.
    pub fn new(packed: &[u8], codebook: &[f32], ic: usize, oc: usize) -> Self {
        assert_eq!(packed.len(), ic * (oc / 2), "packed must be ic*(oc/2) bytes");
        assert_eq!(codebook.len(), K * oc, "codebook must be K*oc floats");
        assert_eq!(oc % 256, 0, "oc must be a multiple of 256 (kernel tiling)");
        assert_eq!(ic % 2, 0, "ic must be even (packed nibbles)");
        let handle =
            unsafe { qlinear_create(packed.as_ptr(), codebook.as_ptr(), ic as i32, oc as i32) };
        assert!(!handle.is_null(), "qlinear_create returned null (CUDA error?)");
        Self { handle, ic, oc }
    }

    /// `y` (device f32) = `x` (device fp16) W^T. Fully on-device, no host copies.
    pub fn forward_into(&self, x: &DevHalf, y: &mut DevF32) {
        assert_eq!(x.n, self.ic, "x must have length ic");
        assert_eq!(y.n, self.oc, "y must have length oc");
        unsafe { qlinear_forward_dev(self.handle, x.ptr, y.ptr) };
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.ic, self.oc)
    }
}

impl Drop for QuantLinear {
    fn drop(&mut self) {
        unsafe { qlinear_free(self.handle) };
    }
}

/// Number of entries per additive codebook (8-bit AVQ indices).
pub const AVQ_K: usize = 256;
/// Group size for additive VQ (weights per code).
pub const AVQ_D: usize = 8;

/// An additive-codebook (AQLM-style) quantized linear whose weights live on the GPU,
/// used for MoE routed experts. A group of [`AVQ_D`] input weights for output channel `o`
/// is reconstructed as `scale[o] * sum_m C_m[code_m[o,g]]`, with `M` additive codebooks of
/// [`AVQ_K`] vectors (`M=2` -> 2 bit/weight, `M=3` -> 3 bit). Same decode as the CBKA
/// container and `kernels/avq_gemv3.cu`; see the CBKA reader in `load_deepseek_qlora`.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct AvqLinear {
    handle: *mut c_void,
    ic: usize,
    oc: usize,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl AvqLinear {
    /// - `codes`: `M*(cols/AVQ_D)*rows` bytes, layout `[M][ng][rows]` (`ng = cols/AVQ_D`).
    /// - `cb`: `M*AVQ_K*AVQ_D` f32, layout `[M][AVQ_K][AVQ_D]`.
    /// - `scale`: `rows` f32, one per output channel. `rows = oc`, `cols = ic`.
    pub fn new(codes: &[u8], cb: &[f32], scale: &[f32], m: usize, rows: usize, cols: usize) -> Self {
        assert!(m == 2 || m == 3, "AVQ M must be 2 or 3 (only those kernels are instantiated)");
        assert_eq!(cols % AVQ_D, 0, "cols (ic) must be a multiple of AVQ_D={AVQ_D}");
        assert_eq!(rows % 4, 0, "rows (oc) must be a multiple of 4 (uint32 code read)");
        assert_eq!(codes.len(), m * (cols / AVQ_D) * rows, "codes must be M*(cols/AVQ_D)*rows bytes");
        assert_eq!(cb.len(), m * AVQ_K * AVQ_D, "cb must be M*AVQ_K*AVQ_D floats");
        assert_eq!(scale.len(), rows, "scale must be rows floats");
        let handle = unsafe {
            avq_create(codes.as_ptr(), cb.as_ptr(), scale.as_ptr(), m as i32, rows as i32, cols as i32)
        };
        assert!(!handle.is_null(), "avq_create returned null (CUDA error?)");
        Self { handle, ic: cols, oc: rows }
    }

    /// `y` (device f32) = `x` (device fp16) W^T, decoding the additive codebook. On-device.
    pub fn forward_into(&self, x: &DevHalf, y: &mut DevF32) {
        assert_eq!(x.n, self.ic, "x must have length ic");
        assert_eq!(y.n, self.oc, "y must have length oc");
        unsafe { avq_forward_dev(self.handle, x.ptr, y.ptr) };
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.ic, self.oc)
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl Drop for AvqLinear {
    fn drop(&mut self) {
        unsafe { avq_free(self.handle) };
    }
}

/// One projection of a streamed MoE expert: either a 4-bit scalar-codebook [`QuantLinear`]
/// (today's CBKR path) or an additive-codebook [`AvqLinear`] (the CBKA 2/3-bit path). Both
/// map a device fp16 activation to a device f32 output; `MoeBlockOffload::run_ffn` dispatches
/// on the variant through the [`ProjRef`] Copy handle so the streaming/LRU logic is unchanged.
#[cfg(any(feature = "cuda", feature = "metal"))]
enum Proj { Scalar(QuantLinear), Avq(AvqLinear) }

/// A Copy, backend-tagged raw handle to a [`Proj`], so `resident()` can hand the three
/// projections to `run_ffn` without holding a borrow of the expert cache (mirrors the old
/// raw-pointer pattern that let `run_ffn(&mut self, ..)` also touch the scratch buffers).
#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Clone, Copy)]
enum ProjRef { Scalar(*mut c_void), Avq(*mut c_void) }

#[cfg(any(feature = "cuda", feature = "metal"))]
impl Proj {
    fn as_ref(&self) -> ProjRef {
        match self {
            Proj::Scalar(q) => ProjRef::Scalar(q.handle),
            Proj::Avq(a) => ProjRef::Avq(a.handle),
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl ProjRef {
    /// Decode-GEMV forward on device pointers: `y (f32) = x (half) W^T`.
    unsafe fn forward(self, d_x: *const c_void, d_y: *mut c_void) {
        match self {
            ProjRef::Scalar(h) => qlinear_forward_dev(h, d_x, d_y),
            ProjRef::Avq(h) => avq_forward_dev(h, d_x, d_y),
        }
    }
}

/// A streamed MoE expert as a Llama-style gated FFN whose three projections may each be
/// scalar-4bit or additive-codebook (see [`Proj`]). Used by [`MoeBlockOffload`] for both the
/// routed experts (streamed) and the always-resident shared expert.
#[cfg(any(feature = "cuda", feature = "metal"))]
struct OffExpert { gate: Proj, up: Proj, down: Proj }

/// RMSNorm: `out = x / sqrt(mean(x^2) + eps) * w`, on-device.
pub fn rmsnorm(x: &DevHalf, w: &DevF32, out: &mut DevHalf, eps: f32) {
    assert_eq!(x.n, out.n);
    assert_eq!(x.n, w.n);
    unsafe { op_rmsnorm(x.ptr, w.ptr, out.ptr, x.n as i32, eps) };
}

/// SwiGLU activation: `out = silu(gate) * up`, on-device.
pub fn silu_mul(gate: &DevF32, up: &DevF32, out: &mut DevHalf) {
    assert_eq!(gate.n, up.n);
    assert_eq!(gate.n, out.n);
    unsafe { op_silu_mul(gate.ptr, up.ptr, out.ptr, gate.n as i32) };
}

/// Residual add, both fp16 (Gemma post-sublayer-norm output into the stream).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn resadd_h(h: &mut DevHalf, d: &DevHalf) { assert_eq!(h.n, d.n); unsafe { op_resadd_h(h.ptr, d.ptr, h.n as i32) }; }

/// Residual add in place: `h += delta`, on-device.
pub fn residual_add(h: &mut DevHalf, delta: &DevF32) {
    assert_eq!(h.n, delta.n);
    unsafe { op_residual_add(h.ptr, delta.ptr, h.n as i32) };
}

// ============================================================================
// Batched (M-token) API for speculative decoding: verify K+1 draft tokens in
// one forward. Metal-only for now; the kernels are validated bit-exact / ~7e-4
// (see check_mtile / check_rmsnorm_m / check_attn_m / check_rope_m).
// ============================================================================

#[cfg(any(feature = "cuda", feature = "metal"))]
impl QuantLinear {
    /// Batched decode GEMM: `y[M][oc] = x[M][ic] W^T`, one weight read serves all M rows.
    pub fn forward_m(&self, x: &DevHalf, y: &mut DevF32, m: usize) {
        assert_eq!(x.n, m * self.ic, "x must be m*ic");
        assert_eq!(y.n, m * self.oc, "y must be m*oc");
        unsafe { qlinear_forward_m(self.handle, x.ptr, y.ptr, m as i32) };
    }
}

/// Batched RMSNorm over M rows (`x`,`out` are `m*n`).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn rmsnorm_m(x: &DevHalf, w: &DevF32, out: &mut DevHalf, eps: f32, n: usize, m: usize) {
    assert_eq!(x.n, m * n);
    assert_eq!(w.n, n);
    unsafe { op_rmsnorm_m(x.ptr, w.ptr, out.ptr, n as i32, eps, m as i32) };
}

/// Batched RoPE: row `r` rotated at absolute position `base+r` (`x` is `m*n_heads*head_dim`).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn rope_m(x: &mut DevHalf, base: usize, n_heads: usize, head_dim: usize, inv_freq: &DevF32, m: usize) {
    assert_eq!(x.n, m * n_heads * head_dim);
    unsafe { op_rope_m(x.ptr, base as i32, n_heads as i32, head_dim as i32, inv_freq.ptr, m as i32) };
}

/// Append M contiguous new rows to the KV cache at rows `base..base+m`.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn cache_append_m(cache: &mut DevHalf, src: &DevHalf, base: usize, dim: usize, m: usize) {
    assert_eq!(src.n, m * dim);
    unsafe { op_cache_append_m(cache.ptr, src.ptr, base as i32, dim as i32, m as i32) };
}

/// Batched causal decode attention: query `r` attends over `base+r+1` keys (`q`,`out` are `m*n_heads*head_dim`).
#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn attention_m(q: &DevHalf, ck: &DevHalf, cv: &DevHalf, out: &mut DevHalf,
                   n_heads: usize, n_kv: usize, head_dim: usize, base: usize, m: usize) {
    unsafe { op_attn_m(q.ptr, ck.ptr, cv.ptr, out.ptr, n_heads as i32, n_kv as i32, head_dim as i32, base as i32, m as i32) };
}

/// Cached batched-decode scratch (allocated once per M value, then reused). The naive
/// forward_m allocated ~10 device buffers per layer per call — hundreds of device mallocs
/// per verify step, each an implicit sync. The M=1 path never does this; neither should M<=4.
struct MScratchMlp { m: usize, norm: DevHalf, g: DevF32, u: DevF32, act: DevHalf, mlp: DevF32 }

struct MScratchAttn {
    m: usize,
    norm: DevHalf,
    qb: DevF32, kb: DevF32, vb: DevF32,
    qh: DevHalf, kh: DevHalf, vh: DevHalf,
    attn_out: DevHalf, ob: DevF32,
    // qkv bias repeated m times (Qwen-style attention bias on the batched path)
    qbias_r: Option<DevF32>, kbias_r: Option<DevF32>, vbias_r: Option<DevF32>,
}

/// A Llama-style gated MLP block: RMSNorm, then `down(SiLU(gate(h)) * up(h))`, plus a
/// residual. The three projections use the codebook decode kernel; norm, activation and
/// residual are on-device too, so the whole forward can be captured as one CUDA graph.
pub struct MlpBlock {
    norm_w: DevF32,
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
    norm: DevHalf,
    g: DevF32,
    u: DevF32,
    act: DevHalf,
    mlp: DevF32,
    eps: f32,
    m_scratch: Option<MScratchMlp>,
}

impl MlpBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        inter: usize,
        norm_w: &[f32],
        gate_packed: &[u8],
        gate_cb: &[f32],
        up_packed: &[u8],
        up_cb: &[f32],
        down_packed: &[u8],
        down_cb: &[f32],
        eps: f32,
    ) -> Self {
        Self {
            norm_w: DevF32::from_host(norm_w),
            gate: QuantLinear::new(gate_packed, gate_cb, hidden, inter),
            up: QuantLinear::new(up_packed, up_cb, hidden, inter),
            down: QuantLinear::new(down_packed, down_cb, inter, hidden),
            norm: DevHalf::zeros(hidden),
            g: DevF32::zeros(inter),
            u: DevF32::zeros(inter),
            act: DevHalf::zeros(inter),
            mlp: DevF32::zeros(hidden),
            eps,
            m_scratch: None,
        }
    }

    /// `h` is the residual stream `(hidden,)`, updated in place: `h = h + MLP(RMSNorm(h))`.
    pub fn forward(&mut self, h: &mut DevHalf) {
        rmsnorm(h, &self.norm_w, &mut self.norm, self.eps);
        self.gate.forward_into(&self.norm, &mut self.g);
        self.up.forward_into(&self.norm, &mut self.u);
        silu_mul(&self.g, &self.u, &mut self.act);
        self.down.forward_into(&self.act, &mut self.mlp);
        residual_add(h, &self.mlp);
    }
}

/// RoPE (HF Llama rotate-half) in place on `x` (`n_heads * head_dim`) at position `pos`.
pub fn rope(x: &mut DevHalf, pos: usize, n_heads: usize, head_dim: usize, inv_freq: &DevF32) {
    assert_eq!(x.n, n_heads * head_dim);
    unsafe { op_rope(x.ptr, pos as i32, n_heads as i32, head_dim as i32, inv_freq.ptr) };
}
/// Add a bias vector into an f32 accumulator in place: `a += b`.
pub fn vadd(a: &mut DevF32, b: &DevF32) {
    assert_eq!(a.n, b.n);
    unsafe { op_vadd(a.ptr, b.ptr, a.n as i32) };
}

/// Append `src` (a `n_kv*head_dim` fp16 vector) to the KV cache at row `pos`.
pub fn cache_append(cache: &mut DevHalf, src: &DevHalf, pos: usize) {
    unsafe { op_cache_append(cache.ptr, src.ptr, pos as i32, src.n as i32) };
}

/// Batch-1 decode attention over `seqlen` cached positions.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &DevHalf,
    ck: &DevHalf,
    cv: &DevHalf,
    out: &mut DevHalf,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    seqlen: usize,
    softcap: f32,
) {
    unsafe {
        op_attn(q.ptr, ck.ptr, cv.ptr, out.ptr, n_heads as i32, n_kv as i32, head_dim as i32, seqlen as i32, softcap)
    };
}

/// A Llama-style attention block: RMSNorm, q/k/v codebook projections, RoPE on q and k,
/// a growing KV cache, batch-1 attention, the output codebook projection, and a residual.
/// All on-device; the per-token forward can be captured as a CUDA graph (the cache row and
/// RoPE angles change with `pos`, so capture per position).
pub struct AttnBlock {
    norm_w: DevF32,
    q: QuantLinear,
    k: QuantLinear,
    v: QuantLinear,
    o: QuantLinear,
    cache_k: DevHalf,
    cache_v: DevHalf,
    norm: DevHalf,
    qb: DevF32,
    kb: DevF32,
    vb: DevF32,
    qh: DevHalf,
    kh: DevHalf,
    vh: DevHalf,
    attn_out: DevHalf,
    ob: DevF32,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    eps: f32,
    inv_freq: DevF32,
    qbias: Option<DevF32>,
    kbias: Option<DevF32>,
    vbias: Option<DevF32>,
    // host copies kept to build the repeated (m-row) bias for the batched path
    qbias_h: Option<Vec<f32>>,
    kbias_h: Option<Vec<f32>>,
    vbias_h: Option<Vec<f32>>,
    m_scratch: Option<MScratchAttn>,
}

impl AttnBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        max_seq: usize,
        norm_w: &[f32],
        q: (&[u8], &[f32]),
        k: (&[u8], &[f32]),
        v: (&[u8], &[f32]),
        o: (&[u8], &[f32]),
        eps: f32,
        inv_freq: &[f32],
        biases: Option<(&[f32], &[f32], &[f32])>,
    ) -> Self {
        let qdim = n_heads * head_dim; // = hidden for MHA
        let kv_dim = n_kv * head_dim;
        Self {
            norm_w: DevF32::from_host(norm_w),
            q: QuantLinear::new(q.0, q.1, hidden, qdim),
            k: QuantLinear::new(k.0, k.1, hidden, kv_dim),
            v: QuantLinear::new(v.0, v.1, hidden, kv_dim),
            o: QuantLinear::new(o.0, o.1, qdim, hidden),
            cache_k: DevHalf::zeros(max_seq * kv_dim),
            cache_v: DevHalf::zeros(max_seq * kv_dim),
            norm: DevHalf::zeros(hidden),
            qb: DevF32::zeros(qdim),
            kb: DevF32::zeros(kv_dim),
            vb: DevF32::zeros(kv_dim),
            qh: DevHalf::zeros(qdim),
            kh: DevHalf::zeros(kv_dim),
            vh: DevHalf::zeros(kv_dim),
            attn_out: DevHalf::zeros(qdim),
            ob: DevF32::zeros(hidden),
            n_heads,
            n_kv,
            head_dim,
            eps,
            inv_freq: DevF32::from_host(inv_freq),
            qbias: biases.map(|b| DevF32::from_host(b.0)),
            kbias: biases.map(|b| DevF32::from_host(b.1)),
            vbias: biases.map(|b| DevF32::from_host(b.2)),
            qbias_h: biases.map(|b| b.0.to_vec()),
            kbias_h: biases.map(|b| b.1.to_vec()),
            vbias_h: biases.map(|b| b.2.to_vec()),
            m_scratch: None,
        }
    }

    /// One decode step at position `pos`: updates the cache and the residual stream `h`.
    pub fn forward(&mut self, h: &mut DevHalf, pos: usize) {
        rmsnorm(h, &self.norm_w, &mut self.norm, self.eps);
        self.q.forward_into(&self.norm, &mut self.qb);
        if let Some(b) = &self.qbias { vadd(&mut self.qb, b); }
        self.qh.copy_cast_from(&self.qb);
        self.k.forward_into(&self.norm, &mut self.kb);
        if let Some(b) = &self.kbias { vadd(&mut self.kb, b); }
        self.kh.copy_cast_from(&self.kb);
        self.v.forward_into(&self.norm, &mut self.vb);
        if let Some(b) = &self.vbias { vadd(&mut self.vb, b); }
        self.vh.copy_cast_from(&self.vb);
        rope(&mut self.qh, pos, self.n_heads, self.head_dim, &self.inv_freq);
        rope(&mut self.kh, pos, self.n_kv, self.head_dim, &self.inv_freq);
        cache_append(&mut self.cache_k, &self.kh, pos);
        cache_append(&mut self.cache_v, &self.vh, pos);
        attention(
            &self.qh, &self.cache_k, &self.cache_v, &mut self.attn_out,
            self.n_heads, self.n_kv, self.head_dim, pos + 1, 0.0,
        );
        self.o.forward_into(&self.attn_out, &mut self.ob);
        residual_add(h, &self.ob);
    }
}

/// A full Llama decoder layer: attention sub-block then MLP sub-block, both updating the
/// residual stream in place. This is the composition of two independently verified blocks
/// (`AttnBlock`, `MlpBlock`); one forward is a complete transformer layer.
pub struct Layer {
    pub attn: AttnBlock,
    pub mlp: MlpBlock,
}

impl Layer {
    pub fn new(attn: AttnBlock, mlp: MlpBlock) -> Self {
        Self { attn, mlp }
    }
    /// One decode step at position `pos`: `h += attn(h); h += mlp(h)`.
    pub fn forward(&mut self, h: &mut DevHalf, pos: usize) {
        self.attn.forward(h, pos);
        self.mlp.forward(h);
    }
}

// ---- batched (M-token) forward for the transformer blocks -------------------
// Verifies K+1 speculative tokens in a single forward. Scratch is allocated per
// call. Scratch is now allocated ONCE per M value and reused (the naive version paid
// hundreds of device mallocs per verify step). Attention (qkv) bias supported.
#[cfg(any(feature = "cuda", feature = "metal"))]
impl MlpBlock {
    /// `h` is `m*hidden`, updated in place: `h = h + MLP(RMSNorm(h))` for all M rows.
    pub fn forward_m(&mut self, h: &mut DevHalf, m: usize) {
        let hidden = self.norm_w.len();
        let inter = self.g.n;
        if self.m_scratch.as_ref().map(|s| s.m) != Some(m) {
            self.m_scratch = Some(MScratchMlp {
                m,
                norm: DevHalf::zeros(m * hidden),
                g: DevF32::zeros(m * inter),
                u: DevF32::zeros(m * inter),
                act: DevHalf::zeros(m * inter),
                mlp: DevF32::zeros(m * hidden),
            });
        }
        let s = self.m_scratch.as_mut().unwrap();
        rmsnorm_m(h, &self.norm_w, &mut s.norm, self.eps, hidden, m);
        self.gate.forward_m(&s.norm, &mut s.g, m);
        self.up.forward_m(&s.norm, &mut s.u, m);
        silu_mul(&s.g, &s.u, &mut s.act);
        self.down.forward_m(&s.act, &mut s.mlp, m);
        residual_add(h, &s.mlp);
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl AttnBlock {
    /// Batched decode step: `h` is `m*hidden`, the M new tokens sit at positions
    /// `base..base+m`. Each query attends causally over `base+row+1` keys.
    pub fn forward_m(&mut self, h: &mut DevHalf, base: usize, m: usize) {
        let hidden = self.norm_w.len();
        let qdim = self.n_heads * self.head_dim;
        let kvdim = self.n_kv * self.head_dim;
        if self.m_scratch.as_ref().map(|s| s.m) != Some(m) {
            let rep = |b: &Option<Vec<f32>>| {
                b.as_ref().map(|v| DevF32::from_host(&v.repeat(m)))
            };
            self.m_scratch = Some(MScratchAttn {
                m,
                norm: DevHalf::zeros(m * hidden),
                qb: DevF32::zeros(m * qdim),
                kb: DevF32::zeros(m * kvdim),
                vb: DevF32::zeros(m * kvdim),
                qh: DevHalf::zeros(m * qdim),
                kh: DevHalf::zeros(m * kvdim),
                vh: DevHalf::zeros(m * kvdim),
                attn_out: DevHalf::zeros(m * qdim),
                ob: DevF32::zeros(m * hidden),
                qbias_r: rep(&self.qbias_h),
                kbias_r: rep(&self.kbias_h),
                vbias_r: rep(&self.vbias_h),
            });
        }
        let s = self.m_scratch.as_mut().unwrap();
        rmsnorm_m(h, &self.norm_w, &mut s.norm, self.eps, hidden, m);
        self.q.forward_m(&s.norm, &mut s.qb, m);
        self.k.forward_m(&s.norm, &mut s.kb, m);
        self.v.forward_m(&s.norm, &mut s.vb, m);
        // Qwen-style qkv bias: the m-row repeated bias makes it a plain vadd.
        if let Some(b) = &s.qbias_r { vadd(&mut s.qb, b); }
        if let Some(b) = &s.kbias_r { vadd(&mut s.kb, b); }
        if let Some(b) = &s.vbias_r { vadd(&mut s.vb, b); }
        s.qh.copy_cast_from(&s.qb);
        s.kh.copy_cast_from(&s.kb);
        s.vh.copy_cast_from(&s.vb);
        rope_m(&mut s.qh, base, self.n_heads, self.head_dim, &self.inv_freq, m);
        rope_m(&mut s.kh, base, self.n_kv, self.head_dim, &self.inv_freq, m);
        cache_append_m(&mut self.cache_k, &s.kh, base, kvdim, m);
        cache_append_m(&mut self.cache_v, &s.vh, base, kvdim, m);
        attention_m(&s.qh, &self.cache_k, &self.cache_v, &mut s.attn_out,
                    self.n_heads, self.n_kv, self.head_dim, base, m);
        self.o.forward_m(&s.attn_out, &mut s.ob, m);
        residual_add(h, &s.ob);
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl Layer {
    /// One batched decode step over M tokens at positions `base..base+m`.
    pub fn forward_m(&mut self, h: &mut DevHalf, base: usize, m: usize) {
        self.attn.forward_m(h, base, m);
        self.mlp.forward_m(h, m);
    }
}

/// A full decoder model: token embedding, a stack of [`Layer`]s with a shared growing KV
/// cache, a final RMSNorm, and a codebook-quantized LM head. `forward(token, pos)` returns
/// the next-token logits. Pure Rust, on-device, no Python.
pub struct Model {
    embedding: Vec<f32>, // host, vocab*hidden (one row uploaded per token)
    layers: Vec<Layer>,
    final_norm: DevF32,
    lm_head: QuantLinear,
    h: DevHalf,
    normed: DevHalf,
    logits: DevF32,
    hidden: usize,
    vocab: usize,
    eps: f32,
    // cached batched-forward buffers (allocated once per M, reused every verify)
    mk_scratch: Option<(usize, DevHalf, DevHalf, DevF32)>,
}

impl Model {
    pub fn new(
        hidden: usize,
        vocab: usize,
        embedding: Vec<f32>,
        layers: Vec<Layer>,
        final_norm_w: &[f32],
        lm_head: (&[u8], &[f32]),
        eps: f32,
    ) -> Self {
        assert_eq!(embedding.len(), vocab * hidden, "embedding must be vocab*hidden");
        Self {
            embedding,
            layers,
            final_norm: DevF32::from_host(final_norm_w),
            lm_head: QuantLinear::new(lm_head.0, lm_head.1, hidden, vocab),
            h: DevHalf::zeros(hidden),
            normed: DevHalf::zeros(hidden),
            logits: DevF32::zeros(vocab),
            hidden,
            vocab,
            eps,
            mk_scratch: None,
        }
    }

    /// Run one token at position `pos`, leaving the next-token logits in `self.logits`
    /// (on device). Shared body of [`forward`](Self::forward) and
    /// [`forward_argmax`](Self::forward_argmax).
    fn run_forward(&mut self, token: usize, pos: usize) {
        let row = &self.embedding[token * self.hidden..(token + 1) * self.hidden];
        self.h.upload(row);
        for l in &mut self.layers {
            l.forward(&mut self.h, pos);
        }
        rmsnorm(&self.h, &self.final_norm, &mut self.normed, self.eps);
        self.lm_head.forward_into(&self.normed, &mut self.logits);
    }

    /// Process one token at position `pos`, returning the `vocab` next-token logits.
    pub fn forward(&mut self, token: usize, pos: usize) -> Vec<f32> {
        self.run_forward(token, pos);
        self.logits.to_host()
    }

    /// Greedy step: process one token at `pos` and return the argmax over the first
    /// `real_vocab` logits, computed on device. No full-vocab host copy. Use this on
    /// the greedy path (temperature 0); the sampling path needs the host logits and
    /// must call [`forward`](Self::forward) instead.
    pub fn forward_argmax(&mut self, token: usize, pos: usize, real_vocab: usize) -> u32 {
        self.run_forward(token, pos);
        self.logits.argmax_device(real_vocab)
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Batched 2-token forward for speculative verification: processes `t0` at `pos` and
    /// `t1` at `pos+1` in ONE forward (appending KV rows `pos`, `pos+1`), returning the
    /// next-token logits at each position: `(logits0, logits1)`. `logits0` predicts the
    /// token after `t0`; `logits1` predicts the token after `t0,t1`.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn forward_m2(&mut self, t0: usize, t1: usize, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let hid = self.hidden;
        let mut h2 = vec![0f32; 2 * hid];
        h2[..hid].copy_from_slice(&self.embedding[t0 * hid..(t0 + 1) * hid]);
        h2[hid..].copy_from_slice(&self.embedding[t1 * hid..(t1 + 1) * hid]);
        let mut h = DevHalf::from_host(&h2);
        for l in &mut self.layers {
            l.forward_m(&mut h, pos, 2);
        }
        let mut normed = DevHalf::zeros(2 * hid);
        rmsnorm_m(&h, &self.final_norm, &mut normed, self.eps, hid, 2);
        let mut logits = DevF32::zeros(2 * self.vocab);
        self.lm_head.forward_m(&normed, &mut logits, 2);
        let all = logits.to_host();
        (all[..self.vocab].to_vec(), all[self.vocab..].to_vec())
    }

    /// Batched forward over M tokens at positions `pos..pos+M`, returning the M next-token
    /// logit vectors. `logits[j]` predicts the token after `tokens[0..=j]`. M = K+1 for a
    /// K-token speculative verify (M<=4, the validated range of gemm_mtile).
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn forward_mk(&mut self, tokens: &[usize], pos: usize) -> Vec<Vec<f32>> {
        let m = tokens.len();
        let hid = self.hidden;
        let mut hm = vec![0f32; m * hid];
        for (r, &t) in tokens.iter().enumerate() {
            hm[r * hid..(r + 1) * hid].copy_from_slice(&self.embedding[t * hid..(t + 1) * hid]);
        }
        if self.mk_scratch.as_ref().map(|s| s.0) != Some(m) {
            self.mk_scratch = Some((
                m,
                DevHalf::zeros(m * hid),
                DevHalf::zeros(m * hid),
                DevF32::zeros(m * self.vocab),
            ));
        }
        let (_, h, normed, logits) = self.mk_scratch.as_mut().unwrap();
        h.upload(&hm);
        for l in &mut self.layers {
            l.forward_m(h, pos, m);
        }
        rmsnorm_m(h, &self.final_norm, normed, self.eps, hid, m);
        self.lm_head.forward_m(normed, logits, m);
        let all = logits.to_host();
        (0..m).map(|r| all[r * self.vocab..(r + 1) * self.vocab].to_vec()).collect()
    }

    /// Greedy speculative decode with block size K (verify K+1 tokens per target forward).
    /// `drafter(prefix, k) -> k proposed tokens`. LOSSLESS by construction. K<=3 (the verify
    /// needs an M=K+1 batched forward, and gemm_mtile is validated to M=4). Returns
    /// `(tokens, target_forwards)`; fewer forwards = the speedup.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn spec_decode_greedy_k(
        &mut self,
        prompt: &[usize],
        n: usize,
        k: usize,
        mut drafter: impl FnMut(&[usize], usize) -> Vec<usize>,
    ) -> (Vec<usize>, usize) {
        assert!((1..=3).contains(&k), "K must be 1..=3 (verify M=K+1 <= 4)");
        assert!(!prompt.is_empty());
        let mut last = vec![];
        for (i, &t) in prompt.iter().enumerate() { last = self.forward(t, i); }
        let mut out: Vec<usize> = Vec::with_capacity(n);
        let mut pos = prompt.len();
        let mut u0 = argmax(&last);
        let mut fwds = 0usize;
        while out.len() < n {
            let mut prefix = prompt.to_vec();
            prefix.extend_from_slice(&out);
            let drafts = drafter(&prefix, k); // k guesses for the tokens after u0
            assert_eq!(drafts.len(), k, "drafter must return exactly k tokens");
            let mut toks = Vec::with_capacity(k + 1);
            toks.push(u0);
            toks.extend_from_slice(&drafts);
            let logits = self.forward_mk(&toks, pos); // k+1 logit vectors
            fwds += 1;
            out.push(u0); // u0 committed at row pos
            if out.len() >= n { break; }
            // accept the longest matching prefix of drafts
            let mut j = 0usize;
            while j < k {
                let a = argmax(&logits[j]); // target token after u0, d_1..d_j
                if drafts[j] == a {
                    out.push(drafts[j]);
                    j += 1;
                    if out.len() >= n { break; }
                } else { break; }
            }
            if out.len() >= n { break; }
            if j == k {
                u0 = argmax(&logits[k]); // all accepted -> bonus token
                pos += k + 1;
            } else {
                u0 = argmax(&logits[j]); // first mismatch -> the correction
                pos += j + 1;            // stale draft rows overwritten next step
            }
        }
        out.truncate(n);
        (out, fwds)
    }

    /// TWO-MODEL wall-clock speculative decode: the drafter is a real second `Model`
    /// decoding incrementally in its own KV cache (no re-prefill per round). The drafter
    /// is conditioned on the committed prefix PLUS the pending token u0, reusing cache
    /// rows via longest-common-prefix (rejected speculative rows are overwritten in
    /// place, same position-indexed trick as the target's verify). LOSSLESS by
    /// construction. Returns `(tokens, target_forwards, drafter_forwards)`; wall-clock
    /// speed is the caller's job (time this against a plain greedy loop).
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn spec_decode_two_model(
        &mut self,
        drafter: &mut Model,
        prompt: &[usize],
        n: usize,
        k: usize,
    ) -> (Vec<usize>, usize, usize) {
        assert!((1..=3).contains(&k), "K must be 1..=3 (verify M=K+1 <= 4)");
        assert!(!prompt.is_empty());
        let mut last = vec![];
        for (i, &t) in prompt.iter().enumerate() { last = self.forward(t, i); }
        let mut out: Vec<usize> = Vec::with_capacity(n);
        let mut pos = prompt.len();
        let mut u0 = argmax(&last);
        let (mut t_fwds, mut d_fwds) = (0usize, 0usize);
        // drafter cache state: d_toks[i] is the token whose KV row sits at position i;
        // d_last = drafter logits after the last row (valid only when d_last_ok).
        let mut d_toks: Vec<usize> = Vec::new();
        let mut d_last: Vec<f32> = vec![];
        while out.len() < n {
            // the drafter conditions on prompt + out + [u0]
            let mut want = prompt.to_vec();
            want.extend_from_slice(&out);
            want.push(u0);
            // longest common prefix with what the drafter already has in cache
            let mut c = 0usize;
            while c < d_toks.len() && c < want.len() && d_toks[c] == want[c] { c += 1; }
            d_toks.truncate(c);
            let mut fed_any = false;
            for i in c..want.len() {
                d_last = drafter.forward(want[i], i);
                d_fwds += 1;
                d_toks.push(want[i]);
                fed_any = true;
            }
            debug_assert!(fed_any || !d_toks.is_empty());
            // propose k tokens greedily, feeding each into the drafter's cache
            let mut drafts = Vec::with_capacity(k);
            for _ in 0..k {
                let d = argmax(&d_last);
                drafts.push(d);
                d_last = drafter.forward(d, d_toks.len());
                d_fwds += 1;
                d_toks.push(d);
            }
            // target verifies u0 + drafts in one batched forward
            let mut toks = Vec::with_capacity(k + 1);
            toks.push(u0);
            toks.extend_from_slice(&drafts);
            let logits = self.forward_mk(&toks, pos);
            t_fwds += 1;
            out.push(u0);
            if out.len() >= n { break; }
            let mut j = 0usize;
            while j < k {
                let a = argmax(&logits[j]);
                if drafts[j] == a {
                    out.push(drafts[j]);
                    j += 1;
                    if out.len() >= n { break; }
                } else { break; }
            }
            if out.len() >= n { break; }
            if j == k {
                u0 = argmax(&logits[k]);
                pos += k + 1;
            } else {
                u0 = argmax(&logits[j]);
                pos += j + 1; // stale target rows overwritten next step
            }
        }
        out.truncate(n);
        (out, t_fwds, d_fwds)
    }

    /// Greedy speculative decode, K=1. `drafter(prefix) -> proposed next token`. Emits
    /// `n` tokens after the prompt. LOSSLESS by construction: every emitted token is the
    /// target's greedy argmax, so the output equals plain greedy decode regardless of the
    /// drafter's accuracy — a good drafter just emits more tokens per target forward.
    /// Returns `(tokens, target_forwards)` so callers can see the speedup (fewer forwards).
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn spec_decode_greedy(
        &mut self,
        prompt: &[usize],
        n: usize,
        mut drafter: impl FnMut(&[usize]) -> usize,
    ) -> (Vec<usize>, usize) {
        assert!(!prompt.is_empty());
        // Prefill the prompt (M=1), last logits predict the first emitted token.
        let mut last = vec![];
        for (i, &t) in prompt.iter().enumerate() {
            last = self.forward(t, i);
        }
        let mut out: Vec<usize> = Vec::with_capacity(n);
        let mut pos = prompt.len();
        let mut u0 = argmax(&last); // next token to emit, not yet in cache
        let mut fwds = 0usize;
        while out.len() < n {
            let mut prefix = prompt.to_vec();
            prefix.extend_from_slice(&out);
            let d = drafter(&prefix); // draft for the token after u0
            let (l0, l1) = self.forward_m2(u0, d, pos);
            fwds += 1;
            out.push(u0); // u0 committed at row `pos`
            let a = argmax(&l0); // target's true token after u0
            if out.len() >= n { break; }
            if d == a {
                out.push(d); // accepted: d committed at row pos+1
                u0 = argmax(&l1); // bonus token, next to emit
                pos += 2;
            } else {
                u0 = a; // reject: row pos+1 (d) is stale, overwritten next step
                pos += 1;
            }
        }
        out.truncate(n);
        (out, fwds)
    }

    /// Load a model exported in the `.cbk` format (see `model/export_runtime.py`):
    /// magic, config, fp16 embedding, then per layer the RMSNorm weights and the seven
    /// codebook-quantized projections, then the final norm and codebook LM head.
    pub fn load_cbk(path: &str, max_seq: usize) -> std::io::Result<Model> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        let v2 = &magic == b"CBK2";
        let v3 = &magic == b"CBK3";
        assert!(&magic == b"CBK1" || v2 || v3, "bad magic (not a .cbk file)");
        let c: Vec<usize> = (0..7).map(|_| rd_i32(&mut r) as usize).collect();
        let (n_layers, hidden, n_heads, n_kv, head_dim, inter, vocab) =
            (c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
        let eps = rd_f32(&mut r);
        let base = rd_f32(&mut r);
        let scale = if v2 || v3 { rd_f32(&mut r) } else { 1.0 };
        let has_bias = if v3 { rd_i32(&mut r) != 0 } else { false };
        let halfd = head_dim / 2;
        // CBK3 stores the RoPE frequencies (any scaling baked in); older formats compute them
        let inv_freq: Vec<f32> = if v3 {
            rd_f32_vec(&mut r, halfd)
        } else {
            (0..halfd).map(|d| base.powf(-2.0 * d as f32 / head_dim as f32) / scale).collect()
        };
        let embedding = rd_f16_vec(&mut r, vocab * hidden);
        let qdim = n_heads * head_dim;
        let kvdim = n_kv * head_dim;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let an = rd_f32_vec(&mut r, hidden);
            let (qp, qc) = (rd_u8_vec(&mut r, hidden * qdim / 2), rd_f32_vec(&mut r, K * qdim));
            let qb = if has_bias { Some(rd_f32_vec(&mut r, qdim)) } else { None };
            let (kp, kc) = (rd_u8_vec(&mut r, hidden * kvdim / 2), rd_f32_vec(&mut r, K * kvdim));
            let kb = if has_bias { Some(rd_f32_vec(&mut r, kvdim)) } else { None };
            let (vp, vc) = (rd_u8_vec(&mut r, hidden * kvdim / 2), rd_f32_vec(&mut r, K * kvdim));
            let vb = if has_bias { Some(rd_f32_vec(&mut r, kvdim)) } else { None };
            let (op, oc) = (rd_u8_vec(&mut r, qdim * hidden / 2), rd_f32_vec(&mut r, K * hidden));
            let pn = rd_f32_vec(&mut r, hidden);
            let (gp, gc) = (rd_u8_vec(&mut r, hidden * inter / 2), rd_f32_vec(&mut r, K * inter));
            let (up, uc) = (rd_u8_vec(&mut r, hidden * inter / 2), rd_f32_vec(&mut r, K * inter));
            let (dp, dc) = (rd_u8_vec(&mut r, inter * hidden / 2), rd_f32_vec(&mut r, K * hidden));
            let biases = match (&qb, &kb, &vb) {
                (Some(q), Some(k), Some(v)) => Some((q.as_slice(), k.as_slice(), v.as_slice())),
                _ => None,
            };
            let attn = AttnBlock::new(
                hidden, n_heads, n_kv, head_dim, max_seq, &an,
                (&qp, &qc), (&kp, &kc), (&vp, &vc), (&op, &oc), eps, &inv_freq, biases,
            );
            let mlp = MlpBlock::new(hidden, inter, &pn, &gp, &gc, &up, &uc, &dp, &dc, eps);
            layers.push(Layer::new(attn, mlp));
        }
        let final_norm = rd_f32_vec(&mut r, hidden);
        let (lmp, lmc) = (rd_u8_vec(&mut r, hidden * vocab / 2), rd_f32_vec(&mut r, K * vocab));
        Ok(Model::new(hidden, vocab, embedding, layers, &final_norm, (&lmp, &lmc), eps))
    }
}

fn rd_i32(r: &mut impl Read) -> i32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).unwrap();
    i32::from_le_bytes(b)
}
fn rd_f32(r: &mut impl Read) -> f32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).unwrap();
    f32::from_le_bytes(b)
}
fn rd_f32_vec(r: &mut impl Read, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    r.read_exact(&mut b).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn rd_f16_vec(r: &mut impl Read, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 2];
    r.read_exact(&mut b).unwrap();
    b.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
}
fn rd_u8_vec(r: &mut impl Read, n: usize) -> Vec<u8> {
    let mut b = vec![0u8; n];
    r.read_exact(&mut b).unwrap();
    b
}

/// Record the next `len` bytes as a byte range into `mmap` (its absolute offset = the
/// reader's current stream position, since `mmap` covers the same file from byte 0) and seek
/// `r` past them WITHOUT reading -- avoids ever copying a routed expert's packed indices into
/// RAM. See `PackedBytes`/`ExpertHost`/`load_deepseek_qlora`.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn mmap_skip(r: &mut BufReader<File>, mmap: &Arc<memmap2::Mmap>, len: usize) -> std::io::Result<PackedBytes> {
    let off = r.stream_position()? as usize;
    r.seek(SeekFrom::Current(len as i64))?;
    Ok(PackedBytes::Mmap(mmap.clone(), off, len))
}

/// Block until all queued GPU work completes (call before stopping a timer).
pub fn sync() {
    unsafe { dev_sync() };
}

/// A captured CUDA graph of a GPU op sequence, replayable with near-zero CPU launch
/// overhead. This is how serving engines turn a per-op kernel speedup into an
/// end-to-end one.
pub struct Graph {
    exec: *mut c_void,
}

impl Graph {
    /// Capture the GPU work issued inside `f` into a replayable graph. Every buffer `f`
    /// touches must be stable (allocated once); replay reuses the same device memory.
    pub fn capture(f: impl FnOnce()) -> Self {
        unsafe { graph_begin() };
        f();
        let exec = unsafe { graph_end() };
        assert!(!exec.is_null(), "CUDA graph capture failed");
        Self { exec }
    }

    /// Replay the captured graph.
    pub fn launch(&self) {
        unsafe { graph_launch(self.exec) };
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { graph_free(self.exec) };
    }
}

/// C ABI for embedding the engine in a native app (iOS/macOS), no server.
pub mod ffi;

/// CPU forward for scalar 4-bit codebook experts (host-side, no GPU). See module docs.
pub mod cpu_experts;

/// Microbenchmark: fused 4-bit decode GEMV vs dense fp16 GEMV at `ic`x`oc`,
/// averaged over `iters`. Returns (ms_4bit, ms_fp16). Metal backend only.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub fn bench_gemv(ic: usize, oc: usize, iters: usize) -> (f64, f64) {
    unsafe { backend::bench_gemv(ic as i32, oc as i32, iters as i32) }
}

/// M0 microbenchmark: time the small-M fused decode GEMM at `m` columns.
/// If ms(m=6) ~= ms(m=1), verifying K+1 draft tokens costs ~one weight read.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub fn bench_mtile(ic: usize, oc: usize, m: usize, iters: usize) -> f64 {
    unsafe { backend::bench_mtile(ic as i32, oc as i32, m as i32, iters as i32) }
}

/// M0b: optimized small-M decode GEMM (2 chan/thread, no atomics).
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub fn bench_mtile2(ic: usize, oc: usize, m: usize, iters: usize) -> f64 {
    unsafe { backend::bench_mtile2(ic as i32, oc as i32, m as i32, iters as i32) }
}

/// Validate gemm_mtile computes the same as per-column gemv (worst rel err).
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub fn check_mtile(ic: usize, oc: usize, m: usize) -> f64 {
    unsafe { backend::check_mtile(ic as i32, oc as i32, m as i32) }
}

/// Validate batched rmsnorm_m against per-row M=1 rmsnorm (worst rel err).
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub fn check_rmsnorm_m(n: usize, m: usize) -> f64 {
    unsafe { backend::check_rmsnorm_m(n as i32, m as i32) }
}

/// Validate batched causal attention attn_m vs per-query M=1 reference.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub fn check_attn_m(n_heads: usize, n_kv: usize, hd: usize, base: usize, m: usize) -> f64 {
    unsafe { backend::check_attn_m(n_heads as i32, n_kv as i32, hd as i32, base as i32, m as i32) }
}

/// Validate batched rope_m vs per-row M=1 rope.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub fn check_rope_m(n_heads: usize, head_dim: usize, base: usize, m: usize) -> f64 {
    unsafe { backend::check_rope_m(n_heads as i32, head_dim as i32, base as i32, m as i32) }
}

/// End-to-end lossless check of the batched decode path: a full transformer layer
/// (attention + MLP) run once over M tokens must equal M sequential M=1 forwards
/// through the same weights with a causally growing KV cache. Returns worst rel err
/// over the M output rows. This validates the whole spec-dec verify forward.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_batched_layer(hidden: usize, n_heads: usize, n_kv: usize, head_dim: usize,
                           inter: usize, base: usize, m: usize) -> f64 {
    check_batched_impl(hidden, n_heads, n_kv, head_dim, inter, base, m, false)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
fn check_batched_impl(hidden: usize, n_heads: usize, n_kv: usize, head_dim: usize,
                      inter: usize, base: usize, m: usize, attn_only: bool) -> f64 {
    let qdim = n_heads * head_dim;
    let kvdim = n_kv * head_dim;
    let eps = 1e-5f32;
    let mut rng = 0xC0FFEEu64;
    let mut nx = || { rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; (((rng>>40) as f32/(1u64<<24) as f32)*2.0-1.0) };
    let packed = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
    let cbk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
    // host weights (generated once, shared by both paths)
    let anorm: Vec<f32> = (0..hidden).map(|_| nx()*0.1+1.0).collect();
    let mnorm: Vec<f32> = (0..hidden).map(|_| nx()*0.1+1.0).collect();
    let (qp,qc)=(packed(hidden*(qdim/2),&mut nx),  cbk(K*qdim,&mut nx));
    let (kp,kc)=(packed(hidden*(kvdim/2),&mut nx), cbk(K*kvdim,&mut nx));
    let (vp,vc)=(packed(hidden*(kvdim/2),&mut nx), cbk(K*kvdim,&mut nx));
    let (op,oc)=(packed(qdim*(hidden/2),&mut nx),  cbk(K*hidden,&mut nx));
    let (gp,gc)=(packed(hidden*(inter/2),&mut nx), cbk(K*inter,&mut nx));
    let (up,uc)=(packed(hidden*(inter/2),&mut nx), cbk(K*inter,&mut nx));
    let (dp,dc)=(packed(inter*(hidden/2),&mut nx), cbk(K*hidden,&mut nx));
    let inv: Vec<f32> = (0..head_dim/2).map(|d| 10000f32.powf(-2.0*d as f32/head_dim as f32)).collect();
    let build = || {
        let a = AttnBlock::new(hidden, n_heads, n_kv, head_dim, base+m+4, &anorm,
            (&qp,&qc),(&kp,&kc),(&vp,&vc),(&op,&oc), eps, &inv, None);
        let mlp = MlpBlock::new(hidden, inter, &mnorm, &gp,&gc,&up,&uc,&dp,&dc, eps);
        Layer::new(a, mlp)
    };
    // random hidden states for base+m tokens
    let hs: Vec<Vec<f32>> = (0..base+m).map(|_| (0..hidden).map(|_| nx()*0.3).collect()).collect();
    // reference: base+m sequential M=1 forwards; capture the last m outputs
    let mut refl = build();
    let mut refout = vec![vec![0f32; hidden]; m];
    for pos in 0..base+m {
        let mut hp = DevHalf::from_host(&hs[pos]);
        if attn_only { refl.attn.forward(&mut hp, pos); } else { refl.forward(&mut hp, pos); }
        if pos >= base { refout[pos-base] = hp.to_host(); }
    }
    // batched: prefill base tokens M=1, then ONE M=2 forward for the last m
    let mut batl = build();
    for pos in 0..base {
        let mut hp = DevHalf::from_host(&hs[pos]);
        if attn_only { batl.attn.forward(&mut hp, pos); } else { batl.forward(&mut hp, pos); }
    }
    let mut hb = vec![0f32; m*hidden];
    for r in 0..m { hb[r*hidden..(r+1)*hidden].copy_from_slice(&hs[base+r]); }
    let mut hbat = DevHalf::from_host(&hb);
    if attn_only { batl.attn.forward_m(&mut hbat, base, m); } else { batl.forward_m(&mut hbat, base, m); }
    let got = hbat.to_host();
    // Compare per row with a normalized L2 metric: ||got - ref|| / ||ref||. A per-element
    // max metric would blow up on near-zero hidden elements, where the tiny absolute noise
    // from fp16 + non-associative atomic accumulation (identical to normal M=1 decode) reads
    // as a huge relative error. The L2 metric measures true divergence of the whole vector.
    let mut worst = 0f64;
    for r in 0..m {
        let mut num = 0f64; let mut den = 0f64;
        for i in 0..hidden {
            let d = (got[r*hidden+i] - refout[r][i]) as f64;
            num += d*d; den += (refout[r][i] as f64).powi(2);
        }
        worst = worst.max((num / den.max(1e-12)).sqrt());
    }
    worst
}

/// Test wrapper for the batched-layer lossless check.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn batched_layer_relerr(hidden: usize, n_heads: usize, n_kv: usize, head_dim: usize, inter: usize, base: usize, m: usize) -> f64 {
    check_batched_layer(hidden, n_heads, n_kv, head_dim, inter, base, m)
}

/// Attention-sublayer-only variant of the batched lossless check (for bisecting).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn batched_attn_relerr(hidden: usize, n_heads: usize, n_kv: usize, head_dim: usize, base: usize, m: usize) -> f64 {
    check_batched_impl(hidden, n_heads, n_kv, head_dim, 512, base, m, true)
}

/// Index of the maximum element (greedy token).
pub fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize; let mut bv = f32::MIN;
    for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i; } }
    bi
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl Model {
    /// Plain greedy autoregressive decode (the reference the spec loop must match).
    pub fn decode_greedy(&mut self, prompt: &[usize], n: usize) -> Vec<usize> {
        let mut last = vec![];
        for (i, &t) in prompt.iter().enumerate() { last = self.forward(t, i); }
        let mut out = Vec::with_capacity(n);
        let mut pos = prompt.len();
        let mut u = argmax(&last);
        while out.len() < n {
            out.push(u);
            if out.len() >= n { break; }
            let l = self.forward(u, pos);
            u = argmax(&l);
            pos += 1;
        }
        out
    }
}

/// Validate the K=1 speculative loop is lossless: its output must equal plain greedy
/// decode for ANY drafter. Builds a tiny random model, decodes greedily as reference,
/// then runs spec-dec with (a) a perfect drafter (all accepts -> fewer forwards) and
/// (b) an adversarial drafter (many rejects). Returns (ok_oracle, ok_wrong, fwds_oracle,
/// fwds_wrong, n) so the caller can also confirm the accept path actually saves forwards.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_spec_decode() -> (bool, bool, usize, usize, usize) {
    let (hidden, vocab, n_heads, n_kv, head_dim, inter, n_layers) = (256usize, 256usize, 8usize, 8usize, 32usize, 512usize, 2usize);
    let eps = 1e-5f32;
    let max_seq = 128usize;
    let mut rng = 0x5EEDu64;
    let mut nx = move || { rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; (((rng>>40) as f32/(1u64<<24) as f32)*2.0-1.0) };
    // Deterministic weight generators (same seed => identical model each build).
    let build = || {
        let mut r = { let mut s = 0x5EEDu64; move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0) } };
        let packed = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
        let cbk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
        let qdim = n_heads*head_dim; let kvdim = n_kv*head_dim;
        let emb: Vec<f32> = (0..vocab*hidden).map(|_| r()*0.2).collect();
        let inv: Vec<f32> = (0..head_dim/2).map(|d| 10000f32.powf(-2.0*d as f32/head_dim as f32)).collect();
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let an: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
            let mn: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
            let a = AttnBlock::new(hidden, n_heads, n_kv, head_dim, max_seq, &an,
                (&packed(hidden*(qdim/2),&mut r), &cbk(K*qdim,&mut r)),
                (&packed(hidden*(kvdim/2),&mut r), &cbk(K*kvdim,&mut r)),
                (&packed(hidden*(kvdim/2),&mut r), &cbk(K*kvdim,&mut r)),
                (&packed(qdim*(hidden/2),&mut r), &cbk(K*hidden,&mut r)),
                eps, &inv, None);
            let mlp = MlpBlock::new(hidden, inter, &mn,
                &packed(hidden*(inter/2),&mut r), &cbk(K*inter,&mut r),
                &packed(hidden*(inter/2),&mut r), &cbk(K*inter,&mut r),
                &packed(inter*(hidden/2),&mut r), &cbk(K*hidden,&mut r), eps);
            layers.push(Layer::new(a, mlp));
        }
        let fnorm: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
        let lmp = packed(hidden*(vocab/2),&mut r); let lmc = cbk(K*vocab,&mut r);
        Model::new(hidden, vocab, emb, layers, &fnorm, (&lmp,&lmc), eps)
    };
    let _ = &mut nx;
    let prompt = vec![1usize, 5, 9, 2];
    let n = 20usize;
    // reference
    let mut m0 = build();
    let reference = m0.decode_greedy(&prompt, n);
    // full oracle sequence for the perfect drafter (prompt + reference)
    let mut full = prompt.clone(); full.extend_from_slice(&reference);
    // (a) perfect drafter: propose the true next token => all accepts
    let mut m1 = build();
    let full_o = full.clone();
    // Draft predicts the token AFTER u0; prefix excludes u0, so index is prefix.len()+1.
    let (seq_o, fwds_o) = m1.spec_decode_greedy(&prompt, n, move |prefix: &[usize]| {
        full_o.get(prefix.len() + 1).copied().unwrap_or(0)
    });
    // (b) adversarial drafter: propose a token that is usually wrong => many rejects
    let mut m2 = build();
    let (seq_w, fwds_w) = m2.spec_decode_greedy(&prompt, n, |prefix: &[usize]| (prefix.len()*7 + 3) % vocab);
    (seq_o == reference, seq_w == reference, fwds_o, fwds_w, n)
}

/// K-general lossless check of the speculative loop (K=1..3). Same tiny random model as
/// `check_spec_decode`, but verifies K+1 tokens per forward. Returns
/// (ok_oracle, ok_wrong, fwds_oracle, fwds_wrong, n).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_spec_decode_k(k: usize) -> (bool, bool, usize, usize, usize) {
    let (hidden, vocab, n_heads, n_kv, head_dim, inter, n_layers) = (256usize, 256usize, 8usize, 8usize, 32usize, 512usize, 2usize);
    let eps = 1e-5f32; let max_seq = 128usize;
    let build = || {
        let mut r = { let mut s = 0x5EEDu64; move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0) } };
        let packed = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
        let cbk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
        let qdim = n_heads*head_dim; let kvdim = n_kv*head_dim;
        let emb: Vec<f32> = (0..vocab*hidden).map(|_| r()*0.2).collect();
        let inv: Vec<f32> = (0..head_dim/2).map(|d| 10000f32.powf(-2.0*d as f32/head_dim as f32)).collect();
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let an: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
            let mn: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
            let a = AttnBlock::new(hidden, n_heads, n_kv, head_dim, max_seq, &an,
                (&packed(hidden*(qdim/2),&mut r), &cbk(K*qdim,&mut r)),
                (&packed(hidden*(kvdim/2),&mut r), &cbk(K*kvdim,&mut r)),
                (&packed(hidden*(kvdim/2),&mut r), &cbk(K*kvdim,&mut r)),
                (&packed(qdim*(hidden/2),&mut r), &cbk(K*hidden,&mut r)), eps, &inv, None);
            let mlp = MlpBlock::new(hidden, inter, &mn,
                &packed(hidden*(inter/2),&mut r), &cbk(K*inter,&mut r),
                &packed(hidden*(inter/2),&mut r), &cbk(K*inter,&mut r),
                &packed(inter*(hidden/2),&mut r), &cbk(K*hidden,&mut r), eps);
            layers.push(Layer::new(a, mlp));
        }
        let fnorm: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
        let lmp = packed(hidden*(vocab/2),&mut r); let lmc = cbk(K*vocab,&mut r);
        Model::new(hidden, vocab, emb, layers, &fnorm, (&lmp,&lmc), eps)
    };
    let prompt = vec![1usize, 5, 9, 2];
    let n = 21usize;
    let mut m0 = build();
    let reference = m0.decode_greedy(&prompt, n);
    let mut full = prompt.clone(); full.extend_from_slice(&reference);
    // oracle: the true next k tokens after u0 (u0 = full[prefix.len()]) => all accept
    let mut m1 = build();
    let full_o = full.clone();
    let (seq_o, fwds_o) = m1.spec_decode_greedy_k(&prompt, n, k, move |prefix: &[usize], kk: usize| {
        (0..kk).map(|i| full_o.get(prefix.len()+1+i).copied().unwrap_or(0)).collect()
    });
    // adversarial: k arbitrary tokens => frequent rejects
    let mut m2 = build();
    let (seq_w, fwds_w) = m2.spec_decode_greedy_k(&prompt, n, k, |prefix: &[usize], kk: usize| {
        (0..kk).map(|i| (prefix.len()*7 + 3 + i*11) % vocab).collect()
    });
    (seq_o == reference, seq_w == reference, fwds_o, fwds_w, n)
}

impl Model {
    /// Confidence-scheduled speculative decode (DSpark-style): the drafter returns up to
    /// `kmax` (token, confidence) proposals, and each step drafts a DYNAMIC K = the leading
    /// run of proposals with confidence >= `thresh` (at least 1, capped at min(kmax,3) since
    /// the verify needs M=K+1 <= 4). Draft more when the drafter is sure, fewer when not.
    /// Still LOSSLESS (the verify guarantees the target's greedy output). Returns
    /// `(tokens, target_forwards, avg_k)`.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn spec_decode_greedy_conf(
        &mut self,
        prompt: &[usize],
        n: usize,
        kmax: usize,
        thresh: f32,
        mut drafter: impl FnMut(&[usize], usize) -> Vec<(usize, f32)>,
    ) -> (Vec<usize>, usize, f64) {
        assert!(!prompt.is_empty());
        let kcap = kmax.min(3);
        let mut last = vec![];
        for (i, &t) in prompt.iter().enumerate() { last = self.forward(t, i); }
        let mut out: Vec<usize> = Vec::with_capacity(n);
        let mut pos = prompt.len();
        let mut u0 = argmax(&last);
        let (mut fwds, mut ksum) = (0usize, 0usize);
        while out.len() < n {
            let mut prefix = prompt.to_vec();
            prefix.extend_from_slice(&out);
            let prop = drafter(&prefix, kcap);
            // dynamic K: leading run above the confidence threshold, >=1, <=kcap
            let mut k = 1usize;
            while k < prop.len().min(kcap) && prop[k].1 >= thresh { k += 1; }
            let drafts: Vec<usize> = prop[..k].iter().map(|(t, _)| *t).collect();
            let mut toks = Vec::with_capacity(k + 1);
            toks.push(u0); toks.extend_from_slice(&drafts);
            let logits = self.forward_mk(&toks, pos);
            fwds += 1; ksum += k;
            out.push(u0);
            if out.len() >= n { break; }
            let mut j = 0usize;
            while j < k {
                if drafts[j] == argmax(&logits[j]) { out.push(drafts[j]); j += 1; if out.len() >= n { break; } }
                else { break; }
            }
            if out.len() >= n { break; }
            if j == k { u0 = argmax(&logits[k]); pos += k + 1; }
            else { u0 = argmax(&logits[j]); pos += j + 1; }
        }
        out.truncate(n);
        (out, fwds, ksum as f64 / fwds.max(1) as f64)
    }
}

/// Lossless check of confidence-scheduled decode: output must equal plain greedy for any
/// confidence signal. Returns (ok_highconf, ok_mixed, fwds_high, avg_k_high, n).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_spec_decode_conf() -> (bool, bool, usize, f64, usize) {
    let (hidden, vocab, n_heads, n_kv, head_dim, inter, n_layers) = (256usize, 256usize, 8usize, 8usize, 32usize, 512usize, 2usize);
    let eps = 1e-5f32; let max_seq = 128usize;
    let build = || {
        let mut r = { let mut s = 0x5EEDu64; move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0) } };
        let packed = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
        let cbk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
        let qdim = n_heads*head_dim; let kvdim = n_kv*head_dim;
        let emb: Vec<f32> = (0..vocab*hidden).map(|_| r()*0.2).collect();
        let inv: Vec<f32> = (0..head_dim/2).map(|d| 10000f32.powf(-2.0*d as f32/head_dim as f32)).collect();
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let an: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
            let mn: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
            let a = AttnBlock::new(hidden, n_heads, n_kv, head_dim, max_seq, &an,
                (&packed(hidden*(qdim/2),&mut r), &cbk(K*qdim,&mut r)),
                (&packed(hidden*(kvdim/2),&mut r), &cbk(K*kvdim,&mut r)),
                (&packed(hidden*(kvdim/2),&mut r), &cbk(K*kvdim,&mut r)),
                (&packed(qdim*(hidden/2),&mut r), &cbk(K*hidden,&mut r)), eps, &inv, None);
            let mlp = MlpBlock::new(hidden, inter, &mn,
                &packed(hidden*(inter/2),&mut r), &cbk(K*inter,&mut r),
                &packed(hidden*(inter/2),&mut r), &cbk(K*inter,&mut r),
                &packed(inter*(hidden/2),&mut r), &cbk(K*hidden,&mut r), eps);
            layers.push(Layer::new(a, mlp));
        }
        let fnorm: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
        let lmp = packed(hidden*(vocab/2),&mut r); let lmc = cbk(K*vocab,&mut r);
        Model::new(hidden, vocab, emb, layers, &fnorm, (&lmp,&lmc), eps)
    };
    let prompt = vec![1usize, 5, 9, 2];
    let n = 21usize;
    let mut m0 = build();
    let reference = m0.decode_greedy(&prompt, n);
    let mut full = prompt.clone(); full.extend_from_slice(&reference);
    // high-confidence oracle: true tokens, conf=1.0 => always drafts kmax
    let mut m1 = build(); let fo = full.clone();
    let (seq_h, fwds_h, avgk) = m1.spec_decode_greedy_conf(&prompt, n, 3, 0.6, move |p: &[usize], kk| {
        (0..kk).map(|i| (fo.get(p.len()+1+i).copied().unwrap_or(0), 1.0f32)).collect()
    });
    // mixed confidence: alternating high/low conf, arbitrary tokens => dynamic K + rejects
    let mut m2 = build();
    let (seq_m, _fm, _ak) = m2.spec_decode_greedy_conf(&prompt, n, 3, 0.6, |p: &[usize], kk| {
        (0..kk).map(|i| ((p.len()*7+3+i*11)%vocab, if i%2==0 {0.9} else {0.3})).collect()
    });
    (seq_h == reference, seq_m == reference, fwds_h, avgk, n)
}

/// Two-model end-to-end speculative decode with real KV rollback: a separate `drafter`
/// proposes K tokens (its own cache), the `target` verifies K+1 in one batched forward,
/// and the longest matching prefix is committed. Stale draft rows in BOTH caches are
/// overwritten on the next step (implicit rollback); the accept-all case resyncs the
/// drafter by feeding it the bonus token's predecessor. LOSSLESS: output == plain greedy.
/// Both models decode from position 0, so re-running reuses the same loaded models (the
/// cache is position-addressed and causal, so old rows are never read). Returns
/// `(tokens, target_forwards, drafter_forwards)`.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn spec_decode_two_model(
    target: &mut Model,
    drafter: &mut Model,
    prompt: &[usize],
    n: usize,
    k: usize,
) -> (Vec<usize>, usize, usize) {
    assert!((1..=3).contains(&k));
    assert_eq!(target.vocab(), drafter.vocab(), "target and drafter must share a tokenizer");
    let mut tl = vec![];
    for (i, &t) in prompt.iter().enumerate() {
        tl = target.forward(t, i);
        let _ = drafter.forward(t, i);
    }
    let mut out: Vec<usize> = Vec::with_capacity(n);
    let mut pos = prompt.len();
    let mut u0 = argmax(&tl);
    let (mut tf, mut df) = (0usize, 0usize);
    while out.len() < n {
        // drafter proposes k tokens after u0 (it consumes u0 first), advancing its own cache
        let mut drafts = Vec::with_capacity(k);
        let mut dcur = u0;
        for j in 0..k {
            let dl = drafter.forward(dcur, pos + j);
            df += 1;
            dcur = argmax(&dl);
            drafts.push(dcur);
        }
        // target verifies u0,d_1..d_k in one forward
        let mut toks = Vec::with_capacity(k + 1);
        toks.push(u0);
        toks.extend_from_slice(&drafts);
        let logits = target.forward_mk(&toks, pos);
        tf += 1;
        out.push(u0);
        if out.len() >= n { break; }
        let mut acc = 0;
        while acc < k {
            if drafts[acc] == argmax(&logits[acc]) {
                out.push(drafts[acc]);
                acc += 1;
                if out.len() >= n { break; }
            } else { break; }
        }
        if out.len() >= n { break; }
        if acc == k {
            // all accepted: drafter cache is missing d_k (only consumed u0..d_{k-1}) -> sync it
            let _ = drafter.forward(drafts[k - 1], pos + k);
            df += 1;
            u0 = argmax(&logits[k]); // bonus token
            pos += k + 1;
        } else {
            u0 = argmax(&logits[acc]); // correction; stale draft rows overwritten next step
            pos += acc + 1;
        }
    }
    out.truncate(n);
    (out, tf, df)
}

/// Validate the MLA (DeepSeek-V2/V3) decode attention kernel vs a CPU reference.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub fn check_mla_attn(n_heads: usize, d_c: usize, d_rope: usize, seqlen: usize) -> f64 {
    unsafe { backend::check_mla_attn(n_heads as i32, d_c as i32, d_rope as i32, seqlen as i32) }
}

/// Scaled add into an f32 accumulator: `acc += alpha * y`. Combines MoE expert outputs.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn saxpy(acc: &mut DevF32, y: &DevF32, alpha: f32) {
    assert_eq!(acc.n, y.n);
    unsafe { op_saxpy(acc.ptr, y.ptr, alpha, acc.n as i32) };
}

/// One expert = a Llama-style gated FFN (gate/up/down codebook projections).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct Expert { gate: QuantLinear, up: QuantLinear, down: QuantLinear }

/// DeepSeek-V3/R1 router scoring ("noaux_tc", `scoring_func="sigmoid"`, `topk_method=
/// "noaux_tc"`): `s = sigmoid(logits)`; `sb = s + score_bias` (the learned per-expert
/// correction, decoupled from the loss so it does NOT bias the combine weights); experts
/// are split into `n_group` equal groups, each group's score is the sum of its top-2 `sb`
/// values, and only the `topk_group` best groups stay eligible; the final `top_k` experts
/// are the highest-`sb` experts among the eligible groups; combine weights are `s` (NOT
/// `sb`) renormalized to sum to 1 over the selection, then scaled by `rscale`. Returns
/// `(expert, weight)` pairs, `top_k` of them.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn moe_route_v3(logits: &[f32], score_bias: &[f32], n_experts: usize, n_group: usize,
                 topk_group: usize, top_k: usize, rscale: f32) -> Vec<(usize, f32)> {
    assert_eq!(logits.len(), n_experts);
    assert_eq!(score_bias.len(), n_experts);
    let s: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let sb: Vec<f32> = s.iter().zip(score_bias.iter()).map(|(a, b)| a + b).collect();
    let group_size = n_experts / n_group;
    let group_score = |g: usize| -> f32 {
        let (start, end) = (g * group_size, (g + 1) * group_size);
        let mut top2 = [f32::MIN, f32::MIN];
        for &v in &sb[start..end] {
            if v > top2[0] { top2[1] = top2[0]; top2[0] = v; } else if v > top2[1] { top2[1] = v; }
        }
        if group_size >= 2 { top2[0] + top2[1] } else { top2[0] }
    };
    let mut gidx: Vec<usize> = (0..n_group).collect();
    let gscores: Vec<f32> = (0..n_group).map(group_score).collect();
    gidx.sort_by(|&a, &b| gscores[b].partial_cmp(&gscores[a]).unwrap());
    let eligible: std::collections::HashSet<usize> = gidx[..topk_group].iter().cloned().collect();
    let mut idx: Vec<usize> = (0..n_experts).filter(|&e| eligible.contains(&(e / group_size))).collect();
    idx.sort_by(|&a, &b| sb[b].partial_cmp(&sb[a]).unwrap());
    idx.truncate(top_k);
    let wsum: f32 = idx.iter().map(|&e| s[e]).sum();
    idx.iter().map(|&e| (e, rscale * s[e] / wsum)).collect()
}

/// A Mixture-of-Experts decoder block (DeepSeek-V2/V3 style): RMSNorm, a router that
/// scores `n_experts`, top-k selection, the k selected expert FFNs run and combined by
/// their (renormalized) router weights, plus an always-on shared expert, then a residual.
/// At batch-1 decode only k of n_experts run, which is what makes huge MoE models cheap
/// per token (and what the memory-offload path below exploits). Solves the "dense runtime"
/// wall: the router + top-k + expert combine are the missing pieces.
/// A routed expert kept HOST-side (never uploaded to the GPU) for the CPU-experts hybrid
/// path (`TRAPETUM_CPU_EXPERTS=1`). Owns the exact six slices `ExpertHost::Scalar` stores;
/// evaluated by [`cpu_experts::expert_forward_cpu`]. This is the VRAM-savings variant: the
/// bulk of a MoE model's weights (the routed experts) stays in RAM.
// Packed indices are stored OUTPUT-major (`pack_to_rowmajor`) and codebooks PRE-TRANSPOSED
// (`cb_t[o*K+k]`, `transpose_codebook`), so the row-major work-stealing kernel streams each
// output row's indices contiguously with its 16-entry codebook table resident in registers.
#[cfg(any(feature = "cuda", feature = "metal"))]
struct CpuExpert { gp_t: Vec<u8>, gc_t: Vec<f32>, up_t: Vec<u8>, uc_t: Vec<f32>, dp_t: Vec<u8>, dc_t: Vec<f32> }

#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct MoeBlock {
    norm_w: DevF32,
    router: DenseLinear,
    experts: Vec<Expert>,
    /// When `Some`, routed experts run on the CPU (weights host-resident, not on the GPU) and
    /// `experts` is empty. `None` = today's all-on-GPU path (byte-identical when the flag is off).
    cpu_experts: Option<Vec<CpuExpert>>,
    shared: Option<Expert>,
    top_k: usize,
    rscale: f32,
    hidden: usize,
    inter: usize,
    shared_inter: usize,
    n_experts: usize,
    eps: f32,
    norm: DevHalf,
    rlogits: DevF32,
    g: DevF32,
    u: DevF32,
    act: DevHalf,
    ey: DevF32,
    g_sh: DevF32,
    u_sh: DevF32,
    act_sh: DevHalf,
    // V3 (DeepSeek-V3/R1) sigmoid+bias grouped router ("noaux_tc"); None/false reproduces
    // the V2 plain-softmax top-k path exactly. See `moe_route_v3`.
    score_bias: Option<Vec<f32>>,
    n_group: usize,
    topk_group: usize,
    sigmoid: bool,
    // CPU-experts mode only: overlap the GPU shared expert with the CPU routed experts
    // (Metal). Default true; the micro-bench flips it to measure sequential vs overlapped.
    overlap_shared: bool,
    // Last routed_cpu component timings (ms), for the micro-bench. In sequential mode these
    // are the isolated CPU-routed and GPU-shared costs; in overlapped mode shared is the
    // residual wait after the CPU finished.
    last_cpu_ms: f64,
    last_shared_ms: f64,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl MoeBlock {
    /// `TRAPETUM_CPU_EXPERTS=1` -> routed experts run on the CPU (host-resident weights).
    fn cpu_experts_flag() -> bool {
        std::env::var("TRAPETUM_CPU_EXPERTS").map(|v| v == "1").unwrap_or(false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(hidden: usize, inter: usize, n_experts: usize, top_k: usize, eps: f32,
               norm_w: &[f32], router_w: &[f32],
               experts: Vec<(&[u8],&[f32],&[u8],&[f32],&[u8],&[f32])>,
               shared: Option<(&[u8],&[f32],&[u8],&[f32],&[u8],&[f32])>,
               shared_inter: usize, rscale: f32) -> Self {
        Self::new_mode(hidden, inter, n_experts, top_k, eps, norm_w, router_w, experts, shared,
                       shared_inter, rscale, Self::cpu_experts_flag())
    }

    /// Like [`Self::new`], but the caller decides whether routed experts live on the CPU
    /// (`cpu = true`, host-resident, not uploaded) or the GPU (`cpu = false`, today's path).
    /// The public `new` picks `cpu` from `TRAPETUM_CPU_EXPERTS`; validation uses both.
    #[allow(clippy::too_many_arguments)]
    pub fn new_mode(hidden: usize, inter: usize, n_experts: usize, top_k: usize, eps: f32,
                    norm_w: &[f32], router_w: &[f32],
                    experts: Vec<(&[u8],&[f32],&[u8],&[f32],&[u8],&[f32])>,
                    shared: Option<(&[u8],&[f32],&[u8],&[f32],&[u8],&[f32])>,
                    shared_inter: usize, rscale: f32, cpu: bool) -> Self {
        assert_eq!(experts.len(), n_experts);
        let mk = |e: &(&[u8],&[f32],&[u8],&[f32],&[u8],&[f32]), inter: usize| Expert {
            gate: QuantLinear::new(e.0, e.1, hidden, inter),
            up:   QuantLinear::new(e.2, e.3, hidden, inter),
            down: QuantLinear::new(e.4, e.5, inter, hidden),
        };
        // Routed experts: host-side copies (CPU mode) or GPU-uploaded QuantLinears (GPU mode).
        // The shared expert always stays on the GPU.
        let (gpu_experts, cpu_experts) = if cpu {
            // gate/up: oc=inter, ic=hidden; down: oc=hidden, ic=inter. Re-tile packed to
            // output-major and transpose each codebook once here (amortized over all tokens).
            let ce: Vec<CpuExpert> = experts.iter().map(|e| CpuExpert {
                gp_t: cpu_experts::pack_to_rowmajor(e.0, inter, hidden), gc_t: cpu_experts::transpose_codebook(e.1, inter),
                up_t: cpu_experts::pack_to_rowmajor(e.2, inter, hidden), uc_t: cpu_experts::transpose_codebook(e.3, inter),
                dp_t: cpu_experts::pack_to_rowmajor(e.4, hidden, inter), dc_t: cpu_experts::transpose_codebook(e.5, hidden),
            }).collect();
            (Vec::new(), Some(ce))
        } else {
            (experts.iter().map(|e| mk(e, inter)).collect::<Vec<_>>(), None)
        };
        Self {
            norm_w: DevF32::from_host(norm_w),
            router: DenseLinear::new(router_w, hidden, n_experts),
            experts: gpu_experts,
            cpu_experts,
            shared: shared.as_ref().map(|e| mk(e, shared_inter)),  // DeepSeek shared expert is bigger (n_shared*moe_inter)
            top_k, rscale, hidden, inter, shared_inter, n_experts, eps,
            norm: DevHalf::zeros(hidden),
            rlogits: DevF32::zeros(n_experts),
            g: DevF32::zeros(inter), u: DevF32::zeros(inter),
            act: DevHalf::zeros(inter), ey: DevF32::zeros(hidden),
            g_sh: DevF32::zeros(shared_inter.max(1)), u_sh: DevF32::zeros(shared_inter.max(1)),
            act_sh: DevHalf::zeros(shared_inter.max(1)),
            score_bias: None, n_group: 0, topk_group: 0, sigmoid: false,
            overlap_shared: true, last_cpu_ms: 0.0, last_shared_ms: 0.0,
        }
    }

    /// Enable/disable overlapping the GPU shared expert with the CPU routed experts
    /// (CPU-experts mode, Metal). Used by the micro-bench to compare sequential vs overlapped.
    pub fn set_overlap_shared(&mut self, on: bool) { self.overlap_shared = on; }

    /// Switch this block's router to DeepSeek-V3's sigmoid+bias grouped top-k ("noaux_tc"):
    /// `score_bias` is the learned `e_score_correction_bias` (length `n_experts`), experts
    /// are split into `n_group` equal groups and only the `topk_group` best-scoring groups
    /// are eligible. Leaves `top_k`/`rscale` (set at construction) unchanged.
    pub fn set_v3_scoring(&mut self, score_bias: Vec<f32>, n_group: usize, topk_group: usize) {
        assert_eq!(score_bias.len(), self.n_experts);
        assert_eq!(self.n_experts % n_group, 0, "n_experts must be divisible by n_group");
        self.score_bias = Some(score_bias);
        self.n_group = n_group;
        self.topk_group = topk_group;
        self.sigmoid = true;
    }

    /// Router top-k selection: `(expert, weight)` pairs, `weight` already `rscale`-scaled and
    /// normalized to sum to `rscale` over the selected experts. Shared between `forward` and
    /// `forward_dense_ref` so both paths score identically.
    fn route(&self, rl: &[f32]) -> Vec<(usize, f32)> {
        if self.sigmoid {
            let bias = self.score_bias.as_ref().expect("v3 scoring requires score_bias");
            moe_route_v3(rl, bias, self.n_experts, self.n_group, self.topk_group, self.top_k, self.rscale)
        } else {
            let mx = rl.iter().cloned().fold(f32::MIN, f32::max);
            let ex: Vec<f32> = rl.iter().map(|x| (x - mx).exp()).collect();
            let sum: f32 = ex.iter().sum();
            let probs: Vec<f32> = ex.iter().map(|x| x / sum).collect();
            let mut idx: Vec<usize> = (0..self.n_experts).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let topk = &idx[..self.top_k];
            let wsum: f32 = topk.iter().map(|&e| probs[e]).sum();
            topk.iter().map(|&e| (e, self.rscale * probs[e] / wsum)).collect()
        }
    }

    fn ffn(&mut self, e: usize, acc: &mut DevF32, w: f32) {
        // run routed expert e FFN on self.norm (inter-sized scratch), scaled-add into acc
        let ex = &self.experts[e];
        let (g_ptr, u_ptr, d_ptr) = (ex.gate.handle, ex.up.handle, ex.down.handle);
        unsafe {
            qlinear_forward_dev(g_ptr, self.norm.ptr, self.g.ptr);
            qlinear_forward_dev(u_ptr, self.norm.ptr, self.u.ptr);
        }
        silu_mul(&self.g, &self.u, &mut self.act);
        unsafe { qlinear_forward_dev(d_ptr, self.act.ptr, self.ey.ptr); }
        saxpy(acc, &self.ey, w);
    }

    /// Encode the shared expert's GPU work (gate/up/SiLU/down) so its output lands in
    /// `self.ey`. Does NOT drain or accumulate -- the caller either `saxpy`s it into an
    /// accumulator (`run_shared`) or commits it without waiting to overlap the CPU routed
    /// experts (`routed_cpu`). Uses the shared expert's larger `shared_inter` scratch.
    fn encode_shared_into_ey(&mut self) {
        let ex = self.shared.as_ref().unwrap();
        let (g_ptr, u_ptr, d_ptr) = (ex.gate.handle, ex.up.handle, ex.down.handle);
        unsafe {
            qlinear_forward_dev(g_ptr, self.norm.ptr, self.g_sh.ptr);
            qlinear_forward_dev(u_ptr, self.norm.ptr, self.u_sh.ptr);
        }
        silu_mul(&self.g_sh, &self.u_sh, &mut self.act_sh);
        unsafe { qlinear_forward_dev(d_ptr, self.act_sh.ptr, self.ey.ptr); }
    }

    fn run_shared(&mut self, acc: &mut DevF32) {
        self.encode_shared_into_ey();
        saxpy(acc, &self.ey, 1.0);
    }

    /// `h` (hidden,) updated in place: `h += MoE(RMSNorm(h))`.
    pub fn forward(&mut self, h: &mut DevHalf) {
        rmsnorm(h, &self.norm_w, &mut self.norm, self.eps);
        self.router.forward_into(&self.norm, &mut self.rlogits);
        let rl = self.rlogits.to_host();
        let picks = self.route(&rl);
        if self.cpu_experts.is_some() {
            // CPU-experts mode: routed_cpu owns both the routed (CPU) and the shared (GPU)
            // contribution, overlapping them; it returns routed+shared already combined.
            let acc = self.routed_cpu(&picks);
            residual_add(h, &acc);
        } else {
            let mut acc = DevF32::from_host(&vec![0f32; self.hidden]);
            for (e, w) in picks { self.ffn(e, &mut acc, w); }
            if self.shared.is_some() { self.run_shared(&mut acc); }
            residual_add(h, &acc);
        }
    }

    /// CPU-experts hybrid (`TRAPETUM_CPU_EXPERTS=1`): run the top-k routed experts on the CPU
    /// while the shared expert runs on the GPU, then combine `routed + shared`. The MoE
    /// output is a commutative sum, so the two are independent: on Metal we commit the
    /// shared-expert GPU work with `dev_flush` (no wait) and run the routed experts on CPU
    /// threads concurrently, joining with `dev_wait` before the combine. Returns the
    /// combined accumulator (routed + shared). `TRAPETUM_CPU_EXPERTS_TIMING=1` prints the
    /// three timings (routed CPU, shared GPU wait-residual, and overlapped/sequential mode).
    fn routed_cpu(&mut self, picks: &[(usize, f32)]) -> DevF32 {
        // The GPU experts consume fp16 activations; mirror that by reading the fp16 `norm`
        // back to host (already rounded) so the CPU result tracks the GPU path's input.
        // Read it BEFORE flushing the shared work, so this drain doesn't wait on the GPU.
        let x = self.norm.to_host();

        // Kick the shared expert onto the GPU concurrently (Metal only; CUDA stays
        // sequential -- see the else branch below).
        let overlapped;
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            overlapped = self.shared.is_some() && self.overlap_shared;
            if overlapped {
                self.encode_shared_into_ey();
                unsafe { dev_flush(); } // commit, no wait: GPU runs shared while CPU runs routed
            }
        }
        #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
        { overlapped = false; }

        // Routed experts on the CPU: row-chunk work-stealing across all picked experts
        // (`routed_experts_worksteal`), the C-probe design. One thread batch per token cooperatively
        // drains gate+up / SiLU / down phases; the row-major kernel streams packed contiguously.
        let t_cpu = std::time::Instant::now();
        let (hidden, inter) = (self.hidden, self.inter);
        let cpu = self.cpu_experts.as_ref().expect("routed_cpu without cpu_experts");
        let refs: Vec<cpu_experts::RoutedExpert> = picks.iter().map(|&(e, w)| {
            let ce = &cpu[e];
            cpu_experts::RoutedExpert {
                gp_t: &ce.gp_t, gc_t: &ce.gc_t, up_t: &ce.up_t, uc_t: &ce.uc_t, dp_t: &ce.dp_t, dc_t: &ce.dc_t, weight: w,
            }
        }).collect();
        let mut acc_host = vec![0f32; hidden];
        cpu_experts::routed_experts_worksteal(&x, &refs, hidden, inter, &mut acc_host);
        let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1e3;

        // Combine routed + shared.
        let mut shared_ms = 0f64;
        let acc = if overlapped {
            // Join the shared GPU work (residual time after the CPU finished), then add it.
            let tw = std::time::Instant::now();
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            unsafe { dev_wait(); }
            shared_ms = tw.elapsed().as_secs_f64() * 1e3;
            let mut acc = DevF32::from_host(&acc_host);
            if self.shared.is_some() { saxpy(&mut acc, &self.ey, 1.0); } // shared already in self.ey
            acc
        } else {
            // Sequential: routed uploaded, then shared runs on the GPU after it.
            let mut acc = DevF32::from_host(&acc_host);
            if self.shared.is_some() {
                let ts = std::time::Instant::now();
                self.run_shared(&mut acc);
                unsafe { dev_sync(); } // force completion so the measured shared time is real
                shared_ms = ts.elapsed().as_secs_f64() * 1e3;
            }
            acc
        };

        self.last_cpu_ms = cpu_ms;
        self.last_shared_ms = shared_ms;
        if std::env::var("TRAPETUM_CPU_EXPERTS_TIMING").map(|v| v == "1").unwrap_or(false) {
            eprintln!("[cpu_experts] routed_cpu={cpu_ms:.3} ms  shared_gpu={shared_ms:.3} ms  \
                       ({} experts, hidden={}, inter={}, {})",
                      picks.len(), self.hidden, self.inter,
                      if overlapped { "overlapped" } else { "sequential" });
        }
        acc
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl MoeBlock {
    /// Reference path: run ALL n_experts, weighting each by its top-k-masked router weight
    /// (0 if not selected). Must equal `forward` (which runs only the top-k). Catches
    /// top-k selection / weight / accumulation bugs.
    pub fn forward_dense_ref(&mut self, h: &mut DevHalf) {
        rmsnorm(h, &self.norm_w, &mut self.norm, self.eps);
        self.router.forward_into(&self.norm, &mut self.rlogits);
        let rl = self.rlogits.to_host();
        let picks = self.route(&rl);
        let weights: std::collections::HashMap<usize, f32> = picks.into_iter().collect();
        let mut acc = DevF32::from_host(&vec![0f32; self.hidden]);
        for e in 0..self.n_experts {
            let w = *weights.get(&e).unwrap_or(&0.0);
            if w > 0.0 { self.ffn(e, &mut acc, w); }
        }
        if self.shared.is_some() { self.run_shared(&mut acc); }
        residual_add(h, &acc);
    }
}

/// Validate the MoE block: the top-k `forward` must equal the dense reference, and saxpy
/// must accumulate correctly. Builds a small synthetic MoE (256 experts like DeepSeek-V3,
/// top-k=8). Returns worst rel err over the hidden output.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_moe() -> f64 {
    let (hidden, inter, n_experts, top_k) = (256usize, 256usize, 256usize, 8usize);
    let eps = 1e-5f32;
    let mut s = 0x1234_0E0Fu64;
    let mut r = move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0) };
    let packed = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
    let cbk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
    let nw: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
    let rw: Vec<f32> = (0..n_experts*hidden).map(|_| r()*0.05).collect();
    // experts
    let mut exp_store: Vec<(Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>)> = Vec::new();
    for _ in 0..n_experts {
        exp_store.push((
            packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
            packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
            packed(inter*(hidden/2),&mut r), cbk(K*hidden,&mut r)));
    }
    let experts: Vec<_> = exp_store.iter().map(|e| (e.0.as_slice(),e.1.as_slice(),e.2.as_slice(),e.3.as_slice(),e.4.as_slice(),e.5.as_slice())).collect();
    let sh = (packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
              packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
              packed(inter*(hidden/2),&mut r), cbk(K*hidden,&mut r));
    let shared = Some((sh.0.as_slice(),sh.1.as_slice(),sh.2.as_slice(),sh.3.as_slice(),sh.4.as_slice(),sh.5.as_slice()));
    let mut moe = MoeBlock::new(hidden, inter, n_experts, top_k, eps, &nw, &rw, experts, shared, inter, 1.0);
    let h0: Vec<f32> = (0..hidden).map(|_| r()*0.3).collect();
    let mut ha = DevHalf::from_host(&h0); moe.forward(&mut ha); let a = ha.to_host();
    let mut hb = DevHalf::from_host(&h0); moe.forward_dense_ref(&mut hb); let b = hb.to_host();
    let mut worst = 0f64;
    for i in 0..hidden { let den=(b[i] as f64).abs().max(1e-3); worst=worst.max(((a[i]-b[i]) as f64).abs()/den); }
    worst
}

/// Validate the CPU-experts hybrid path against the all-on-GPU path: build the SAME
/// synthetic MoE both ways (`new_mode(cpu=false)` vs `new_mode(cpu=true)`) and compare the
/// forward output over one token. The GPU experts consume fp16 activations while the CPU
/// path is f32, so a small diff is expected. Returns `(worst_rel, l2_rel, cpu_ms)`; also
/// prints them. No model artifact needed.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_moe_cpu_experts() -> (f64, f64, f64) {
    let (hidden, inter, n_experts, top_k) = (256usize, 256usize, 256usize, 8usize);
    let eps = 1e-5f32;
    let mut s = 0x1234_0E0Fu64;
    let mut r = move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0) };
    let packed = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
    let cbk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
    let nw: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
    let rw: Vec<f32> = (0..n_experts*hidden).map(|_| r()*0.05).collect();
    let mut exp_store: Vec<(Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>)> = Vec::new();
    for _ in 0..n_experts {
        exp_store.push((
            packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
            packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
            packed(inter*(hidden/2),&mut r), cbk(K*hidden,&mut r)));
    }
    let experts: Vec<_> = exp_store.iter().map(|e| (e.0.as_slice(),e.1.as_slice(),e.2.as_slice(),e.3.as_slice(),e.4.as_slice(),e.5.as_slice())).collect();
    let sh = (packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
              packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
              packed(inter*(hidden/2),&mut r), cbk(K*hidden,&mut r));
    let shared = Some((sh.0.as_slice(),sh.1.as_slice(),sh.2.as_slice(),sh.3.as_slice(),sh.4.as_slice(),sh.5.as_slice()));
    // Same weights, two backends for the routed experts.
    let mut gpu = MoeBlock::new_mode(hidden, inter, n_experts, top_k, eps, &nw, &rw, experts.clone(), shared.clone(), inter, 1.0, false);
    let mut cpu = MoeBlock::new_mode(hidden, inter, n_experts, top_k, eps, &nw, &rw, experts, shared, inter, 1.0, true);
    let h0: Vec<f32> = (0..hidden).map(|_| r()*0.3).collect();
    let mut hg = DevHalf::from_host(&h0); gpu.forward(&mut hg); let a = hg.to_host();
    // Time only the CPU forward (routed section dominates); a couple of warm iters.
    let mut hc = DevHalf::from_host(&h0); cpu.forward(&mut hc);
    let t0 = std::time::Instant::now();
    let mut hc2 = DevHalf::from_host(&h0); cpu.forward(&mut hc2);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let b = hc2.to_host();
    let mut worst = 0f64; let (mut num, mut den) = (0f64, 0f64);
    for i in 0..hidden {
        let d = (a[i]-b[i]) as f64;
        worst = worst.max(d.abs() / (a[i] as f64).abs().max(1e-3));
        num += d*d; den += (a[i] as f64)*(a[i] as f64);
    }
    let l2 = (num.sqrt()) / (den.sqrt().max(1e-9));
    eprintln!("[check_moe_cpu_experts] worst_rel={worst:.3e} l2_rel={l2:.3e} cpu_forward={cpu_ms:.3} ms (top_k={top_k}, hidden={hidden}, inter={inter})");
    (worst, l2, cpu_ms)
}

/// Build a CPU-experts MoE (with a shared expert) and run one token through it twice from
/// the SAME input: once with the shared expert overlapped onto the GPU (`overlap_shared=on`)
/// and once sequential. The two must agree -- overlap only reschedules the commutative sum,
/// it changes no arithmetic. Returns the worst rel err (expected ~0). Also the correctness
/// guard for the concurrency mechanism (dev_flush/dev_wait ordering).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_moe_cpu_overlap() -> f64 {
    let (hidden, inter, n_experts, top_k) = (256usize, 256usize, 64usize, 6usize);
    let eps = 1e-5f32;
    let mut s = 0x51DE_0FF5u64;
    let mut r = move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0) };
    let packed = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
    let cbk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
    let nw: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
    let rw: Vec<f32> = (0..n_experts*hidden).map(|_| r()*0.05).collect();
    let mut estore: Vec<(Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>)> = Vec::new();
    for _ in 0..n_experts {
        estore.push((packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
                     packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
                     packed(inter*(hidden/2),&mut r), cbk(K*hidden,&mut r)));
    }
    let experts: Vec<_> = estore.iter().map(|e| (e.0.as_slice(),e.1.as_slice(),e.2.as_slice(),e.3.as_slice(),e.4.as_slice(),e.5.as_slice())).collect();
    let sh = (packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
              packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
              packed(inter*(hidden/2),&mut r), cbk(K*hidden,&mut r));
    let shared = Some((sh.0.as_slice(),sh.1.as_slice(),sh.2.as_slice(),sh.3.as_slice(),sh.4.as_slice(),sh.5.as_slice()));
    let mut moe = MoeBlock::new_mode(hidden, inter, n_experts, top_k, eps, &nw, &rw, experts, shared, inter, 1.0, true);
    let h0: Vec<f32> = (0..hidden).map(|_| r()*0.3).collect();
    moe.set_overlap_shared(false);
    let mut hs = DevHalf::from_host(&h0); moe.forward(&mut hs); let seq = hs.to_host();
    moe.set_overlap_shared(true);
    let mut ho = DevHalf::from_host(&h0); moe.forward(&mut ho); let ov = ho.to_host();
    let mut worst = 0f64;
    for i in 0..hidden { worst = worst.max(((seq[i]-ov[i]) as f64).abs() / (seq[i] as f64).abs().max(1e-3)); }
    eprintln!("[check_moe_cpu_overlap] overlapped vs sequential worst_rel={worst:.3e}");
    worst
}

/// Micro-bench the CPU-experts hybrid at a meaningful size (default: V2-Lite dims). Builds a
/// CPU-experts MoE with a GPU shared expert and reports four numbers: the isolated CPU-routed
/// cost, the isolated GPU-shared cost, the sequential total, and the overlapped total. Overlap
/// works iff overlapped_total ~ max(cpu, shared) rather than their sum. Returns
/// `(cpu_ms, shared_ms, sequential_ms, overlapped_ms)` (best of `passes`).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn bench_moe_cpu_experts(hidden: usize, inter: usize, n_experts: usize, top_k: usize,
                             shared_inter: usize, passes: usize) -> (f64, f64, f64, f64) {
    assert_eq!(hidden % 256, 0, "hidden must be %256 (shared expert is on the GPU)");
    assert_eq!(shared_inter % 256, 0, "shared_inter must be %256 (GPU QuantLinear tiling)");
    let eps = 1e-5f32;
    let mut s = 0x0B16_5EEDu64;
    let mut r = move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0) };
    let packed = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
    let cbk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
    let nw: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
    let rw: Vec<f32> = (0..n_experts*hidden).map(|_| r()*0.05).collect();
    let mut estore: Vec<(Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>)> = Vec::new();
    for _ in 0..n_experts {
        estore.push((packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
                     packed(hidden*(inter/2),&mut r), cbk(K*inter,&mut r),
                     packed(inter*(hidden/2),&mut r), cbk(K*hidden,&mut r)));
    }
    let experts: Vec<_> = estore.iter().map(|e| (e.0.as_slice(),e.1.as_slice(),e.2.as_slice(),e.3.as_slice(),e.4.as_slice(),e.5.as_slice())).collect();
    let sh = (packed(hidden*(shared_inter/2),&mut r), cbk(K*shared_inter,&mut r),
              packed(hidden*(shared_inter/2),&mut r), cbk(K*shared_inter,&mut r),
              packed(shared_inter*(hidden/2),&mut r), cbk(K*hidden,&mut r));
    let shared = Some((sh.0.as_slice(),sh.1.as_slice(),sh.2.as_slice(),sh.3.as_slice(),sh.4.as_slice(),sh.5.as_slice()));
    let mut moe = MoeBlock::new_mode(hidden, inter, n_experts, top_k, eps, &nw, &rw, experts, shared, shared_inter, 1.0, true);
    let h0: Vec<f32> = (0..hidden).map(|_| r()*0.3).collect();

    // Best-of-`passes` wall time; also returns the CPU/shared component split of the BEST
    // pass (last_cpu_ms/last_shared_ms), so totals and components come from the same pass.
    let time_forward = |moe: &mut MoeBlock, passes: usize| -> (f64, f64, f64) {
        let (mut best, mut bc, mut bs) = (f64::MAX, 0f64, 0f64);
        for _ in 0..passes {
            let mut hh = DevHalf::from_host(&h0);
            let t = std::time::Instant::now();
            moe.forward(&mut hh);
            unsafe { dev_sync(); } // include the shared-expert GPU completion in the wall time
            let el = t.elapsed().as_secs_f64() * 1e3;
            if el < best { best = el; bc = moe.last_cpu_ms; bs = moe.last_shared_ms; }
        }
        (best, bc, bs)
    };
    // warm both modes (first Metal command buffer / pipeline JIT is not representative).
    moe.set_overlap_shared(false); let _ = time_forward(&mut moe, 2);
    moe.set_overlap_shared(true);  let _ = time_forward(&mut moe, 2);

    moe.set_overlap_shared(false);
    let (sequential_ms, cpu_ms, shared_ms) = time_forward(&mut moe, passes);
    moe.set_overlap_shared(true);
    let (overlapped_ms, _, _) = time_forward(&mut moe, passes);

    // Routed packed bytes streamed per token = top_k * 3 projections (gate+up+down), each
    // ic*(oc/2) bytes: 2 * hidden*(inter/2) + inter*(hidden/2) = 3 * hidden*inter/2 per expert.
    let routed_bytes = top_k as f64 * 3.0 * (hidden as f64) * (inter as f64) / 2.0;
    let gbs = routed_bytes / (cpu_ms * 1e-3) / 1e9;
    eprintln!("[bench_moe_cpu_experts] hidden={hidden} inter={inter} experts={n_experts} top_k={top_k} shared_inter={shared_inter} passes={passes}");
    eprintln!("  cpu_routed={cpu_ms:.3} ms ({:.0} packed bytes -> {gbs:.2} GB/s)   gpu_shared={shared_ms:.3} ms   ->  sum={:.3} ms   max={:.3} ms",
              routed_bytes, cpu_ms + shared_ms, cpu_ms.max(shared_ms));
    eprintln!("  sequential_total={sequential_ms:.3} ms   overlapped_total={overlapped_ms:.3} ms");
    (cpu_ms, shared_ms, sequential_ms, overlapped_ms)
}

#[cfg(all(test, any(feature = "cuda", feature = "metal")))]
mod moe_cpu_tests {
    #[test]
    fn cpu_experts_match_gpu() {
        let (worst, l2, _ms) = super::check_moe_cpu_experts();
        // GPU experts consume fp16 activations, CPU path is f32: the worst-case elementwise
        // rel err is dominated by a single near-zero output element (small denominator),
        // while the L2 rel err reflects the true agreement. Pass on either (per spec).
        assert!(worst < 1e-2 || l2 < 1e-3, "CPU vs GPU MoE mismatch: worst_rel={worst:e} l2_rel={l2:e}");
    }

    #[test]
    fn overlap_matches_sequential() {
        // Overlapping the shared expert must not change the result (only its scheduling).
        let worst = super::check_moe_cpu_overlap();
        assert!(worst < 1e-4, "overlapped != sequential MoE output: worst_rel={worst:e}");
    }

    #[test]
    fn overlap_bench_v2lite() {
        // V2-Lite dims: hidden=2048, moe_inter=1408, 64 routed experts, top_k=6, 2 shared
        // experts (shared_inter=2*1408=2816, a multiple of 256). Loaded M4, best of 5.
        // NO wall-clock ratio assertion here: timing is machine-load dependent and a unit
        // test must not gate on it (an earlier `ov <= seq*1.15` assert flaked under load).
        // The prints (cpu_routed/gpu_shared/sequential/overlapped + GB/s) are the deliverable;
        // overlap CORRECTNESS is proven bit-identical by `overlap_matches_sequential`.
        let (cpu, shared, _seq, _ov) = super::bench_moe_cpu_experts(2048, 1408, 64, 6, 2816, 5);
        assert!(cpu > 0.0 && shared > 0.0, "component timings not captured: cpu={cpu} shared={shared}");
    }
}

/// A packed (4-bit indices) tensor, either owned in RAM or a zero-copy byte range into a
/// shared file `Mmap` -- the OS pages mmap-backed bytes in from disk on first touch (and can
/// evict them under memory pressure), which is exactly the "stream experts from storage"
/// behavior `MoeBlockOffload` wants at 671B scale, without ever copying the whole tensor set
/// into the process's RSS. `Owned` is used by `gen_moe`'s synthetic check models (and would
/// be used by any small in-RAM export); `Mmap` is what `load_deepseek_qlora` uses for the
/// routed experts' packed indices (the bulk of a 671B CBKR file).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub enum PackedBytes {
    Owned(Vec<u8>),
    Mmap(Arc<memmap2::Mmap>, usize, usize), // (mmap, byte offset, byte length)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl PackedBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            PackedBytes::Owned(v) => v.as_slice(),
            PackedBytes::Mmap(m, off, len) => &m[*off..*off + *len],
        }
    }

    /// Hint the kernel to read this mmap range in the background (madvise WILLNEED).
    /// Turns the serial page faults of expert streaming into parallel queued disk
    /// reads: issue it for every routed expert as soon as the router has picked
    /// them, before any expert is touched. No-op for owned buffers.
    fn prefetch(&self) {
        #[cfg(unix)]
        if let PackedBytes::Mmap(m, off, len) = self {
            let _ = m.advise_range(memmap2::Advice::WillNeed, *off, *len);
        }
    }
}

/// Host-resident (or mmap-backed, see `PackedBytes`) expert weights, not on the GPU until
/// streamed in. The packed 4-bit indices (`gp`/`up`/`dp`, the bulk of an expert's bytes) may
/// be `PackedBytes::Mmap`; the tiny per-output-channel codebooks (`gc`/`uc`/`dc`, a few KB to
/// a few hundred KB each) always stay owned `Vec<f32>` -- a mmap byte offset is not
/// guaranteed 4-byte aligned, so casting a raw mmap `&[u8]` to `&[f32]` would be unsound, and
/// copying something this small at load time costs nothing.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub enum ExpertHost {
    /// 4-bit scalar-codebook expert (today's CBKR path): packed nibble indices + per-output
    /// codebook for gate/up/down.
    Scalar { gp: PackedBytes, gc: Vec<f32>, up: PackedBytes, uc: Vec<f32>, dp: PackedBytes, dc: Vec<f32> },
    /// Additive-codebook expert (CBKA, 2/3-bit): each projection is an [`AvqProjHost`].
    Avq { gate: AvqProjHost, up: AvqProjHost, down: AvqProjHost },
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl ExpertHost {
    /// Background-read (madvise WILLNEED) the mmap-backed index bytes of all three projections
    /// so expert streaming issues parallel disk reads instead of serial page faults.
    fn prefetch(&self) {
        match self {
            ExpertHost::Scalar { gp, up, dp, .. } => { gp.prefetch(); up.prefetch(); dp.prefetch(); }
            ExpertHost::Avq { gate, up, down } => { gate.idx.prefetch(); up.idx.prefetch(); down.idx.prefetch(); }
        }
    }

    /// Borrow the six scalar-expert slices (packed indices + codebook, for gate/up/down).
    /// Panics on an AVQ host -- used only by the all-resident-vs-offload check, which is
    /// built from `gen_moe` (always `Scalar`).
    #[allow(clippy::type_complexity)]
    fn scalar_slices(&self) -> (&[u8], &[f32], &[u8], &[f32], &[u8], &[f32]) {
        match self {
            ExpertHost::Scalar { gp, gc, up, uc, dp, dc } =>
                (gp.as_slice(), gc.as_slice(), up.as_slice(), uc.as_slice(), dp.as_slice(), dc.as_slice()),
            ExpertHost::Avq { .. } => panic!("scalar_slices called on an AVQ ExpertHost"),
        }
    }
}

/// One additive-codebook (CBKA) projection of a MoE expert, host-resident until streamed to
/// the GPU. The packed `M*(cols/AVQ_D)*rows` u8 indices (`idx`, the bulk) may be
/// `PackedBytes::Mmap` (paged from disk on demand); the tiny codebooks (`cb`, `M*AVQ_K*AVQ_D`
/// f32) and per-output-channel scales (`scale`, `rows` f32) always stay owned -- a mmap byte
/// offset is not guaranteed f32/f16 aligned, and both are only a few KB.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct AvqProjHost { idx: PackedBytes, cb: Vec<f32>, scale: Vec<f32>, m: usize, rows: usize, cols: usize }

#[cfg(any(feature = "cuda", feature = "metal"))]
impl AvqProjHost {
    fn to_linear(&self) -> AvqLinear {
        AvqLinear::new(self.idx.as_slice(), &self.cb, &self.scale, self.m, self.rows, self.cols)
    }
}

// ============================================================================
// CBKA -- additive-codebook expert record (2/3-bit AQLM-style), the on-disk format
// emitted by model/export_deepseek_stream.py --experts-avq {2,3} and decoded by
// `kernels/avq_gemv3.cu` / `AvqLinear`. Only MoE ROUTED experts use it; attention,
// dense FFN, shared expert, router and lm_head stay on the 4-bit scalar CBKR path.
// One record per weight matrix (a routed expert = 3 records: gate, up, down):
//   magic  "CBKA"                 (4 bytes)
//   M      i32                    (2 or 3 additive codebooks)
//   D      i32                    (AVQ_D = 8, group size)
//   K      i32                    (AVQ_K = 256, entries per codebook)
//   rows   i32                    (OC, output channels; rows % 4 == 0)
//   cols   i32                    (IC, input channels; cols % D == 0)
//   codebooks   M*K*D    f16      layout [M][K][D], flat (m*K + k)*D + e
//   scales      rows     f16      per output channel
//   indices     M*ng*rows u8      ng = cols/D, layout [M][ng][rows], flat (m*ng + g)*rows + o
// The indices (the bulk) are recorded as an mmap byte range (never copied to RAM); the
// codebooks/scales are read owned. Reconstruction: W[o, g*D+e] = scale[o]*sum_m CB[m][code][e].
// ============================================================================
#[cfg(any(feature = "cuda", feature = "metal"))]
fn read_cbka(r: &mut BufReader<File>, mmap: &Arc<memmap2::Mmap>,
             exp_m: usize, exp_rows: usize, exp_cols: usize) -> std::io::Result<AvqProjHost> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    assert_eq!(&magic, b"CBKA", "expected a CBKA additive-codebook expert record");
    let m = rd_i32(r) as usize;
    let d = rd_i32(r) as usize;
    let k = rd_i32(r) as usize;
    let rows = rd_i32(r) as usize;
    let cols = rd_i32(r) as usize;
    assert_eq!(m, exp_m, "CBKA M mismatch (header experts_avq={exp_m})");
    assert_eq!(d, AVQ_D, "CBKA D must be {AVQ_D}");
    assert_eq!(k, AVQ_K, "CBKA K must be {AVQ_K}");
    assert_eq!(rows, exp_rows, "CBKA rows mismatch");
    assert_eq!(cols, exp_cols, "CBKA cols mismatch");
    let cb = rd_f16_vec(r, m * k * d);   // codebooks f16 -> f32
    let scale = rd_f16_vec(r, rows);     // per-output scales f16 -> f32
    let ng = cols / d;
    let idx = mmap_skip(r, mmap, m * ng * rows)?; // u8 indices, mmap-backed (the bulk)
    Ok(AvqProjHost { idx, cb, scale, m, rows, cols })
}

/// MoE block with EXPERT OFFLOADING (solves the memory wall). Only the router and the
/// shared expert stay GPU-resident; the routed experts live in host memory and the
/// top-k active ones are streamed to the GPU per token, cached with an LRU of capacity
/// `cap`. Exploiting MoE sparsity (k of n_experts active) + MLA's tiny KV cache, the GPU
/// working set is router + shared + `cap` experts instead of all n_experts: a 671B model's
/// 350 GB of 4-bit weights fit in host RAM / on NVMe while the GPU holds under 20 GB. The
/// cost is streaming ~k experts/token/layer over PCIe (bandwidth-bound, so a few tok/s on
/// one GPU); correctness is identical to the all-resident block.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct MoeBlockOffload {
    norm_w: DevF32,
    router: DenseLinear,
    hosts: Vec<ExpertHost>,
    shared: OffExpert,
    cache: std::collections::HashMap<usize, OffExpert>,
    lru: Vec<usize>,
    cap: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
    n_experts: usize,
    eps: f32,
    norm: DevHalf,
    rlogits: DevF32,
    g: DevF32,
    u: DevF32,
    act: DevHalf,
    ey: DevF32,
    pub uploads: usize, // experts streamed to GPU (perf accounting)
    rscale: f32,
    // V3 (DeepSeek-V3/R1) sigmoid+bias grouped router; see `MoeBlock`/`moe_route_v3`.
    score_bias: Option<Vec<f32>>,
    n_group: usize,
    topk_group: usize,
    sigmoid: bool,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl MoeBlockOffload {
    #[allow(clippy::too_many_arguments)]
    pub fn new(hidden: usize, inter: usize, n_experts: usize, top_k: usize, cap: usize, eps: f32,
               norm_w: &[f32], router_w: &[f32], hosts: Vec<ExpertHost>,
               shared: (&[u8],&[f32],&[u8],&[f32],&[u8],&[f32])) -> Self {
        assert_eq!(hosts.len(), n_experts);
        assert!(cap >= top_k, "cache must hold at least top_k experts");
        Self {
            norm_w: DevF32::from_host(norm_w),
            router: DenseLinear::new(router_w, hidden, n_experts),
            hosts,
            shared: OffExpert { gate: Proj::Scalar(QuantLinear::new(shared.0,shared.1,hidden,inter)),
                             up: Proj::Scalar(QuantLinear::new(shared.2,shared.3,hidden,inter)),
                             down: Proj::Scalar(QuantLinear::new(shared.4,shared.5,inter,hidden)) },
            cache: std::collections::HashMap::new(), lru: Vec::new(), cap,
            top_k, hidden, inter, n_experts, eps,
            norm: DevHalf::zeros(hidden), rlogits: DevF32::zeros(n_experts),
            g: DevF32::zeros(inter), u: DevF32::zeros(inter),
            act: DevHalf::zeros(inter), ey: DevF32::zeros(hidden),
            uploads: 0,
            rscale: 1.0,
            score_bias: None, n_group: 0, topk_group: 0, sigmoid: false,
        }
    }

    /// Set the `routed_scaling_factor` combine-weight multiplier (default 1.0).
    pub fn set_rscale(&mut self, rscale: f32) { self.rscale = rscale; }

    /// Switch to DeepSeek-V3's sigmoid+bias grouped top-k router. See `MoeBlock::set_v3_scoring`.
    pub fn set_v3_scoring(&mut self, score_bias: Vec<f32>, n_group: usize, topk_group: usize) {
        assert_eq!(score_bias.len(), self.n_experts);
        assert_eq!(self.n_experts % n_group, 0, "n_experts must be divisible by n_group");
        self.score_bias = Some(score_bias);
        self.n_group = n_group;
        self.topk_group = topk_group;
        self.sigmoid = true;
    }

    fn route(&self, rl: &[f32]) -> Vec<(usize, f32)> {
        if self.sigmoid {
            let bias = self.score_bias.as_ref().expect("v3 scoring requires score_bias");
            moe_route_v3(rl, bias, self.n_experts, self.n_group, self.topk_group, self.top_k, self.rscale)
        } else {
            let mx = rl.iter().cloned().fold(f32::MIN, f32::max);
            let ex: Vec<f32> = rl.iter().map(|x| (x - mx).exp()).collect();
            let sum: f32 = ex.iter().sum();
            let probs: Vec<f32> = ex.iter().map(|x| x / sum).collect();
            let mut idx: Vec<usize> = (0..self.n_experts).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let topk = &idx[..self.top_k];
            let wsum: f32 = topk.iter().map(|&e| probs[e]).sum();
            topk.iter().map(|&e| (e, self.rscale * probs[e] / wsum)).collect()
        }
    }

    /// Ensure expert `e` is resident (stream from host + LRU-evict if needed); return the three
    /// projection handles (Copy [`ProjRef`], so `run_ffn(&mut self, ..)` can also touch scratch).
    fn resident(&mut self, e: usize) -> (ProjRef, ProjRef, ProjRef) {
        if self.cache.contains_key(&e) {
            let pos = self.lru.iter().position(|&x| x == e).unwrap();
            self.lru.remove(pos);
            self.lru.push(e);
        } else {
            if self.cache.len() >= self.cap {
                let victim = self.lru.remove(0);
                self.cache.remove(&victim); // Drop frees the GPU buffers
            }
            let ex = match &self.hosts[e] {
                ExpertHost::Scalar { gp, gc, up, uc, dp, dc } => OffExpert {
                    gate: Proj::Scalar(QuantLinear::new(gp.as_slice(), gc, self.hidden, self.inter)),
                    up:   Proj::Scalar(QuantLinear::new(up.as_slice(), uc, self.hidden, self.inter)),
                    down: Proj::Scalar(QuantLinear::new(dp.as_slice(), dc, self.inter, self.hidden)),
                },
                ExpertHost::Avq { gate, up, down } => OffExpert {
                    gate: Proj::Avq(gate.to_linear()),
                    up:   Proj::Avq(up.to_linear()),
                    down: Proj::Avq(down.to_linear()),
                },
            };
            self.cache.insert(e, ex);
            self.lru.push(e);
            self.uploads += 1;
        }
        let ex = self.cache.get(&e).unwrap();
        (ex.gate.as_ref(), ex.up.as_ref(), ex.down.as_ref())
    }

    fn run_ffn(&mut self, g: ProjRef, u: ProjRef, d: ProjRef, acc: &mut DevF32, w: f32) {
        unsafe {
            g.forward(self.norm.ptr, self.g.ptr);
            u.forward(self.norm.ptr, self.u.ptr);
        }
        silu_mul(&self.g, &self.u, &mut self.act);
        unsafe { d.forward(self.act.ptr, self.ey.ptr); }
        saxpy(acc, &self.ey, w);
    }

    pub fn forward(&mut self, h: &mut DevHalf) {
        rmsnorm(h, &self.norm_w, &mut self.norm, self.eps);
        self.router.forward_into(&self.norm, &mut self.rlogits);
        let rl = self.rlogits.to_host();
        let picks = self.route(&rl);
        // TRAPETUM_LOG_EXPERTS=<path>: append one line of routed expert ids per MoE
        // call (call order = token-major, layer-minor), to measure the adjacent-token
        // expert overlap that bounds speculative-decode byte amortization.
        if let Ok(path) = std::env::var("TRAPETUM_LOG_EXPERTS") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let ids: Vec<String> = picks.iter().map(|(e, _)| e.to_string()).collect();
                let _ = writeln!(f, "{}", ids.join(","));
            }
        }
        // TRAPETUM_PREFETCH=1: kick off background reads for ALL picked experts not
        // yet GPU-resident before computing the first one, so the disk works on
        // experts 2..k while expert 1 streams and computes.
        static PREFETCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let pf = *PREFETCH.get_or_init(|| {
            std::env::var("TRAPETUM_PREFETCH").map(|v| v != "0").unwrap_or(false)
        });
        if pf {
            for (e, _) in &picks {
                if !self.cache.contains_key(e) {
                    self.hosts[*e].prefetch();
                }
            }
        }
        let mut acc = DevF32::from_host(&vec![0f32; self.hidden]);
        for (e, w) in picks {
            let (g, u, d) = self.resident(e);
            self.run_ffn(g, u, d, &mut acc, w);
        }
        let (sg, su, sd) = (self.shared.gate.as_ref(), self.shared.up.as_ref(), self.shared.down.as_ref());
        self.run_ffn(sg, su, sd, &mut acc, 1.0);
        residual_add(h, &acc);
    }
}

// Deterministic MoE weight generator (same seed -> identical model), for the offload check.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn gen_moe(hidden: usize, inter: usize, n_experts: usize, seed: u64)
    -> (Vec<f32>, Vec<f32>, Vec<ExpertHost>, ExpertHost) {
    let mut s = seed;
    let mut r = move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0) };
    let pk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
    let cb = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
    let nw: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
    let rw: Vec<f32> = (0..n_experts*hidden).map(|_| r()*0.05).collect();
    let mkh = |r: &mut dyn FnMut()->f32| ExpertHost::Scalar {
        gp: PackedBytes::Owned(pk(hidden*(inter/2), r)), gc: cb(K*inter, r),
        up: PackedBytes::Owned(pk(hidden*(inter/2), r)), uc: cb(K*inter, r),
        dp: PackedBytes::Owned(pk(inter*(hidden/2), r)), dc: cb(K*hidden, r) };
    let hosts: Vec<ExpertHost> = (0..n_experts).map(|_| mkh(&mut r)).collect();
    let shared = mkh(&mut r);
    (nw, rw, hosts, shared)
}

/// Validate expert OFFLOADING: the offloaded block (only `cap` experts resident, streamed
/// from host with an LRU) must produce IDENTICAL output to the all-resident block, over
/// several tokens. Returns (worst_rel_err, cap, n_experts, uploads_over_tokens).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_moe_offload() -> (f64, usize, usize, usize) {
    let (hidden, inter, n_experts, top_k, cap) = (256usize, 256usize, 256usize, 8usize, 16usize);
    let eps = 1e-5f32; let seed = 0x0FF10AD5u64;
    // all-resident reference
    let (nw, rw, hosts_r, sh_r) = gen_moe(hidden, inter, n_experts, seed);
    let exps_ref: Vec<_> = hosts_r.iter().map(|e| e.scalar_slices()).collect();
    let shref = sh_r.scalar_slices();
    let mut moe = MoeBlock::new(hidden, inter, n_experts, top_k, eps, &nw, &rw, exps_ref, Some(shref), inter, 1.0);
    // offloaded (identical weights via same seed)
    let (nw2, rw2, hosts_o, sh_o) = gen_moe(hidden, inter, n_experts, seed);
    let sho = sh_o.scalar_slices();
    let mut off = MoeBlockOffload::new(hidden, inter, n_experts, top_k, cap, eps, &nw2, &rw2, hosts_o, sho);
    // run several distinct tokens through both, compare
    let mut worst = 0f64;
    let mut sh = 0xABCDu64;
    let mut rr = move || { sh ^= sh<<13; sh ^= sh>>7; sh ^= sh<<17; (((sh>>40) as f32/(1u64<<24) as f32)*2.0-1.0) };
    for _ in 0..12 {
        let h0: Vec<f32> = (0..hidden).map(|_| rr()*0.3).collect();
        let mut ha = DevHalf::from_host(&h0); moe.forward(&mut ha); let a = ha.to_host();
        let mut hb = DevHalf::from_host(&h0); off.forward(&mut hb); let b = hb.to_host();
        for i in 0..hidden { let den=(a[i] as f64).abs().max(1e-3); worst=worst.max(((a[i]-b[i]) as f64).abs()/den); }
    }
    (worst, cap, n_experts, off.uploads)
}

/// Dense fp16 GEMV: `y = W x` (W is `[oc][ic]` fp16). `saxpy`-free helper for the MLA
/// projection matrices, which are small and kept dense rather than codebook-quantized.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn gemv_fp16(w: &DevHalf, x: &DevHalf, y: &mut DevF32, ic: usize, oc: usize) {
    assert_eq!(w.n, ic*oc); assert_eq!(x.n, ic); assert_eq!(y.n, oc);
    unsafe { op_gemv_fp16(w.ptr, x.ptr, y.ptr, ic as i32, oc as i32) };
}

/// A dense fp16 linear `y = W x`, weights resident on the GPU (`[oc][ic]` row-major).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct DenseLinear { w: DevHalf, ic: usize, oc: usize }
#[cfg(any(feature = "cuda", feature = "metal"))]
impl DenseLinear {
    pub fn new(w_rowmajor: &[f32], ic: usize, oc: usize) -> Self {
        assert_eq!(w_rowmajor.len(), ic*oc);
        Self { w: DevHalf::from_host(w_rowmajor), ic, oc }
    }
    pub fn forward_into(&self, x: &DevHalf, y: &mut DevF32) { gemv_fp16(&self.w, x, y, self.ic, self.oc); }
    pub fn oc(&self) -> usize { self.oc }
}

/// MLA (DeepSeek) decode attention on-device: `out_latent[h] = softmax(absorbed_q[h]·c_KV +
/// q_R[h]·k_R) · c_KV`. Inputs/outputs are device fp16; a W_UV GEMM turns out_latent into
/// per-head values. See the `mla_attn` kernel.
#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn mla_attention(aq: &DevHalf, qr: &DevHalf, ckv: &DevHalf, kr: &DevHalf, out: &mut DevHalf,
                     n_heads: usize, d_c: usize, d_rope: usize, seqlen: usize, scale: f32) {
    unsafe { op_mla_attn(aq.ptr, qr.ptr, ckv.ptr, kr.ptr, out.ptr, n_heads as i32, d_c as i32, d_rope as i32, seqlen as i32, scale) };
}

/// A DeepSeek MLA (Multi-head Latent Attention) decode block. q_lora_rank=0 variant
/// (DeepSeek-V2-Lite): q_proj is a single dense. The KV cache stores only the shared
/// low-rank latent c_KV (d_c) + decoupled rope key k_R (d_rope) per token. Uses the
/// validated `mla_attn` kernel with absorbed queries; the small projection matrices are
/// dense fp16. Correctness-first: the per-head absorption (W_UK/W_UV) + rope + splits run
/// on the host, the heavy projections + attention on the GPU. Solves the attention wall.
///
/// A second variant (`new_qlora`, DeepSeek-V3/R1) replaces the single dense `q_proj` with
/// q_lora: `q_a_proj` (dense) -> RMSNorm -> `q_b_proj` (4-bit codebook, since q_lora_rank ->
/// qdim is a much bigger matrix at 671B scale); `o_proj` can likewise be 4-bit (`o_quant`).
/// Both variants share the same cache/attention/absorption code below `q_lora`/`o_quant`
/// being `None` reproduces the V2-Lite dense path exactly.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct MlaAttn {
    q_proj: Option<DenseLinear>,
    kv_a: DenseLinear,
    o_proj: Option<DenseLinear>,
    kv_a_norm: Vec<f32>,          // [d_c]
    kv_b: Vec<f32>,               // [n_heads*(nope+v_head_dim)][d_c]  (W_UK ++ W_UV per head)
    inv_freq: Vec<f32>,           // [d_rope/2]
    cache_ckv: DevHalf,           // [max_seq][d_c]
    cache_kr: DevHalf,            // [max_seq][d_rope]
    n_heads: usize, d_c: usize, d_rope: usize, nope: usize, v_head_dim: usize, hidden: usize,
    pub eps: f32, softmax_scale: f32,
    aq_dev: DevHalf, qr_dev: DevHalf, outl_dev: DevHalf,
    ckv_h: DevHalf, kr_h: DevHalf,
    qf: DevF32, kvf: DevF32, attn_dev: DevHalf, o_out: DevF32,
    pub last_attn: Vec<f32>,      // pre-o_proj per-head values (for validation)
    q_lora: Option<QLoraQ>,
    o_quant: Option<QuantLinear>,
}

/// q_lora scratch + weights for `MlaAttn::new_qlora` (DeepSeek-V3/R1 q path):
/// `h -> q_a (dense) -> RMSNorm(q_a_norm) -> q_b (4-bit codebook) -> qdim`.
#[cfg(any(feature = "cuda", feature = "metal"))]
struct QLoraQ {
    q_a: DenseLinear,       // hidden -> q_lora_rank
    q_a_norm: Vec<f32>,     // [q_lora_rank], host RMSNorm weight
    q_b: QuantLinear,       // q_lora_rank -> qdim
    qa_dev: DevF32,         // [q_lora_rank] scratch (q_a output)
    qa_normed: DevHalf,     // [q_lora_rank] scratch (normed, re-uploaded for q_b)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl MlaAttn {
    #[allow(clippy::too_many_arguments)]
    pub fn new(hidden: usize, n_heads: usize, d_c: usize, d_rope: usize, nope: usize, v_head_dim: usize,
               max_seq: usize, eps: f32, softmax_scale: f32, q_w: &[f32], kv_a_w: &[f32], kv_a_norm: &[f32], kv_b: &[f32],
               o_w: &[f32], inv_freq: &[f32]) -> Self {
        let qdim = n_heads*(nope+d_rope);
        Self {
            q_proj: Some(DenseLinear::new(q_w, hidden, qdim)),
            kv_a: DenseLinear::new(kv_a_w, hidden, d_c + d_rope),
            o_proj: Some(DenseLinear::new(o_w, n_heads*v_head_dim, hidden)),
            kv_a_norm: kv_a_norm.to_vec(), kv_b: kv_b.to_vec(), inv_freq: inv_freq.to_vec(),
            cache_ckv: DevHalf::zeros(max_seq*d_c), cache_kr: DevHalf::zeros(max_seq*d_rope),
            n_heads, d_c, d_rope, nope, v_head_dim, hidden, eps, softmax_scale,
            aq_dev: DevHalf::zeros(n_heads*d_c), qr_dev: DevHalf::zeros(n_heads*d_rope),
            outl_dev: DevHalf::zeros(n_heads*d_c),
            ckv_h: DevHalf::zeros(d_c), kr_h: DevHalf::zeros(d_rope),
            qf: DevF32::zeros(qdim), kvf: DevF32::zeros(d_c + d_rope),
            attn_dev: DevHalf::zeros(n_heads*v_head_dim), o_out: DevF32::zeros(hidden),
            last_attn: vec![0f32; n_heads*v_head_dim],
            q_lora: None, o_quant: None,
        }
    }

    /// DeepSeek-V3/R1 q_lora + quantized-o_proj variant. `q_b` and `o` are `(packed, codebook)`
    /// 4-bit codebook pairs (see `QuantLinear::new`); everything else matches `new`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_qlora(hidden: usize, n_heads: usize, d_c: usize, d_rope: usize, nope: usize, v_head_dim: usize,
               max_seq: usize, eps: f32, softmax_scale: f32, q_lora_rank: usize,
               q_a_w: &[f32], q_a_norm: &[f32], q_b: (&[u8], &[f32]),
               kv_a_w: &[f32], kv_a_norm: &[f32], kv_b: &[f32],
               o: (&[u8], &[f32]), inv_freq: &[f32]) -> Self {
        let qdim = n_heads*(nope+d_rope);
        Self {
            q_proj: None,
            kv_a: DenseLinear::new(kv_a_w, hidden, d_c + d_rope),
            o_proj: None,
            kv_a_norm: kv_a_norm.to_vec(), kv_b: kv_b.to_vec(), inv_freq: inv_freq.to_vec(),
            cache_ckv: DevHalf::zeros(max_seq*d_c), cache_kr: DevHalf::zeros(max_seq*d_rope),
            n_heads, d_c, d_rope, nope, v_head_dim, hidden, eps, softmax_scale,
            aq_dev: DevHalf::zeros(n_heads*d_c), qr_dev: DevHalf::zeros(n_heads*d_rope),
            outl_dev: DevHalf::zeros(n_heads*d_c),
            ckv_h: DevHalf::zeros(d_c), kr_h: DevHalf::zeros(d_rope),
            qf: DevF32::zeros(qdim), kvf: DevF32::zeros(d_c + d_rope),
            attn_dev: DevHalf::zeros(n_heads*v_head_dim), o_out: DevF32::zeros(hidden),
            last_attn: vec![0f32; n_heads*v_head_dim],
            q_lora: Some(QLoraQ {
                q_a: DenseLinear::new(q_a_w, hidden, q_lora_rank),
                q_a_norm: q_a_norm.to_vec(),
                q_b: QuantLinear::new(q_b.0, q_b.1, q_lora_rank, qdim),
                qa_dev: DevF32::zeros(q_lora_rank),
                qa_normed: DevHalf::zeros(q_lora_rank),
            }),
            o_quant: Some(QuantLinear::new(o.0, o.1, n_heads*v_head_dim, hidden)),
        }
    }

    fn rope(&self, v: &mut [f32], pos: usize) {
        // DeepSeek-V2 uses INTERLEAVED RoPE (adjacent pairs 2i,2i+1), not Llama split-half:
        // apply_rotary_pos_emb reshapes view(d/2,2).transpose before rotate_half, equivalent
        // to rotating adjacent pairs (q.k is permutation-invariant, so rotate in place).
        for d in 0..self.d_rope/2 {
            let ang = pos as f32 * self.inv_freq[d];
            let (c, s) = (ang.cos(), ang.sin());
            let (x0, x1) = (v[2*d], v[2*d+1]);
            v[2*d] = x0*c - x1*s; v[2*d+1] = x1*c + x0*s;
        }
    }

    /// `h_normed` = RMSNorm(h). Returns the attention output (hidden,), to be residual-added.
    pub fn forward(&mut self, h_normed: &DevHalf, pos: usize) -> &DevF32 {
        let (nh, dc, dr, nope, vhd) = (self.n_heads, self.d_c, self.d_rope, self.nope, self.v_head_dim);
        if let Some(ql) = self.q_lora.as_mut() {
            // q_lora path: h -> q_a (dense) -> host RMSNorm -> re-upload -> q_b (4-bit) -> self.qf
            ql.q_a.forward_into(h_normed, &mut ql.qa_dev);
            let qa = ql.qa_dev.to_host();
            let ss: f32 = qa.iter().map(|x| x*x).sum::<f32>() / qa.len() as f32;
            let sc = 1.0 / (ss + self.eps).sqrt();
            let qa_normed: Vec<f32> = qa.iter().zip(ql.q_a_norm.iter()).map(|(x, w)| x * sc * w).collect();
            ql.qa_normed.upload(&qa_normed);
            ql.q_b.forward_into(&ql.qa_normed, &mut self.qf);
        } else {
            self.q_proj.as_ref().unwrap().forward_into(h_normed, &mut self.qf);
        }
        self.kv_a.forward_into(h_normed, &mut self.kvf);
        let q = self.qf.to_host();
        let kv = self.kvf.to_host();
        // latent c_KV: RMSNorm; decoupled rope key
        let mut ckv: Vec<f32> = kv[..dc].to_vec();
        let ss: f32 = ckv.iter().map(|x| x*x).sum::<f32>() / dc as f32;
        let sc = 1.0/(ss + self.eps).sqrt();
        for (i, x) in ckv.iter_mut().enumerate() { *x = *x * sc * self.kv_a_norm[i]; }
        let mut krope: Vec<f32> = kv[dc..dc+dr].to_vec();
        self.rope(&mut krope, pos);
        // per-head absorbed query + rope query
        let mut aq = vec![0f32; nh*dc];
        let mut qr = vec![0f32; nh*dr];
        let hd = nope + dr;
        for h in 0..nh {
            let mut qn = q[h*hd..h*hd+nope].to_vec();
            let mut qrope = q[h*hd+nope..h*hd+hd].to_vec();
            self.rope(&mut qrope, pos);
            qr[h*dr..(h+1)*dr].copy_from_slice(&qrope);
            // absorbed_q[h][j] = sum_i qn[i] * W_UK[h][i][j] ; W_UK[h] rows = kv_b[h*(nope+vhd)+i]
            let base = h*(nope+vhd);
            for j in 0..dc {
                let mut acc = 0f32;
                for i in 0..nope { acc += qn[i] * self.kv_b[(base+i)*dc + j]; }
                aq[h*dc + j] = acc;
            }
            let _ = &mut qn;
        }
        // upload + append cache
        self.aq_dev.upload(&aq); self.qr_dev.upload(&qr);
        self.ckv_h.upload(&ckv); self.kr_h.upload(&krope);
        unsafe {
            op_cache_append(self.cache_ckv.ptr, self.ckv_h.ptr, pos as i32, dc as i32);
            op_cache_append(self.cache_kr.ptr, self.kr_h.ptr, pos as i32, dr as i32);
        }
        // MLA attention on device (DeepSeek softmax_scale = qk_head_dim^-0.5 * mscale^2)
        let scale = self.softmax_scale;
        mla_attention(&self.aq_dev, &self.qr_dev, &self.cache_ckv, &self.cache_kr, &mut self.outl_dev, nh, dc, dr, pos+1, scale);
        let outl = self.outl_dev.to_host();
        // per-head value: attn[h][v] = sum_j outl[h][j] * W_UV[h][v][j] ; W_UV[h] rows = kv_b[h*(nope+vhd)+nope+v]
        for h in 0..nh {
            let base = h*(nope+vhd) + nope;
            for v in 0..vhd {
                let mut acc = 0f32;
                for j in 0..dc { acc += outl[h*dc + j] * self.kv_b[(base+v)*dc + j]; }
                self.last_attn[h*vhd + v] = acc;
            }
        }
        self.attn_dev.upload(&self.last_attn);
        if let Some(oq) = self.o_quant.as_ref() {
            oq.forward_into(&self.attn_dev, &mut self.o_out);
        } else {
            self.o_proj.as_ref().unwrap().forward_into(&self.attn_dev, &mut self.o_out);
        }
        &self.o_out
    }
}

/// Host-side per-output-channel K-means quantizer (mirrors `model/export_runtime.py`'s
/// `quantize()`, on the CPU, for tiny synthetic-check dims). `w` is `[oc][ic]` row-major
/// (`nn.Linear.weight` layout). Returns `(packed, codebook, w_dq)`: `packed` is `ic*(oc/2)`
/// bytes (low nibble first, per `QuantLinear::new`), `codebook` is `K*oc` f32, and `w_dq` is
/// the dequantized `[oc][ic]` matrix — the effective weight `QuantLinear` computes with.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn quantize_host(w: &[f32], oc: usize, ic: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    assert_eq!(w.len(), oc * ic);
    let mut cb = vec![0f32; K * oc];
    let mut idx = vec![0u8; oc * ic]; // idx[o*ic+i]
    for o in 0..oc {
        let col: Vec<f32> = (0..ic).map(|i| w[o * ic + i]).collect();
        let lo = col.iter().cloned().fold(f32::MAX, f32::min);
        let hi = col.iter().cloned().fold(f32::MIN, f32::max);
        let mut centroids: Vec<f32> = (0..K).map(|k| lo + (hi - lo) * (k as f32 / (K as f32 - 1.0))).collect();
        let mut assign = vec![0usize; ic];
        for _ in 0..12 {
            for (i, &v) in col.iter().enumerate() {
                let mut best = 0usize; let mut bd = f32::MAX;
                for (k, &c) in centroids.iter().enumerate() { let d = (v - c).powi(2); if d < bd { bd = d; best = k; } }
                assign[i] = best;
            }
            let mut sum = [0f32; K]; let mut cnt = [0f32; K];
            for (i, &v) in col.iter().enumerate() { sum[assign[i]] += v; cnt[assign[i]] += 1.0; }
            for k in 0..K { if cnt[k] > 0.0 { centroids[k] = sum[k] / cnt[k]; } }
        }
        for i in 0..ic { idx[o * ic + i] = assign[i] as u8; }
        for k in 0..K { cb[k * oc + o] = centroids[k]; }
    }
    // packed[i][j] = idx[i][2j] | (idx[i][2j+1] << 4), matching export_runtime.py's quantize().
    let mut packed = vec![0u8; ic * (oc / 2)];
    for i in 0..ic {
        for j in 0..oc / 2 {
            let (lo, hi) = (idx[(2 * j) * ic + i], idx[(2 * j + 1) * ic + i]);
            packed[i * (oc / 2) + j] = lo | (hi << 4);
        }
    }
    let mut w_dq = vec![0f32; oc * ic];
    for o in 0..oc { for i in 0..ic { w_dq[o * ic + i] = cb[(idx[o * ic + i] as usize) * oc + o]; } }
    (packed, cb, w_dq)
}

/// Validate the DeepSeek-V3/R1 q_lora MLA path (`MlaAttn::new_qlora`): a tiny synthetic
/// config with q_b/o 4-bit codebook-quantized (host k-means via `quantize_host`), compared
/// against a CPU reference that uses the SAME dequantized q_b/o weights (so this isolates
/// the q_lora wiring + RMSNorm + absorption from quantization noise, which `check_moe`/
/// `check_mla_block` already cover elsewhere). Returns worst rel err over `o_proj` output.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_qlora_mla() -> f64 {
    let (hidden, nh, dc, dr, nope, vhd, q_lora_rank) =
        (512usize, 8usize, 128usize, 32usize, 64usize, 64usize, 128usize);
    let qdim = nh * (nope + dr); // 8*(64+32) = 768, %256==0
    assert_eq!(qdim % 256, 0); assert_eq!(hidden % 256, 0); assert_eq!(nh * vhd % 256, 0);
    let eps = 1e-6f32; let max_seq = 24usize; let tpos = 6usize;
    let mut s = 0x9101_ADEDu64;
    // small magnitude (like check_mla_block): keeps fp16 rounding negligible on the dense parts.
    let mut r = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 0.03 };
    let q_a_w: Vec<f32> = (0..q_lora_rank * hidden).map(|_| r()).collect();
    let q_a_norm: Vec<f32> = (0..q_lora_rank).map(|_| r() * 0.1 + 1.0).collect();
    let q_b_w: Vec<f32> = (0..qdim * q_lora_rank).map(|_| r()).collect(); // [qdim][q_lora_rank]
    let kv_a_w: Vec<f32> = (0..(dc + dr) * hidden).map(|_| r()).collect();
    let kv_a_norm: Vec<f32> = (0..dc).map(|_| r() * 0.1 + 1.0).collect();
    let kv_b: Vec<f32> = (0..nh * (nope + vhd) * dc).map(|_| r()).collect();
    let o_w: Vec<f32> = (0..hidden * (nh * vhd)).map(|_| r()).collect(); // [hidden][nh*vhd]
    let inv_freq: Vec<f32> = (0..dr / 2).map(|d| 10000f32.powf(-2.0 * d as f32 / dr as f32)).collect();
    let (qbp, qbc, qb_dq) = quantize_host(&q_b_w, qdim, q_lora_rank);
    let (op, ocb, o_dq) = quantize_host(&o_w, hidden, nh * vhd);
    let sms = 1.0f32 / ((nope + dr) as f32).sqrt();
    let mut mla = MlaAttn::new_qlora(hidden, nh, dc, dr, nope, vhd, max_seq, eps, sms, q_lora_rank,
        &q_a_w, &q_a_norm, (&qbp, &qbc), &kv_a_w, &kv_a_norm, &kv_b, (&op, &ocb), &inv_freq);
    // host reference state (mirrors check_mla_block's cache-of-(ckv,krope))
    let mut cache_ckv: Vec<Vec<f32>> = Vec::new();
    let mut cache_kr: Vec<Vec<f32>> = Vec::new();
    let rope = |v: &mut [f32], pos: usize| { // interleaved (DeepSeek), matches MlaAttn::rope
        for d in 0..dr / 2 { let a = pos as f32 * inv_freq[d]; let (c, sn) = (a.cos(), a.sin());
            let (x0, x1) = (v[2 * d], v[2 * d + 1]); v[2 * d] = x0 * c - x1 * sn; v[2 * d + 1] = x1 * c + x0 * sn; }
    };
    let mv = |w: &[f32], x: &[f32], ic: usize, oc: usize| -> Vec<f32> { // y[oc] = W[oc][ic] x
        (0..oc).map(|o| (0..ic).map(|i| w[o * ic + i] * x[i]).sum()).collect()
    };
    let h16 = |v: f32| half::f16::from_f32(v).to_f32();
    let mut worst = 0f64;
    for pos in 0..tpos {
        let hn: Vec<f32> = (0..hidden).map(|_| r()).collect();
        let mut hd = DevHalf::from_host(&hn);
        let got = mla.forward(&hd, pos).to_host();
        let _ = &mut hd;
        // reference: q_lora front-end using the DEQUANTIZED q_b, then the same absorbed
        // attention as check_mla_block, then o_proj using the DEQUANTIZED o.
        let qa = mv(&q_a_w, &hn, hidden, q_lora_rank);
        let ss: f32 = qa.iter().map(|x| x * x).sum::<f32>() / q_lora_rank as f32;
        let scn = 1.0 / (ss + eps).sqrt();
        let qa_normed: Vec<f32> = qa.iter().zip(q_a_norm.iter()).map(|(x, w)| x * scn * w).collect();
        let q = mv(&qb_dq, &qa_normed, q_lora_rank, qdim);
        let kv = mv(&kv_a_w, &hn, hidden, dc + dr);
        let mut ckv: Vec<f32> = kv[..dc].to_vec();
        let ss2: f32 = ckv.iter().map(|x| x * x).sum::<f32>() / dc as f32; let sc2 = 1.0 / (ss2 + eps).sqrt();
        for (i, x) in ckv.iter_mut().enumerate() { *x = h16(*x * sc2 * kv_a_norm[i]); }
        let mut krope: Vec<f32> = kv[dc..dc + dr].to_vec(); rope(&mut krope, pos);
        for x in krope.iter_mut() { *x = h16(*x); }
        cache_ckv.push(ckv.clone()); cache_kr.push(krope.clone());
        let seqlen = pos + 1; let scale = sms; let hdw = nope + dr;
        let mut attn_vec = vec![0f32; nh * vhd];
        for h in 0..nh {
            let qn: Vec<f32> = (0..nope).map(|i| h16(q[h * hdw + i])).collect();
            let mut qrope: Vec<f32> = (0..dr).map(|i| q[h * hdw + nope + i]).collect(); rope(&mut qrope, pos);
            for x in qrope.iter_mut() { *x = h16(*x); }
            let base = h * (nope + vhd);
            let mut sc = vec![0f32; seqlen];
            for t in 0..seqlen {
                let mut s_nope = 0f32;
                for i in 0..nope { let kni: f32 = (0..dc).map(|j| h16(kv_b[(base + i) * dc + j]) * cache_ckv[t][j]).sum(); s_nope += qn[i] * h16(kni); }
                let s_rope: f32 = (0..dr).map(|i| qrope[i] * cache_kr[t][i]).sum();
                sc[t] = (s_nope + s_rope) * scale;
            }
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0f32; for x in sc.iter_mut() { *x = (*x - mx).exp(); sum += *x; } for x in sc.iter_mut() { *x /= sum; }
            for v in 0..vhd {
                let mut acc = 0f32;
                for t in 0..seqlen { let vv: f32 = (0..dc).map(|j| h16(kv_b[(base + nope + v) * dc + j]) * cache_ckv[t][j]).sum(); acc += sc[t] * h16(vv); }
                attn_vec[h * vhd + v] = acc;
            }
        }
        let o_ref = mv(&o_dq, &attn_vec, nh * vhd, hidden);
        for i in 0..hidden {
            let den = (o_ref[i] as f64).abs().max(1e-2);
            worst = worst.max(((got[i] - o_ref[i]) as f64).abs() / den);
        }
    }
    worst
}

/// Validate the V3 (DeepSeek-V3/R1) sigmoid+bias grouped router (`moe_route_v3`) against an
/// independent CPU implementation (selection-sort based, no shared code with `moe_route_v3`
/// beyond the algorithm description). Returns true if both agree on the selected experts AND
/// their combine weights, across several random logit/bias draws.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_moe_route_v3() -> bool {
    fn route_v3_ref(logits: &[f32], bias: &[f32], n_experts: usize, n_group: usize,
                     topk_group: usize, top_k: usize, rscale: f32) -> Vec<(usize, f32)> {
        let group_size = n_experts / n_group;
        let s: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let sb: Vec<f32> = s.iter().zip(bias.iter()).map(|(a, b)| a + b).collect();
        let mut gscore = vec![0f32; n_group];
        for g in 0..n_group {
            let (mut best1, mut best2) = (f32::MIN, f32::MIN);
            for &v in &sb[g * group_size..(g + 1) * group_size] {
                if v > best1 { best2 = best1; best1 = v; } else if v > best2 { best2 = v; }
            }
            gscore[g] = best1 + best2;
        }
        let mut chosen = vec![false; n_group];
        let mut gs = gscore.clone();
        for _ in 0..topk_group {
            let (mut bi, mut bv) = (0usize, f32::MIN);
            for g in 0..n_group { if !chosen[g] && gs[g] > bv { bv = gs[g]; bi = g; } }
            chosen[bi] = true; gs[bi] = f32::MIN;
        }
        let elig: Vec<usize> = (0..n_experts).filter(|&e| chosen[e / group_size]).collect();
        let mut used = vec![false; elig.len()];
        let mut picked: Vec<usize> = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            let (mut bi, mut bv) = (0usize, f32::MIN);
            for (ii, &e) in elig.iter().enumerate() { if !used[ii] && sb[e] > bv { bv = sb[e]; bi = ii; } }
            used[bi] = true; picked.push(elig[bi]);
        }
        let wsum: f32 = picked.iter().map(|&e| s[e]).sum();
        picked.iter().map(|&e| (e, rscale * s[e] / wsum)).collect()
    }

    let (n_experts, n_group, topk_group, top_k) = (32usize, 4usize, 2usize, 4usize);
    let rscale = 2.5f32;
    let mut s = 0x7047_A1E3u64;
    let mut r = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0 };
    let mut ok = true;
    for _ in 0..8 {
        let logits: Vec<f32> = (0..n_experts).map(|_| r()).collect();
        let bias: Vec<f32> = (0..n_experts).map(|_| r() * 0.5).collect();
        let mut a = moe_route_v3(&logits, &bias, n_experts, n_group, topk_group, top_k, rscale);
        let mut b = route_v3_ref(&logits, &bias, n_experts, n_group, topk_group, top_k, rscale);
        a.sort_by_key(|&(e, _)| e); b.sort_by_key(|&(e, _)| e);
        if a.len() != b.len() { ok = false; break; }
        for ((ea, wa), (eb, wb)) in a.iter().zip(b.iter()) {
            if ea != eb || (wa - wb).abs() > 1e-5 { ok = false; break; }
        }
        if !ok { break; }
    }
    ok
}

/// Validate the full MLA attention block (absorbed kernel path) vs a full-reconstruction
/// CPU reference (rebuild per-head K/V from the latent, standard attention). Lossless up
/// to fp16 noise. Returns worst rel err over the per-head attention output.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_mla_block() -> f64 {
    let (hidden, nh, dc, dr, nope, vhd) = (512usize, 4usize, 128usize, 32usize, 64usize, 64usize);
    let eps = 1e-6f32; let max_seq = 24usize; let tpos = 6usize;
    let mut s = 0xDEE5EE0Fu64;
    // realistic (small) weight magnitudes: the absorbed query folds W_UK over `nope` terms,
    // so large synthetic weights blow up fp16 precision (real model weights are ~0.02).
    let mut r = move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0)*0.03 };
    let qdim = nh*(nope+dr);
    let q_w: Vec<f32> = (0..qdim*hidden).map(|_| r()).collect();
    let kv_a_w: Vec<f32> = (0..(dc+dr)*hidden).map(|_| r()).collect();
    let kv_a_norm: Vec<f32> = (0..dc).map(|_| r()*0.1+1.0).collect();
    let kv_b: Vec<f32> = (0..nh*(nope+vhd)*dc).map(|_| r()).collect();
    let o_w: Vec<f32> = (0..hidden*(nh*vhd)).map(|_| r()).collect();
    let inv_freq: Vec<f32> = (0..dr/2).map(|d| 10000f32.powf(-2.0*d as f32/dr as f32)).collect();
    let sms = 1.0f32/((nope+dr) as f32).sqrt();
    let mut mla = MlaAttn::new(hidden, nh, dc, dr, nope, vhd, max_seq, eps, sms, &q_w, &kv_a_w, &kv_a_norm, &kv_b, &o_w, &inv_freq);
    // host reference state: cache of (ckv, krope) per position
    let mut cache_ckv: Vec<Vec<f32>> = Vec::new();
    let mut cache_kr: Vec<Vec<f32>> = Vec::new();
    let rope = |v: &mut [f32], pos: usize| {  // interleaved (DeepSeek), matches MlaAttn::rope
        for d in 0..dr/2 { let a = pos as f32*inv_freq[d]; let (c,s)=(a.cos(),a.sin());
            let (x0,x1)=(v[2*d],v[2*d+1]); v[2*d]=x0*c-x1*s; v[2*d+1]=x1*c+x0*s; }
    };
    let mv = |w: &[f32], x: &[f32], ic: usize, oc: usize| -> Vec<f32> {  // y[oc] = W[oc][ic] x
        (0..oc).map(|o| (0..ic).map(|i| w[o*ic+i]*x[i]).sum()).collect()
    };
    let h16 = |v: f32| half::f16::from_f32(v).to_f32();
    let mut worst = 0f64;
    for pos in 0..tpos {
        let hn: Vec<f32> = (0..hidden).map(|_| r()).collect();
        // block
        let mut hd = DevHalf::from_host(&hn);
        mla.forward(&hd, pos);
        let got = mla.last_attn.clone();
        let _ = &mut hd;
        // reference (host, f32 but round key parts to fp16 to match the kernel's inputs)
        let q = mv(&q_w, &hn, hidden, qdim);
        let kv = mv(&kv_a_w, &hn, hidden, dc+dr);
        let mut ckv: Vec<f32> = kv[..dc].to_vec();
        let ss: f32 = ckv.iter().map(|x| x*x).sum::<f32>()/dc as f32; let scn = 1.0/(ss+eps).sqrt();
        for (i,x) in ckv.iter_mut().enumerate() { *x = h16(*x*scn*kv_a_norm[i]); }
        let mut krope: Vec<f32> = kv[dc..dc+dr].to_vec(); rope(&mut krope, pos);
        for x in krope.iter_mut() { *x = h16(*x); }
        cache_ckv.push(ckv.clone()); cache_kr.push(krope.clone());
        let seqlen = pos+1; let scale = 1.0/((nope+dr) as f32).sqrt();
        let hdw = nope+dr;
        for h in 0..nh {
            let qn: Vec<f32> = (0..nope).map(|i| h16(q[h*hdw+i])).collect();
            let mut qrope: Vec<f32> = (0..dr).map(|i| q[h*hdw+nope+i]).collect(); rope(&mut qrope, pos);
            for x in qrope.iter_mut() { *x = h16(*x); }
            let base = h*(nope+vhd);
            let mut sc = vec![0f32; seqlen];
            for t in 0..seqlen {
                // k_nope[h][t] = W_UK[h] @ ckv[t] ; W_UK[h] rows = kv_b[(base+i)*dc..]
                let mut s_nope = 0f32;
                for i in 0..nope { let kni: f32 = (0..dc).map(|j| h16(kv_b[(base+i)*dc+j])*cache_ckv[t][j]).sum(); s_nope += qn[i]*h16(kni); }
                let s_rope: f32 = (0..dr).map(|i| qrope[i]*cache_kr[t][i]).sum();
                sc[t] = (s_nope + s_rope)*scale;
            }
            let mx = sc.iter().cloned().fold(f32::MIN,f32::max);
            let mut sum=0f32; for x in sc.iter_mut(){*x=(*x-mx).exp();sum+=*x;} for x in sc.iter_mut(){*x/=sum;}
            for v in 0..vhd {
                // attn[h][v] = sum_t p_t * (W_UV[h][v] @ ckv[t]) ; W_UV[h] rows = kv_b[(base+nope+v)*dc..]
                let mut acc=0f32;
                for t in 0..seqlen { let vv: f32 = (0..dc).map(|j| h16(kv_b[(base+nope+v)*dc+j])*cache_ckv[t][j]).sum(); acc += sc[t]*h16(vv); }
                let g = got[h*vhd+v]; let den=(acc as f64).abs().max(1e-2);
                worst = worst.max(((g-acc) as f64).abs()/den);
            }
        }
    }
    worst
}

/// A full DeepSeek decoder layer: RMSNorm -> MLA attention -> residual, then a MoE block
/// (which norms + residual-adds internally). Composes the validated MlaAttn + MoeBlock.
/// (DeepSeek-V2/V3 use a dense MLP for the first `first_k_dense` layers; swap MoeBlock for
/// MlpBlock there. The MoE variant can be MoeBlock or MoeBlockOffload for the memory wall.)
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct DeepSeekLayer {
    attn_norm: DevF32,
    pub attn: MlaAttn,
    pub moe: MoeBlock,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl DeepSeekLayer {
    pub fn new(attn_norm: &[f32], attn: MlaAttn, moe: MoeBlock) -> Self {
        Self { attn_norm: DevF32::from_host(attn_norm), attn, moe }
    }
    /// One decode step at position `pos`: `h += attn(norm(h)); h += MoE(norm(h))`.
    pub fn forward(&mut self, h: &mut DevHalf, pos: usize) {
        let mut normed = DevHalf::zeros(h.n);
        rmsnorm(h, &self.attn_norm, &mut normed, self.attn.eps);
        {
            let o = self.attn.forward(&normed, pos);
            residual_add(h, o);
        }
        self.moe.forward(h); // MoeBlock norms + residual-adds internally
    }
}

/// The feed-forward of a DeepSeek layer: a dense MLP (first_k_dense layers), an
/// all-resident MoE, or an OFFLOADED MoE (`MoeOffload`, required at 671B scale: routed
/// experts stream from host RAM instead of all sitting on the GPU).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub enum DsFfn { Dense(MlpBlock), Moe(MoeBlock), MoeOffload(MoeBlockOffload) }

/// A DeepSeek decoder layer at the model level (dense or MoE FFN).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct DsLayer { attn_norm: DevF32, attn: MlaAttn, ffn: DsFfn }
#[cfg(any(feature = "cuda", feature = "metal"))]
impl DsLayer {
    fn forward(&mut self, h: &mut DevHalf, pos: usize) {
        let mut normed = DevHalf::zeros(h.n);
        rmsnorm(h, &self.attn_norm, &mut normed, self.attn.eps);
        { let o = self.attn.forward(&normed, pos); residual_add(h, o); }
        match &mut self.ffn {
            DsFfn::Dense(m) => m.forward(h),
            DsFfn::Moe(m) => m.forward(h),
            DsFfn::MoeOffload(m) => m.forward(h),
        }
    }
}

/// A full DeepSeek-V2/V3 (MLA + MoE) model, loaded from the `CBKD` .cbk format. MLA
/// projections dense fp16; experts/router/dense-MLP/LM-head 4-bit codebook. Pure Rust.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct DeepSeekModel {
    embedding: Vec<f32>,
    layers: Vec<DsLayer>,
    final_norm: DevF32,
    lm_head: QuantLinear,
    h: DevHalf,
    normed: DevHalf,
    logits: DevF32,
    hidden: usize,
    vocab: usize,
    eps: f32,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl DeepSeekModel {
    pub fn vocab(&self) -> usize { self.vocab }

    /// Process one token at `pos`, returning the `vocab` next-token logits.
    pub fn forward(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let row = &self.embedding[token*self.hidden..(token+1)*self.hidden];
        self.h.upload(row);
        for l in &mut self.layers { l.forward(&mut self.h, pos); }
        rmsnorm(&self.h, &self.final_norm, &mut self.normed, self.eps);
        self.lm_head.forward_into(&self.normed, &mut self.logits);
        self.logits.to_host()
    }

    /// Load a DeepSeek `.cbk` exported by `model/export_deepseek.py`: `CBKD` (V2-Lite style,
    /// plain dense q_proj, all-resident MoE) or `CBKR` (V3/R1, q_lora + sigmoid-router MoE,
    /// offloaded experts — see `load_deepseek_qlora`).
    pub fn load_deepseek(path: &str, max_seq: usize) -> std::io::Result<DeepSeekModel> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4]; r.read_exact(&mut magic)?;
        // CBKR = original 18-i32 header, routed experts 4-bit scalar (byte-identical to old
        // artifacts). CBKV = 19-i32 header with a trailing `experts_avq` field (0/2/3) for
        // additive-codebook routed experts. Old scalar files keep the CBKR magic and load
        // unchanged; only avq exports use CBKV.
        if &magic == b"CBKR" { return Self::load_deepseek_qlora(path, r, max_seq, false); }
        if &magic == b"CBKV" { return Self::load_deepseek_qlora(path, r, max_seq, true); }
        assert_eq!(&magic, b"CBKD", "not a DeepSeek .cbk (expected CBKD, CBKR or CBKV)");
        let cfg: Vec<usize> = (0..14).map(|_| rd_i32(&mut r) as usize).collect();
        let (n_layers, hidden, n_heads, kv_lora, nope, rope, vhd, inter_dense, moe_inter,
             n_routed, n_shared, top_k, vocab, first_k_dense) =
            (cfg[0],cfg[1],cfg[2],cfg[3],cfg[4],cfg[5],cfg[6],cfg[7],cfg[8],cfg[9],cfg[10],cfg[11],cfg[12],cfg[13]);
        let eps = rd_f32(&mut r); let softmax_scale = rd_f32(&mut r); let rscale = rd_f32(&mut r);
        let inv_freq = rd_f32_vec(&mut r, rope/2);
        let embedding = rd_f16_vec(&mut r, vocab*hidden);
        let qdim = n_heads*(nope+rope);
        let mut layers = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let attn_norm = rd_f32_vec(&mut r, hidden);
            let q_w = rd_f16_vec(&mut r, qdim*hidden);
            let kv_a_w = rd_f16_vec(&mut r, (kv_lora+rope)*hidden);
            let kv_a_norm = rd_f32_vec(&mut r, kv_lora);
            let kv_b = rd_f16_vec(&mut r, n_heads*(nope+vhd)*kv_lora);
            let o_w = rd_f16_vec(&mut r, hidden*(n_heads*vhd));
            let post_norm = rd_f32_vec(&mut r, hidden);
            let attn = MlaAttn::new(hidden, n_heads, kv_lora, rope, nope, vhd, max_seq, eps, softmax_scale,
                &q_w, &kv_a_w, &kv_a_norm, &kv_b, &o_w, &inv_freq);
            let ffn = if li < first_k_dense {
                let (gp,gc) = (rd_u8_vec(&mut r, hidden*(inter_dense/2)), rd_f32_vec(&mut r, K*inter_dense));
                let (up,uc) = (rd_u8_vec(&mut r, hidden*(inter_dense/2)), rd_f32_vec(&mut r, K*inter_dense));
                let (dp,dc) = (rd_u8_vec(&mut r, inter_dense*(hidden/2)), rd_f32_vec(&mut r, K*hidden));
                DsFfn::Dense(MlpBlock::new(hidden, inter_dense, &post_norm, &gp,&gc,&up,&uc,&dp,&dc, eps))
            } else {
                let rw = rd_f16_vec(&mut r, n_routed*hidden);
                let mut estore: Vec<(Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>,Vec<u8>,Vec<f32>)> = Vec::with_capacity(n_routed);
                for _ in 0..n_routed {
                    estore.push((
                        rd_u8_vec(&mut r, hidden*(moe_inter/2)), rd_f32_vec(&mut r, K*moe_inter),
                        rd_u8_vec(&mut r, hidden*(moe_inter/2)), rd_f32_vec(&mut r, K*moe_inter),
                        rd_u8_vec(&mut r, moe_inter*(hidden/2)), rd_f32_vec(&mut r, K*hidden)));
                }
                let si = ((n_shared*moe_inter + 255)/256)*256; // shared expert inter is padded to %256 in export
                let sh = (rd_u8_vec(&mut r, hidden*(si/2)), rd_f32_vec(&mut r, K*si),
                          rd_u8_vec(&mut r, hidden*(si/2)), rd_f32_vec(&mut r, K*si),
                          rd_u8_vec(&mut r, si*(hidden/2)), rd_f32_vec(&mut r, K*hidden));
                let experts: Vec<_> = estore.iter().map(|e| (e.0.as_slice(),e.1.as_slice(),e.2.as_slice(),e.3.as_slice(),e.4.as_slice(),e.5.as_slice())).collect();
                let shared = Some((sh.0.as_slice(),sh.1.as_slice(),sh.2.as_slice(),sh.3.as_slice(),sh.4.as_slice(),sh.5.as_slice()));
                DsFfn::Moe(MoeBlock::new(hidden, moe_inter, n_routed, top_k, eps, &post_norm, &rw, experts, shared, si, rscale))
            };
            layers.push(DsLayer { attn_norm: DevF32::from_host(&attn_norm), attn, ffn });
            eprintln!("  loaded layer {}/{}", li+1, n_layers);
        }
        let final_norm = rd_f32_vec(&mut r, hidden);
        let (lp, lc) = (rd_u8_vec(&mut r, hidden*(vocab/2)), rd_f32_vec(&mut r, K*vocab));
        Ok(DeepSeekModel {
            embedding, layers,
            final_norm: DevF32::from_host(&final_norm),
            lm_head: QuantLinear::new(&lp, &lc, hidden, vocab),
            h: DevHalf::zeros(hidden), normed: DevHalf::zeros(hidden), logits: DevF32::zeros(vocab),
            hidden, vocab, eps,
        })
    }

    /// Load the `CBKR` format (DeepSeek-V3/R1: q_lora MLA + V3 sigmoid/grouped router,
    /// 671B-scale). Header: 18 i32 `[n_layers, hidden, n_heads, kv_lora, nope, rope, vhd,
    /// inter_dense, moe_inter, n_routed, n_shared, top_k, vocab, first_k_dense, q_lora_rank,
    /// n_group, topk_group, sigmoid_flag]`, then 3 f32 (eps, softmax_scale, rscale), inv_freq,
    /// f16 embedding. Per layer: attn_norm, q_a/q_a_norm/q_b (q_lora), kv_a/kv_a_norm/kv_b,
    /// o (4-bit), post_norm, then the dense-or-MoE FFN (MoE layers read `e_score_bias`
    /// BEFORE the router). Routed experts are NOT uploaded to the GPU here — they stay in
    /// host `ExpertHost`s and stream through `MoeBlockOffload` (the memory wall at 671B).
    fn load_deepseek_qlora(path: &str, mut r: BufReader<File>, max_seq: usize, avq_header: bool) -> std::io::Result<DeepSeekModel> {
        // mmap the whole file once: the routed experts' packed indices (the ~300GB bulk at
        // 671B) are handed to ExpertHost as zero-copy byte ranges into this, instead of being
        // read into owned Vecs -- the OS pages them in from disk on demand (and can evict
        // under memory pressure), which is the actual "stream experts from storage" behavior.
        // Safety: the file is a static export artifact, not mutated while the model is loaded.
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&File::open(path)?)? });
        // Header is 18 i32 for CBKR (byte-identical to the original scalar format) or 19 i32 for
        // CBKV, where the trailing field is `experts_avq` (2 or 3 = additive CBKA routed experts
        // at that M). Everything else (attention, dense FFN, shared expert, router, lm_head) stays
        // on the 4-bit scalar path regardless of `experts_avq`.
        let n_fields = if avq_header { 19 } else { 18 };
        let cfg: Vec<usize> = (0..n_fields).map(|_| rd_i32(&mut r) as usize).collect();
        let (n_layers, hidden, n_heads, kv_lora, nope, rope, vhd, inter_dense, moe_inter,
             n_routed, n_shared, top_k, vocab, first_k_dense, q_lora_rank, n_group, topk_group, sigmoid_flag) =
            (cfg[0],cfg[1],cfg[2],cfg[3],cfg[4],cfg[5],cfg[6],cfg[7],cfg[8],cfg[9],cfg[10],cfg[11],cfg[12],cfg[13],cfg[14],cfg[15],cfg[16],cfg[17]);
        let experts_avq = if avq_header { cfg[18] } else { 0 };
        assert!(experts_avq == 0 || experts_avq == 2 || experts_avq == 3, "experts_avq must be 0, 2 or 3");
        let eps = rd_f32(&mut r); let softmax_scale = rd_f32(&mut r); let rscale = rd_f32(&mut r);
        let inv_freq = rd_f32_vec(&mut r, rope/2);
        let embedding = rd_f16_vec(&mut r, vocab*hidden);
        let qdim = n_heads*(nope+rope);
        let mut layers = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let attn_norm = rd_f32_vec(&mut r, hidden);
            let q_a_w = rd_f16_vec(&mut r, q_lora_rank*hidden);
            let q_a_norm = rd_f32_vec(&mut r, q_lora_rank);
            let (qbp, qbc) = (rd_u8_vec(&mut r, q_lora_rank*(qdim/2)), rd_f32_vec(&mut r, K*qdim));
            let kv_a_w = rd_f16_vec(&mut r, (kv_lora+rope)*hidden);
            let kv_a_norm = rd_f32_vec(&mut r, kv_lora);
            let kv_b = rd_f16_vec(&mut r, n_heads*(nope+vhd)*kv_lora);
            let (op, oc) = (rd_u8_vec(&mut r, (n_heads*vhd)*(hidden/2)), rd_f32_vec(&mut r, K*hidden));
            let post_norm = rd_f32_vec(&mut r, hidden);
            let attn = MlaAttn::new_qlora(hidden, n_heads, kv_lora, rope, nope, vhd, max_seq, eps, softmax_scale,
                q_lora_rank, &q_a_w, &q_a_norm, (&qbp, &qbc), &kv_a_w, &kv_a_norm, &kv_b, (&op, &oc), &inv_freq);
            let ffn = if li < first_k_dense {
                let (gp,gc) = (rd_u8_vec(&mut r, hidden*(inter_dense/2)), rd_f32_vec(&mut r, K*inter_dense));
                let (up,uc) = (rd_u8_vec(&mut r, hidden*(inter_dense/2)), rd_f32_vec(&mut r, K*inter_dense));
                let (dp,dc) = (rd_u8_vec(&mut r, inter_dense*(hidden/2)), rd_f32_vec(&mut r, K*hidden));
                DsFfn::Dense(MlpBlock::new(hidden, inter_dense, &post_norm, &gp,&gc,&up,&uc,&dp,&dc, eps))
            } else {
                let score_bias = rd_f32_vec(&mut r, n_routed);
                let rw = rd_f16_vec(&mut r, n_routed*hidden);
                let mut hosts: Vec<ExpertHost> = Vec::with_capacity(n_routed);
                for _ in 0..n_routed {
                    if experts_avq == 0 {
                        // 4-bit scalar: packed indices SKIP (seek past, record the mmap byte
                        // range); the tiny codebooks are read into RAM as before.
                        let gp = mmap_skip(&mut r, &mmap, hidden * (moe_inter / 2))?;
                        let gc = rd_f32_vec(&mut r, K * moe_inter);
                        let up = mmap_skip(&mut r, &mmap, hidden * (moe_inter / 2))?;
                        let uc = rd_f32_vec(&mut r, K * moe_inter);
                        let dp = mmap_skip(&mut r, &mmap, moe_inter * (hidden / 2))?;
                        let dc = rd_f32_vec(&mut r, K * hidden);
                        hosts.push(ExpertHost::Scalar { gp, gc, up, uc, dp, dc });
                    } else {
                        // additive CBKA: gate/up are [moe_inter][hidden], down is [hidden][moe_inter].
                        // Indices are mmap-backed inside read_cbka; codebooks/scales read owned.
                        let gate = read_cbka(&mut r, &mmap, experts_avq, moe_inter, hidden)?;
                        let up   = read_cbka(&mut r, &mmap, experts_avq, moe_inter, hidden)?;
                        let down = read_cbka(&mut r, &mmap, experts_avq, hidden, moe_inter)?;
                        hosts.push(ExpertHost::Avq { gate, up, down });
                    }
                }
                let si = ((n_shared*moe_inter + 255)/256)*256; // shared expert inter, padded to %256
                let sh = (rd_u8_vec(&mut r, hidden*(si/2)), rd_f32_vec(&mut r, K*si),
                          rd_u8_vec(&mut r, hidden*(si/2)), rd_f32_vec(&mut r, K*si),
                          rd_u8_vec(&mut r, si*(hidden/2)), rd_f32_vec(&mut r, K*hidden));
                let shared = (sh.0.as_slice(), sh.1.as_slice(), sh.2.as_slice(), sh.3.as_slice(), sh.4.as_slice(), sh.5.as_slice());
                // cap = top_k: minimal GPU-resident expert cache, experts stream from host every token.
                let mut off = MoeBlockOffload::new(hidden, moe_inter, n_routed, top_k, top_k, eps, &post_norm, &rw, hosts, shared);
                off.set_rscale(rscale);
                if sigmoid_flag != 0 { off.set_v3_scoring(score_bias, n_group, topk_group); }
                DsFfn::MoeOffload(off)
            };
            layers.push(DsLayer { attn_norm: DevF32::from_host(&attn_norm), attn, ffn });
            eprintln!("  loaded layer {}/{} (q_lora)", li+1, n_layers);
        }
        let final_norm = rd_f32_vec(&mut r, hidden);
        let (lp, lc) = (rd_u8_vec(&mut r, hidden*(vocab/2)), rd_f32_vec(&mut r, K*vocab));
        Ok(DeepSeekModel {
            embedding, layers,
            final_norm: DevF32::from_host(&final_norm),
            lm_head: QuantLinear::new(&lp, &lc, hidden, vocab),
            h: DevHalf::zeros(hidden), normed: DevHalf::zeros(hidden), logits: DevF32::zeros(vocab),
            hidden, vocab, eps,
        })
    }
}

/// Read a little-endian i32 binary file into a Vec<i32> (prompt.bin / cont.bin).
pub fn read_i32s(path: &str) -> Vec<i32> {
    let mut f = BufReader::new(File::open(path).unwrap());
    let mut buf = Vec::new(); f.read_to_end(&mut buf).unwrap();
    buf.chunks_exact(4).map(|c| i32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}

/// GeGLU (Gemma): `out = gelu_tanh(gate) * up`, on-device.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn gelu_mul(gate: &DevF32, up: &DevF32, out: &mut DevHalf) {
    assert_eq!(gate.n, up.n); assert_eq!(gate.n, out.n);
    unsafe { op_gelu_mul(gate.ptr, up.ptr, out.ptr, gate.n as i32) };
}

// ============================================================================
// Gemma-2 support. Differs from Llama: GeGLU (not SiLU), RMSNorm (1+w) [baked at
// export], embedding * sqrt(hidden) [baked], attention logit softcapping, final
// logit softcapping, and a 4-norm residual (post-norm on each sublayer OUTPUT
// before the residual add). q_head_dim may differ from hidden (n_heads*head_dim).
// Sliding-window attention is a no-op for short context, so it is not modeled.
// ============================================================================
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct GemmaAttn {
    q: QuantLinear, k: QuantLinear, v: QuantLinear, o: QuantLinear,
    cache_k: DevHalf, cache_v: DevHalf,
    qb: DevF32, kb: DevF32, vb: DevF32, qh: DevHalf, kh: DevHalf, vh: DevHalf,
    attn_out: DevHalf, ob: DevF32, oh: DevHalf,
    inv_freq: DevF32, n_heads: usize, n_kv: usize, head_dim: usize, softcap: f32,
}
#[cfg(any(feature = "cuda", feature = "metal"))]
impl GemmaAttn {
    #[allow(clippy::too_many_arguments)]
    pub fn new(hidden: usize, n_heads: usize, n_kv: usize, head_dim: usize, max_seq: usize,
               q: (&[u8],&[f32]), k: (&[u8],&[f32]), v: (&[u8],&[f32]), o: (&[u8],&[f32]),
               inv_freq: &[f32], softcap: f32) -> Self {
        let qdim = n_heads*head_dim; let kvdim = n_kv*head_dim;
        Self {
            q: QuantLinear::new(q.0,q.1,hidden,qdim), k: QuantLinear::new(k.0,k.1,hidden,kvdim),
            v: QuantLinear::new(v.0,v.1,hidden,kvdim), o: QuantLinear::new(o.0,o.1,qdim,hidden),
            cache_k: DevHalf::zeros(max_seq*kvdim), cache_v: DevHalf::zeros(max_seq*kvdim),
            qb: DevF32::zeros(qdim), kb: DevF32::zeros(kvdim), vb: DevF32::zeros(kvdim),
            qh: DevHalf::zeros(qdim), kh: DevHalf::zeros(kvdim), vh: DevHalf::zeros(kvdim),
            attn_out: DevHalf::zeros(qdim), ob: DevF32::zeros(hidden), oh: DevHalf::zeros(hidden),
            inv_freq: DevF32::from_host(inv_freq), n_heads, n_kv, head_dim, softcap,
        }
    }
    /// `x` is pre-normed (input_layernorm applied by the layer). Returns o_proj output (half).
    pub fn forward(&mut self, x: &DevHalf, pos: usize) -> &DevHalf {
        self.q.forward_into(x, &mut self.qb); self.qh.copy_cast_from(&self.qb);
        self.k.forward_into(x, &mut self.kb); self.kh.copy_cast_from(&self.kb);
        self.v.forward_into(x, &mut self.vb); self.vh.copy_cast_from(&self.vb);
        rope(&mut self.qh, pos, self.n_heads, self.head_dim, &self.inv_freq);
        rope(&mut self.kh, pos, self.n_kv, self.head_dim, &self.inv_freq);
        cache_append(&mut self.cache_k, &self.kh, pos);
        cache_append(&mut self.cache_v, &self.vh, pos);
        attention(&self.qh, &self.cache_k, &self.cache_v, &mut self.attn_out,
                  self.n_heads, self.n_kv, self.head_dim, pos+1, self.softcap);
        self.o.forward_into(&self.attn_out, &mut self.ob);
        self.oh.copy_cast_from(&self.ob);
        &self.oh
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct GemmaMlp { gate: QuantLinear, up: QuantLinear, down: QuantLinear,
    g: DevF32, u: DevF32, act: DevHalf, mb: DevF32, mh: DevHalf }
#[cfg(any(feature = "cuda", feature = "metal"))]
impl GemmaMlp {
    pub fn new(hidden: usize, inter: usize, gate: (&[u8],&[f32]), up: (&[u8],&[f32]), down: (&[u8],&[f32])) -> Self {
        Self { gate: QuantLinear::new(gate.0,gate.1,hidden,inter), up: QuantLinear::new(up.0,up.1,hidden,inter),
            down: QuantLinear::new(down.0,down.1,inter,hidden),
            g: DevF32::zeros(inter), u: DevF32::zeros(inter), act: DevHalf::zeros(inter),
            mb: DevF32::zeros(hidden), mh: DevHalf::zeros(hidden) }
    }
    /// `x` pre-normed. Returns mlp output (half). Uses GeGLU (gelu_tanh).
    pub fn forward(&mut self, x: &DevHalf) -> &DevHalf {
        self.gate.forward_into(x, &mut self.g); self.up.forward_into(x, &mut self.u);
        gelu_mul(&self.g, &self.u, &mut self.act);
        self.down.forward_into(&self.act, &mut self.mb); self.mh.copy_cast_from(&self.mb);
        &self.mh
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct GemmaLayer {
    input_norm: DevF32, post_attn_norm: DevF32, pre_ff_norm: DevF32, post_ff_norm: DevF32,
    attn: GemmaAttn, mlp: GemmaMlp, eps: f32,
    normed: DevHalf, tmp: DevHalf,
}
#[cfg(any(feature = "cuda", feature = "metal"))]
impl GemmaLayer {
    fn forward(&mut self, h: &mut DevHalf, pos: usize) {
        // attention sub-block: h += post_attn_norm(attn(input_norm(h)))
        rmsnorm(h, &self.input_norm, &mut self.normed, self.eps);
        { let ao_ptr = self.attn.forward(&self.normed, pos) as *const DevHalf;
          let ao = unsafe { &*ao_ptr };
          rmsnorm(ao, &self.post_attn_norm, &mut self.tmp, self.eps); }
        resadd_h(h, &self.tmp);
        // mlp sub-block: h += post_ff_norm(geglu_mlp(pre_ff_norm(h)))
        rmsnorm(h, &self.pre_ff_norm, &mut self.normed, self.eps);
        { let mo_ptr = self.mlp.forward(&self.normed) as *const DevHalf;
          let mo = unsafe { &*mo_ptr };
          rmsnorm(mo, &self.post_ff_norm, &mut self.tmp, self.eps); }
        resadd_h(h, &self.tmp);
    }
}

/// A full Gemma-2 model (CBKG format). Pure Rust; GeGLU + softcaps + 4-norm layers.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct GemmaModel {
    embedding: Vec<f32>, layers: Vec<GemmaLayer>, final_norm: DevF32, lm_head: QuantLinear,
    h: DevHalf, normed: DevHalf, logits: DevF32,
    hidden: usize, vocab: usize, eps: f32, final_softcap: f32,
}
#[cfg(any(feature = "cuda", feature = "metal"))]
impl GemmaModel {
    pub fn vocab(&self) -> usize { self.vocab }
    pub fn forward(&mut self, token: usize, pos: usize) -> Vec<f32> {
        // embedding was scaled by sqrt(hidden) at export
        self.h.upload(&self.embedding[token*self.hidden..(token+1)*self.hidden]);
        for l in &mut self.layers { l.forward(&mut self.h, pos); }
        rmsnorm(&self.h, &self.final_norm, &mut self.normed, self.eps);
        self.lm_head.forward_into(&self.normed, &mut self.logits);
        let mut lg = self.logits.to_host();
        if self.final_softcap > 0.0 { let c = self.final_softcap; for x in lg.iter_mut() { *x = c * (*x / c).tanh(); } }
        lg
    }
    /// Load a Gemma-2 `.cbk` (CBKG) from `model/export_gemma.py`.
    pub fn load_gemma(path: &str, max_seq: usize) -> std::io::Result<GemmaModel> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8;4]; r.read_exact(&mut magic)?;
        assert_eq!(&magic, b"CBKG", "not a Gemma .cbk");
        let cfg: Vec<usize> = (0..7).map(|_| rd_i32(&mut r) as usize).collect();
        let (n_layers, hidden, n_heads, n_kv, head_dim, inter, vocab) =
            (cfg[0],cfg[1],cfg[2],cfg[3],cfg[4],cfg[5],cfg[6]);
        let eps = rd_f32(&mut r); let _rope_theta = rd_f32(&mut r);
        let attn_softcap = rd_f32(&mut r); let final_softcap = rd_f32(&mut r);
        let inv_freq = rd_f32_vec(&mut r, head_dim/2);
        let embedding = rd_f16_vec(&mut r, vocab*hidden);
        let qdim = n_heads*head_dim; let kvdim = n_kv*head_dim;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let input_norm = rd_f32_vec(&mut r, hidden);
            let q = (rd_u8_vec(&mut r, hidden*(qdim/2)), rd_f32_vec(&mut r, K*qdim));
            let k = (rd_u8_vec(&mut r, hidden*(kvdim/2)), rd_f32_vec(&mut r, K*kvdim));
            let v = (rd_u8_vec(&mut r, hidden*(kvdim/2)), rd_f32_vec(&mut r, K*kvdim));
            let o = (rd_u8_vec(&mut r, qdim*(hidden/2)), rd_f32_vec(&mut r, K*hidden));
            let post_attn_norm = rd_f32_vec(&mut r, hidden);
            let pre_ff_norm = rd_f32_vec(&mut r, hidden);
            let gate = (rd_u8_vec(&mut r, hidden*(inter/2)), rd_f32_vec(&mut r, K*inter));
            let up = (rd_u8_vec(&mut r, hidden*(inter/2)), rd_f32_vec(&mut r, K*inter));
            let down = (rd_u8_vec(&mut r, inter*(hidden/2)), rd_f32_vec(&mut r, K*hidden));
            let post_ff_norm = rd_f32_vec(&mut r, hidden);
            let attn = GemmaAttn::new(hidden, n_heads, n_kv, head_dim, max_seq,
                (&q.0,&q.1),(&k.0,&k.1),(&v.0,&v.1),(&o.0,&o.1), &inv_freq, attn_softcap);
            let mlp = GemmaMlp::new(hidden, inter, (&gate.0,&gate.1),(&up.0,&up.1),(&down.0,&down.1));
            layers.push(GemmaLayer {
                input_norm: DevF32::from_host(&input_norm), post_attn_norm: DevF32::from_host(&post_attn_norm),
                pre_ff_norm: DevF32::from_host(&pre_ff_norm), post_ff_norm: DevF32::from_host(&post_ff_norm),
                attn, mlp, eps, normed: DevHalf::zeros(hidden), tmp: DevHalf::zeros(hidden),
            });
        }
        let final_norm = rd_f32_vec(&mut r, hidden);
        let (lp, lc) = (rd_u8_vec(&mut r, hidden*(vocab/2)), rd_f32_vec(&mut r, K*vocab));
        Ok(GemmaModel { embedding, layers, final_norm: DevF32::from_host(&final_norm),
            lm_head: QuantLinear::new(&lp,&lc,hidden,vocab),
            h: DevHalf::zeros(hidden), normed: DevHalf::zeros(hidden), logits: DevF32::zeros(vocab),
            hidden, vocab, eps, final_softcap })
    }
}
