//! Minimal Rust runtime over the fused 4-bit codebook decode CUDA kernel.
//!
//! A [`QuantLinear`] holds codebook-quantized weights resident on the GPU. Activations
//! live in caller-owned device buffers ([`DevHalf`], [`DevF32`]), so layers chain
//! **on-device** with no host<->device copy between them: the kernel writes f32 and
//! [`DevHalf::copy_cast_from`] converts it to half for the next layer. No Python.
use half::f16;
use std::fs::File;
use std::io::{BufReader, Read};
use std::os::raw::c_void;

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
        pub fn dev_alloc_half(n: i32) -> *mut c_void;
        pub fn dev_alloc_f32(n: i32) -> *mut c_void;
        pub fn dev_free(p: *mut c_void);
        pub fn dev_upload_to_half(d_half: *mut c_void, x: *const f32, n: i32);
        pub fn dev_cast_f32_to_half(d_half: *mut c_void, d_f32: *const c_void, n: i32);
        pub fn dev_download_f32(x: *mut f32, d_f32: *const c_void, n: i32);
        pub fn dev_download_half(x: *mut f32, d_half: *const c_void, n: i32);
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
        );
        // batched (M-token) ops for speculative decoding
        pub fn qlinear_forward_m(handle: *mut c_void, d_x: *const c_void, d_y: *mut c_void, m: i32);
        pub fn op_rmsnorm_m(x_half: *const c_void, w_f32: *const c_void, out_half: *mut c_void, n: i32, eps: f32, m: i32);
        pub fn op_rope_m(x_half: *mut c_void, base: i32, n_heads: i32, head_dim: i32, inv_freq: *const c_void, m: i32);
        pub fn op_cache_append_m(cache_half: *mut c_void, src_half: *const c_void, base: i32, dim: i32, m: i32);
        pub fn op_saxpy(acc_f32: *mut c_void, y_f32: *const c_void, alpha: f32, n: i32);
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
) {
    unsafe {
        op_attn(q.ptr, ck.ptr, cv.ptr, out.ptr, n_heads as i32, n_kv as i32, head_dim as i32, seqlen as i32)
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
            self.n_heads, self.n_kv, self.head_dim, pos + 1,
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
// call (a test-grade path; production would preallocate). No attention bias yet.
#[cfg(any(feature = "cuda", feature = "metal"))]
impl MlpBlock {
    /// `h` is `m*hidden`, updated in place: `h = h + MLP(RMSNorm(h))` for all M rows.
    pub fn forward_m(&mut self, h: &mut DevHalf, m: usize) {
        let hidden = self.norm_w.len();
        let inter = self.g.n;
        let mut norm = DevHalf::zeros(m * hidden);
        let mut g = DevF32::zeros(m * inter);
        let mut u = DevF32::zeros(m * inter);
        let mut act = DevHalf::zeros(m * inter);
        let mut mlp = DevF32::zeros(m * hidden);
        rmsnorm_m(h, &self.norm_w, &mut norm, self.eps, hidden, m);
        self.gate.forward_m(&norm, &mut g, m);
        self.up.forward_m(&norm, &mut u, m);
        silu_mul(&g, &u, &mut act);
        self.down.forward_m(&act, &mut mlp, m);
        residual_add(h, &mlp);
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl AttnBlock {
    /// Batched decode step: `h` is `m*hidden`, the M new tokens sit at positions
    /// `base..base+m`. Each query attends causally over `base+row+1` keys.
    pub fn forward_m(&mut self, h: &mut DevHalf, base: usize, m: usize) {
        assert!(self.qbias.is_none(), "batched forward_m does not support attention bias yet");
        let hidden = self.norm_w.len();
        let qdim = self.n_heads * self.head_dim;
        let kvdim = self.n_kv * self.head_dim;
        let mut norm = DevHalf::zeros(m * hidden);
        let mut qb = DevF32::zeros(m * qdim);
        let mut kb = DevF32::zeros(m * kvdim);
        let mut vb = DevF32::zeros(m * kvdim);
        let mut qh = DevHalf::zeros(m * qdim);
        let mut kh = DevHalf::zeros(m * kvdim);
        let mut vh = DevHalf::zeros(m * kvdim);
        let mut attn_out = DevHalf::zeros(m * qdim);
        let mut ob = DevF32::zeros(m * hidden);
        rmsnorm_m(h, &self.norm_w, &mut norm, self.eps, hidden, m);
        self.q.forward_m(&norm, &mut qb, m);
        self.k.forward_m(&norm, &mut kb, m);
        self.v.forward_m(&norm, &mut vb, m);
        qh.copy_cast_from(&qb);
        kh.copy_cast_from(&kb);
        vh.copy_cast_from(&vb);
        rope_m(&mut qh, base, self.n_heads, self.head_dim, &self.inv_freq, m);
        rope_m(&mut kh, base, self.n_kv, self.head_dim, &self.inv_freq, m);
        cache_append_m(&mut self.cache_k, &kh, base, kvdim, m);
        cache_append_m(&mut self.cache_v, &vh, base, kvdim, m);
        attention_m(&qh, &self.cache_k, &self.cache_v, &mut attn_out,
                    self.n_heads, self.n_kv, self.head_dim, base, m);
        self.o.forward_m(&attn_out, &mut ob, m);
        residual_add(h, &ob);
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
        }
    }

    /// Process one token at position `pos`, returning the `vocab` next-token logits.
    pub fn forward(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let row = &self.embedding[token * self.hidden..(token + 1) * self.hidden];
        self.h.upload(row);
        for l in &mut self.layers {
            l.forward(&mut self.h, pos);
        }
        rmsnorm(&self.h, &self.final_norm, &mut self.normed, self.eps);
        self.lm_head.forward_into(&self.normed, &mut self.logits);
        self.logits.to_host()
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
        let mut h = DevHalf::from_host(&hm);
        for l in &mut self.layers {
            l.forward_m(&mut h, pos, m);
        }
        let mut normed = DevHalf::zeros(m * hid);
        rmsnorm_m(&h, &self.final_norm, &mut normed, self.eps, hid, m);
        let mut logits = DevF32::zeros(m * self.vocab);
        self.lm_head.forward_m(&normed, &mut logits, m);
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

/// A Mixture-of-Experts decoder block (DeepSeek-V2/V3 style): RMSNorm, a router that
/// scores `n_experts`, top-k selection, the k selected expert FFNs run and combined by
/// their (renormalized) router weights, plus an always-on shared expert, then a residual.
/// At batch-1 decode only k of n_experts run, which is what makes huge MoE models cheap
/// per token (and what the memory-offload path below exploits). Solves the "dense runtime"
/// wall: the router + top-k + expert combine are the missing pieces.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct MoeBlock {
    norm_w: DevF32,
    router: QuantLinear,
    experts: Vec<Expert>,
    shared: Option<Expert>,
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
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl MoeBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(hidden: usize, inter: usize, n_experts: usize, top_k: usize, eps: f32,
               norm_w: &[f32], router: (&[u8], &[f32]),
               experts: Vec<(&[u8],&[f32],&[u8],&[f32],&[u8],&[f32])>,
               shared: Option<(&[u8],&[f32],&[u8],&[f32],&[u8],&[f32])>) -> Self {
        assert_eq!(experts.len(), n_experts);
        let mk = |e: &(&[u8],&[f32],&[u8],&[f32],&[u8],&[f32])| Expert {
            gate: QuantLinear::new(e.0, e.1, hidden, inter),
            up:   QuantLinear::new(e.2, e.3, hidden, inter),
            down: QuantLinear::new(e.4, e.5, inter, hidden),
        };
        Self {
            norm_w: DevF32::from_host(norm_w),
            router: QuantLinear::new(router.0, router.1, hidden, n_experts),
            experts: experts.iter().map(mk).collect(),
            shared: shared.as_ref().map(mk),
            top_k, hidden, inter, n_experts, eps,
            norm: DevHalf::zeros(hidden),
            rlogits: DevF32::zeros(n_experts),
            g: DevF32::zeros(inter), u: DevF32::zeros(inter),
            act: DevHalf::zeros(inter), ey: DevF32::zeros(hidden),
        }
    }

    fn ffn(&mut self, e: usize, shared: bool, acc: &mut DevF32, w: f32) {
        // run expert e (or the shared expert) FFN on self.norm, scaled-add into acc
        let ex = if shared { self.shared.as_ref().unwrap() } else { &self.experts[e] };
        // borrow split: copy raw handles out to avoid double-borrow of self
        let (g_ptr, u_ptr, d_ptr) = (ex.gate.handle, ex.up.handle, ex.down.handle);
        unsafe {
            qlinear_forward_dev(g_ptr, self.norm.ptr, self.g.ptr);
            qlinear_forward_dev(u_ptr, self.norm.ptr, self.u.ptr);
        }
        silu_mul(&self.g, &self.u, &mut self.act);
        unsafe { qlinear_forward_dev(d_ptr, self.act.ptr, self.ey.ptr); }
        saxpy(acc, &self.ey, w);
    }

    /// `h` (hidden,) updated in place: `h += MoE(RMSNorm(h))`.
    pub fn forward(&mut self, h: &mut DevHalf) {
        rmsnorm(h, &self.norm_w, &mut self.norm, self.eps);
        self.router.forward_into(&self.norm, &mut self.rlogits);
        let rl = self.rlogits.to_host();
        // softmax over all experts
        let mx = rl.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = rl.iter().map(|x| (x - mx).exp()).collect();
        let sum: f32 = ex.iter().sum();
        let probs: Vec<f32> = ex.iter().map(|x| x / sum).collect();
        // top-k experts
        let mut idx: Vec<usize> = (0..self.n_experts).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let topk = &idx[..self.top_k];
        let wsum: f32 = topk.iter().map(|&e| probs[e]).sum();
        let mut acc = DevF32::from_host(&vec![0f32; self.hidden]);
        let picks: Vec<(usize, f32)> = topk.iter().map(|&e| (e, probs[e] / wsum)).collect();
        for (e, w) in picks { self.ffn(e, false, &mut acc, w); }
        if self.shared.is_some() { self.ffn(0, true, &mut acc, 1.0); }
        residual_add(h, &acc);
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
        let mx = rl.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = rl.iter().map(|x| (x - mx).exp()).collect();
        let sum: f32 = ex.iter().sum();
        let probs: Vec<f32> = ex.iter().map(|x| x / sum).collect();
        let mut idx: Vec<usize> = (0..self.n_experts).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let topk: std::collections::HashSet<usize> = idx[..self.top_k].iter().cloned().collect();
        let wsum: f32 = idx[..self.top_k].iter().map(|&e| probs[e]).sum();
        let mut acc = DevF32::from_host(&vec![0f32; self.hidden]);
        for e in 0..self.n_experts {
            let w = if topk.contains(&e) { probs[e] / wsum } else { 0.0 };
            if w > 0.0 { self.ffn(e, false, &mut acc, w); }
        }
        if self.shared.is_some() { self.ffn(0, true, &mut acc, 1.0); }
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
    let rp = packed(hidden*(n_experts/2), &mut r); let rc = cbk(K*n_experts, &mut r);
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
    let mut moe = MoeBlock::new(hidden, inter, n_experts, top_k, eps, &nw, (&rp,&rc), experts, shared);
    let h0: Vec<f32> = (0..hidden).map(|_| r()*0.3).collect();
    let mut ha = DevHalf::from_host(&h0); moe.forward(&mut ha); let a = ha.to_host();
    let mut hb = DevHalf::from_host(&h0); moe.forward_dense_ref(&mut hb); let b = hb.to_host();
    let mut worst = 0f64;
    for i in 0..hidden { let den=(b[i] as f64).abs().max(1e-3); worst=worst.max(((a[i]-b[i]) as f64).abs()/den); }
    worst
}

/// Host-resident expert weights (not on the GPU until streamed in).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct ExpertHost { gp: Vec<u8>, gc: Vec<f32>, up: Vec<u8>, uc: Vec<f32>, dp: Vec<u8>, dc: Vec<f32> }

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
    router: QuantLinear,
    hosts: Vec<ExpertHost>,
    shared: Expert,
    cache: std::collections::HashMap<usize, Expert>,
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
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl MoeBlockOffload {
    #[allow(clippy::too_many_arguments)]
    pub fn new(hidden: usize, inter: usize, n_experts: usize, top_k: usize, cap: usize, eps: f32,
               norm_w: &[f32], router: (&[u8], &[f32]), hosts: Vec<ExpertHost>,
               shared: (&[u8],&[f32],&[u8],&[f32],&[u8],&[f32])) -> Self {
        assert_eq!(hosts.len(), n_experts);
        assert!(cap >= top_k, "cache must hold at least top_k experts");
        Self {
            norm_w: DevF32::from_host(norm_w),
            router: QuantLinear::new(router.0, router.1, hidden, n_experts),
            hosts,
            shared: Expert { gate: QuantLinear::new(shared.0,shared.1,hidden,inter),
                             up: QuantLinear::new(shared.2,shared.3,hidden,inter),
                             down: QuantLinear::new(shared.4,shared.5,inter,hidden) },
            cache: std::collections::HashMap::new(), lru: Vec::new(), cap,
            top_k, hidden, inter, n_experts, eps,
            norm: DevHalf::zeros(hidden), rlogits: DevF32::zeros(n_experts),
            g: DevF32::zeros(inter), u: DevF32::zeros(inter),
            act: DevHalf::zeros(inter), ey: DevF32::zeros(hidden),
            uploads: 0,
        }
    }

    /// Ensure expert `e` is resident (stream from host + LRU-evict if needed); return handles.
    fn resident(&mut self, e: usize) -> (*mut c_void, *mut c_void, *mut c_void) {
        if self.cache.contains_key(&e) {
            let pos = self.lru.iter().position(|&x| x == e).unwrap();
            self.lru.remove(pos);
            self.lru.push(e);
        } else {
            if self.cache.len() >= self.cap {
                let victim = self.lru.remove(0);
                self.cache.remove(&victim); // Drop frees the GPU buffers
            }
            let h = &self.hosts[e];
            let ex = Expert {
                gate: QuantLinear::new(&h.gp, &h.gc, self.hidden, self.inter),
                up:   QuantLinear::new(&h.up, &h.uc, self.hidden, self.inter),
                down: QuantLinear::new(&h.dp, &h.dc, self.inter, self.hidden),
            };
            self.cache.insert(e, ex);
            self.lru.push(e);
            self.uploads += 1;
        }
        let ex = self.cache.get(&e).unwrap();
        (ex.gate.handle, ex.up.handle, ex.down.handle)
    }

    fn run_ffn(&mut self, g: *mut c_void, u: *mut c_void, d: *mut c_void, acc: &mut DevF32, w: f32) {
        unsafe {
            qlinear_forward_dev(g, self.norm.ptr, self.g.ptr);
            qlinear_forward_dev(u, self.norm.ptr, self.u.ptr);
        }
        silu_mul(&self.g, &self.u, &mut self.act);
        unsafe { qlinear_forward_dev(d, self.act.ptr, self.ey.ptr); }
        saxpy(acc, &self.ey, w);
    }

    pub fn forward(&mut self, h: &mut DevHalf) {
        rmsnorm(h, &self.norm_w, &mut self.norm, self.eps);
        self.router.forward_into(&self.norm, &mut self.rlogits);
        let rl = self.rlogits.to_host();
        let mx = rl.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = rl.iter().map(|x| (x - mx).exp()).collect();
        let sum: f32 = ex.iter().sum();
        let probs: Vec<f32> = ex.iter().map(|x| x / sum).collect();
        let mut idx: Vec<usize> = (0..self.n_experts).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let wsum: f32 = idx[..self.top_k].iter().map(|&e| probs[e]).sum();
        let picks: Vec<(usize, f32)> = idx[..self.top_k].iter().map(|&e| (e, probs[e] / wsum)).collect();
        let mut acc = DevF32::from_host(&vec![0f32; self.hidden]);
        for (e, w) in picks {
            let (g, u, d) = self.resident(e);
            self.run_ffn(g, u, d, &mut acc, w);
        }
        let (sg, su, sd) = (self.shared.gate.handle, self.shared.up.handle, self.shared.down.handle);
        self.run_ffn(sg, su, sd, &mut acc, 1.0);
        residual_add(h, &acc);
    }
}

// Deterministic MoE weight generator (same seed -> identical model), for the offload check.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn gen_moe(hidden: usize, inter: usize, n_experts: usize, seed: u64)
    -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<ExpertHost>, ExpertHost) {
    let mut s = seed;
    let mut r = move || { s ^= s<<13; s ^= s>>7; s ^= s<<17; (((s>>40) as f32/(1u64<<24) as f32)*2.0-1.0) };
    let pk = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<u8> { (0..n).map(|_| ((r()*0.5+0.5)*255.0) as u8).collect() };
    let cb = |n: usize, r: &mut dyn FnMut()->f32| -> Vec<f32> { (0..n).map(|_| r()*0.05).collect() };
    let nw: Vec<f32> = (0..hidden).map(|_| r()*0.1+1.0).collect();
    let rp = pk(hidden*(n_experts/2), &mut r); let rc = cb(K*n_experts, &mut r);
    let mkh = |r: &mut dyn FnMut()->f32| ExpertHost {
        gp: pk(hidden*(inter/2), r), gc: cb(K*inter, r),
        up: pk(hidden*(inter/2), r), uc: cb(K*inter, r),
        dp: pk(inter*(hidden/2), r), dc: cb(K*hidden, r) };
    let hosts: Vec<ExpertHost> = (0..n_experts).map(|_| mkh(&mut r)).collect();
    let shared = mkh(&mut r);
    (nw, rp, rc, hosts, shared)
}

/// Validate expert OFFLOADING: the offloaded block (only `cap` experts resident, streamed
/// from host with an LRU) must produce IDENTICAL output to the all-resident block, over
/// several tokens. Returns (worst_rel_err, cap, n_experts, uploads_over_tokens).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn check_moe_offload() -> (f64, usize, usize, usize) {
    let (hidden, inter, n_experts, top_k, cap) = (256usize, 256usize, 256usize, 8usize, 16usize);
    let eps = 1e-5f32; let seed = 0x0FF10AD5u64;
    // all-resident reference
    let (nw, rp, rc, hosts_r, sh_r) = gen_moe(hidden, inter, n_experts, seed);
    let exps_ref: Vec<_> = hosts_r.iter().map(|e| (e.gp.as_slice(),e.gc.as_slice(),e.up.as_slice(),e.uc.as_slice(),e.dp.as_slice(),e.dc.as_slice())).collect();
    let shref = (sh_r.gp.as_slice(),sh_r.gc.as_slice(),sh_r.up.as_slice(),sh_r.uc.as_slice(),sh_r.dp.as_slice(),sh_r.dc.as_slice());
    let mut moe = MoeBlock::new(hidden, inter, n_experts, top_k, eps, &nw, (&rp,&rc), exps_ref, Some(shref));
    // offloaded (identical weights via same seed)
    let (nw2, rp2, rc2, hosts_o, sh_o) = gen_moe(hidden, inter, n_experts, seed);
    let sho = (sh_o.gp.as_slice(),sh_o.gc.as_slice(),sh_o.up.as_slice(),sh_o.uc.as_slice(),sh_o.dp.as_slice(),sh_o.dc.as_slice());
    let mut off = MoeBlockOffload::new(hidden, inter, n_experts, top_k, cap, eps, &nw2, (&rp2,&rc2), hosts_o, sho);
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
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct MlaAttn {
    q_proj: DenseLinear,
    kv_a: DenseLinear,
    o_proj: DenseLinear,
    kv_a_norm: Vec<f32>,          // [d_c]
    kv_b: Vec<f32>,               // [n_heads*(nope+v_head_dim)][d_c]  (W_UK ++ W_UV per head)
    inv_freq: Vec<f32>,           // [d_rope/2]
    cache_ckv: DevHalf,           // [max_seq][d_c]
    cache_kr: DevHalf,            // [max_seq][d_rope]
    n_heads: usize, d_c: usize, d_rope: usize, nope: usize, v_head_dim: usize, hidden: usize,
    eps: f32,
    aq_dev: DevHalf, qr_dev: DevHalf, outl_dev: DevHalf,
    ckv_h: DevHalf, kr_h: DevHalf,
    qf: DevF32, kvf: DevF32, attn_dev: DevHalf, o_out: DevF32,
    pub last_attn: Vec<f32>,      // pre-o_proj per-head values (for validation)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl MlaAttn {
    #[allow(clippy::too_many_arguments)]
    pub fn new(hidden: usize, n_heads: usize, d_c: usize, d_rope: usize, nope: usize, v_head_dim: usize,
               max_seq: usize, eps: f32, q_w: &[f32], kv_a_w: &[f32], kv_a_norm: &[f32], kv_b: &[f32],
               o_w: &[f32], inv_freq: &[f32]) -> Self {
        let qdim = n_heads*(nope+d_rope);
        Self {
            q_proj: DenseLinear::new(q_w, hidden, qdim),
            kv_a: DenseLinear::new(kv_a_w, hidden, d_c + d_rope),
            o_proj: DenseLinear::new(o_w, n_heads*v_head_dim, hidden),
            kv_a_norm: kv_a_norm.to_vec(), kv_b: kv_b.to_vec(), inv_freq: inv_freq.to_vec(),
            cache_ckv: DevHalf::zeros(max_seq*d_c), cache_kr: DevHalf::zeros(max_seq*d_rope),
            n_heads, d_c, d_rope, nope, v_head_dim, hidden, eps,
            aq_dev: DevHalf::zeros(n_heads*d_c), qr_dev: DevHalf::zeros(n_heads*d_rope),
            outl_dev: DevHalf::zeros(n_heads*d_c),
            ckv_h: DevHalf::zeros(d_c), kr_h: DevHalf::zeros(d_rope),
            qf: DevF32::zeros(qdim), kvf: DevF32::zeros(d_c + d_rope),
            attn_dev: DevHalf::zeros(n_heads*v_head_dim), o_out: DevF32::zeros(hidden),
            last_attn: vec![0f32; n_heads*v_head_dim],
        }
    }

    fn rope(&self, v: &mut [f32], pos: usize) {
        let half = self.d_rope/2;
        for d in 0..half {
            let ang = pos as f32 * self.inv_freq[d];
            let (c, s) = (ang.cos(), ang.sin());
            let (x0, x1) = (v[d], v[d+half]);
            v[d] = x0*c - x1*s; v[d+half] = x1*c + x0*s;
        }
    }

    /// `h_normed` = RMSNorm(h). Returns the attention output (hidden,), to be residual-added.
    pub fn forward(&mut self, h_normed: &DevHalf, pos: usize) -> &DevF32 {
        let (nh, dc, dr, nope, vhd) = (self.n_heads, self.d_c, self.d_rope, self.nope, self.v_head_dim);
        self.q_proj.forward_into(h_normed, &mut self.qf);
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
        // MLA attention on device
        let scale = 1.0/((nope+dr) as f32).sqrt();
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
        self.o_proj.forward_into(&self.attn_dev, &mut self.o_out);
        &self.o_out
    }
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
    let mut mla = MlaAttn::new(hidden, nh, dc, dr, nope, vhd, max_seq, eps, &q_w, &kv_a_w, &kv_a_norm, &kv_b, &o_w, &inv_freq);
    // host reference state: cache of (ckv, krope) per position
    let mut cache_ckv: Vec<Vec<f32>> = Vec::new();
    let mut cache_kr: Vec<Vec<f32>> = Vec::new();
    let rope = |v: &mut [f32], pos: usize| {
        let half = dr/2;
        for d in 0..half { let a = pos as f32*inv_freq[d]; let (c,s)=(a.cos(),a.sin());
            let (x0,x1)=(v[d],v[d+half]); v[d]=x0*c-x1*s; v[d+half]=x1*c+x0*s; }
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
