//! Minimal Rust runtime over the fused 4-bit codebook decode CUDA kernel.
//!
//! A [`QuantLinear`] holds codebook-quantized weights resident on the GPU. Activations
//! live in caller-owned device buffers ([`DevHalf`], [`DevF32`]), so layers chain
//! **on-device** with no host<->device copy between them: the kernel writes f32 and
//! [`DevHalf::copy_cast_from`] converts it to half for the next layer. No Python.
use std::os::raw::c_void;

extern "C" {
    fn qlinear_create(packed: *const u8, cb: *const f32, ic: i32, oc: i32) -> *mut c_void;
    fn qlinear_forward_dev(h: *mut c_void, d_x: *const c_void, d_y: *mut c_void);
    fn qlinear_free(h: *mut c_void);
    fn dev_alloc_half(n: i32) -> *mut c_void;
    fn dev_alloc_f32(n: i32) -> *mut c_void;
    fn dev_free(p: *mut c_void);
    fn dev_upload_to_half(d_half: *mut c_void, x: *const f32, n: i32);
    fn dev_cast_f32_to_half(d_half: *mut c_void, d_f32: *const c_void, n: i32);
    fn dev_download_f32(x: *mut f32, d_f32: *const c_void, n: i32);
    fn dev_download_half(x: *mut f32, d_half: *const c_void, n: i32);
    fn dev_sync();
    fn graph_begin();
    fn graph_end() -> *mut c_void;
    fn graph_launch(exec: *mut c_void);
    fn graph_free(exec: *mut c_void);
    fn dev_upload_f32(d_f32: *mut c_void, x: *const f32, n: i32);
    fn op_rmsnorm(x_half: *const c_void, w_f32: *const c_void, out_half: *mut c_void, n: i32, eps: f32);
    fn op_silu_mul(gate_f32: *const c_void, up_f32: *const c_void, out_half: *mut c_void, n: i32);
    fn op_residual_add(h_half: *mut c_void, delta_f32: *const c_void, n: i32);
    fn op_rope(x_half: *mut c_void, pos: i32, n_heads: i32, head_dim: i32, base: f32);
    fn op_cache_append(cache_half: *mut c_void, src_half: *const c_void, pos: i32, dim: i32);
    fn op_attn(
        q_half: *const c_void,
        ck_half: *const c_void,
        cv_half: *const c_void,
        out_half: *mut c_void,
        n_heads: i32,
        n_kv: i32,
        head_dim: i32,
        seqlen: i32,
    );
}

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
pub fn rope(x: &mut DevHalf, pos: usize, n_heads: usize, head_dim: usize, base: f32) {
    assert_eq!(x.n, n_heads * head_dim);
    unsafe { op_rope(x.ptr, pos as i32, n_heads as i32, head_dim as i32, base) };
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
    base: f32,
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
        base: f32,
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
            base,
        }
    }

    /// One decode step at position `pos`: updates the cache and the residual stream `h`.
    pub fn forward(&mut self, h: &mut DevHalf, pos: usize) {
        rmsnorm(h, &self.norm_w, &mut self.norm, self.eps);
        self.q.forward_into(&self.norm, &mut self.qb);
        self.qh.copy_cast_from(&self.qb);
        self.k.forward_into(&self.norm, &mut self.kb);
        self.kh.copy_cast_from(&self.kb);
        self.v.forward_into(&self.norm, &mut self.vb);
        self.vh.copy_cast_from(&self.vb);
        rope(&mut self.qh, pos, self.n_heads, self.head_dim, self.base);
        rope(&mut self.kh, pos, self.n_kv, self.head_dim, self.base);
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
