//! Minimal Rust runtime over the fused 4-bit codebook decode CUDA kernel.
//!
//! A [`QuantLinear`] holds codebook-quantized weights resident on the GPU and runs a
//! fast batch-1 decode GEMV through the CUDA kernel, with no Python in the loop. This is
//! the foundation of a deployable inference runtime: the weight matrix is never
//! materialized, only the 4-bit codes are read.
use std::os::raw::c_void;

extern "C" {
    fn qlinear_create(packed: *const u8, cb: *const f32, ic: i32, oc: i32) -> *mut c_void;
    fn qlinear_forward(h: *mut c_void, x: *const f32, y: *mut f32);
    fn qlinear_free(h: *mut c_void);
}

/// Number of codebook entries (4-bit indices).
pub const K: usize = 16;

/// A codebook-quantized linear layer whose weights live on the GPU.
pub struct QuantLinear {
    handle: *mut c_void,
    ic: usize,
    oc: usize,
}

impl QuantLinear {
    /// Upload quantized weights to the GPU.
    ///
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

    /// Decode `y = x W^T` from the quantized weights. `x` is `(ic,)`, returns `(oc,)`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.ic, "x must have length ic");
        let mut y = vec![0f32; self.oc];
        unsafe { qlinear_forward(self.handle, x.as_ptr(), y.as_mut_ptr()) };
        y
    }

    /// `(in_features, out_features)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.ic, self.oc)
    }
}

impl Drop for QuantLinear {
    fn drop(&mut self) {
        unsafe { qlinear_free(self.handle) };
    }
}

// The CUDA handle is owned and used single-threaded here; do not auto-derive Send/Sync.
