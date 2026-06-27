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
    fn dev_sync();
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
    /// Device-side cast: fill this fp16 buffer from a device f32 buffer (no host copy).
    /// This is the inter-layer conversion when chaining.
    pub fn copy_cast_from(&mut self, src: &DevF32) {
        assert_eq!(self.n, src.n, "length mismatch in copy_cast_from");
        unsafe { dev_cast_f32_to_half(self.ptr, src.ptr, self.n as i32) };
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

/// Block until all queued GPU work completes (call before stopping a timer).
pub fn sync() {
    unsafe { dev_sync() };
}
