//! CPU forward for scalar 4-bit codebook projections and MoE experts.
//!
//! This is a pure-CPU mirror of the GPU `gemv4` kernel and the SwiGLU expert path.
//! It exists so an expert can be evaluated on the host (no device) with bit-exact
//! agreement (up to summation order) against the runtime's own dequantization.
//!
//! ## Format (mirrors `quantize_host` in `lib.rs` and the `gemv4` Metal kernel)
//! A projection is `oc` output channels x `ic` input channels, stored as:
//!  - `packed`: `ic * (oc/2)` bytes. `packed[i*(oc/2) + j]` holds two 4-bit indices,
//!    the LOW nibble for output column `2*j`, the HIGH nibble for output column `2*j+1`
//!    (`lib.rs:2438-2444`). Equivalently, reading a `u32` at `packed[i*(oc/2) + o/2]`
//!    yields 8 nibbles for output columns `o..o+8`, nibble `c` -> column `o+c`
//!    (the `gemv4` kernel, `kernels.metal:57-62`).
//!  - `cb`: `K*oc` f32, per OUTPUT column. The centroid for index `id` at output `o`
//!    is `cb[id*oc + o]` (`lib.rs:2436,2447`; kernel `cb[k*OC + jj]`).
//!  - Dequantized weight: `w_dq[o,i] = cb[idx[o,i]*oc + o]` (`lib.rs:2447`).
//!
//! So `y[o] = sum_i cb[idx[o,i]*oc + o] * x[i]`, accumulated in f32 (the GPU also
//! accumulates the decode in f32; only summation order differs from a naive matmul).

use std::thread;

/// Number of codebook entries (must match `crate::K`).
const K: usize = 16;

fn cpu_threads() -> usize {
    std::env::var("TRAPETUM_CPU_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(8)
}

/// Decode-GEMV for one output-column range `[o0, o1)`: `y[o] = sum_i w_dq[o,i]*x[i]`.
/// Scalar f32 reference (the correctness anchor). `x.len() == ic`, `y.len() >= o1`.
fn gemv_range_scalar(packed: &[u8], cb: &[f32], oc: usize, ic: usize, x: &[f32], y: &mut [f32], o0: usize, o1: usize) {
    let half = oc / 2;
    for o in o0..o1 {
        let j = o / 2;
        let shift = if o & 1 == 0 { 0 } else { 4 };
        let mut acc = 0f32;
        for i in 0..ic {
            let id = ((packed[i * half + j] >> shift) & 0xF) as usize;
            acc += cb[id * oc + o] * x[i];
        }
        y[o] = acc;
    }
}

/// Exact f32 reference decode-GEMV matching `quantize_host`'s dequant semantics.
/// `packed` is `ic*(oc/2)` bytes, `cb` is `K*oc` f32, `x` is `ic`, writes `y` (`oc`).
/// Multi-threaded over an output-column split (`TRAPETUM_CPU_THREADS`, default 8);
/// threads own disjoint `y` ranges so no synchronization is needed.
pub fn gemv_cpu_f32(packed: &[u8], cb: &[f32], oc: usize, ic: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(packed.len(), ic * (oc / 2), "packed size");
    assert_eq!(cb.len(), K * oc, "codebook size");
    assert_eq!(x.len(), ic, "activation size");
    assert!(y.len() >= oc, "output size");
    assert_eq!(oc % 2, 0, "oc must be even (nibble packing)");

    let nthreads = cpu_threads().min(oc.max(1));
    if nthreads <= 1 || oc < 64 {
        gemv_range_scalar(packed, cb, oc, ic, x, y, 0, oc);
        return;
    }
    // Even output-column split; keep each chunk even so nibble pairs never straddle threads.
    let mut chunk = (oc + nthreads - 1) / nthreads;
    if chunk % 2 == 1 { chunk += 1; }
    thread::scope(|s| {
        let mut o0 = 0usize;
        let mut rest = &mut y[..oc];
        while o0 < oc {
            let o1 = (o0 + chunk).min(oc);
            let (head, tail) = rest.split_at_mut(o1 - o0);
            rest = tail;
            let (packed, cb, x) = (&packed, &cb, &x);
            s.spawn(move || {
                // `head` covers columns [o0, o1); shift indices back into it.
                for (li, o) in (o0..o1).enumerate() {
                    let j = o / 2;
                    let shift = if o & 1 == 0 { 0 } else { 4 };
                    let mut acc = 0f32;
                    let half = oc / 2;
                    for i in 0..ic {
                        let id = ((packed[i * half + j] >> shift) & 0xF) as usize;
                        acc += cb[id * oc + o] * x[i];
                    }
                    head[li] = acc;
                }
            });
            o0 = o1;
        }
    });
}

/// NEON-accelerated decode-GEMV for a `[o0, o1)` output range (aarch64 only).
/// Processes 8 output columns per `u32` of packed indices, mirroring `gemv4`:
/// gather 8 codebook centroids into two `float32x4_t`, FMA the broadcast activation.
/// The gather is scalar (no NEON f32 gather); the multiply-accumulate is vector.
#[cfg(target_arch = "aarch64")]
fn gemv_range_neon(packed: &[u8], cb: &[f32], oc: usize, ic: usize, x: &[f32], y: &mut [f32], o0: usize, o1: usize) {
    use std::arch::aarch64::*;
    let half = oc / 2;
    // Vector body over blocks of 8 output columns fully inside [o0, o1) and 8-aligned.
    let vstart = (o0 + 7) & !7;
    let vend = o1 & !7;
    unsafe {
        // Scalar head (columns before the first 8-aligned block).
        gemv_range_scalar(packed, cb, oc, ic, x, y, o0, vstart.min(o1));
        let mut o = vstart;
        while o < vend {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let byte_col = o / 2; // = o*4/8; u32 spanning 8 nibbles for columns o..o+8
            for i in 0..ic {
                let word = u32::from_le_bytes([
                    packed[i * half + byte_col],
                    packed[i * half + byte_col + 1],
                    packed[i * half + byte_col + 2],
                    packed[i * half + byte_col + 3],
                ]);
                let xx = vdupq_n_f32(x[i]);
                let g = [
                    cb[(((word >> 0) & 0xF) as usize) * oc + o],
                    cb[(((word >> 4) & 0xF) as usize) * oc + o + 1],
                    cb[(((word >> 8) & 0xF) as usize) * oc + o + 2],
                    cb[(((word >> 12) & 0xF) as usize) * oc + o + 3],
                    cb[(((word >> 16) & 0xF) as usize) * oc + o + 4],
                    cb[(((word >> 20) & 0xF) as usize) * oc + o + 5],
                    cb[(((word >> 24) & 0xF) as usize) * oc + o + 6],
                    cb[(((word >> 28) & 0xF) as usize) * oc + o + 7],
                ];
                // Non-fused mul+add (not vfmaq) so each lane rounds exactly like the
                // scalar `acc += cb*x`, giving bit-identical agreement with gemv_cpu_f32.
                acc0 = vaddq_f32(acc0, vmulq_f32(xx, vld1q_f32(g.as_ptr())));
                acc1 = vaddq_f32(acc1, vmulq_f32(xx, vld1q_f32(g.as_ptr().add(4))));
            }
            vst1q_f32(y.as_mut_ptr().add(o), acc0);
            vst1q_f32(y.as_mut_ptr().add(o + 4), acc1);
            o += 8;
        }
        // Scalar tail.
        gemv_range_scalar(packed, cb, oc, ic, x, y, vend.max(vstart), o1);
    }
}

/// NEON multi-threaded decode-GEMV (aarch64). Same result as [`gemv_cpu_f32`] to ~1e-6.
#[cfg(target_arch = "aarch64")]
pub fn gemv_cpu_neon(packed: &[u8], cb: &[f32], oc: usize, ic: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(packed.len(), ic * (oc / 2), "packed size");
    assert_eq!(cb.len(), K * oc, "codebook size");
    assert_eq!(x.len(), ic, "activation size");
    assert!(y.len() >= oc, "output size");
    assert_eq!(oc % 2, 0, "oc must be even (nibble packing)");

    let nthreads = cpu_threads().min(oc.max(1));
    if nthreads <= 1 || oc < 64 {
        gemv_range_neon(packed, cb, oc, ic, x, y, 0, oc);
        return;
    }
    let mut chunk = (oc + nthreads - 1) / nthreads;
    // Keep each chunk a multiple of 8 so the NEON blocks never straddle a thread boundary.
    chunk = (chunk + 7) & !7;
    thread::scope(|s| {
        let mut o0 = 0usize;
        while o0 < oc {
            let o1 = (o0 + chunk).min(oc);
            let yptr = SendPtr(y.as_mut_ptr());
            let (packed, cb, x) = (&packed, &cb, &x);
            s.spawn(move || {
                let _ = &yptr;
                // Safety: disjoint output ranges per thread; no overlap in [o0, o1).
                let yslice = unsafe { std::slice::from_raw_parts_mut(yptr.0, oc) };
                gemv_range_neon(packed, cb, oc, ic, x, yslice, o0, o1);
            });
            o0 = o1;
        }
    });
}

/// Raw-pointer wrapper so a thread can write a disjoint sub-range of `y` under NEON.
#[cfg(target_arch = "aarch64")]
struct SendPtr(*mut f32);
#[cfg(target_arch = "aarch64")]
unsafe impl Send for SendPtr {}

fn silu(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

/// Full SwiGLU expert forward on the CPU, all f32, matching the GPU expert path
/// (`gate`/`up`: `[inter][hidden]`, SwiGLU `silu(g)*u`, `down`: `[hidden][inter]`).
/// Takes exactly the six slices an [`crate::ExpertHost::Scalar`] stores
/// (`gp,gc, up,uc, dp,dc`) so runtime integration is a direct call.
///
/// `x` is `hidden`, `y` is written `hidden`. Uses [`gemv_cpu_f32`] for each GEMV.
#[allow(clippy::too_many_arguments)]
pub fn expert_forward_cpu(
    x: &[f32],
    gp: &[u8], gc: &[f32],
    up: &[u8], uc: &[f32],
    dp: &[u8], dc: &[f32],
    hidden: usize,
    inter: usize,
    y: &mut [f32],
) {
    assert_eq!(x.len(), hidden, "expert input width");
    assert!(y.len() >= hidden, "expert output width");
    let mut g = vec![0f32; inter];
    let mut u = vec![0f32; inter];
    // gate/up: oc=inter, ic=hidden
    gemv_cpu_f32(gp, gc, inter, hidden, x, &mut g);
    gemv_cpu_f32(up, uc, inter, hidden, x, &mut u);
    let mut act = vec![0f32; inter];
    for i in 0..inter {
        act[i] = silu(g[i]) * u[i];
    }
    // down: oc=hidden, ic=inter
    gemv_cpu_f32(dp, dc, hidden, inter, &act, y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize_host;

    // Small deterministic LCG so tests don't depend on rand.
    struct Lcg(u64);
    impl Lcg {
        fn f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn naive_matmul(w_dq: &[f32], oc: usize, ic: usize, x: &[f32]) -> Vec<f32> {
        (0..oc).map(|o| (0..ic).map(|i| w_dq[o * ic + i] * x[i]).sum()).collect()
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f64 {
        let mut worst = 0f64;
        for (&x, &y) in a.iter().zip(b) {
            let d = (x - y).abs() as f64;
            let s = (x.abs().max(y.abs())) as f64;
            worst = worst.max(if s > 1e-6 { d / s } else { d });
        }
        worst
    }

    #[test]
    fn gemv_matches_dequant_matmul() {
        // 64 output x 128 input, quantized with the runtime's own quantize_host.
        let (oc, ic) = (64usize, 128usize);
        let mut r = Lcg(0xDEAD_BEEF_1234_5678);
        let w: Vec<f32> = (0..oc * ic).map(|_| r.f32()).collect();
        let x: Vec<f32> = (0..ic).map(|_| r.f32()).collect();
        let (packed, cb, w_dq) = quantize_host(&w, oc, ic);
        let reference = naive_matmul(&w_dq, oc, ic, &x);
        let mut y = vec![0f32; oc];
        gemv_cpu_f32(&packed, &cb, oc, ic, &x, &mut y);
        let e = rel_err(&y, &reference);
        assert!(e < 1e-5, "gemv_cpu_f32 vs dequant matmul rel err {e:e}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar() {
        // Non-multiple-of-8 offsets exercised via a range split inside the threaded path.
        for (oc, ic) in [(64usize, 128usize), (72, 96), (256, 200)] {
            let mut r = Lcg(0x0C7A_11E5_u64 ^ (oc as u64));
            let w: Vec<f32> = (0..oc * ic).map(|_| r.f32()).collect();
            let x: Vec<f32> = (0..ic).map(|_| r.f32()).collect();
            let (packed, cb, _w_dq) = quantize_host(&w, oc, ic);
            let mut ys = vec![0f32; oc];
            let mut yn = vec![0f32; oc];
            gemv_cpu_f32(&packed, &cb, oc, ic, &x, &mut ys);
            gemv_cpu_neon(&packed, &cb, oc, ic, &x, &mut yn);
            let e = rel_err(&yn, &ys);
            assert!(e < 1e-6, "NEON vs scalar rel err {e:e} at oc={oc} ic={ic}");
        }
    }

    #[test]
    fn expert_forward_matches_reference() {
        let (hidden, inter) = (128usize, 256usize);
        let mut r = Lcg(0xFEED_FACE_C0DE_0001);
        // gate/up: [inter][hidden]; down: [hidden][inter]
        let gw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
        let uw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
        let dw: Vec<f32> = (0..hidden * inter).map(|_| r.f32() * 0.5).collect();
        let x: Vec<f32> = (0..hidden).map(|_| r.f32()).collect();
        let (gp, gc, g_dq) = quantize_host(&gw, inter, hidden);
        let (up, uc, u_dq) = quantize_host(&uw, inter, hidden);
        let (dp, dc, d_dq) = quantize_host(&dw, hidden, inter);

        let mut y = vec![0f32; hidden];
        expert_forward_cpu(&x, &gp, &gc, &up, &uc, &dp, &dc, hidden, inter, &mut y);

        // Naive f32 reference on the dequantized weights.
        let g = naive_matmul(&g_dq, inter, hidden, &x);
        let u = naive_matmul(&u_dq, inter, hidden, &x);
        let act: Vec<f32> = (0..inter).map(|i| silu(g[i]) * u[i]).collect();
        let reference = naive_matmul(&d_dq, hidden, inter, &act);

        let e = rel_err(&y, &reference);
        assert!(e < 1e-4, "expert_forward_cpu vs reference rel err {e:e}");
    }
}
