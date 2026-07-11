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

/// Worker-thread count for the CPU expert paths (`TRAPETUM_CPU_THREADS`, default 8).
pub fn cpu_threads() -> usize {
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

// ============================================================================
// Layout-aware streaming kernel (the fast path).
//
// The reference `gemv_cpu_f32` above walks output-columns outermost, so its inner
// loop reads `packed[i*(oc/2)+j]` with stride `oc/2` -- a cache miss per input at
// realistic widths (704-byte stride at inter=1408), which is why it measured
// ~0.13 GB/s. The kernel below instead walks INPUT outermost and output-column
// pairs innermost, so `packed` is streamed CONTIGUOUSLY (base+j sequential), read
// exactly once in order; the `y` accumulator (oc f32) stays L1-resident; and the
// per-column codebook is a contiguous 16-entry table via a one-time transpose.
// ============================================================================

/// Transpose a codebook from the stored `cb[k*oc + o]` layout to `cb_t[o*K + k]`, so the
/// 16-entry table for output column `o` is 16 contiguous f32 (64 bytes, L1-friendly) instead
/// of a stride-`oc` gather. Done ONCE per projection at expert construction.
pub fn transpose_codebook(cb: &[f32], oc: usize) -> Vec<f32> {
    assert_eq!(cb.len(), K * oc, "codebook size");
    let mut t = vec![0f32; oc * K];
    for o in 0..oc {
        for k in 0..K {
            t[o * K + k] = cb[k * oc + o];
        }
    }
    t
}

/// Streaming decode-GEMV for an INPUT range `[i0, i1)`, accumulating into `y` (NOT zeroed
/// here -- caller owns init). i-outer / output-pairs-inner: `packed` is read contiguously,
/// `cb_t` is the transposed codebook (`cb_t[o*K + k]`). Same math as `gemv_cpu_f32`, only
/// the summation order differs. This is the kernel the hot path uses (see `stream_range`).
fn gemv_stream_range(packed: &[u8], cb_t: &[f32], oc: usize, x: &[f32], y: &mut [f32], i0: usize, i1: usize) {
    let half = oc / 2;
    for i in i0..i1 {
        let xi = x[i];
        let base = i * half;
        let row = &packed[base..base + half];
        for j in 0..half {
            let b = row[j];
            let lo = (b & 0xF) as usize;
            let hi = (b >> 4) as usize;
            // output columns 2j (low nibble) and 2j+1 (high nibble)
            y[2 * j]     += cb_t[(2 * j) * K + lo] * xi;
            y[2 * j + 1] += cb_t[(2 * j + 1) * K + hi] * xi;
        }
    }
}

/// NEON i-outer streaming range (aarch64). Processes 8 output columns per 4 packed bytes:
/// one broadcast `xi`, 8 scalar codebook gathers into two `float32x4`, two fused MACs into
/// the `y` accumulator. Streams `packed` contiguously like the scalar version; the vector
/// MAC is the speedup, the gather stays scalar (no f32 gather across 8 distinct tables).
/// Measured SLOWER than the scalar kernel here (gather-bound), so wired out; kept + tested
/// for a future revisit (prefetch / wider tile). See `stream_range`.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn gemv_stream_range_neon(packed: &[u8], cb_t: &[f32], oc: usize, x: &[f32], y: &mut [f32], i0: usize, i1: usize) {
    use std::arch::aarch64::*;
    let half = oc / 2;
    let ocv = oc & !7; // largest multiple of 8 <= oc
    unsafe {
        for i in i0..i1 {
            let xi = vdupq_n_f32(x[i]);
            let base = i * half;
            let mut o = 0usize;
            let mut jb = base; // byte offset of the packed pair for columns o..o+1
            while o < ocv {
                let b0 = packed[jb]; let b1 = packed[jb + 1]; let b2 = packed[jb + 2]; let b3 = packed[jb + 3];
                // packed[i*half + j] holds column 2j (low nibble), 2j+1 (high nibble); the 4
                // bytes at jb cover output columns o..o+8.
                let g = [
                    cb_t[(o    ) * K + (b0 & 0xF) as usize],
                    cb_t[(o + 1) * K + (b0 >> 4)  as usize],
                    cb_t[(o + 2) * K + (b1 & 0xF) as usize],
                    cb_t[(o + 3) * K + (b1 >> 4)  as usize],
                    cb_t[(o + 4) * K + (b2 & 0xF) as usize],
                    cb_t[(o + 5) * K + (b2 >> 4)  as usize],
                    cb_t[(o + 6) * K + (b3 & 0xF) as usize],
                    cb_t[(o + 7) * K + (b3 >> 4)  as usize],
                ];
                let y0 = vld1q_f32(y.as_ptr().add(o));
                let y1 = vld1q_f32(y.as_ptr().add(o + 4));
                vst1q_f32(y.as_mut_ptr().add(o),     vfmaq_f32(y0, vld1q_f32(g.as_ptr()),       xi));
                vst1q_f32(y.as_mut_ptr().add(o + 4), vfmaq_f32(y1, vld1q_f32(g.as_ptr().add(4)), xi));
                o += 8; jb += 4;
            }
            // scalar tail for the last (oc % 8) columns
            let xif = x[i];
            for oo in ocv..oc {
                let j = oo / 2;
                let b = packed[base + j];
                let id = (if oo & 1 == 0 { b & 0xF } else { b >> 4 }) as usize;
                y[oo] += cb_t[oo * K + id] * xif;
            }
        }
    }
}

/// Layout-aware streaming decode-GEMV: `y[o] = sum_i cb[idx[o,i]*oc + o] * x[i]`, computed
/// i-outer so `packed` streams contiguously. `cb_t` is the transposed codebook from
/// [`transpose_codebook`]. Multi-threaded over an INPUT-range split with per-thread `y`
/// accumulators reduced at the end (each partial is only `oc` f32, a few KB) -- input split
/// is used because the packed layout makes output-column blocks non-contiguous. On aarch64
/// each range uses the NEON kernel.
pub fn gemv_cpu_stream(packed: &[u8], cb_t: &[f32], oc: usize, ic: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(packed.len(), ic * (oc / 2), "packed size");
    assert_eq!(cb_t.len(), K * oc, "transposed codebook size");
    assert_eq!(x.len(), ic, "activation size");
    assert!(y.len() >= oc, "output size");
    assert_eq!(oc % 2, 0, "oc must be even (nibble packing)");

    for v in y.iter_mut().take(oc) { *v = 0.0; }
    let nthreads = cpu_threads().min(ic.max(1));
    if nthreads <= 1 || ic < 64 {
        stream_range(packed, cb_t, oc, x, y, 0, ic);
        return;
    }
    let chunk = (ic + nthreads - 1) / nthreads;
    // Each thread streams its own input range into a private y accumulator; reduce at the end.
    let partials: Vec<Vec<f32>> = thread::scope(|s| {
        let mut handles = Vec::new();
        let mut i0 = 0usize;
        while i0 < ic {
            let i1 = (i0 + chunk).min(ic);
            let (packed, cb_t, x) = (&packed, &cb_t, &x);
            handles.push(s.spawn(move || {
                let mut yp = vec![0f32; oc];
                stream_range(packed, cb_t, oc, x, &mut yp, i0, i1);
                yp
            }));
            i0 = i1;
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for yp in &partials {
        for o in 0..oc { y[o] += yp[o]; }
    }
}

/// The streaming range kernel used by the hot path. We use the SCALAR i-outer version: on
/// this M4, release+LTO, it measured 10.5 ms for the V2-Lite routed block vs 12.6 ms for the
/// NEON variant ([`gemv_stream_range_neon`]). The NEON version is gather-bound -- the 8
/// dependent scalar codebook loads per vector dominate and the vectorized FMA can't hide
/// them -- so it is kept (tested) but wired out, per "only add NEON if scalar doesn't
/// saturate". Revisit with software prefetch / a wider codebook tile before rewiring it in.
#[inline]
fn stream_range(packed: &[u8], cb_t: &[f32], oc: usize, x: &[f32], y: &mut [f32], i0: usize, i1: usize) {
    gemv_stream_range(packed, cb_t, oc, x, y, i0, i1);
}

/// Single-threaded streaming decode-GEMV (zeroes `y`, streams all `ic` inputs on this thread).
/// The building block for `expert_forward_cpu_stream`, which is parallelized ACROSS experts by
/// the caller -- so the per-GEMV path must not itself spawn threads (nested spawning per GEMV
/// is exactly what capped throughput at ~1.8 GB/s: the spawn overhead dwarfed the tiny GEMV).
fn gemv_stream_single(packed: &[u8], cb_t: &[f32], oc: usize, ic: usize, x: &[f32], y: &mut [f32]) {
    for v in y.iter_mut().take(oc) { *v = 0.0; }
    stream_range(packed, cb_t, oc, x, y, 0, ic);
}

/// Full SwiGLU expert forward using the streaming kernel and PRE-TRANSPOSED codebooks
/// (`gc_t`, `uc_t`, `dc_t` from [`transpose_codebook`]). SINGLE-THREADED: the runtime runs
/// several of these concurrently, one expert per thread (see `MoeBlock::routed_cpu`), which
/// keeps each expert's packed stream hot in one core's cache and spawns threads once per
/// token instead of once per GEMV. Same math as [`expert_forward_cpu`] to summation tolerance.
#[allow(clippy::too_many_arguments)]
pub fn expert_forward_cpu_stream(
    x: &[f32],
    gp: &[u8], gc_t: &[f32],
    up: &[u8], uc_t: &[f32],
    dp: &[u8], dc_t: &[f32],
    hidden: usize,
    inter: usize,
    y: &mut [f32],
) {
    assert_eq!(x.len(), hidden, "expert input width");
    assert!(y.len() >= hidden, "expert output width");
    let mut g = vec![0f32; inter];
    let mut u = vec![0f32; inter];
    gemv_stream_single(gp, gc_t, inter, hidden, x, &mut g);
    gemv_stream_single(up, uc_t, inter, hidden, x, &mut u);
    let mut act = vec![0f32; inter];
    for i in 0..inter { act[i] = silu(g[i]) * u[i]; }
    gemv_stream_single(dp, dc_t, hidden, inter, &act, y);
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

    // L2 relative error: robust to near-zero elements (where worst-element rel err blows up
    // on a tiny denominator). Used for fusion comparisons where the diff is absolute-tiny.
    fn l2_rel(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let d = (x - y) as f64; num += d * d; den += (x as f64) * (x as f64);
        }
        num.sqrt() / den.sqrt().max(1e-12)
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

    #[test]
    fn stream_matches_scalar_reference() {
        // The layout-aware streaming GEMV must equal the scalar reference (same math, only
        // summation order differs: i-outer + input-range reduction vs o-outer). Cover a
        // realistic width and a small one; the MT input-split path is exercised at ic>=64.
        for (oc, ic) in [(256usize, 512usize), (64, 128), (128, 2048)] {
            let mut r = Lcg(0x57EA_11u64 ^ ((oc as u64) << 8) ^ ic as u64);
            let w: Vec<f32> = (0..oc * ic).map(|_| r.f32()).collect();
            let x: Vec<f32> = (0..ic).map(|_| r.f32()).collect();
            let (packed, cb, _w_dq) = quantize_host(&w, oc, ic);
            let cb_t = transpose_codebook(&cb, oc);
            let mut yref = vec![0f32; oc];
            let mut ystream = vec![0f32; oc];
            gemv_cpu_f32(&packed, &cb, oc, ic, &x, &mut yref);
            gemv_cpu_stream(&packed, &cb_t, oc, ic, &x, &mut ystream);
            let e = rel_err(&ystream, &yref);
            // Same arithmetic, different summation order. Single-thread i-outer is bit-identical
            // to the o-outer reference (both sum i-ascending); the residual (~5e-5 measured) comes
            // from the multi-thread per-input-range partial-sum reduction. Tolerance, not equality.
            assert!(e < 5e-4, "stream vs scalar rel err {e:e} at oc={oc} ic={ic}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_stream_matches_scalar_stream() {
        // The NEON i-outer kernel must equal the scalar i-outer kernel (single-thread, same
        // input order) to ~1e-6; the only difference is vfmaq fusion vs separate mul/add.
        for (oc, ic) in [(256usize, 512usize), (72, 96), (1408, 200)] {
            let mut r = Lcg(0xBEEF_5731u64 ^ ((oc as u64) << 16) ^ ic as u64);
            let w: Vec<f32> = (0..oc * ic).map(|_| r.f32()).collect();
            let x: Vec<f32> = (0..ic).map(|_| r.f32()).collect();
            let (packed, cb, _) = quantize_host(&w, oc, ic);
            let cb_t = transpose_codebook(&cb, oc);
            let mut ys = vec![0f32; oc];
            let mut yn = vec![0f32; oc];
            gemv_stream_range(&packed, &cb_t, oc, &x, &mut ys, 0, ic);
            gemv_stream_range_neon(&packed, &cb_t, oc, &x, &mut yn, 0, ic);
            // vfmaq (fused, kept for speed) doesn't round the intermediate product like the
            // scalar `y += cb*xi`, so they aren't bit-equal. On near-zero output elements the
            // worst-element rel err blows up (tiny denominator, ~5e-4 at oc=1408) even though
            // the diff is absolute-tiny, so gate on the L2 relative error (~1e-6 measured).
            let e = l2_rel(&yn, &ys);
            assert!(e < 1e-5, "NEON stream vs scalar stream L2 rel err {e:e} at oc={oc} ic={ic}");
        }
    }

    // Kernel-level before/after: the OLD o-outer `gemv_cpu_f32` vs the NEW i-outer streaming
    // `gemv_cpu_stream`, at the V2-Lite gate GEMV shape. Run in RELEASE for meaningful numbers:
    //   cargo test --release --features metal -- --nocapture stream_vs_scalar_throughput
    // Prints ms and packed-bytes GB/s for both. No timing assert (machine-load dependent).
    #[test]
    fn stream_vs_scalar_throughput() {
        let (oc, ic) = (1408usize, 2048usize); // gate: [inter][hidden]
        let mut r = Lcg(0x7EA_57A11u64);
        let w: Vec<f32> = (0..oc * ic).map(|_| r.f32()).collect();
        let x: Vec<f32> = (0..ic).map(|_| r.f32()).collect();
        let (packed, cb, _) = quantize_host(&w, oc, ic);
        let cb_t = transpose_codebook(&cb, oc);
        let bytes = packed.len();
        let mut y = vec![0f32; oc];
        let best = |mut f: Box<dyn FnMut()>| -> f64 {
            for _ in 0..3 { f(); }
            let mut b = f64::MAX;
            for _ in 0..20 {
                let t = std::time::Instant::now(); f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let (p2, x2) = (packed.clone(), x.clone());
        let ms_old = best(Box::new({ let cb = cb.clone(); let mut y = y.clone();
            move || gemv_cpu_f32(&p2, &cb, oc, ic, &x2, &mut y) })) * 1e3;
        let ms_new = best(Box::new(move || gemv_cpu_stream(&packed, &cb_t, oc, ic, &x, &mut y))) * 1e3;
        let gbs = |ms: f64| bytes as f64 / (ms * 1e-3) / 1e9;
        eprintln!("[stream_vs_scalar_throughput] oc={oc} ic={ic} packed={} KB", bytes / 1024);
        eprintln!("  OLD o-outer gemv_cpu_f32 : {ms_old:.3} ms  ({:.2} GB/s)", gbs(ms_old));
        eprintln!("  NEW i-outer gemv_cpu_stream: {ms_new:.3} ms  ({:.2} GB/s)  speedup x{:.1}", gbs(ms_new), ms_old / ms_new);
    }

    #[test]
    fn expert_stream_matches_expert_reference() {
        let (hidden, inter) = (256usize, 512usize);
        let mut r = Lcg(0xC0FFEE_5715_EA33);
        let gw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
        let uw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
        let dw: Vec<f32> = (0..hidden * inter).map(|_| r.f32() * 0.5).collect();
        let x: Vec<f32> = (0..hidden).map(|_| r.f32()).collect();
        let (gp, gc, _) = quantize_host(&gw, inter, hidden);
        let (up, uc, _) = quantize_host(&uw, inter, hidden);
        let (dp, dc, _) = quantize_host(&dw, hidden, inter);
        let (gct, uct, dct) = (transpose_codebook(&gc, inter), transpose_codebook(&uc, inter), transpose_codebook(&dc, hidden));

        let mut yref = vec![0f32; hidden];
        let mut ystream = vec![0f32; hidden];
        expert_forward_cpu(&x, &gp, &gc, &up, &uc, &dp, &dc, hidden, inter, &mut yref);
        expert_forward_cpu_stream(&x, &gp, &gct, &up, &uct, &dp, &dct, hidden, inter, &mut ystream);
        let e = rel_err(&ystream, &yref);
        // Summation-order residual (multi-thread reduction); measured ~1e-4 through the 3 GEMVs.
        assert!(e < 5e-4, "expert stream vs reference rel err {e:e}");
    }
}
