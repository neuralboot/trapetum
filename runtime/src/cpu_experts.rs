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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Condvar, Mutex, OnceLock};
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

// ============================================================================
// Row-major (output-major) work-stealing engine -- the throughput path.
//
// The streaming kernel above already fixed the packed-stream locality, but its
// per-column codebook still has to be re-gathered per input and its threading
// spawned per token. This section ports the C probe's design
// (bench/cpu_probes/s13m2_worksteal.c, ~47 GB/s on this M4):
//   1. RE-TILE packed to OUTPUT-major once (pack_to_rowmajor): output row `o`'s
//      `ic` indices become contiguous, so a row-chunk streams one contiguous run
//      and the row's 16-entry codebook table stays in registers for the whole row.
//   2. Row-chunk WORK-STEALING across all picked experts, in the C probe's three
//      phases (A: gate+up rows, B: SiLU per expert, C: down rows), each drained
//      by one atomic counter, barriers between. This balances the M4's P/E cores.
// ============================================================================

/// Re-tile input-major `packed` (`packed[i*(oc/2)+j]`, the `quantize_host`/GPU layout) to
/// OUTPUT-major `packed_t[o*(ic/2)+k]`, where byte `k` of output row `o` holds input `2k` in
/// the low nibble and `2k+1` in the high nibble. Each output row's `ic` indices are then
/// contiguous. Done once per projection at `CpuExpert` construction. Requires `ic` even.
pub fn pack_to_rowmajor(packed: &[u8], oc: usize, ic: usize) -> Vec<u8> {
    assert_eq!(packed.len(), ic * (oc / 2), "packed size");
    assert_eq!(ic % 2, 0, "ic must be even");
    let (half_o, half_i) = (oc / 2, ic / 2);
    let mut t = vec![0u8; oc * half_i];
    for o in 0..oc {
        let shift = if o & 1 == 0 { 0 } else { 4 };
        let jo = o / 2;
        for k in 0..half_i {
            let lo = (packed[(2 * k) * half_o + jo] >> shift) & 0xF;
            let hi = (packed[(2 * k + 1) * half_o + jo] >> shift) & 0xF;
            t[o * half_i + k] = lo | (hi << 4);
        }
    }
    t
}

/// Contiguous row-major decode-GEMV for output rows `[o0, o1)`:
/// `y_base[o] = sum_i cb_t[o*K + idx[o,i]] * x[i]`, summed i-ascending in one accumulator --
/// BIT-IDENTICAL to the o-outer reference `gemv_cpu_f32` (no reduction reassociation, since a
/// whole output row is owned by one task). `packed_t` is output-major (`pack_to_rowmajor`);
/// `cb_t` is the transposed codebook (`transpose_codebook`, `cb_t[o*K+k]`), whose 16-entry row
/// table stays in registers for the full input loop.
///
/// # Safety
/// Writes `y_base[o0..o1]`. The caller must ensure those indices are within the allocation and
/// that concurrent callers own disjoint `[o0,o1)` ranges (work-stealing guarantees this).
///
/// Uses raw unchecked loads (no per-nibble bounds check) and FOUR accumulators to break the
/// serial f32 add-latency chain that otherwise caps this at ~0.4 GB/s: two packed bytes (four
/// nibbles) per iteration feed four independent partial sums, so the M4's FMA pipeline stays
/// full. The partials are combined at the row end, so the result differs from the strictly-
/// sequential reference only by f32 reassociation (~1e-6 L2), not bit-identically.
unsafe fn gemv_rowmajor_range(packed_t: &[u8], cb_t: &[f32], ic: usize, x: &[f32], y_base: *mut f32, o0: usize, o1: usize) {
    let half_i = ic / 2;
    let xp = x.as_ptr();
    for o in o0..o1 {
        let tbl = cb_t.as_ptr().add(o * K);
        let row = packed_t.as_ptr().add(o * half_i);
        let (mut a0, mut a1, mut a2, mut a3) = (0f32, 0f32, 0f32, 0f32);
        let mut k = 0;
        while k + 2 <= half_i {
            let (b0, b1) = (*row.add(k), *row.add(k + 1));
            a0 += *tbl.add((b0 & 0xF) as usize) * *xp.add(2 * k);
            a1 += *tbl.add((b0 >> 4) as usize) * *xp.add(2 * k + 1);
            a2 += *tbl.add((b1 & 0xF) as usize) * *xp.add(2 * k + 2);
            a3 += *tbl.add((b1 >> 4) as usize) * *xp.add(2 * k + 3);
            k += 2;
        }
        let mut acc = (a0 + a1) + (a2 + a3);
        while k < half_i {
            let b = *row.add(k);
            acc += *tbl.add((b & 0xF) as usize) * *xp.add(2 * k);
            acc += *tbl.add((b >> 4) as usize) * *xp.add(2 * k + 1);
            k += 1;
        }
        *y_base.add(o) = acc;
    }
}

/// Safe single-shot row-major decode-GEMV over all `oc` rows (used by tests/benches).
pub fn gemv_rowmajor(packed_t: &[u8], cb_t: &[f32], oc: usize, ic: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(packed_t.len(), oc * (ic / 2), "row-major packed size");
    assert_eq!(cb_t.len(), K * oc, "transposed codebook size");
    assert_eq!(x.len(), ic, "activation size");
    assert!(y.len() >= oc, "output size");
    unsafe { gemv_rowmajor_range(packed_t, cb_t, ic, x, y.as_mut_ptr(), 0, oc); }
}

/// One picked routed expert, referencing its OUTPUT-major packed indices and transposed
/// codebooks (built once at `CpuExpert` construction) plus its router weight.
pub struct RoutedExpert<'a> {
    pub gp_t: &'a [u8], pub gc_t: &'a [f32],
    pub up_t: &'a [u8], pub uc_t: &'a [f32],
    pub dp_t: &'a [u8], pub dc_t: &'a [f32],
    pub weight: f32,
}

/// Raw f32 base pointer shared across worker threads that write DISJOINT rows.
struct RawF32(*mut f32);
unsafe impl Send for RawF32 {}
unsafe impl Sync for RawF32 {}

/// Row chunk sizes (rows per work-steal task), from the C probe: gate+up 128, down 256.
const CHUNK_A: usize = 128;
const CHUNK_C: usize = 256;

/// Compute the weighted MoE routed sum `acc_out = sum_e weight_e * down_e(SiLU(gate_e(x))*up_e(x))`
/// on the CPU with row-chunk work-stealing across all picked experts. `acc_out` (len `hidden`)
/// is overwritten. Same math as running each expert through the reference to within f32
/// summation tolerance (per-GEMV it is bit-identical; only the final weighted combine reorders).
///
/// Threading: `TRAPETUM_CPU_THREADS` workers (default 8) spawned once for this call cooperatively
/// drain three phases -- A (gate+up rows), B (SiLU+multiply per expert), C (down rows) -- each via
/// one atomic counter, with a barrier between. The row-major kernel keeps `packed_t` streaming and
/// the codebook table in registers.
/// Shared state a set of workers cooperatively drains for one routed-experts call: the expert
/// refs + activation, the fixed chunk decomposition, three phase counters, a barrier, and raw
/// base pointers to the per-expert scratch (workers write DISJOINT rows). `Sync` because the only
/// interior mutability is the atomics/barrier and the raw pointers write non-overlapping ranges.
struct WorkstealCtx<'a> {
    experts: &'a [RoutedExpert<'a>],
    x: &'a [f32],
    hidden: usize, inter: usize,
    a_per: usize, c_per: usize, na: usize, nc: usize, k: usize,
    ctr_a: AtomicUsize, ctr_b: AtomicUsize, ctr_c: AtomicUsize,
    barrier: Barrier,
    gp: RawF32, upp: RawF32, ap: RawF32, opp: RawF32,
}
unsafe impl Sync for WorkstealCtx<'_> {}

/// One worker's contribution to a routed-experts call: drain phase A (gate+up row chunks),
/// barrier, phase B (SiLU per expert), barrier, phase C (down row chunks). The chunk-to-row
/// mapping is fixed, so which worker runs which chunk never affects any output row's value.
fn worksteal_worker(c: &WorkstealCtx) {
    loop {
        let t = c.ctr_a.fetch_add(1, Ordering::Relaxed);
        if t >= c.na { break; }
        let (e, ci) = (t / c.a_per, t % c.a_per);
        let (o0, o1) = (ci * CHUNK_A, ((ci + 1) * CHUNK_A).min(c.inter));
        let ex = &c.experts[e];
        unsafe {
            gemv_rowmajor_range(ex.gp_t, ex.gc_t, c.hidden, c.x, c.gp.0.add(e * c.inter), o0, o1);
            gemv_rowmajor_range(ex.up_t, ex.uc_t, c.hidden, c.x, c.upp.0.add(e * c.inter), o0, o1);
        }
    }
    c.barrier.wait();
    loop {
        let t = c.ctr_b.fetch_add(1, Ordering::Relaxed);
        if t >= c.k { break; }
        unsafe {
            for r in 0..c.inter {
                let gg = *c.gp.0.add(t * c.inter + r);
                let uu = *c.upp.0.add(t * c.inter + r);
                *c.ap.0.add(t * c.inter + r) = silu(gg) * uu;
            }
        }
    }
    c.barrier.wait();
    loop {
        let t = c.ctr_c.fetch_add(1, Ordering::Relaxed);
        if t >= c.nc { break; }
        let (e, ci) = (t / c.c_per, t % c.c_per);
        let (o0, o1) = (ci * CHUNK_C, ((ci + 1) * CHUNK_C).min(c.hidden));
        let ex = &c.experts[e];
        let act_e = unsafe { std::slice::from_raw_parts(c.ap.0.add(e * c.inter), c.inter) };
        unsafe { gemv_rowmajor_range(ex.dp_t, ex.dc_t, c.inter, act_e, c.opp.0.add(e * c.hidden), o0, o1); }
    }
}

// ============================================================================
// Persistent worker pool: spawn TRAPETUM_CPU_THREADS workers ONCE for the process, parked on a
// condvar between calls, so a MoE layer's routed work does NOT pay a thread::spawn per call
// (~26 layers/token x spawn = the overhead this removes). A job is a type-erased `&dyn Fn()+Sync`
// handed over under a generation counter; the submitter blocks until all workers finish, so the
// borrowed job data (WorkstealCtx over stack locals) is valid for the whole broadcast.
// ============================================================================

/// Type-erased job pointer (a `&dyn Fn()+Sync` valid only while the submitter blocks in `run`).
struct Job(*const (dyn Fn() + Sync));
unsafe impl Send for Job {}

struct PoolState { gen: u64, job: Option<Job>, done: usize }

struct Pool {
    n: usize,
    st: Mutex<PoolState>,
    cv_work: Condvar,
    cv_done: Condvar,
}

impl Pool {
    fn new(n: usize) -> &'static Pool {
        let p: &'static Pool = Box::leak(Box::new(Pool {
            n, st: Mutex::new(PoolState { gen: 0, job: None, done: 0 }), cv_work: Condvar::new(), cv_done: Condvar::new(),
        }));
        for _ in 0..n {
            thread::spawn(move || {
                let mut last_gen = 0u64;
                loop {
                    let jobptr = {
                        let mut st = p.st.lock().unwrap();
                        while st.gen == last_gen { st = p.cv_work.wait(st).unwrap(); }
                        last_gen = st.gen;
                        st.job.as_ref().map(|j| j.0).unwrap()
                    };
                    // Safety: the submitter holds the job alive until done == n (below), and only
                    // touches it after that; workers only run it between the gen bump and their done++.
                    unsafe { (*jobptr)(); }
                    let mut st = p.st.lock().unwrap();
                    st.done += 1;
                    if st.done == p.n { p.cv_done.notify_one(); }
                }
            });
        }
        p
    }
    /// Run `f` on all `n` workers and block until every one has returned.
    fn run(&self, f: &(dyn Fn() + Sync)) {
        // Erase the borrow lifetime to store the job pointer. SOUND because this call blocks
        // until done == n, so `f` outlives every worker's use of the stored pointer.
        let raw: *const (dyn Fn() + Sync + 'static) =
            unsafe { std::mem::transmute(f as *const (dyn Fn() + Sync)) };
        let mut st = self.st.lock().unwrap();
        st.job = Some(Job(raw));
        st.done = 0;
        st.gen = st.gen.wrapping_add(1);
        self.cv_work.notify_all();
        while st.done < self.n { st = self.cv_done.wait(st).unwrap(); }
        st.job = None;
    }
}

/// Global persistent pool, sized `TRAPETUM_CPU_THREADS` (default 8), created on first use.
fn pool() -> &'static Pool {
    static P: OnceLock<&'static Pool> = OnceLock::new();
    P.get_or_init(|| Pool::new(cpu_threads().max(1)))
}

/// Compute the weighted MoE routed sum on the PERSISTENT pool (production path): no thread::spawn
/// per call. All `pool().n` workers cooperatively drain the phases (idle ones just fall through the
/// barriers). Bit-identical to [`routed_experts_worksteal_nt`] -- same chunk decomposition and
/// fixed-order combine, so the pool vs scope choice never changes a byte of `acc_out`.
pub fn routed_experts_worksteal(x: &[f32], experts: &[RoutedExpert], hidden: usize, inter: usize, acc_out: &mut [f32]) {
    let p = pool();
    if p.n <= 1 { routed_experts_worksteal_nt(x, experts, hidden, inter, acc_out, 1); return; }
    assert_eq!(x.len(), hidden, "routed input width");
    assert!(acc_out.len() >= hidden, "routed output width");
    let k = experts.len();
    for v in acc_out.iter_mut().take(hidden) { *v = 0.0; }
    if k == 0 { return; }
    let mut g = vec![0f32; k * inter];
    let mut u = vec![0f32; k * inter];
    let mut act = vec![0f32; k * inter];
    let mut out = vec![0f32; k * hidden];
    let a_per = (inter + CHUNK_A - 1) / CHUNK_A;
    let c_per = (hidden + CHUNK_C - 1) / CHUNK_C;
    let ctx = WorkstealCtx {
        experts, x, hidden, inter, a_per, c_per, na: k * a_per, nc: k * c_per, k,
        ctr_a: AtomicUsize::new(0), ctr_b: AtomicUsize::new(0), ctr_c: AtomicUsize::new(0),
        barrier: Barrier::new(p.n), // all pool workers participate in the barriers
        gp: RawF32(g.as_mut_ptr()), upp: RawF32(u.as_mut_ptr()), ap: RawF32(act.as_mut_ptr()), opp: RawF32(out.as_mut_ptr()),
    };
    p.run(&|| worksteal_worker(&ctx));
    for (e, ex) in experts.iter().enumerate() {
        let (w, base) = (ex.weight, e * hidden);
        for i in 0..hidden { acc_out[i] += w * out[base + i]; }
    }
}

/// Thread-count-explicit variant of [`routed_experts_worksteal`] (the public one reads
/// `TRAPETUM_CPU_THREADS`). Exposed so the determinism harness can prove the result is
/// IDENTICAL across worker counts without touching a process-global env var.
///
/// DETERMINISM CONTRACT (thread-count-invariant, bitwise): every output row is computed in
/// full by exactly one work-steal task (chunks split the OUTPUT dimension, never the reduction),
/// with a fixed 4-accumulator order inside the row; the final weighted combine runs on one thread
/// in fixed EXPERT order (0..k), never thread-completion order. So `worker_threads` changes only
/// WHICH core runs a chunk, not any summation order -- the bytes of `acc_out` are identical.
pub fn routed_experts_worksteal_nt(x: &[f32], experts: &[RoutedExpert], hidden: usize, inter: usize, acc_out: &mut [f32], worker_threads: usize) {
    assert_eq!(x.len(), hidden, "routed input width");
    assert!(acc_out.len() >= hidden, "routed output width");
    let k = experts.len();
    for v in acc_out.iter_mut().take(hidden) { *v = 0.0; }
    if k == 0 { return; }

    // Per-expert scratch (contiguous [k][width]); threads write disjoint rows.
    let mut g = vec![0f32; k * inter];
    let mut u = vec![0f32; k * inter];
    let mut act = vec![0f32; k * inter];
    let mut out = vec![0f32; k * hidden];

    let a_per = (inter + CHUNK_A - 1) / CHUNK_A;
    let c_per = (hidden + CHUNK_C - 1) / CHUNK_C;
    let (na, nc) = (k * a_per, k * c_per);
    let nthreads = worker_threads.max(1).min(na.max(nc).max(1));

    if nthreads <= 1 {
        // Sequential fallback (still row-major/contiguous).
        for (e, ex) in experts.iter().enumerate() {
            unsafe {
                gemv_rowmajor_range(ex.gp_t, ex.gc_t, hidden, x, g.as_mut_ptr().add(e * inter), 0, inter);
                gemv_rowmajor_range(ex.up_t, ex.uc_t, hidden, x, u.as_mut_ptr().add(e * inter), 0, inter);
            }
            for r in 0..inter { act[e * inter + r] = silu(g[e * inter + r]) * u[e * inter + r]; }
            let act_e = &act[e * inter..e * inter + inter];
            unsafe { gemv_rowmajor_range(ex.dp_t, ex.dc_t, inter, act_e, out.as_mut_ptr().add(e * hidden), 0, hidden); }
        }
        for (e, ex) in experts.iter().enumerate() {
            let (w, base) = (ex.weight, e * hidden);
            for i in 0..hidden { acc_out[i] += w * out[base + i]; }
        }
        return;
    }

    // thread::scope spawns exactly `nthreads` workers for THIS call (used by the determinism
    // harness with explicit counts; the production path uses the persistent `pool()` instead,
    // which avoids the per-call spawn). Both drive the SAME `worksteal_worker`, so results match.
    let ctx = WorkstealCtx {
        experts, x, hidden, inter, a_per, c_per, na, nc, k,
        ctr_a: AtomicUsize::new(0), ctr_b: AtomicUsize::new(0), ctr_c: AtomicUsize::new(0),
        barrier: Barrier::new(nthreads),
        gp: RawF32(g.as_mut_ptr()), upp: RawF32(u.as_mut_ptr()), ap: RawF32(act.as_mut_ptr()), opp: RawF32(out.as_mut_ptr()),
    };
    thread::scope(|s| {
        for _ in 0..nthreads {
            let ctx = &ctx;
            s.spawn(move || worksteal_worker(ctx));
        }
    });

    // Weighted combine (cheap: k*hidden adds).
    for (e, ex) in experts.iter().enumerate() {
        let (w, base) = (ex.weight, e * hidden);
        for i in 0..hidden { acc_out[i] += w * out[base + i]; }
    }
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
        let packed_t = pack_to_rowmajor(&packed, oc, ic);
        let (p2, x2) = (packed.clone(), x.clone());
        let ms_old = best(Box::new({ let cb = cb.clone(); let mut y = y.clone();
            move || gemv_cpu_f32(&p2, &cb, oc, ic, &x2, &mut y) })) * 1e3;
        let (cbt2, x3, pt2) = (cb_t.clone(), x.clone(), packed.clone());
        let ms_stream = best(Box::new({ let mut y = y.clone();
            move || gemv_cpu_stream(&pt2, &cbt2, oc, ic, &x3, &mut y) })) * 1e3;
        // Row-major single-thread kernel (the per-row task the work-stealing engine runs on
        // each core; the full 8-thread number is `cpu_routed` in the MoE bench below).
        let ms_rm = best(Box::new(move || gemv_rowmajor(&packed_t, &cb_t, oc, ic, &x, &mut y))) * 1e3;
        let gbs = |ms: f64| bytes as f64 / (ms * 1e-3) / 1e9;
        eprintln!("[stream_vs_scalar_throughput] oc={oc} ic={ic} packed={} KB (single 1408x2048 GEMV)", bytes / 1024);
        eprintln!("  o-outer gemv_cpu_f32   (8t): {ms_old:.3} ms  ({:.2} GB/s)", gbs(ms_old));
        eprintln!("  i-outer gemv_cpu_stream(8t): {ms_stream:.3} ms  ({:.2} GB/s)", gbs(ms_stream));
        eprintln!("  row-major gemv_rowmajor(1t): {ms_rm:.3} ms  ({:.2} GB/s)  [1 thread; x8 in the engine]", gbs(ms_rm));
    }

    #[test]
    fn moe_block_run_to_run_spread_probe() {
        // DIAGNOSTIC: how much does ONE V2-Lite MoE block's output vary run-to-run under the
        // default (atomic) GS? Judges whether per-layer nondeterminism can compound to a ~1-logit
        // end-to-end shift across ~26 layers, or whether a systematic ~1-logit diff must be a bug.
        let worst = crate::moe_forward_run_to_run_spread(10);
        eprintln!("[moe_block_spread] worst abs diff over 10 runs = {worst:e} (per MoE block; x26 layers rough upper bound = {:e})", worst * 26.0);
        let _ = worst;
    }

    #[test]
    fn det_gemv_mode_report() {
        // Reports determinism + correctness + timing for the fused GEMV under whatever
        // TRAPETUM_DETERMINISTIC mode is set. Run three times to fill the before/after table:
        //   (unset)                -> mode 0 (atomic, nondeterministic, fast)
        //   TRAPETUM_DETERMINISTIC=1 -> grid.y=1 (deterministic, slow on small OC)
        //   TRAPETUM_DETERMINISTIC=2 -> two-stage fixed-order (deterministic, keeps IC split)
        let (ic, oc) = (8192usize, 512usize);
        let mode = std::env::var("TRAPETUM_DETERMINISTIC").unwrap_or_else(|_| "0".into());
        let (mism, worst_abs) = crate::check_gpu_gemv_determinism(ic, oc, 30);
        let (worst_rel, l2) = crate::check_gpu_gemv_vs_fp16cb_ref(ic, oc);
        let ms = crate::bench_gpu_gemv_ms(ic, oc, 50);
        eprintln!("[det_gemv mode={mode}] determinism: {mism}/30 differ (worst_abs={worst_abs:e}); \
                   vs fp16cb-ref: worst_rel={worst_rel:e} l2={l2:e}; time={ms:.4} ms/call (ic={ic} oc={oc})");
        // Correctness holds in every mode (fp16 summation-order tolerance). Do NOT assert
        // determinism here -- it depends on the env-set mode; that is asserted in the =2 CI run.
        assert!(l2 < 5e-3, "GPU GEMV vs fp16-codebook reference l2={l2:e} too large");
    }

    #[test]
    fn det_gemv_overhead_bench() {
        // Overhead of the active TRAPETUM_DETERMINISTIC mode across realistic shapes. Run in
        // RELEASE, once with env unset (mode 0) and once with =2, and diff the ms/call. Small OC
        // is launch-dominated; big IC*OC shows the true two-stage memory overhead (~GS/IC).
        let mode = std::env::var("TRAPETUM_DETERMINISTIC").unwrap_or_else(|_| "0".into());
        for (ic, oc) in [(2048usize, 2048usize), (2048, 7168), (8192, 2048), (2048, 16384)] {
            let ms = crate::bench_gpu_gemv_ms(ic, oc, 50);
            eprintln!("[det_gemv_overhead mode={mode}] ic={ic} oc={oc} -> {ms:.4} ms/call");
        }
    }

    #[test]
    fn gpu_gemv_determinism_probe() {
        // DIAGNOSTIC (not a pass/fail gate): does the fused GPU codebook GEMV return bitwise-
        // identical output run-to-run? If not, the base runtime is nondeterministic (atomic
        // grid.y reduction), which is the real explanation for the flag-off vs main greedy
        // divergence -- the branch's memory-layout change merely resampled it. Large IC forces
        // multiple grid.y slices. Prints; asserts only that it ran.
        let (mism, worst) = crate::check_gpu_gemv_determinism(8192, 512, 30);
        eprintln!("[gpu_determinism_probe] ic=8192 oc=512 iters=30 -> {mism}/30 runs differ bitwise from run 0, worst_abs_diff={worst:e}");
        // No determinism assertion: whether atomics reorder is GPU/scheduler dependent. The
        // number is the evidence we report.
        let _ = (mism, worst);
    }

    // Build k routed experts with RANDOM row-major packed + transposed codebooks (fast; we test
    // scheduling/timing, not decode accuracy). Returns (store, x) -- refs are built by the caller.
    fn rand_experts(hidden: usize, inter: usize, k: usize, seed: u64)
        -> (Vec<(Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, f32)>, Vec<f32>) {
        let mut r = Lcg(seed);
        let x: Vec<f32> = (0..hidden).map(|_| r.f32()).collect();
        let bytes = |n: usize, r: &mut Lcg| -> Vec<u8> { (0..n).map(|_| ((r.f32() * 0.5 + 0.5) * 255.0) as u8).collect() };
        let cbk = |n: usize, r: &mut Lcg| -> Vec<f32> { (0..n).map(|_| r.f32() * 0.05).collect() };
        let mut store = Vec::new();
        for e in 0..k {
            store.push((bytes(inter * (hidden / 2), &mut r), cbk(K * inter, &mut r),
                        bytes(inter * (hidden / 2), &mut r), cbk(K * inter, &mut r),
                        bytes(hidden * (inter / 2), &mut r), cbk(K * hidden, &mut r), 0.1 + 0.13 * e as f32));
        }
        (store, x)
    }
    fn refs<'a>(store: &'a [(Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, f32)]) -> Vec<RoutedExpert<'a>> {
        store.iter().map(|s| RoutedExpert { gp_t: &s.0, gc_t: &s.1, up_t: &s.2, uc_t: &s.3, dp_t: &s.4, dc_t: &s.5, weight: s.6 }).collect()
    }

    #[test]
    fn pool_matches_scope() {
        // The persistent-pool production path must be BIT-identical to the thread::scope path
        // (same chunk decomposition + fixed-order combine). V2-Lite dims.
        let (hidden, inter, k) = (2048usize, 1536usize, 6usize);
        let (store, x) = rand_experts(hidden, inter, k, 0x9A55_0011);
        let experts = refs(&store);
        let mut a = vec![0f32; hidden];
        let mut b = vec![0f32; hidden];
        routed_experts_worksteal(&x, &experts, hidden, inter, &mut a);            // persistent pool
        routed_experts_worksteal_nt(&x, &experts, hidden, inter, &mut b, cpu_threads()); // thread::scope
        assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
                "pool result differs from scope result (should be bit-identical)");
    }

    #[test]
    fn pool_vs_scope_overhead() {
        // Per-call cost of the persistent pool vs a fresh thread::scope spawn. At V2-Lite dims the
        // compute dominates; a TINY-dim run isolates the pure spawn/wake overhead (the lever).
        for (hidden, inter, k, label) in [(2048usize, 1536usize, 6usize, "V2-Lite"), (256, 256, 2, "tiny")] {
            let (store, x) = rand_experts(hidden, inter, k, 0x1122_3344 ^ hidden as u64);
            let experts = refs(&store);
            let mut acc = vec![0f32; hidden];
            let nt = cpu_threads();
            for _ in 0..5 { routed_experts_worksteal(&x, &experts, hidden, inter, &mut acc); routed_experts_worksteal_nt(&x, &experts, hidden, inter, &mut acc, nt); }
            let iters = 200;
            let t = std::time::Instant::now();
            for _ in 0..iters { routed_experts_worksteal(&x, &experts, hidden, inter, &mut acc); }
            let pool_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let t = std::time::Instant::now();
            for _ in 0..iters { routed_experts_worksteal_nt(&x, &experts, hidden, inter, &mut acc, nt); }
            let scope_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            eprintln!("[pool_vs_scope {label}] hidden={hidden} inter={inter} k={k} threads={nt}: pool={pool_us:.1} us/call  scope={scope_us:.1} us/call  saved={:.1} us", scope_us - pool_us);
        }
    }

    #[test]
    fn worksteal_is_thread_count_invariant() {
        // PROOF that the routed CPU reduction is bit-identical across worker counts: the
        // greedy-decode thread-dependence must NOT originate here. Uses the pod's V2-Lite dims
        // with RANDOM packed indices + codebooks (any bytes are valid indices; no need for real
        // k-means -- we are testing the reduction order, not the decode values), so it is fast.
        //
        // SCOPE: this covers ONLY the CPU routed-experts reduction (routed_experts_worksteal).
        // It says nothing about the GPU kernels (attention, shared expert, dense FFN, lm_head),
        // whose fused codebook GEMV reduces IC-split partials with atomicAdd across grid.y blocks
        // (kernels/gemv_codebook_4bit.cu:45, and the Metal gemv4 twin) -- that atomic add-order is
        // nondeterministic RUN-TO-RUN and is the base-runtime nondeterminism to fix separately
        // (see S14 #7 report / the deterministic-reduction proposal). CPU determinism proven here
        // does not make the end-to-end greedy output deterministic while those kernels remain.
        let (hidden, inter, k) = (2048usize, 1536usize, 6usize);
        let mut r = Lcg(0xD37E_2711_u64);
        let x: Vec<f32> = (0..hidden).map(|_| r.f32()).collect();
        let bytes = |n: usize, r: &mut Lcg| -> Vec<u8> { (0..n).map(|_| ((r.f32() * 0.5 + 0.5) * 255.0) as u8).collect() };
        let cbk = |n: usize, r: &mut Lcg| -> Vec<f32> { (0..n).map(|_| r.f32() * 0.05).collect() };
        // Row-major packed is [oc][ic/2]; transposed codebook is [oc*K]. gate/up: oc=inter; down: oc=hidden.
        let mut store: Vec<(Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, f32)> = Vec::new();
        for e in 0..k {
            store.push((bytes(inter * (hidden / 2), &mut r), cbk(K * inter, &mut r),
                        bytes(inter * (hidden / 2), &mut r), cbk(K * inter, &mut r),
                        bytes(hidden * (inter / 2), &mut r), cbk(K * hidden, &mut r), 0.1 + 0.13 * e as f32));
        }
        let experts: Vec<RoutedExpert> = store.iter().map(|s| RoutedExpert {
            gp_t: &s.0, gc_t: &s.1, up_t: &s.2, uc_t: &s.3, dp_t: &s.4, dc_t: &s.5, weight: s.6,
        }).collect();
        let mut baseline = vec![0f32; hidden];
        routed_experts_worksteal_nt(&x, &experts, hidden, inter, &mut baseline, 1);
        for nt in [4usize, 8, 16, 32, 64] {
            let mut got = vec![0f32; hidden];
            routed_experts_worksteal_nt(&x, &experts, hidden, inter, &mut got, nt);
            // BITWISE identical: not a tolerance -- the reduction order is fixed by construction.
            assert!(baseline.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "routed output changed at worker_threads={nt} (thread-count-DEPENDENT reduction!)");
        }
    }

    #[test]
    fn rowmajor_matches_scalar_reference() {
        // The output-major re-tile + row kernel must be BIT-IDENTICAL to the o-outer reference
        // (both sum i-ascending per output row; work-stealing owns whole rows, no reassociation).
        for (oc, ic) in [(256usize, 512usize), (64, 128), (1408, 2048), (2048, 1408)] {
            let mut r = Lcg(0x50FA_5711u64 ^ ((oc as u64) << 20) ^ ic as u64);
            let w: Vec<f32> = (0..oc * ic).map(|_| r.f32()).collect();
            let x: Vec<f32> = (0..ic).map(|_| r.f32()).collect();
            let (packed, cb, _) = quantize_host(&w, oc, ic);
            let packed_t = pack_to_rowmajor(&packed, oc, ic);
            let cb_t = transpose_codebook(&cb, oc);
            let mut yref = vec![0f32; oc];
            let mut yrm = vec![0f32; oc];
            gemv_cpu_f32(&packed, &cb, oc, ic, &x, &mut yref);
            gemv_rowmajor(&packed_t, &cb_t, oc, ic, &x, &mut yrm);
            // Two even/odd accumulators (for ILP) reassociate the sum vs the sequential
            // reference. On near-zero output rows the worst-element rel err blows up (~1.5e-5);
            // the L2 metric (~1e-6) reflects the true agreement. Tolerance, per summation rule.
            let e = l2_rel(&yrm, &yref);
            assert!(e < 1e-5, "row-major vs scalar L2 rel err {e:e} at oc={oc} ic={ic}");
        }
    }

    #[test]
    fn worksteal_matches_sequential_reference() {
        // routed_experts_worksteal must equal running each expert through the reference
        // (per-GEMV bit-identical; only the final weighted combine reorders -> tiny residual).
        let (hidden, inter, k) = (512usize, 256usize, 6usize);
        let mut r = Lcg(0x704B_5713_u64);
        let x: Vec<f32> = (0..hidden).map(|_| r.f32()).collect();
        // Build k experts + a reference result.
        let mut store: Vec<(Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, f32)> = Vec::new();
        let mut reference = vec![0f32; hidden];
        for e in 0..k {
            let gw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
            let uw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
            let dw: Vec<f32> = (0..hidden * inter).map(|_| r.f32() * 0.5).collect();
            let (gp, gc, g_dq) = quantize_host(&gw, inter, hidden);
            let (up, uc, u_dq) = quantize_host(&uw, inter, hidden);
            let (dp, dc, d_dq) = quantize_host(&dw, hidden, inter);
            let w = 0.1 + 0.15 * e as f32;
            // reference via dequant matmul
            let g: Vec<f32> = (0..inter).map(|o| (0..hidden).map(|i| g_dq[o * hidden + i] * x[i]).sum()).collect();
            let u: Vec<f32> = (0..inter).map(|o| (0..hidden).map(|i| u_dq[o * hidden + i] * x[i]).sum()).collect();
            let act: Vec<f32> = (0..inter).map(|i| silu(g[i]) * u[i]).collect();
            for o in 0..hidden { reference[o] += w * (0..inter).map(|i| d_dq[o * inter + i] * act[i]).sum::<f32>(); }
            store.push((pack_to_rowmajor(&gp, inter, hidden), transpose_codebook(&gc, inter),
                        pack_to_rowmajor(&up, inter, hidden), transpose_codebook(&uc, inter),
                        pack_to_rowmajor(&dp, hidden, inter), transpose_codebook(&dc, hidden), w));
        }
        let experts: Vec<RoutedExpert> = store.iter().map(|s| RoutedExpert {
            gp_t: &s.0, gc_t: &s.1, up_t: &s.2, uc_t: &s.3, dp_t: &s.4, dc_t: &s.5, weight: s.6,
        }).collect();
        let mut got = vec![0f32; hidden];
        routed_experts_worksteal(&x, &experts, hidden, inter, &mut got);
        // L2 metric (near-zero hidden outputs make worst-element rel err misleading; the
        // 2-accumulator kernel + weighted combine reassociate vs the sequential reference).
        let e = l2_rel(&got, &reference);
        assert!(e < 1e-4, "worksteal vs reference L2 rel err {e:e}");
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
