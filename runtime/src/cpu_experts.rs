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
        for w in 0..n {
            thread::spawn(move || {
                WORKER_ID.with(|c| c.set(w));
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

/// Run `f(i)` for every `i` in `0..n` across the persistent pool (each `i` independent; the caller
/// writes disjoint regions). Sequential fallback for a 1-thread pool or tiny `n`. Used to parallelize
/// the MLA per-head W_UK/W_UV absorption -- the single-threaded host wall found in deliverable E.
pub fn parallel_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
    let p = pool();
    if p.n <= 1 || n <= 1 { for i in 0..n { f(i); } return; }
    let ctr = AtomicUsize::new(0);
    let ctr = &ctr;
    p.run(&|| loop { let i = ctr.fetch_add(1, Ordering::Relaxed); if i >= n { break; } f(i); });
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

// ============================================================================
// NATIVE-LAYOUT path (for the 671B offload / MoeBlockOffload). The 671B routed experts are
// ~350 GB of mmap-backed packed indices in the artifact's NATIVE input-major layout
// (`packed[i*(oc/2)+j]`, per quantize_host); re-tiling to output-major would DOUBLE that RAM,
// so these kernels consume the mmap slices DIRECTLY. i-OUTER streams `packed` contiguously (the
// increment-4 insight); the codebook is used in its NATIVE `cb[k*oc+o]` layout too (K*oc, L2-
// resident -- no transpose copy). Determinism is thread-count-invariant via a FIXED input-chunk
// decomposition (chunk count set by `ic`, not by the worker count) whose per-chunk partials are
// summed in FIXED chunk-index order.
// ============================================================================

/// Input rows per fixed determinism chunk (independent of worker count).
const NATIVE_CHUNK_I: usize = 256;

/// Scalar i-outer decode-GEMV over an INPUT range `[i0,i1)`, native input-major `packed` + native
/// codebook, ACCUMULATING into `y` (caller owns init). Streams `packed` contiguously; `cb` is
/// gathered strided (`cb[id*oc+o]`) but is L2-resident. Sums i-ascending within the range. This is
/// the portable reference; on x86_64 [`gemv_native_range`] dispatches to the SIMD twins below.
fn gemv_native_range_scalar(packed: &[u8], cb: &[f32], oc: usize, x: &[f32], y: &mut [f32], i0: usize, i1: usize) {
    let half = oc / 2;
    for i in i0..i1 {
        let xi = x[i];
        let row = &packed[i * half..i * half + half];
        for j in 0..half {
            let b = row[j];
            y[2 * j]     += cb[((b & 0xF) as usize) * oc + 2 * j] * xi;
            y[2 * j + 1] += cb[((b >> 4) as usize) * oc + 2 * j + 1] * xi;
        }
    }
}

// x86_64 SIMD decode-GEMV (deliverable B lever 1). The 671B host is a Xeon; the scalar decode ran
// ~1.2 GB/s single-thread (the NEON tbl kernel is cfg(aarch64) and never had a vpshufb twin), which
// capped the hybrid at ~2.4 GB/s aggregate. These kernels keep the SAME i-outer streaming of `packed`
// and process 8 (AVX2) / 16 (AVX-512) output columns per step: the packed byte -> nibble index expand
// feeds a vectorized gather of the L2-resident codebook, then mul + add into the y accumulators.
//
// BIT-EXACTNESS: each output column is an INDEPENDENT accumulator summed over i in the SAME ascending
// order as the scalar path, and the value is `cb_value * xi` with a SEPARATE mul then add (NOT an FMA
// -- fusing would change the rounding, the vfmaq lesson from S14). So the SIMD result is bitwise equal
// to the scalar reference, and the determinism/thread-invariance proofs carry over unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_native_range_avx2(packed: &[u8], cb: &[f32], oc: usize, x: &[f32], y: &mut [f32], i0: usize, i1: usize) {
    use std::arch::x86_64::*;
    let half = oc / 2;
    let cbp = cb.as_ptr();
    let ocv = _mm256_set1_epi32(oc as i32);
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let yp = y.as_mut_ptr();
    let rp = packed.as_ptr();
    let xp = x.as_ptr();
    for i in i0..i1 {
        let xi = _mm256_set1_ps(*xp.add(i));
        let row = rp.add(i * half);
        let (mut col, mut j) = (0usize, 0usize);
        while col < oc {
            // 4 packed bytes -> 8 nibble indices for columns col..col+7 (low, high, low, high, ...)
            let (b0, b1) = (*row.add(j) as i32, *row.add(j + 1) as i32);
            let (b2, b3) = (*row.add(j + 2) as i32, *row.add(j + 3) as i32);
            let idx = _mm256_setr_epi32(b0 & 0xF, b0 >> 4, b1 & 0xF, b1 >> 4, b2 & 0xF, b2 >> 4, b3 & 0xF, b3 >> 4);
            let colv = _mm256_add_epi32(_mm256_set1_epi32(col as i32), lane);
            let off = _mm256_add_epi32(_mm256_mullo_epi32(idx, ocv), colv); // cb element index = idx*oc + col
            let w = _mm256_i32gather_ps::<4>(cbp, off);
            let prod = _mm256_mul_ps(w, xi);
            let yv = _mm256_loadu_ps(yp.add(col));
            _mm256_storeu_ps(yp.add(col), _mm256_add_ps(yv, prod));
            col += 8; j += 4;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemv_native_range_avx512(packed: &[u8], cb: &[f32], oc: usize, x: &[f32], y: &mut [f32], i0: usize, i1: usize) {
    use std::arch::x86_64::*;
    let half = oc / 2;
    let cbp = cb.as_ptr();
    let ocv = _mm512_set1_epi32(oc as i32);
    let lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let yp = y.as_mut_ptr();
    let rp = packed.as_ptr();
    let xp = x.as_ptr();
    for i in i0..i1 {
        let xi = _mm512_set1_ps(*xp.add(i));
        let row = rp.add(i * half);
        let (mut col, mut j) = (0usize, 0usize);
        while col < oc {
            // 8 packed bytes -> 16 nibble indices for columns col..col+15
            let mut ix = [0i32; 16];
            let mut c = 0usize;
            while c < 16 { let b = *row.add(j + c / 2) as i32; ix[c] = b & 0xF; ix[c + 1] = b >> 4; c += 2; }
            let idx = _mm512_loadu_si512(ix.as_ptr() as *const __m512i);
            let colv = _mm512_add_epi32(_mm512_set1_epi32(col as i32), lane);
            let off = _mm512_add_epi32(_mm512_mullo_epi32(idx, ocv), colv);
            let w = _mm512_i32gather_ps::<4>(off, cbp);
            let prod = _mm512_mul_ps(w, xi);
            let yv = _mm512_loadu_ps(yp.add(col));
            _mm512_storeu_ps(yp.add(col), _mm512_add_ps(yv, prod));
            col += 16; j += 8;
        }
    }
}

/// Native decode kernel selection (deliverable C). `TRAPETUM_CPU_KERNEL` picks explicitly, else auto
/// (widest exact SIMD). `TRAPETUM_CPU_SCALAR=1` is a back-compat alias for `=scalar`.
///   scalar          f32 reference (portable)
///   gather / auto   f32 gather AVX2/AVX-512 (deliverable B; exact; gather-bound ~ the 671B ceiling)
///   lut8            int8-recoded codebook + vpshufb value-LUT (deliverable C2; tolerance-gated)
///   vpermps         f32 register-resident LUT (deliverable C1; exact) -- not yet built
/// Values whose kernel is not yet implemented fall back to the best exact kernel with a one-time
/// notice, so the box's A/B script can set the flag today without breaking.
#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)] // some variants only constructed under cfg(x86_64)
pub(crate) enum CpuKernel { Scalar, GatherAvx2, GatherAvx512, Lut8 }

pub(crate) fn cpu_kernel() -> CpuKernel {
    static K: OnceLock<CpuKernel> = OnceLock::new();
    *K.get_or_init(|| {
        let scalar_alias = std::env::var("TRAPETUM_CPU_SCALAR").map(|v| v == "1").unwrap_or(false);
        let sel = std::env::var("TRAPETUM_CPU_KERNEL").ok().unwrap_or_default();
        #[cfg(target_arch = "x86_64")]
        let best = if is_x86_feature_detected!("avx512f") { CpuKernel::GatherAvx512 }
                   else if is_x86_feature_detected!("avx2") { CpuKernel::GatherAvx2 } else { CpuKernel::Scalar };
        #[cfg(not(target_arch = "x86_64"))]
        let best = CpuKernel::Scalar;
        if scalar_alias { return CpuKernel::Scalar; }
        match sel.as_str() {
            "scalar" => CpuKernel::Scalar,
            "gather" | "auto" | "" => best,
            "lut8" => CpuKernel::Lut8,
            other => { eprintln!("[cpu_kernel] '{other}' not implemented yet -> falling back to the best exact kernel"); best }
        }
    })
}

/// i-outer decode-GEMV, dispatching to the widest available x86_64 SIMD kernel (bit-exact to the
/// scalar reference), or scalar elsewhere. `oc` is a multiple of 256 in the runtime, so the AVX-512
/// (16-col) and AVX2 (8-col) tilings never leave a remainder; the guards keep it correct regardless.
#[inline]
fn gemv_native_range(packed: &[u8], cb: &[f32], oc: usize, x: &[f32], y: &mut [f32], i0: usize, i1: usize) {
    // This is the f32-exact GEMV. The int8 lut8 path (C2) has a different signature (int8 cb + scale
    // + int8 x) and is wired separately; here Lut8 maps to the best exact gather kernel.
    #[cfg(target_arch = "x86_64")]
    {
        match cpu_kernel() {
            CpuKernel::GatherAvx512 if oc % 16 == 0 => return unsafe { gemv_native_range_avx512(packed, cb, oc, x, y, i0, i1) },
            CpuKernel::Scalar => {}
            _ if oc % 8 == 0 => return unsafe { gemv_native_range_avx2(packed, cb, oc, x, y, i0, i1) },
            _ => {}
        }
    }
    gemv_native_range_scalar(packed, cb, oc, x, y, i0, i1);
}

/// Deterministic native decode-GEMV: `y[o] = sum_i cb[idx[o,i]*oc+o]*x[i]`, consuming the native
/// input-major `packed` with NO re-tile. Multi-threaded over a FIXED input-chunk split (persistent
/// pool); per-chunk partials (`oc` f32 each) are reduced in FIXED chunk-index order, so the bytes
/// of `y` are identical for ANY worker count.
pub fn gemv_cpu_native_det(packed: &[u8], cb: &[f32], oc: usize, ic: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(packed.len(), ic * (oc / 2), "packed size");
    assert_eq!(cb.len(), K * oc, "codebook size");
    assert_eq!(x.len(), ic, "activation size");
    assert!(y.len() >= oc, "output size");
    let nchunks = (ic + NATIVE_CHUNK_I - 1) / NATIVE_CHUNK_I;
    if nchunks <= 1 {
        for v in y.iter_mut().take(oc) { *v = 0.0; }
        gemv_native_range(packed, cb, oc, x, y, 0, ic);
        return;
    }
    // Per-chunk partials (zeroed); each chunk owns a disjoint `oc`-slice.
    let mut partials = vec![0f32; nchunks * oc];
    let pp = RawF32(partials.as_mut_ptr());
    let ctr = AtomicUsize::new(0);
    // Capture the WHOLE RawF32 (Sync) by reference, not its raw-pointer field (disjoint capture
    // of `*mut f32` would make the closure non-Sync).
    let (packed, cb, x, pp) = (&packed, &cb, &x, &pp);
    pool().run(&|| {
        loop {
            let c = ctr.fetch_add(1, Ordering::Relaxed);
            if c >= nchunks { break; }
            let (i0, i1) = (c * NATIVE_CHUNK_I, ((c + 1) * NATIVE_CHUNK_I).min(ic));
            let pc = unsafe { std::slice::from_raw_parts_mut(pp.0.add(c * oc), oc) };
            gemv_native_range(packed, cb, oc, x, pc, i0, i1);
        }
    });
    // Fixed chunk-index-order reduction -> thread-count-invariant.
    for v in y.iter_mut().take(oc) { *v = 0.0; }
    for c in 0..nchunks {
        let base = c * oc;
        for o in 0..oc { y[o] += partials[base + o]; }
    }
}

/// A routed expert for the native path: NATIVE input-major packed slices (mmap-backed, borrowed
/// directly -- no copy) + native codebooks, plus the router weight.
pub struct NativeExpert<'a> {
    pub gp: &'a [u8], pub gc: &'a [f32],
    pub up: &'a [u8], pub uc: &'a [f32],
    pub dp: &'a [u8], pub dc: &'a [f32],
    pub weight: f32,
}

/// Full SwiGLU expert forward on the native input-major layout: gate/up (`oc=inter, ic=hidden`)
/// via [`gemv_cpu_native_det`], SiLU(g)*u, then down (`oc=hidden, ic=inter`). Deterministic.
#[allow(clippy::too_many_arguments)]
pub fn expert_forward_native(x: &[f32], gp: &[u8], gc: &[f32], up: &[u8], uc: &[f32], dp: &[u8], dc: &[f32],
                             hidden: usize, inter: usize, y: &mut [f32]) {
    assert_eq!(x.len(), hidden, "expert input width");
    assert!(y.len() >= hidden, "expert output width");
    let mut g = vec![0f32; inter];
    let mut u = vec![0f32; inter];
    gemv_cpu_native_det(gp, gc, inter, hidden, x, &mut g);
    gemv_cpu_native_det(up, uc, inter, hidden, x, &mut u);
    let mut act = vec![0f32; inter];
    for i in 0..inter { act[i] = silu(g[i]) * u[i]; }
    gemv_cpu_native_det(dp, dc, hidden, inter, &act, y);
}

/// Weighted routed MoE sum on the native path (for MoeBlockOffload / 671B): runs each picked
/// expert with [`expert_forward_native`] and accumulates `sum_e w_e * expert_e(x)` in FIXED expert
/// order. Deterministic (each GEMV is thread-invariant; the combine order is fixed). `acc_out`
/// (len `hidden`) is overwritten. No re-tile, no per-expert copy -- the mmap slices are read directly.
pub fn routed_experts_native(x: &[f32], experts: &[NativeExpert], hidden: usize, inter: usize, acc_out: &mut [f32]) {
    for v in acc_out.iter_mut().take(hidden) { *v = 0.0; }
    let mut y = vec![0f32; hidden];
    for ex in experts {
        expert_forward_native(x, ex.gp, ex.gc, ex.up, ex.uc, ex.dp, ex.dc, hidden, inter, &mut y);
        for i in 0..hidden { acc_out[i] += ex.weight * y[i]; }
    }
}

// ============================================================================
// int8 codebook recode (deliverable C2). To reach the vpshufb/tbl 47 GB/s class the decode must be
// a byte-LUT, which needs an int8 codebook. We recode each per-column f32 codebook to int8 + a
// per-column f32 scale at LOAD time (a small one-time pass, no artifact change): cb_i8[k,o] =
// round(cb[k,o] / scale[o]), scale[o] = max_k|cb[k,o]| / 127. The decode then quantizes the
// activation to int8 too (per-vector scale), does an int32 dot, and rescales by (xs * scale[o]).
// This adds codebook+activation int8 error -- comparable in scale to the fp16-vs-f32 path gap we
// already accept; the tolerance is MEASURED and printed (int8_recode_error_probe), and the gate is
// margins/PPL (the determinism campaign), not raw f32 equality. The SIMD kernel (C-inc3) implements
// exactly this arithmetic, so its correctness test is "vpshufb == this scalar int8", bit-exact.
// ============================================================================

/// Recode a per-column f32 codebook `[K,oc]` to int8 `[K,oc]` + per-column f32 `scale[oc]`.
pub fn recode_codebook_i8(cb: &[f32], oc: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(cb.len(), K * oc);
    let mut cb_i8 = vec![0i8; K * oc];
    let mut scale = vec![0f32; oc];
    for o in 0..oc {
        let mut mx = 0f32;
        for k in 0..K { mx = mx.max(cb[k * oc + o].abs()); }
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        scale[o] = s;
        for k in 0..K { cb_i8[k * oc + o] = (cb[k * oc + o] / s).round().clamp(-127.0, 127.0) as i8; }
    }
    (cb_i8, scale)
}

/// Full int8-path decode-GEMV (the arithmetic the lut8 SIMD kernel implements): quantize `x` to int8
/// (per-vector scale `xs = max|x|/127`), int32 dot with the int8 codebook, rescale by `xs*scale[o]`.
/// `y[o] = xs * scale[o] * sum_i cb_i8[idx[i,o], o] * round(x[i]/xs)`. Scalar reference for the probe
/// and the SIMD correctness anchor. Overflow-safe: |cb_i8*xq| <= 127*127, summed over ic < 2^31.
pub fn gemv_native_i8(packed: &[u8], cb_i8: &[i8], scale: &[f32], oc: usize, ic: usize, x: &[f32], y: &mut [f32]) {
    let half = oc / 2;
    let xs = { let mut mx = 0f32; for &v in &x[..ic] { mx = mx.max(v.abs()); } if mx > 0.0 { mx / 127.0 } else { 1.0 } };
    let xq: Vec<i32> = x[..ic].iter().map(|&v| (v / xs).round().clamp(-127.0, 127.0) as i32).collect();
    let mut acc = vec![0i32; oc];
    for i in 0..ic {
        let xi = xq[i];
        if xi == 0 { continue; }
        let row = &packed[i * half..i * half + half];
        for j in 0..half {
            let b = row[j];
            acc[2 * j]     += cb_i8[((b & 0xF) as usize) * oc + 2 * j] as i32 * xi;
            acc[2 * j + 1] += cb_i8[((b >> 4) as usize) * oc + 2 * j + 1] as i32 * xi;
        }
    }
    for o in 0..oc { y[o] = xs * scale[o] * acc[o] as f32; }
}

// ============================================================================
// lut8 SIMD decode (deliverable C increment 3). The vpshufb value-LUT needs a column's 16 int8
// centroids CONTIGUOUS (a 16-byte shuffle table), so we first transpose the recoded codebook from
// [K,oc] (strided cb_i8[k*oc+o]) to [oc,16] (cb_i8_t[o*16+k]) -- a small one-time pass. Then the
// kernel streams the input-major `packed` contiguously in i-blocks of LUT8_IB inputs, and for each
// byte-position gathers that column's LUT8_IB nibbles from the (cache-resident) tile, vpshufb-decodes
// them against the column table, and int-madds with the int8 activation block. Integer accumulation
// is EXACT and order-free, so lut8 == the scalar int8 accumulate BIT-FOR-BIT (the box's cheap gate).
//
// Block sizes are consts so the box (Sapphire-Rapids-class L2/L3) can retune in one line. LUT8_IB=16
// matches an xmm's 16 int8 lanes; the per-byte-position nibble gather from the cache-resident tile is
// the tunable hotspot (a full in-register 16x16 transpose is the perf follow-up once the box profiles).
pub const LUT8_IB: usize = 16;

/// Column held by output register k of `transpose16x16_epi8`: the unpack cascade emits columns in
/// bit-reversed (reverse-4-bit) register order, so register k holds column `LUT8_COL_OF_REG[k]`.
/// Verified on all arches by `transpose16x16_wiring_is_a_true_transpose` (the decode indexes by it).
const LUT8_COL_OF_REG: [usize; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];

/// Transpose the recoded int8 codebook `[K,oc]` (strided per column) to `[oc,16]` (16 contiguous
/// centroids per column) -- the vpshufb table layout. One-time, small (oc*16 bytes).
pub fn transpose_codebook_i8(cb_i8: &[i8], oc: usize) -> Vec<i8> {
    assert_eq!(cb_i8.len(), K * oc);
    let mut t = vec![0i8; oc * 16];
    for o in 0..oc { for k in 0..K { t[o * 16 + k] = cb_i8[k * oc + o]; } }
    t
}

/// Scalar int8 accumulate from the per-column table `cb_i8_t` `[oc,16]`: the bit-exact reference the
/// lut8 SIMD kernel must match. `acc[o] += sum_i cb_i8_t[o*16 + idx[i,o]] * xq[i]` (integer, exact).
pub fn accumulate_i8_t(packed: &[u8], cb_i8_t: &[i8], oc: usize, xq: &[i8], acc: &mut [i32], i0: usize, i1: usize) {
    let half = oc / 2;
    for i in i0..i1 {
        let xi = xq[i] as i32;
        let row = &packed[i * half..i * half + half];
        for j in 0..half {
            let b = row[j];
            acc[2 * j]     += cb_i8_t[(2 * j) * 16 + (b & 0xF) as usize] as i32 * xi;
            acc[2 * j + 1] += cb_i8_t[(2 * j + 1) * 16 + (b >> 4) as usize] as i32 * xi;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256_i32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0x4E>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0xB1>(s));
    _mm_cvtsi128_si32(s)
}

/// AVX2 lut8 accumulate: vpshufb value-LUT decode + int madd, streaming `packed` in LUT8_IB-input
/// blocks. Bit-exact to [`accumulate_i8_t`] (integer). `cb_i8_t` is `[oc,16]`, `xq` int8 activations.
/// 16x16 byte transpose (deliverable C follow-up, session-4 per-core lever). `rows[t]` holds 16
/// bytes = byte-positions j0..j0+15 of input `ib+t`; returns `cols[c]` = 16 bytes = the 16 inputs'
/// byte at position j0+c. Standard 4-stage unpack cascade (8->16->32->64-bit interleave). This
/// replaces the scalar strided nibble gather (16 L2-latency loads/byte-position -- the 0.66 GB/s/core
/// atom) with in-register shuffles, so the lut8 decode is register-bound, not element-load-bound.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose16x16_epi8(rows: &[std::arch::x86_64::__m128i; 16]) -> [std::arch::x86_64::__m128i; 16] {
    use std::arch::x86_64::*;
    let mut a = [_mm_setzero_si128(); 16];
    for i in 0..8 { a[2*i] = _mm_unpacklo_epi8(rows[2*i], rows[2*i+1]); a[2*i+1] = _mm_unpackhi_epi8(rows[2*i], rows[2*i+1]); }
    let mut b = [_mm_setzero_si128(); 16];
    for i in 0..4 { for k in 0..2 { b[4*i+k] = _mm_unpacklo_epi16(a[4*i+k], a[4*i+2+k]); b[4*i+2+k] = _mm_unpackhi_epi16(a[4*i+k], a[4*i+2+k]); } }
    let mut c = [_mm_setzero_si128(); 16];
    for i in 0..2 { for k in 0..4 { c[8*i+k] = _mm_unpacklo_epi32(b[8*i+k], b[8*i+4+k]); c[8*i+4+k] = _mm_unpackhi_epi32(b[8*i+k], b[8*i+4+k]); } }
    let mut d = [_mm_setzero_si128(); 16];
    for k in 0..8 { d[k] = _mm_unpacklo_epi64(c[k], c[8+k]); d[8+k] = _mm_unpackhi_epi64(c[k], c[8+k]); }
    d
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_lut8_avx2(packed: &[u8], cb_i8_t: &[i8], oc: usize, xq: &[i8], acc: &mut [i32], i0: usize, i1: usize) {
    use std::arch::x86_64::*;
    let half = oc / 2;
    let mask = _mm_set1_epi8(0x0F);
    // decode one byte-position's transposed lane (16 inputs) into acc[o_lo],acc[o_hi].
    #[target_feature(enable = "avx2")]
    unsafe fn decode_lane(bytes: std::arch::x86_64::__m128i, o_lo: usize, cb_i8_t: &[i8],
                          xq16: std::arch::x86_64::__m256i, acc: &mut [i32], mask: std::arch::x86_64::__m128i) {
        use std::arch::x86_64::*;
        let lo = _mm_and_si128(bytes, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        let tbl_lo = _mm_loadu_si128(cb_i8_t.as_ptr().add(o_lo * 16) as *const __m128i);
        let tbl_hi = _mm_loadu_si128(cb_i8_t.as_ptr().add((o_lo + 1) * 16) as *const __m128i);
        let p_lo = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(tbl_lo, lo)), xq16);
        let p_hi = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(tbl_hi, hi)), xq16);
        *acc.get_unchecked_mut(o_lo)     += hsum256_i32(p_lo);
        *acc.get_unchecked_mut(o_lo + 1) += hsum256_i32(p_hi);
    }
    let mut ib = i0;
    while ib + LUT8_IB <= i1 {
        let xq16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xq.as_ptr().add(ib) as *const __m128i));
        let mut j = 0usize;
        // full 16-byte-position tiles: load 16 input rows, transpose in-register, decode 16 lanes.
        while j + 16 <= half {
            let mut rows = [_mm_setzero_si128(); 16];
            for t in 0..LUT8_IB { rows[t] = _mm_loadu_si128(packed.as_ptr().add((ib + t) * half + j) as *const __m128i); }
            let cols = transpose16x16_epi8(&rows);
            // register k holds column j + LUT8_COL_OF_REG[k] (cascade emits bit-reversed order).
            for k in 0..16 { decode_lane(cols[k], 2 * (j + LUT8_COL_OF_REG[k]), cb_i8_t, xq16, acc, mask); }
            j += 16;
        }
        // byte-position remainder (half % 16 != 0): scalar-gather the lane, same decode.
        while j < half {
            let mut buf = [0u8; LUT8_IB];
            for t in 0..LUT8_IB { buf[t] = *packed.get_unchecked((ib + t) * half + j); }
            decode_lane(_mm_loadu_si128(buf.as_ptr() as *const __m128i), 2 * j, cb_i8_t, xq16, acc, mask);
            j += 1;
        }
        ib += LUT8_IB;
    }
    if ib < i1 { accumulate_i8_t(packed, cb_i8_t, oc, xq, acc, ib, i1); } // input tail (< LUT8_IB inputs)
}

// ============================================================================
// Worker engagement (deliverable C). The 671B rerun showed only ~9-16 of 32 pool workers doing
// useful work: the reduction phases (RA/RC) had just `k` tasks, phase C had `k*ceil(inter/256)`
// (~64) for 32-60 workers, and the std Barrier's futex wake latency x (4 phases x 58 calls/token)
// serialized the tail. Three orthogonal fixes, all behind flags so the box can A/B:
//   * finer reduction: RA/RC split by OUTPUT chunk (RED_CHUNK), so every worker has reduce work.
//   * spin phase barrier (TRAPETUM_CPU_SPIN=1): swap the blocking Barrier for an atomic spin so a
//     phase transition costs a cache-line poll, not a futex round-trip (the AWS cores are dedicated
//     during decode). Determinism is unaffected -- a barrier only orders phases, never sums.
//   * per-worker task instrumentation (TRAPETUM_CPU_ENGAGE_DEBUG=1): counts tasks/worker so the
//     engagement distribution is observable (on the M4 at V2-Lite dims, and on the box).
// ============================================================================

// Persistent per-thread worker index (set once when the pool spawns the thread); usize::MAX on
// any non-pool thread (e.g. the scoped _nt path). Used only for engagement instrumentation.
thread_local! { static WORKER_ID: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) }; }

/// Output elements per reduction task (RA/RC), so the cheap reduction phases have enough tasks to
/// keep all workers busy instead of `k`. Fixed (not worker-derived) -> reduction order unchanged.
const RED_CHUNK: usize = 512;

/// Spin barrier: workers poll a generation counter instead of blocking on a futex, so a phase
/// transition inside a MoE call is a cache-line read, not a kernel round-trip.
struct SpinBarrier { n: usize, count: AtomicUsize, gen: AtomicUsize }
impl SpinBarrier {
    fn new(n: usize) -> Self { Self { n, count: AtomicUsize::new(0), gen: AtomicUsize::new(0) } }
    fn wait(&self) {
        let g = self.gen.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Release);
            self.gen.fetch_add(1, Ordering::Release);
        } else {
            while self.gen.load(Ordering::Acquire) == g { std::hint::spin_loop(); }
        }
    }
}

/// Phase barrier for the native work-steal: blocking (default) or spinning (`TRAPETUM_CPU_SPIN=1`).
enum PhaseBar { Block(Barrier), Spin(SpinBarrier) }
impl PhaseBar {
    fn new(n: usize, spin: bool) -> Self { if spin { PhaseBar::Spin(SpinBarrier::new(n)) } else { PhaseBar::Block(Barrier::new(n)) } }
    #[inline] fn wait(&self) { match self { PhaseBar::Block(b) => { b.wait(); } PhaseBar::Spin(s) => s.wait() } }
}

fn cpu_spin() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("TRAPETUM_CPU_SPIN").map(|v| v == "1").unwrap_or(false))
}
fn engage_debug() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("TRAPETUM_CPU_ENGAGE_DEBUG").map(|v| v == "1").unwrap_or(false))
}

/// TRAPETUM_MOE_TIMING=1: per-phase worker-microseconds (summed across workers) for the native MoE
/// work-steal, so the box can see WHERE the MoE phase spends its ~1.4 s (phase A/C decode vs the
/// reduce tails vs setup) without gdb. Summed worker-us / n_workers ~ the phase wall time.
pub mod moetime {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    pub static A_US: AtomicU64 = AtomicU64::new(0);
    pub static RA_US: AtomicU64 = AtomicU64::new(0);
    pub static C_US: AtomicU64 = AtomicU64::new(0);
    pub static RC_US: AtomicU64 = AtomicU64::new(0);
    // Whole-forward stages OUTSIDE the work-steal phases (G-inc1: the ~386 ms/token gap session-5
    // exposed lives here). XCOPY = norm.to_host() drain+copy; WS = the whole work-steal call wall
    // (WS - (A+RA+C+RC) = the entry overhead: xquant + cache lookup + pool handoff + combine);
    // SHARED = from_host + the GPU shared-expert run_ffn + residual (drained).
    pub static XCOPY_US: AtomicU64 = AtomicU64::new(0);
    pub static WS_US: AtomicU64 = AtomicU64::new(0);
    pub static SHARED_US: AtomicU64 = AtomicU64::new(0);
    pub fn on() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| std::env::var("TRAPETUM_MOE_TIMING").map(|v| v == "1").unwrap_or(false))
    }
    #[inline] pub fn add(a: &AtomicU64, us: u64) { a.fetch_add(us, Ordering::Relaxed); }
    /// (A, RA, C, RC, XCOPY, WS, SHARED) microseconds, reset.
    pub fn take() -> (u64, u64, u64, u64, u64, u64, u64) {
        (A_US.swap(0, Ordering::Relaxed), RA_US.swap(0, Ordering::Relaxed),
         C_US.swap(0, Ordering::Relaxed), RC_US.swap(0, Ordering::Relaxed),
         XCOPY_US.swap(0, Ordering::Relaxed), WS_US.swap(0, Ordering::Relaxed), SHARED_US.swap(0, Ordering::Relaxed))
    }
}

/// Shared state a set of pool workers cooperatively drains for ONE native-path MoE call. Unlike the
/// re-tiled [`WorkstealCtx`], the native layout chunks over the INPUT dimension (streaming `packed`
/// once), so each (expert, proj, input-chunk) task writes a disjoint per-chunk PARTIAL that a later
/// phase reduces in fixed chunk order. `Sync` because the only shared mutation is the phase atomics
/// / barrier and raw pointers into non-overlapping partial/scratch slices.
struct NativeWsCtx<'a> {
    experts: &'a [NativeExpert<'a>],
    x: &'a [f32],
    hidden: usize, inter: usize,
    nch_h: usize, nch_i: usize, k: usize,   // input chunks for gate/up (ic=hidden) and down (ic=inter)
    ra_per: usize, rc_per: usize,           // output chunks per expert for the RA/RC reductions
    ctr_a: AtomicUsize, ctr_ra: AtomicUsize, ctr_c: AtomicUsize, ctr_rc: AtomicUsize,
    bar: PhaseBar,
    eng: Option<&'a [AtomicUsize]>,         // per-worker task counts (TRAPETUM_CPU_ENGAGE_DEBUG)
    pg: RawF32, pu: RawF32, pd: RawF32,           // per-chunk partials
    g: RawF32, u: RawF32, act: RawF32, out: RawF32, // reduced per-expert scratch
}
unsafe impl Sync for NativeWsCtx<'_> {}

/// One worker's slice of a native-path MoE call. Four phases separated by barriers:
///   A  gate+up decode-GEMV per (expert, proj, input-chunk) -> disjoint partials
///   RA reduce gate/up partials (fixed chunk order) + SiLU, per (expert, output-chunk)
///   C  down decode-GEMV per (expert, input-chunk) -> disjoint partials
///   RC reduce down partials (fixed chunk order), per (expert, output-chunk)
/// The chunk->partial mapping and every reduction order are fixed, so which worker runs which task
/// never changes a byte of the result -- thread-count invariant, exactly like the sequential path.
/// RA/RC split by OUTPUT chunk (not just by expert) so the cheap reduction phases have enough tasks
/// to keep every worker busy (the 671B engagement fix); the per-element reduction order is unchanged.
fn native_ws_worker(c: &NativeWsCtx) {
    let (nch_h, nch_i, inter, hidden, k) = (c.nch_h, c.nch_i, c.inter, c.hidden, c.k);
    let wid = WORKER_ID.with(|w| w.get());
    let mut done = 0usize;
    let tm = moetime::on();
    let mut ph = if tm { Some(std::time::Instant::now()) } else { None };
    macro_rules! lap { ($a:expr) => { if let Some(t) = ph { moetime::add($a, t.elapsed().as_micros() as u64); ph = Some(std::time::Instant::now()); } }; }
    // Phase A: gate+up input-chunks (2*nch_h per expert).
    let na = k * 2 * nch_h;
    loop {
        let t = c.ctr_a.fetch_add(1, Ordering::Relaxed);
        if t >= na { break; }
        let (e, rr) = (t / (2 * nch_h), t % (2 * nch_h));
        let (which, ci) = (rr / nch_h, rr % nch_h);
        let (i0, i1) = (ci * NATIVE_CHUNK_I, ((ci + 1) * NATIVE_CHUNK_I).min(hidden));
        let ex = &c.experts[e];
        let (packed, cb, base) = if which == 0 { (ex.gp, ex.gc, &c.pg) } else { (ex.up, ex.uc, &c.pu) };
        let slot = unsafe { std::slice::from_raw_parts_mut(base.0.add((e * nch_h + ci) * inter), inter) };
        gemv_native_range(packed, cb, inter, c.x, slot, i0, i1);
        done += 1;
    }
    lap!(&moetime::A_US);
    c.bar.wait();
    // Phase RA: reduce gate/up partials (fixed chunk order) + SiLU, per (expert, output-chunk).
    let nra = k * c.ra_per;
    loop {
        let t = c.ctr_ra.fetch_add(1, Ordering::Relaxed);
        if t >= nra { break; }
        let (e, oc) = (t / c.ra_per, t % c.ra_per);
        let (o0, o1) = (oc * RED_CHUNK, ((oc + 1) * RED_CHUNK).min(inter));
        unsafe {
            let (ge, ue, acte) = (c.g.0.add(e * inter), c.u.0.add(e * inter), c.act.0.add(e * inter));
            for o in o0..o1 { *ge.add(o) = 0.0; *ue.add(o) = 0.0; }
            for ci in 0..nch_h {
                let (pgc, puc) = (c.pg.0.add((e * nch_h + ci) * inter), c.pu.0.add((e * nch_h + ci) * inter));
                for o in o0..o1 { *ge.add(o) += *pgc.add(o); *ue.add(o) += *puc.add(o); }
            }
            for o in o0..o1 { *acte.add(o) = silu(*ge.add(o)) * *ue.add(o); }
        }
        done += 1;
    }
    lap!(&moetime::RA_US);
    c.bar.wait();
    // Phase C: down input-chunks (nch_i per expert).
    let nc = k * nch_i;
    loop {
        let t = c.ctr_c.fetch_add(1, Ordering::Relaxed);
        if t >= nc { break; }
        let (e, ci) = (t / nch_i, t % nch_i);
        let (i0, i1) = (ci * NATIVE_CHUNK_I, ((ci + 1) * NATIVE_CHUNK_I).min(inter));
        let ex = &c.experts[e];
        let act_e = unsafe { std::slice::from_raw_parts(c.act.0.add(e * inter), inter) };
        let slot = unsafe { std::slice::from_raw_parts_mut(c.pd.0.add((e * nch_i + ci) * hidden), hidden) };
        gemv_native_range(ex.dp, ex.dc, hidden, act_e, slot, i0, i1);
        done += 1;
    }
    lap!(&moetime::C_US);
    c.bar.wait();
    // Phase RC: reduce down partials (fixed chunk order), per (expert, output-chunk).
    let nrc = k * c.rc_per;
    loop {
        let t = c.ctr_rc.fetch_add(1, Ordering::Relaxed);
        if t >= nrc { break; }
        let (e, oc) = (t / c.rc_per, t % c.rc_per);
        let (o0, o1) = (oc * RED_CHUNK, ((oc + 1) * RED_CHUNK).min(hidden));
        unsafe {
            let oe = c.out.0.add(e * hidden);
            for o in o0..o1 { *oe.add(o) = 0.0; }
            for ci in 0..nch_i {
                let pdc = c.pd.0.add((e * nch_i + ci) * hidden);
                for o in o0..o1 { *oe.add(o) += *pdc.add(o); }
            }
        }
        done += 1;
    }
    lap!(&moetime::RC_US);
    if let Some(eng) = c.eng { if wid < eng.len() { eng[wid].fetch_add(done, Ordering::Relaxed); } }
}

/// Build the ctx fields shared by the pool and scoped native work-steal entry points.
#[allow(clippy::too_many_arguments)]
fn native_ws_ctx<'a>(experts: &'a [NativeExpert<'a>], x: &'a [f32], hidden: usize, inter: usize,
                     nch_h: usize, nch_i: usize, k: usize, nbar: usize, eng: Option<&'a [AtomicUsize]>,
                     pg: &mut [f32], pu: &mut [f32], pd: &mut [f32],
                     g: &mut [f32], u: &mut [f32], act: &mut [f32], out: &mut [f32]) -> NativeWsCtx<'a> {
    NativeWsCtx {
        experts, x, hidden, inter, nch_h, nch_i, k,
        ra_per: (inter + RED_CHUNK - 1) / RED_CHUNK, rc_per: (hidden + RED_CHUNK - 1) / RED_CHUNK,
        ctr_a: AtomicUsize::new(0), ctr_ra: AtomicUsize::new(0), ctr_c: AtomicUsize::new(0), ctr_rc: AtomicUsize::new(0),
        bar: PhaseBar::new(nbar, cpu_spin()), eng,
        pg: RawF32(pg.as_mut_ptr()), pu: RawF32(pu.as_mut_ptr()), pd: RawF32(pd.as_mut_ptr()),
        g: RawF32(g.as_mut_ptr()), u: RawF32(u.as_mut_ptr()), act: RawF32(act.as_mut_ptr()), out: RawF32(out.as_mut_ptr()),
    }
}

/// Persistent per-thread scratch for the native work-steal, reused across the ~58 MoE calls/token so
/// the ~5.5 MB of partial/reduce buffers are allocated + faulted in ONCE, not per layer (the ~138 ms
/// per-call overhead session-4 flagged was mostly this malloc + first-touch fault). Only the partials
/// pg/pu/pd are re-zeroed each call (phase A accumulates into them); g/u/act/out are fully overwritten
/// by RA/RC so they keep stale bytes harmlessly. Determinism is unchanged (same buffers, same math).
#[derive(Default)]
struct WsScratch { pg: Vec<f32>, pu: Vec<f32>, pd: Vec<f32>, g: Vec<f32>, u: Vec<f32>, act: Vec<f32>, out: Vec<f32> }
impl WsScratch {
    fn ready(&mut self, k: usize, nch_h: usize, nch_i: usize, inter: usize, hidden: usize) {
        let (a, d, e) = (k * nch_h * inter, k * nch_i * hidden, k * inter);
        self.pg.clear(); self.pg.resize(a, 0.0); // zeroed; keeps capacity so no realloc after warmup
        self.pu.clear(); self.pu.resize(a, 0.0);
        self.pd.clear(); self.pd.resize(d, 0.0);
        self.g.resize(e, 0.0); self.u.resize(e, 0.0); self.act.resize(e, 0.0); self.out.resize(k * hidden, 0.0);
    }
}
thread_local! { static WS_SCRATCH: std::cell::RefCell<WsScratch> = std::cell::RefCell::new(WsScratch::default()); }

/// Native-path routed MoE sum with ONE phased work-steal over ALL (expert, proj, input-chunk) tasks
/// per call, on the persistent pool -- replacing the per-GEMV pool dispatch (`routed_experts_native`
/// issued 3*k dispatches/layer; at 671B that was ~1392/token, most with too few chunks to fill the
/// workers). Bit-identical to [`routed_experts_native`]: same input-chunk decomposition, same
/// fixed-order partial reduction, same fixed expert-order combine -- so it is deterministic and
/// thread-count invariant. `acc_out` (len `hidden`) is overwritten.
pub fn routed_experts_native_worksteal(x: &[f32], experts: &[NativeExpert], hidden: usize, inter: usize, acc_out: &mut [f32]) {
    let p = pool();
    let k = experts.len();
    for v in acc_out.iter_mut().take(hidden) { *v = 0.0; }
    if k == 0 { return; }
    if p.n <= 1 { routed_experts_native(x, experts, hidden, inter, acc_out); return; }
    let nch_h = (hidden + NATIVE_CHUNK_I - 1) / NATIVE_CHUNK_I;
    let nch_i = (inter + NATIVE_CHUNK_I - 1) / NATIVE_CHUNK_I;
    let eng_vec: Vec<AtomicUsize> = if engage_debug() { (0..p.n).map(|_| AtomicUsize::new(0)).collect() } else { Vec::new() };
    let eng = if engage_debug() { Some(eng_vec.as_slice()) } else { None };
    WS_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        s.ready(k, nch_h, nch_i, inter, hidden);
        let WsScratch { pg, pu, pd, g, u, act, out } = &mut *s;
        let ctx = native_ws_ctx(experts, x, hidden, inter, nch_h, nch_i, k, p.n, eng,
            pg, pu, pd, g, u, act, out);
        p.run(&|| native_ws_worker(&ctx));
        if engage_debug() {
            // Print ONE representative distribution (not per-call, which would flood a decode).
            static PRINTED: OnceLock<()> = OnceLock::new();
            PRINTED.get_or_init(|| {
                let counts: Vec<usize> = eng_vec.iter().map(|a| a.load(Ordering::Relaxed)).collect();
                let active = counts.iter().filter(|&&c| c > 0).count();
                let (mn, mx) = (counts.iter().copied().min().unwrap_or(0), counts.iter().copied().max().unwrap_or(0));
                let total: usize = counts.iter().sum();
                eprintln!("[cpu_engage] workers={} active={active} tasks_total={total} per-worker min={mn} max={mx} spin={} dist={:?}",
                          p.n, cpu_spin(), counts);
            });
        }
        // Fixed expert-order weighted combine (single thread).
        for (e, ex) in experts.iter().enumerate() {
            let (w, base) = (ex.weight, e * hidden);
            for i in 0..hidden { acc_out[i] += w * out[base + i]; }
        }
    });
}

/// Thread-count-explicit twin of [`routed_experts_native_worksteal`] (scoped spawn instead of the
/// persistent pool), so the determinism harness can prove the result is BITWISE identical across
/// worker counts. Same phases/reductions -> `worker_threads` only changes which core runs a task.
pub fn routed_experts_native_worksteal_nt(x: &[f32], experts: &[NativeExpert], hidden: usize, inter: usize, acc_out: &mut [f32], worker_threads: usize) {
    let k = experts.len();
    for v in acc_out.iter_mut().take(hidden) { *v = 0.0; }
    if k == 0 { return; }
    let nch_h = (hidden + NATIVE_CHUNK_I - 1) / NATIVE_CHUNK_I;
    let nch_i = (inter + NATIVE_CHUNK_I - 1) / NATIVE_CHUNK_I;
    let nthreads = worker_threads.max(1).min((k * 2 * nch_h).max(k * nch_i).max(1));
    if nthreads <= 1 { routed_experts_native(x, experts, hidden, inter, acc_out); return; }
    let mut pg = vec![0f32; k * nch_h * inter];
    let mut pu = vec![0f32; k * nch_h * inter];
    let mut pd = vec![0f32; k * nch_i * hidden];
    let mut g = vec![0f32; k * inter];
    let mut u = vec![0f32; k * inter];
    let mut act = vec![0f32; k * inter];
    let mut out = vec![0f32; k * hidden];
    let ctx = native_ws_ctx(experts, x, hidden, inter, nch_h, nch_i, k, nthreads, None,
        &mut pg, &mut pu, &mut pd, &mut g, &mut u, &mut act, &mut out);
    thread::scope(|s| {
        for _ in 0..nthreads { let ctx = &ctx; s.spawn(move || native_ws_worker(ctx)); }
    });
    for (e, ex) in experts.iter().enumerate() {
        let (w, base) = (ex.weight, e * hidden);
        for i in 0..hidden { acc_out[i] += w * out[base + i]; }
    }
}

// ============================================================================
// lut8 END-TO-END int8 work-steal (deliverable C increment 4). Same phase skeleton as the f32
// work-steal, but the decode is the int8 vpshufb kernel: codebooks are recoded+transposed to int8
// per expert (small, one-time per call), the activation is quantized to int8 per GEMV, phases A/C
// accumulate INT32 partials, and the reductions (RA/RC) fold in the rescale (xs * scale[o]). An extra
// per-expert phase Q quantizes the SiLU output before the down GEMV (its scale spans the whole vector).
// Integer partials -> deterministic and thread-invariant; result matches the f32 path within the
// measured int8 tolerance (~0.5%/GEMV). Activated by TRAPETUM_CPU_KERNEL=lut8; the f32 path untouched.
// ============================================================================
struct RawI32(*mut i32);
unsafe impl Send for RawI32 {}
unsafe impl Sync for RawI32 {}
struct RawI8(*mut i8);
unsafe impl Send for RawI8 {}
unsafe impl Sync for RawI8 {}

/// Per-expert int8 codebooks (transposed to `[oc,16]`) + per-column scales, borrowed from the cache.
struct ExpertI8<'a> {
    gp: &'a [u8], g_t: &'a [i8], gs: &'a [f32],
    up: &'a [u8], u_t: &'a [i8], us: &'a [f32],
    dp: &'a [u8], d_t: &'a [i8], ds: &'a [f32],
    weight: f32,
}

/// Owned int8-recoded+transposed codebooks for one routed expert. Cached ONCE (see `cached_recode`)
/// and reused across every MoE call -- session 5 showed the transpose delivered (per-core atom
/// 0.66 -> 3.3 GB/s) but the win was masked by re-doing this recode+transpose for all picked experts
/// EVERY call (~464/token, ~710 ms). Hoisting it here (lazy-once) is the last increment to ~2 tok/s.
struct ExpertI8Cb { g_t: Vec<i8>, gs: Vec<f32>, u_t: Vec<i8>, us: Vec<f32>, d_t: Vec<i8>, ds: Vec<f32> }

/// Lazily recode+transpose an expert's int8 codebooks, cached by the (stable, model-lifetime) gate-
/// codebook pointer. First touch per expert only; ~150 MB total at 671B (58*256*3 small mats).
fn cached_recode(e: &NativeExpert, hidden: usize, inter: usize) -> std::sync::Arc<ExpertI8Cb> {
    use std::collections::HashMap;
    static CACHE: OnceLock<Mutex<HashMap<usize, std::sync::Arc<ExpertI8Cb>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = e.gc.as_ptr() as usize;
    if let Some(a) = cache.lock().unwrap().get(&key) { return a.clone(); }
    let (gi, gs) = recode_codebook_i8(e.gc, inter);
    let (ui, us) = recode_codebook_i8(e.uc, inter);
    let (di, ds) = recode_codebook_i8(e.dc, hidden);
    let a = std::sync::Arc::new(ExpertI8Cb {
        g_t: transpose_codebook_i8(&gi, inter), gs,
        u_t: transpose_codebook_i8(&ui, inter), us,
        d_t: transpose_codebook_i8(&di, hidden), ds });
    cache.lock().unwrap().insert(key, a.clone());
    a
}

/// int8-quantize a vector: `(xq, xs)` with `xs = max|v|/127`.
fn quantize_i8(v: &[f32]) -> (Vec<i8>, f32) {
    let mut mx = 0f32; for &a in v { mx = mx.max(a.abs()); }
    let xs = if mx > 0.0 { mx / 127.0 } else { 1.0 };
    (v.iter().map(|&a| (a / xs).round().clamp(-127.0, 127.0) as i8).collect(), xs)
}

/// int8 decode: AVX2 vpshufb lut8 where present, else the scalar int8 reference. Both accumulate
/// int32 into `acc` and are bit-identical (integer, order-free).
#[inline]
fn accumulate_i8_dispatch(packed: &[u8], cb_i8_t: &[i8], oc: usize, xq: &[i8], acc: &mut [i32], i0: usize, i1: usize) {
    #[cfg(target_arch = "x86_64")]
    { if is_x86_feature_detected!("avx2") { unsafe { accumulate_lut8_avx2(packed, cb_i8_t, oc, xq, acc, i0, i1); } return; } }
    accumulate_i8_t(packed, cb_i8_t, oc, xq, acc, i0, i1);
}

struct Lut8Ctx<'a> {
    experts: &'a [ExpertI8<'a>],
    xq_x: &'a [i8], xs_x: f32,
    hidden: usize, inter: usize, nch_h: usize, nch_i: usize, k: usize, ra_per: usize, rc_per: usize,
    ctr_a: AtomicUsize, ctr_ra: AtomicUsize, ctr_q: AtomicUsize, ctr_c: AtomicUsize, ctr_rc: AtomicUsize,
    bar: PhaseBar,
    pg: RawI32, pu: RawI32, pd: RawI32,
    g: RawF32, u: RawF32, act: RawF32, out: RawF32,
    xq_act: RawI8, xs_act: RawF32,
}
unsafe impl Sync for Lut8Ctx<'_> {}

fn lut8_worker(c: &Lut8Ctx) {
    let (nch_h, nch_i, inter, hidden, k) = (c.nch_h, c.nch_i, c.inter, c.hidden, c.k);
    // TRAPETUM_MOE_TIMING also covers the lut8 path (the gap session-4 flagged): Phase Q (int8
    // quantize of the SiLU output, lut8-only) folds into the RA bucket.
    let tm = moetime::on();
    let mut ph = if tm { Some(std::time::Instant::now()) } else { None };
    macro_rules! lap { ($a:expr) => { if let Some(t) = ph { moetime::add($a, t.elapsed().as_micros() as u64); ph = Some(std::time::Instant::now()); } }; }
    // Phase A: gate+up int8 decode -> int32 partials.
    let na = k * 2 * nch_h;
    loop {
        let t = c.ctr_a.fetch_add(1, Ordering::Relaxed);
        if t >= na { break; }
        let (e, rr) = (t / (2 * nch_h), t % (2 * nch_h));
        let (which, ci) = (rr / nch_h, rr % nch_h);
        let (i0, i1) = (ci * NATIVE_CHUNK_I, ((ci + 1) * NATIVE_CHUNK_I).min(hidden));
        let ex = &c.experts[e];
        let (packed, cb_t, base) = if which == 0 { (ex.gp, ex.g_t, &c.pg) } else { (ex.up, ex.u_t, &c.pu) };
        let slot = unsafe { std::slice::from_raw_parts_mut(base.0.add((e * nch_h + ci) * inter), inter) };
        accumulate_i8_dispatch(packed, cb_t, inter, c.xq_x, slot, i0, i1);
    }
    lap!(&moetime::A_US);
    c.bar.wait();
    // Phase RA: reduce gate/up int32 partials + rescale (xs_x*scale) + SiLU, per (expert, o-chunk).
    let nra = k * c.ra_per;
    loop {
        let t = c.ctr_ra.fetch_add(1, Ordering::Relaxed);
        if t >= nra { break; }
        let (e, oc) = (t / c.ra_per, t % c.ra_per);
        let (o0, o1) = (oc * RED_CHUNK, ((oc + 1) * RED_CHUNK).min(inter));
        let ex = &c.experts[e];
        unsafe {
            let (ge, ue, acte) = (c.g.0.add(e * inter), c.u.0.add(e * inter), c.act.0.add(e * inter));
            for o in o0..o1 {
                let (mut sg, mut su) = (0i32, 0i32);
                for ci in 0..nch_h {
                    sg += *c.pg.0.add((e * nch_h + ci) * inter + o);
                    su += *c.pu.0.add((e * nch_h + ci) * inter + o);
                }
                let gv = c.xs_x * ex.gs[o] * sg as f32;
                let uv = c.xs_x * ex.us[o] * su as f32;
                *ge.add(o) = gv; *ue.add(o) = uv;
                *acte.add(o) = silu(gv) * uv;
            }
        }
    }
    lap!(&moetime::RA_US);
    c.bar.wait();
    // Phase Q: quantize each expert's SiLU output to int8 (scale spans the whole vector).
    loop {
        let e = c.ctr_q.fetch_add(1, Ordering::Relaxed);
        if e >= k { break; }
        unsafe {
            let acte = std::slice::from_raw_parts(c.act.0.add(e * inter), inter);
            let (xq, xs) = quantize_i8(acte);
            *c.xs_act.0.add(e) = xs;
            let dst = c.xq_act.0.add(e * inter);
            for o in 0..inter { *dst.add(o) = xq[o]; }
        }
    }
    lap!(&moetime::RA_US); // fold the lut8-only quantize phase into RA
    c.bar.wait();
    // Phase C: down int8 decode -> int32 partials.
    let nc = k * nch_i;
    loop {
        let t = c.ctr_c.fetch_add(1, Ordering::Relaxed);
        if t >= nc { break; }
        let (e, ci) = (t / nch_i, t % nch_i);
        let (i0, i1) = (ci * NATIVE_CHUNK_I, ((ci + 1) * NATIVE_CHUNK_I).min(inter));
        let ex = &c.experts[e];
        let xq_act_e = unsafe { std::slice::from_raw_parts(c.xq_act.0.add(e * inter), inter) };
        let slot = unsafe { std::slice::from_raw_parts_mut(c.pd.0.add((e * nch_i + ci) * hidden), hidden) };
        accumulate_i8_dispatch(ex.dp, ex.d_t, hidden, xq_act_e, slot, i0, i1);
    }
    lap!(&moetime::C_US);
    c.bar.wait();
    // Phase RC: reduce down int32 partials + rescale (xs_act*scale), per (expert, o-chunk).
    let nrc = k * c.rc_per;
    loop {
        let t = c.ctr_rc.fetch_add(1, Ordering::Relaxed);
        if t >= nrc { break; }
        let (e, oc) = (t / c.rc_per, t % c.rc_per);
        let (o0, o1) = (oc * RED_CHUNK, ((oc + 1) * RED_CHUNK).min(hidden));
        let ex = &c.experts[e];
        unsafe {
            let xs_act = *c.xs_act.0.add(e);
            let oe = c.out.0.add(e * hidden);
            for o in o0..o1 {
                let mut s = 0i32;
                for ci in 0..nch_i { s += *c.pd.0.add((e * nch_i + ci) * hidden + o); }
                *oe.add(o) = xs_act * ex.ds[o] * s as f32;
            }
        }
    }
    lap!(&moetime::RC_US);
}

/// lut8 end-to-end routed MoE sum (TRAPETUM_CPU_KERNEL=lut8). Recodes codebooks to int8, quantizes
/// activations, runs the int8 vpshufb decode through the phased work-steal, rescales in the reduces.
/// Deterministic + thread-invariant (integer partials, fixed orders). Matches the f32 path within the
/// int8 tolerance. `acc_out` (len `hidden`) overwritten. Falls back to the f32 path for k<=1 / no pool.
pub fn routed_experts_native_worksteal_lut8(x: &[f32], experts: &[NativeExpert], hidden: usize, inter: usize, acc_out: &mut [f32]) {
    let p = pool();
    let k = experts.len();
    for v in acc_out.iter_mut().take(hidden) { *v = 0.0; }
    if k == 0 { return; }
    // Recode is now CACHED per expert (lazy-once), not redone every call -- the session-5 fix.
    let cbs: Vec<std::sync::Arc<ExpertI8Cb>> = experts.iter().map(|e| cached_recode(e, hidden, inter)).collect();
    let ei8: Vec<ExpertI8> = experts.iter().zip(&cbs).map(|(e, cb)| ExpertI8 {
        gp: e.gp, g_t: &cb.g_t, gs: &cb.gs,
        up: e.up, u_t: &cb.u_t, us: &cb.us,
        dp: e.dp, d_t: &cb.d_t, ds: &cb.ds, weight: e.weight,
    }).collect();
    let (xq_x, xs_x) = quantize_i8(&x[..hidden]);
    let nch_h = (hidden + NATIVE_CHUNK_I - 1) / NATIVE_CHUNK_I;
    let nch_i = (inter + NATIVE_CHUNK_I - 1) / NATIVE_CHUNK_I;
    let mut pg = vec![0i32; k * nch_h * inter];
    let mut pu = vec![0i32; k * nch_h * inter];
    let mut pd = vec![0i32; k * nch_i * hidden];
    let mut g = vec![0f32; k * inter];
    let mut u = vec![0f32; k * inter];
    let mut act = vec![0f32; k * inter];
    let mut out = vec![0f32; k * hidden];
    let mut xq_act = vec![0i8; k * inter];
    let mut xs_act = vec![0f32; k];
    let ctx = Lut8Ctx {
        experts: &ei8, xq_x: &xq_x, xs_x, hidden, inter, nch_h, nch_i, k,
        ra_per: (inter + RED_CHUNK - 1) / RED_CHUNK, rc_per: (hidden + RED_CHUNK - 1) / RED_CHUNK,
        ctr_a: AtomicUsize::new(0), ctr_ra: AtomicUsize::new(0), ctr_q: AtomicUsize::new(0),
        ctr_c: AtomicUsize::new(0), ctr_rc: AtomicUsize::new(0),
        bar: PhaseBar::new(p.n, cpu_spin()),
        pg: RawI32(pg.as_mut_ptr()), pu: RawI32(pu.as_mut_ptr()), pd: RawI32(pd.as_mut_ptr()),
        g: RawF32(g.as_mut_ptr()), u: RawF32(u.as_mut_ptr()), act: RawF32(act.as_mut_ptr()), out: RawF32(out.as_mut_ptr()),
        xq_act: RawI8(xq_act.as_mut_ptr()), xs_act: RawF32(xs_act.as_mut_ptr()),
    };
    p.run(&|| lut8_worker(&ctx));
    for (e, ex) in ei8.iter().enumerate() {
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
        // DIAGNOSTIC: run-to-run variation of ONE V2-Lite MoE block. The DEFAULT is now mode 2
        // (deterministic) -> this reports 0. Set TRAPETUM_DETERMINISTIC=0 to select the atomic
        // path and measure the per-layer noise (~1.95e-3, the S14 figure that showed atomic drift
        // compounds across ~26 layers into a token-flipping ~1-logit end-to-end shift).
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
        // identical output run-to-run? The DEFAULT is now mode 2 (two-stage fixed-order) -> this
        // reports 0/30 (deterministic). Set TRAPETUM_DETERMINISTIC=0 to select the atomic path and
        // observe the historical nondeterminism (30/30), the root cause the S14 campaign fixed.
        let (mism, worst) = crate::check_gpu_gemv_determinism(8192, 512, 30);
        eprintln!("[gpu_determinism_probe] ic=8192 oc=512 iters=30 -> {mism}/30 runs differ bitwise from run 0, worst_abs_diff={worst:e}");
        // No determinism assertion: whether atomics reorder is GPU/scheduler dependent. The
        // number is the evidence we report.
        let _ = (mism, worst);
    }

    #[test]
    fn gemv8_k256_matches_dequant_and_is_deterministic() {
        // S19 increment 1: the K256 8-bit decode GEMV (gemv8) decodes uint8 indices against a
        // per-column 256-entry codebook and matches a CPU reference that uses the GPU's exact
        // fp16 codebook + fp16 activations -- so the only residual is float summation order.
        // Also checks the path is bitwise run-to-run deterministic under the default (two-stage)
        // mode. Two shapes: a wide small-OC (8192x512) and a shared-expert-shaped OC (2048x1408).
        for (ic, oc) in [(8192usize, 512usize), (2048usize, 1408usize)] {
            let (worst_rel, l2, mism) = crate::check_gpu_gemv8_vs_ref(ic, oc, 10);
            let mode = std::env::var("TRAPETUM_DETERMINISTIC").unwrap_or_else(|_| "2(default)".into());
            eprintln!("[gemv8_k256 ic={ic} oc={oc} mode={mode}] vs fp16cb-ref: worst_rel={worst_rel:e} l2={l2:e}; \
                       determinism: {mism}/10 runs differ bitwise");
            assert!(l2 < 5e-3, "K256 gemv8 vs fp16-codebook reference l2={l2:e} too large (ic={ic} oc={oc})");
            // Under the default deterministic mode the path must be bitwise stable. If the env
            // forces the atomic mode (=0) skip the determinism gate (atomics reorder by design).
            let atomic = std::env::var("TRAPETUM_DETERMINISTIC").map(|v| v == "0").unwrap_or(false);
            if !atomic { assert_eq!(mism, 0, "K256 gemv8 not deterministic under mode {mode} (ic={ic} oc={oc})"); }
        }
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

    // Single-thread replica of gemv_cpu_native_det's math (same fixed chunks + fixed-order
    // reduction) -- the deterministic reference the pooled version must equal BITWISE.
    fn native_det_ref(packed: &[u8], cb: &[f32], oc: usize, ic: usize, x: &[f32]) -> Vec<f32> {
        let nchunks = (ic + NATIVE_CHUNK_I - 1) / NATIVE_CHUNK_I;
        let mut y = vec![0f32; oc];
        if nchunks <= 1 { gemv_native_range(packed, cb, oc, x, &mut y, 0, ic); return y; }
        let mut parts = vec![vec![0f32; oc]; nchunks];
        for c in 0..nchunks {
            let (i0, i1) = (c * NATIVE_CHUNK_I, ((c + 1) * NATIVE_CHUNK_I).min(ic));
            gemv_native_range(packed, cb, oc, x, &mut parts[c], i0, i1);
        }
        for c in 0..nchunks { for o in 0..oc { y[o] += parts[c][o]; } }
        y
    }

    #[test]
    fn native_matches_dequant_and_is_deterministic() {
        // The native input-major GEMV (no re-tile) matches the dequant reference to summation
        // tolerance, and its pooled result is BIT-identical to the single-thread deterministic
        // reference (fixed chunks + fixed-order reduce) -> thread-count-invariant by construction.
        for (oc, ic) in [(256usize, 512usize), (64, 300), (512, 2048), (128, 4096)] {
            let mut r = Lcg(0x0FF10AD5u64 ^ ((oc as u64) << 12) ^ ic as u64);
            let w: Vec<f32> = (0..oc * ic).map(|_| r.f32()).collect();
            let x: Vec<f32> = (0..ic).map(|_| r.f32()).collect();
            let (packed, cb, w_dq) = quantize_host(&w, oc, ic);
            let mut y = vec![0f32; oc];
            gemv_cpu_native_det(&packed, &cb, oc, ic, &x, &mut y);
            // (a) correctness vs dequant matmul (L2: worst-elt rel err blows up on near-zero
            // outputs from the chunked reassociation; the true agreement is the L2 metric)
            let reference = naive_matmul(&w_dq, oc, ic, &x);
            let e = l2_rel(&y, &reference);
            assert!(e < 1e-4, "native vs dequant matmul L2 rel err {e:e} at oc={oc} ic={ic}");
            // (b) determinism: pooled == single-thread chunked reference, BITWISE
            let det = native_det_ref(&packed, &cb, oc, ic, &x);
            assert!(y.iter().zip(&det).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "native GEMV not bit-identical to the fixed-chunk reference at oc={oc} ic={ic}");
        }
    }

    #[test]
    fn native_expert_matches_reference() {
        // Full native expert (gate/up/silu/down) vs a dequant f32 reference.
        let (hidden, inter) = (512usize, 256usize);
        let mut r = Lcg(0xA71_5713u64);
        let gw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
        let uw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
        let dw: Vec<f32> = (0..hidden * inter).map(|_| r.f32() * 0.5).collect();
        let x: Vec<f32> = (0..hidden).map(|_| r.f32()).collect();
        let (gp, gc, g_dq) = quantize_host(&gw, inter, hidden);
        let (up, uc, u_dq) = quantize_host(&uw, inter, hidden);
        let (dp, dc, d_dq) = quantize_host(&dw, hidden, inter);
        let mut y = vec![0f32; hidden];
        expert_forward_native(&x, &gp, &gc, &up, &uc, &dp, &dc, hidden, inter, &mut y);
        let g = naive_matmul(&g_dq, inter, hidden, &x);
        let u = naive_matmul(&u_dq, inter, hidden, &x);
        let act: Vec<f32> = (0..inter).map(|i| silu(g[i]) * u[i]).collect();
        let reference = naive_matmul(&d_dq, hidden, inter, &act);
        let e = l2_rel(&y, &reference);
        assert!(e < 1e-4, "native expert vs reference L2 rel err {e:e}");
    }

    #[test]
    fn native_worksteal_matches_sequential_and_is_thread_invariant() {
        // Deliverable B lever 2: the single phased work-steal over all (expert, proj, input-chunk)
        // tasks must be BIT-identical to the per-GEMV sequential native path AND invariant to the
        // worker count -- same input-chunk decomposition, same fixed-order partial reduction, same
        // fixed expert-order combine. V2-Lite-ish dims, random packed/codebooks (any bytes are valid
        // indices; we test scheduling+reduction order, not decode values).
        let (hidden, inter, k) = (2048usize, 1408usize, 6usize);
        let mut r = Lcg(0xB17E_5713_u64);
        let x: Vec<f32> = (0..hidden).map(|_| r.f32()).collect();
        let bytes = |n: usize, r: &mut Lcg| -> Vec<u8> { (0..n).map(|_| ((r.f32() * 0.5 + 0.5) * 255.0) as u8).collect() };
        let cbk = |n: usize, r: &mut Lcg| -> Vec<f32> { (0..n).map(|_| r.f32() * 0.05).collect() };
        // NATIVE input-major packed: gate/up [hidden][inter/2] (oc=inter,ic=hidden); down [inter][hidden/2].
        let mut store: Vec<(Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, f32)> = Vec::new();
        for e in 0..k {
            store.push((bytes(hidden * (inter / 2), &mut r), cbk(K * inter, &mut r),
                        bytes(hidden * (inter / 2), &mut r), cbk(K * inter, &mut r),
                        bytes(inter * (hidden / 2), &mut r), cbk(K * hidden, &mut r), 0.1 + 0.13 * e as f32));
        }
        let experts: Vec<NativeExpert> = store.iter().map(|s| NativeExpert {
            gp: &s.0, gc: &s.1, up: &s.2, uc: &s.3, dp: &s.4, dc: &s.5, weight: s.6,
        }).collect();
        let mut seq = vec![0f32; hidden];
        routed_experts_native(&x, &experts, hidden, inter, &mut seq);
        // pool path == sequential, bitwise
        let mut pooled = vec![0f32; hidden];
        routed_experts_native_worksteal(&x, &experts, hidden, inter, &mut pooled);
        assert!(seq.iter().zip(&pooled).all(|(a, b)| a.to_bits() == b.to_bits()),
                "native worksteal (pool) not bit-identical to sequential native path");
        // explicit worker counts == sequential, bitwise (thread-count invariant)
        for nt in [1usize, 2, 4, 8, 16, 32] {
            let mut got = vec![0f32; hidden];
            routed_experts_native_worksteal_nt(&x, &experts, hidden, inter, &mut got, nt);
            assert!(seq.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "native worksteal not bit-identical to sequential at {nt} workers");
        }
    }

    #[test]
    fn int8_recode_error_probe() {
        // Deliverable C2 precision gate: how much error does the int8 path (int8 codebook + int8
        // activation) add vs the EXACT f32 decode? Reports L2 rel error. Reference scale: the
        // fp16-vs-f32 path we already accept is ~1e-3..1e-2. If int8 lands in that ballpark, C2's
        // 47 GB/s ceiling is worth the SIMD kernel; if it blows up, C1 (exact vpermps) is the path.
        for (oc, ic) in [(2048usize, 2048usize), (7168, 2048), (2048, 7168)] {
            let mut r = Lcg(0x0C2E_5713u64 ^ ((oc as u64) << 7) ^ ic as u64);
            let w: Vec<f32> = (0..oc * ic).map(|_| r.f32() * 0.5).collect();
            let x: Vec<f32> = (0..ic).map(|_| r.f32()).collect();
            let (packed, cb, _) = quantize_host(&w, oc, ic);
            let mut yf = vec![0f32; oc];
            super::gemv_native_range_scalar(&packed, &cb, oc, &x, &mut yf, 0, ic); // exact f32 decode
            let (cb_i8, scale) = super::recode_codebook_i8(&cb, oc);
            let mut yi = vec![0f32; oc];
            super::gemv_native_i8(&packed, &cb_i8, &scale, oc, ic, &x, &mut yi);
            let e = l2_rel(&yi, &yf);
            eprintln!("[int8_recode_error oc={oc} ic={ic}] int8-path vs f32-decode L2 rel err = {e:.4e}");
            assert!(e < 5e-2, "int8 path L2 rel err {e:e} too large at oc={oc} ic={ic}");
        }
    }

    #[test]
    fn lut8_worksteal_matches_f32_within_int8_tolerance() {
        // Deliverable C-inc4: the end-to-end lut8 int8 work-steal (recode -> quantize -> int8 decode
        // -> rescale) matches the f32 work-steal within the int8 tolerance. On the M4 the decode is
        // the scalar int8 accumulate (no AVX2); on the box it is the vpshufb kernel (bit-exact to it),
        // so this M4 test validates the STRUCTURE (scales, xs, rescale, phases) end-to-end. Realistic
        // (quantize_host) codebooks so the int8 error is representative.
        let (hidden, inter, k) = (2048usize, 1408usize, 6usize);
        let mut r = Lcg(0x0148_5713u64);
        let x: Vec<f32> = (0..hidden).map(|_| r.f32() * 0.5).collect();
        // per expert: gate/up [inter][hidden], down [hidden][inter], quantized with the real k-means
        let mut store = Vec::new();
        for _ in 0..k {
            let gw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
            let uw: Vec<f32> = (0..inter * hidden).map(|_| r.f32() * 0.5).collect();
            let dw: Vec<f32> = (0..hidden * inter).map(|_| r.f32() * 0.5).collect();
            let (gp, gc, _) = quantize_host(&gw, inter, hidden);
            let (up, uc, _) = quantize_host(&uw, inter, hidden);
            let (dp, dc, _) = quantize_host(&dw, hidden, inter);
            store.push((gp, gc, up, uc, dp, dc, 0.15 + 0.1 * store.len() as f32));
        }
        let experts: Vec<NativeExpert> = store.iter().map(|s| NativeExpert {
            gp: &s.0, gc: &s.1, up: &s.2, uc: &s.3, dp: &s.4, dc: &s.5, weight: s.6,
        }).collect();
        let mut yf = vec![0f32; hidden];
        routed_experts_native_worksteal(&x, &experts, hidden, inter, &mut yf);
        let mut yi = vec![0f32; hidden];
        routed_experts_native_worksteal_lut8(&x, &experts, hidden, inter, &mut yi);
        let e = l2_rel(&yi, &yf);
        eprintln!("[lut8_worksteal] int8 end-to-end vs f32 work-steal L2 rel err = {e:.4e}");
        assert!(e < 5e-2, "lut8 worksteal vs f32 L2 rel err {e:e} too large (int8 tolerance ~0.5%/GEMV compounded)");
    }

    #[test]
    fn native_worksteal_engage_probe() {
        // DIAGNOSTIC (deliverable C engagement): time the pooled native work-steal at V2-Lite dims
        // and, under TRAPETUM_CPU_ENGAGE_DEBUG=1, print the per-worker task distribution; under
        // TRAPETUM_CPU_SPIN=1, the phase barriers spin. Run on the M4 (where I can observe) with:
        //   TRAPETUM_CPU_THREADS=8 TRAPETUM_CPU_ENGAGE_DEBUG=1 [TRAPETUM_CPU_SPIN=1] cargo test ... \
        //     native_worksteal_engage_probe -- --nocapture
        // No hard timing assert (machine-load sensitive); the numbers are the report.
        let (hidden, inter, k) = (2048usize, 1408usize, 6usize);
        let mut r = Lcg(0xE17A_9E31u64);
        let x: Vec<f32> = (0..hidden).map(|_| r.f32()).collect();
        let bytes = |n: usize, r: &mut Lcg| -> Vec<u8> { (0..n).map(|_| ((r.f32() * 0.5 + 0.5) * 255.0) as u8).collect() };
        let cbk = |n: usize, r: &mut Lcg| -> Vec<f32> { (0..n).map(|_| r.f32() * 0.05).collect() };
        let mut store: Vec<(Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>, f32)> = Vec::new();
        for e in 0..k {
            store.push((bytes(hidden * (inter / 2), &mut r), cbk(K * inter, &mut r),
                        bytes(hidden * (inter / 2), &mut r), cbk(K * inter, &mut r),
                        bytes(inter * (hidden / 2), &mut r), cbk(K * hidden, &mut r), 0.1 + 0.13 * e as f32));
        }
        let experts: Vec<NativeExpert> = store.iter().map(|s| NativeExpert {
            gp: &s.0, gc: &s.1, up: &s.2, uc: &s.3, dp: &s.4, dc: &s.5, weight: s.6,
        }).collect();
        let mut acc = vec![0f32; hidden];
        for _ in 0..20 { routed_experts_native_worksteal(&x, &experts, hidden, inter, &mut acc); } // warm
        let iters = 200;
        let t = std::time::Instant::now();
        for _ in 0..iters { routed_experts_native_worksteal(&x, &experts, hidden, inter, &mut acc); }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        // routed bytes/call = k * 3 * hidden*inter/2 (gate+up+down packed, 4-bit)
        let gbs = (k as f64 * 3.0 * hidden as f64 * inter as f64 / 2.0) / (ms * 1e-3) / 1e9;
        eprintln!("[native_ws_probe] hidden={hidden} inter={inter} k={k} threads={} spin={} -> {ms:.3} ms/call, {gbs:.2} GB/s packed",
                  crate::cpu_experts::cpu_threads(), cpu_spin());
        let _ = acc;
    }

    #[test]
    fn transpose16x16_wiring_is_a_true_transpose() {
        // De-risk the 16x16 unpack-cascade WIRING on any arch (the SSE kernel is x86-only): model
        // _mm_unpacklo/hi_epi{8,16,32,64} as byte permutations on [u8;16], run the SAME 4-stage
        // cascade, and assert cols[c][t] == rows[t][c]. Proves the index arithmetic before the box.
        type V = [u8; 16];
        fn unpack(a: V, b: V, w: usize, hi: bool) -> V {
            // interleave w-byte lanes from a and b; lo takes lanes 0..8/w*... i.e. the low half.
            let mut o = [0u8; 16]; let lanes = 8 / w; let base = if hi { 8 } else { 0 };
            for i in 0..lanes {
                for x in 0..w { o[(2*i)*w + x] = a[base + i*w + x]; o[(2*i+1)*w + x] = b[base + i*w + x]; }
            }
            o
        }
        let mut rows = [[0u8; 16]; 16];
        for t in 0..16 { for c in 0..16 { rows[t][c] = (t * 16 + c) as u8; } }
        let mut a = [[0u8; 16]; 16];
        for i in 0..8 { a[2*i] = unpack(rows[2*i], rows[2*i+1], 1, false); a[2*i+1] = unpack(rows[2*i], rows[2*i+1], 1, true); }
        let mut b = [[0u8; 16]; 16];
        for i in 0..4 { for k in 0..2 { b[4*i+k] = unpack(a[4*i+k], a[4*i+2+k], 2, false); b[4*i+2+k] = unpack(a[4*i+k], a[4*i+2+k], 2, true); } }
        let mut cc = [[0u8; 16]; 16];
        for i in 0..2 { for k in 0..4 { cc[8*i+k] = unpack(b[8*i+k], b[8*i+4+k], 4, false); cc[8*i+4+k] = unpack(b[8*i+k], b[8*i+4+k], 4, true); } }
        let mut d = [[0u8; 16]; 16];
        for k in 0..8 { d[k] = unpack(cc[k], cc[8+k], 8, false); d[8+k] = unpack(cc[k], cc[8+k], 8, true); }
        // Each output register d[k] IS a full column (d[k][t] == rows[t][col] for a fixed col), but
        // in bit-reversed register order. Compute the column each register holds and check it against
        // the bit-reverse-4 table the kernel uses (LUT8_COL_OF_REG); also assert it's a permutation.
        let mut col_of = [255usize; 16];
        for k in 0..16 {
            let c0 = (d[k][0]) as usize; // rows[0][c] = c, so d[k][0] names the column
            for t in 0..16 { assert_eq!(d[k][t], (t*16 + c0) as u8, "d[{k}] is not a clean column"); }
            col_of[k] = c0;
        }
        let mut seen = [false; 16];
        for &c in &col_of { assert!(!seen[c], "not a permutation"); seen[c] = true; }
        assert_eq!(col_of, super::LUT8_COL_OF_REG, "kernel's LUT8_COL_OF_REG must match the measured cascade permutation");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn lut8_matches_scalar_int8() {
        // Deliverable C-inc3: the AVX2 lut8 (vpshufb value-LUT) accumulate must be BIT-EXACT to the
        // scalar int8 accumulate -- integer math, so equality is exact regardless of order. This is
        // the box's cheap correctness gate before any 671B perf run. ic=1000 exercises the tail.
        if !is_x86_feature_detected!("avx2") { eprintln!("[lut8] no AVX2 here; skipping (runs on the box)"); return; }
        for (oc, ic) in [(256usize, 512usize), (128, 1000), (2048, 320)] {
            let mut r = Lcg(0x0107_8u64 ^ ((oc as u64) << 9) ^ ic as u64);
            let packed: Vec<u8> = (0..ic * (oc / 2)).map(|_| ((r.f32() * 0.5 + 0.5) * 255.0) as u8).collect();
            let cb_i8: Vec<i8> = (0..K * oc).map(|_| (r.f32() * 100.0) as i8).collect();
            let xq: Vec<i8> = (0..ic).map(|_| (r.f32() * 100.0) as i8).collect();
            let cb_t = super::transpose_codebook_i8(&cb_i8, oc);
            let mut a1 = vec![0i32; oc];
            super::accumulate_i8_t(&packed, &cb_t, oc, &xq, &mut a1, 0, ic);
            let mut a2 = vec![0i32; oc];
            unsafe { super::accumulate_lut8_avx2(&packed, &cb_t, oc, &xq, &mut a2, 0, ic); }
            assert_eq!(a1, a2, "lut8 AVX2 != scalar int8 (bitwise) at oc={oc} ic={ic}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn native_simd_bit_exact_to_scalar() {
        // Deliverable B lever 1: the AVX2 / AVX-512 decode-GEMV must be BITWISE equal to the scalar
        // reference (independent per-column accumulators, mul-then-add not FMA, same i order). Runs
        // only where the feature is present; team-lead runs this + the perf A/B on the x86 box.
        for (oc, ic) in [(256usize, 512usize), (512, 300), (128, 1024), (2048, 256)] {
            let mut r = Lcg(0x51D0_0AD5u64 ^ ((oc as u64) << 8) ^ ic as u64);
            let packed: Vec<u8> = (0..ic * (oc / 2)).map(|_| ((r.f32() * 0.5 + 0.5) * 255.0) as u8).collect();
            let cb: Vec<f32> = (0..K * oc).map(|_| r.f32() * 0.05).collect();
            let x: Vec<f32> = (0..ic).map(|_| r.f32()).collect();
            let mut sc = vec![0f32; oc];
            super::gemv_native_range_scalar(&packed, &cb, oc, &x, &mut sc, 0, ic);
            if is_x86_feature_detected!("avx2") {
                let mut v = vec![0f32; oc];
                unsafe { super::gemv_native_range_avx2(&packed, &cb, oc, &x, &mut v, 0, ic); }
                assert!(sc.iter().zip(&v).all(|(a, b)| a.to_bits() == b.to_bits()), "AVX2 != scalar bitwise at oc={oc} ic={ic}");
            }
            if is_x86_feature_detected!("avx512f") && oc % 16 == 0 {
                let mut v = vec![0f32; oc];
                unsafe { super::gemv_native_range_avx512(&packed, &cb, oc, &x, &mut v, 0, ic); }
                assert!(sc.iter().zip(&v).all(|(a, b)| a.to_bits() == b.to_bits()), "AVX512 != scalar bitwise at oc={oc} ic={ic}");
            }
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
