//! Metal backend: the same 24-function seam as the CUDA wrapper, implemented on
//! MTLBuffer + a single MTLCommandQueue. Buffers are storageModeShared (Apple
//! unified memory), so downloads are plain memcpy after a queue drain.
//!
//! Correctness-first: one command buffer per op, in-order queue. Batching a whole
//! decode step into one command buffer (the CUDA-graph equivalent) is the next
//! work package; graph_* therefore panics with a clear message for now.
#![allow(clippy::missing_safety_doc)]
use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::{Mutex, OnceLock};

use metal_rs::{
    Buffer, CommandBuffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize, NSRange,
};

const K: usize = 16;
const CPB: usize = 256;
const TY: usize = 8;

// IC-split grid.y for the fused GEMV. 20 was tuned for Ampere's ~108 SMs; Apple
// GPUs have far fewer cores, so fewer, fatter threadgroups (less atomic-reduce
// contention) win. Overridable via TRAPETUM_GS for tuning; default 8.
fn gs() -> u64 {
    static G: OnceLock<u64> = OnceLock::new();
    *G.get_or_init(|| {
        std::env::var("TRAPETUM_GS").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
    })
}

const METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.metallib"));
const KERNELS: &[&str] = &[
    "gemv4", "cast_f2h", "rmsnorm_k", "silu_mul_k", "resadd_k", "rope_k", "vadd_k", "attn_k",
    "gemv_fp16", "gemm_mtile", "gemm_mtile2",
];

struct Ctx {
    _device: Device,
    queue: CommandQueue,
    pl: HashMap<&'static str, ComputePipelineState>,
    // Current OPEN (uncommitted) command buffer. Every op encodes into it and
    // does NOT commit; drain() (called by dev_sync and any host read-back)
    // commits it once. So a whole decode step between two sync() calls becomes a
    // SINGLE command buffer, the Metal equivalent of the CUDA-graph lesson that
    // turns a per-op kernel win into an end-to-end one.
    cur: Mutex<Option<CommandBuffer>>,
}
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

fn ctx() -> &'static Ctx {
    static CTX: OnceLock<Ctx> = OnceLock::new();
    CTX.get_or_init(|| {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_data(METALLIB)
            .expect("failed to load kernels.metallib");
        let mut pl = HashMap::new();
        for name in KERNELS {
            let f = lib.get_function(name, None).expect("missing kernel");
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .expect("pipeline creation failed");
            pl.insert(*name, p);
        }
        Ctx { _device: device, queue, pl, cur: Mutex::new(None) }
    })
}

/// The current open command buffer, created lazily. Ops append encoders to it
/// and never commit; only drain() commits.
fn cur_cb() -> CommandBuffer {
    let mut g = ctx().cur.lock().unwrap();
    if g.is_none() {
        *g = Some(ctx().queue.new_command_buffer().to_owned());
    }
    g.as_ref().unwrap().to_owned()
}

fn drain() {
    if let Some(cb) = ctx().cur.lock().unwrap().take() {
        cb.commit();
        cb.wait_until_completed();
    }
}

unsafe fn bufref<'a>(p: *const c_void) -> &'a Buffer {
    &*(p as *const Buffer)
}

fn alloc_bytes(len: usize) -> *mut c_void {
    let b = ctx()
        ._device
        .new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
    Box::into_raw(Box::new(b)) as *mut c_void
}

/// Ceil to the 16-byte granularity Metal requires for threadgroup memory lengths.
fn tg(len: usize) -> u64 {
    (((len + 15) / 16) * 16) as u64
}

// ---- 1D elementwise dispatch helper -----------------------------------------
fn dispatch1d(name: &'static str, bufs: &[&Buffer], scalars: &[u8], n: usize) {
    let c = ctx();
    let cb = cur_cb();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl[name]);
    for (i, b) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.set_bytes(
        bufs.len() as u64,
        scalars.len() as u64,
        scalars.as_ptr() as *const c_void,
    );
    let tpb = 256u64;
    let grid = MTLSize::new((n as u64 + tpb - 1) / tpb, 1, 1);
    enc.dispatch_thread_groups(grid, MTLSize::new(tpb, 1, 1));
    enc.end_encoding();
}

// ---- QLinear -----------------------------------------------------------------
struct QLin {
    packed: Buffer,
    cb: Buffer,
    ic: i32,
    oc: i32,
}

pub unsafe fn qlinear_create(packed: *const u8, cb_f32: *const f32, ic: i32, oc: i32) -> *mut c_void {
    let c = ctx();
    let np = ic as usize * (oc as usize / 2);
    let pb = c
        ._device
        .new_buffer(np as u64, MTLResourceOptions::StorageModeShared);
    std::ptr::copy_nonoverlapping(packed, pb.contents() as *mut u8, np);
    let ncb = K * oc as usize;
    let cbb = c
        ._device
        .new_buffer((ncb * 2) as u64, MTLResourceOptions::StorageModeShared);
    let dst = cbb.contents() as *mut u16;
    for i in 0..ncb {
        *dst.add(i) = half::f16::from_f32(*cb_f32.add(i)).to_bits();
    }
    Box::into_raw(Box::new(QLin { packed: pb, cb: cbb, ic, oc })) as *mut c_void
}

pub unsafe fn qlinear_forward_dev(h: *mut c_void, d_x: *const c_void, d_y: *mut c_void) {
    let q = &*(h as *const QLin);
    let c = ctx();
    let y = bufref(d_y);
    let cb = cur_cb();
    // zero the f32 accumulator (cudaMemsetAsync equivalent)
    let blit = cb.new_blit_command_encoder();
    blit.fill_buffer(y, NSRange::new(0, q.oc as u64 * 4), 0);
    blit.end_encoding();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["gemv4"]);
    enc.set_buffer(0, Some(bufref(d_x)), 0);
    enc.set_buffer(1, Some(&q.packed), 0);
    enc.set_buffer(2, Some(&q.cb), 0);
    enc.set_buffer(3, Some(y), 0);
    let p: [i32; 2] = [q.ic, q.oc];
    enc.set_bytes(4, 8, p.as_ptr() as *const c_void);
    enc.set_threadgroup_memory_length(0, tg(K * CPB * 2));
    enc.set_threadgroup_memory_length(1, tg(TY * CPB * 4));
    enc.dispatch_thread_groups(
        MTLSize::new(q.oc as u64 / CPB as u64, gs(), 1),
        MTLSize::new(32, TY as u64, 1),
    );
    enc.end_encoding();
}

pub unsafe fn qlinear_free(h: *mut c_void) {
    drop(Box::from_raw(h as *mut QLin));
}

// ---- device buffers ----------------------------------------------------------
pub unsafe fn dev_alloc_half(n: i32) -> *mut c_void { alloc_bytes(n as usize * 2) }
pub unsafe fn dev_alloc_f32(n: i32) -> *mut c_void { alloc_bytes(n as usize * 4) }
pub unsafe fn dev_free(p: *mut c_void) {
    drain();
    drop(Box::from_raw(p as *mut Buffer));
}

pub unsafe fn dev_upload_to_half(d_half: *mut c_void, x: *const f32, n: i32) {
    drain();
    let dst = bufref(d_half).contents() as *mut u16;
    for i in 0..n as usize {
        *dst.add(i) = half::f16::from_f32(*x.add(i)).to_bits();
    }
}

pub unsafe fn dev_upload_f32(d_f32: *mut c_void, x: *const f32, n: i32) {
    drain();
    std::ptr::copy_nonoverlapping(x, bufref(d_f32).contents() as *mut f32, n as usize);
}

pub unsafe fn dev_cast_f32_to_half(d_half: *mut c_void, d_f32: *const c_void, n: i32) {
    dispatch1d(
        "cast_f2h",
        &[bufref(d_f32), bufref(d_half)],
        &n.to_ne_bytes(),
        n as usize,
    );
}

pub unsafe fn dev_download_f32(x: *mut f32, d_f32: *const c_void, n: i32) {
    drain();
    std::ptr::copy_nonoverlapping(bufref(d_f32).contents() as *const f32, x, n as usize);
}

pub unsafe fn dev_download_half(x: *mut f32, d_half: *const c_void, n: i32) {
    drain();
    let src = bufref(d_half).contents() as *const u16;
    for i in 0..n as usize {
        *x.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
    }
}

pub unsafe fn dev_sync() { drain(); }

// ---- transformer-block ops ----------------------------------------------------
pub unsafe fn op_rmsnorm(x_half: *const c_void, w_f32: *const c_void, out_half: *mut c_void, n: i32, eps: f32) {
    let c = ctx();
    let cb = cur_cb();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["rmsnorm_k"]);
    enc.set_buffer(0, Some(bufref(x_half)), 0);
    enc.set_buffer(1, Some(bufref(w_f32)), 0);
    enc.set_buffer(2, Some(bufref(out_half)), 0);
    #[repr(C)]
    struct P { n: i32, eps: f32 }
    let p = P { n, eps };
    enc.set_bytes(3, 8, &p as *const P as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
}

pub unsafe fn op_silu_mul(gate_f32: *const c_void, up_f32: *const c_void, out_half: *mut c_void, n: i32) {
    dispatch1d(
        "silu_mul_k",
        &[bufref(gate_f32), bufref(up_f32), bufref(out_half)],
        &n.to_ne_bytes(),
        n as usize,
    );
}

pub unsafe fn op_residual_add(h_half: *mut c_void, delta_f32: *const c_void, n: i32) {
    dispatch1d(
        "resadd_k",
        &[bufref(h_half), bufref(delta_f32)],
        &n.to_ne_bytes(),
        n as usize,
    );
}

pub unsafe fn op_rope(x_half: *mut c_void, pos: i32, n_heads: i32, head_dim: i32, inv_freq: *const c_void) {
    #[repr(C)]
    struct P { pos: i32, n_heads: i32, head_dim: i32 }
    let p = P { pos, n_heads, head_dim };
    let total = (n_heads * (head_dim / 2)) as usize;
    let bytes = std::slice::from_raw_parts(&p as *const P as *const u8, 12);
    dispatch1d("rope_k", &[bufref(x_half), bufref(inv_freq)], bytes, total);
}

pub unsafe fn op_vadd(a_f32: *mut c_void, b_f32: *const c_void, n: i32) {
    dispatch1d(
        "vadd_k",
        &[bufref(a_f32), bufref(b_f32)],
        &n.to_ne_bytes(),
        n as usize,
    );
}

pub unsafe fn op_cache_append(cache_half: *mut c_void, src_half: *const c_void, pos: i32, dim: i32) {
    let cb = cur_cb();
    let blit = cb.new_blit_command_encoder();
    blit.copy_from_buffer(
        bufref(src_half),
        0,
        bufref(cache_half),
        pos as u64 * dim as u64 * 2,
        dim as u64 * 2,
    );
    blit.end_encoding();
}

pub unsafe fn op_attn(
    q_half: *const c_void,
    ck_half: *const c_void,
    cv_half: *const c_void,
    out_half: *mut c_void,
    n_heads: i32,
    n_kv: i32,
    head_dim: i32,
    seqlen: i32,
) {
    let c = ctx();
    let cb = cur_cb();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["attn_k"]);
    enc.set_buffer(0, Some(bufref(q_half)), 0);
    enc.set_buffer(1, Some(bufref(ck_half)), 0);
    enc.set_buffer(2, Some(bufref(cv_half)), 0);
    enc.set_buffer(3, Some(bufref(out_half)), 0);
    let p: [i32; 4] = [n_heads, n_kv, head_dim, seqlen];
    enc.set_bytes(4, 16, p.as_ptr() as *const c_void);
    enc.set_threadgroup_memory_length(0, tg(head_dim as usize * 4));
    enc.set_threadgroup_memory_length(1, tg(seqlen as usize * 4));
    enc.dispatch_thread_groups(
        MTLSize::new(n_heads as u64, 1, 1),
        MTLSize::new(head_dim as u64, 1, 1),
    );
    enc.end_encoding();
}

// ---- graph capture: a WP2 item on Metal ----------------------------------------
// The CUDA path records the decode chain into a replayable graph. The Metal
// equivalent (encoding the whole step into one command buffer) lands with the
// host-layer work package; the direct per-op path above is fully functional.
pub unsafe fn graph_begin() {
    panic!("Metal backend: graph capture arrives with the host-layer work package; use the direct decode path")
}
pub unsafe fn graph_end() -> *mut c_void {
    panic!("Metal backend: graph capture arrives with the host-layer work package")
}
pub unsafe fn graph_launch(_exec: *mut c_void) {
    panic!("Metal backend: graph capture arrives with the host-layer work package")
}
pub unsafe fn graph_free(_exec: *mut c_void) {}

// --- microbenchmark: fused 4-bit decode GEMV vs dense fp16 GEMV --------------
// Returns (ms_4bit, ms_fp16) averaged over `iters` at IC x OC, the Apple
// analogue of the paper's cuBLAS comparison. Random data (perf is data
// independent for a memory-bound GEMV).
pub unsafe fn bench_gemv(ic: i32, oc: i32, iters: i32) -> (f64, f64) {
    use std::time::Instant;
    let c = ctx();
    // 4-bit path: build a QLinear (packed indices + fp16 codebook) and time forward.
    let np = ic as usize * (oc as usize / 2);
    let packed: Vec<u8> = (0..np).map(|i| (i * 37 + 11) as u8).collect();
    let cbk: Vec<f32> = (0..K * oc as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let q = qlinear_create(packed.as_ptr(), cbk.as_ptr(), ic, oc);
    let xf: Vec<f32> = (0..ic as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let dx = dev_alloc_half(ic);
    dev_upload_to_half(dx, xf.as_ptr(), ic);
    let dy = dev_alloc_f32(oc);
    // warmup + time
    for _ in 0..3 { qlinear_forward_dev(q, dx, dy); }
    drain();
    let t = Instant::now();
    for _ in 0..iters { qlinear_forward_dev(q, dx, dy); }
    drain();
    let ms_4bit = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // fp16 dense path: full [oc*ic] fp16 weight, naive GEMV.
    let wbuf = c._device.new_buffer((oc as u64 * ic as u64) * 2, MTLResourceOptions::StorageModeShared);
    {
        let dst = wbuf.contents() as *mut u16;
        for i in 0..(oc as usize * ic as usize) { *dst.add(i) = half::f16::from_f32(((i % 19) as f32 - 9.0) * 0.01).to_bits(); }
    }
    let run_fp16 = || {
        let cb = cur_cb();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.pl["gemv_fp16"]);
        enc.set_buffer(0, Some(&wbuf), 0);
        enc.set_buffer(1, Some(bufref(dx)), 0);
        enc.set_buffer(2, Some(bufref(dy)), 0);
        let p: [i32; 2] = [ic, oc];
        enc.set_bytes(3, 8, p.as_ptr() as *const c_void);
        let tpb = 256u64;
        enc.dispatch_thread_groups(MTLSize::new((oc as u64 + tpb - 1)/tpb, 1, 1), MTLSize::new(tpb, 1, 1));
        enc.end_encoding();
    };
    for _ in 0..3 { run_fp16(); } drain();
    let t = Instant::now();
    for _ in 0..iters { run_fp16(); } drain();
    let ms_fp16 = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    qlinear_free(q); dev_free(dx); dev_free(dy);
    (ms_4bit, ms_fp16)
}

// --- M0 microbenchmark: small-M fused decode GEMM stays bandwidth-bound? --------
// Times gemm_mtile at a fixed IC x OC for a given M (columns verified at once).
// If ms(M=6) ~= ms(M=1), the verification of K+1 tokens costs ~one weight read:
// the whole speculative-decoding speedup is unlocked. Returns avg ms.
pub unsafe fn bench_mtile(ic: i32, oc: i32, m: i32, iters: i32) -> f64 {
    use std::time::Instant;
    let c = ctx();
    let np = ic as usize * (oc as usize / 2);
    let packed: Vec<u8> = (0..np).map(|i| (i * 37 + 11) as u8).collect();
    let cbk: Vec<f32> = (0..K * oc as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let q = qlinear_create(packed.as_ptr(), cbk.as_ptr(), ic, oc);
    let ql = &*(q as *const QLin);
    // X is [M][IC] fp16, Y is [M][OC] f32
    let xf: Vec<f32> = (0..(m as usize * ic as usize)).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let dx = dev_alloc_half(m * ic);
    dev_upload_to_half(dx, xf.as_ptr(), m * ic);
    let dy = dev_alloc_f32(m * oc);
    let run = || {
        let cb = cur_cb();
        let blit = cb.new_blit_command_encoder();
        blit.fill_buffer(bufref(dy), NSRange::new(0, (m as u64) * (oc as u64) * 4), 0);
        blit.end_encoding();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.pl["gemm_mtile"]);
        enc.set_buffer(0, Some(bufref(dx)), 0);
        enc.set_buffer(1, Some(&ql.packed), 0);
        enc.set_buffer(2, Some(&ql.cb), 0);
        enc.set_buffer(3, Some(bufref(dy)), 0);
        let p: [i32; 3] = [ic, oc, m];
        enc.set_bytes(4, 12, p.as_ptr() as *const c_void);
        enc.set_threadgroup_memory_length(0, tg(K * CPB * 2));
        enc.dispatch_thread_groups(
            MTLSize::new(oc as u64 / CPB as u64, gs(), 1),
            MTLSize::new(32, TY as u64, 1),
        );
        enc.end_encoding();
    };
    for _ in 0..3 { run(); } drain();
    let t = Instant::now();
    for _ in 0..iters { run(); } drain();
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    qlinear_free(q); dev_free(dx); dev_free(dy);
    ms
}

// M0b: optimized small-M decode GEMM (2 chan/thread, no atomics). Same interface
// as bench_mtile but dispatches gemm_mtile2 (CPB2=64 tiles, one threadgroup owns
// each output tile over the full IC).
pub unsafe fn bench_mtile2(ic: i32, oc: i32, m: i32, iters: i32) -> f64 {
    use std::time::Instant;
    const CPB2: u64 = 64;
    let c = ctx();
    let np = ic as usize * (oc as usize / 2);
    let packed: Vec<u8> = (0..np).map(|i| (i * 37 + 11) as u8).collect();
    let cbk: Vec<f32> = (0..K * oc as usize).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let q = qlinear_create(packed.as_ptr(), cbk.as_ptr(), ic, oc);
    let ql = &*(q as *const QLin);
    let xf: Vec<f32> = (0..(m as usize * ic as usize)).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let dx = dev_alloc_half(m * ic);
    dev_upload_to_half(dx, xf.as_ptr(), m * ic);
    let dy = dev_alloc_f32(m * oc);
    let run = || {
        let cb = cur_cb();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.pl["gemm_mtile2"]);
        enc.set_buffer(0, Some(bufref(dx)), 0);
        enc.set_buffer(1, Some(&ql.packed), 0);
        enc.set_buffer(2, Some(&ql.cb), 0);
        enc.set_buffer(3, Some(bufref(dy)), 0);
        let p: [i32; 3] = [ic, oc, m];
        enc.set_bytes(4, 12, p.as_ptr() as *const c_void);
        enc.set_threadgroup_memory_length(0, tg(K * CPB2 as usize * 2));       // s_cb
        enc.set_threadgroup_memory_length(1, tg(TY * CPB2 as usize * m as usize * 4)); // red
        enc.dispatch_thread_groups(
            MTLSize::new(oc as u64 / CPB2, 1, 1),
            MTLSize::new(32, TY as u64, 1),
        );
        enc.end_encoding();
    };
    for _ in 0..3 { run(); } drain();
    let t = Instant::now();
    for _ in 0..iters { run(); } drain();
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    qlinear_free(q); dev_free(dx); dev_free(dy);
    ms
}
