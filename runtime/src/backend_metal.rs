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
    "gemv_fp16", "gemm_mtile", "gemm_mtile2", "rmsnorm_m", "attn_m", "rope_m", "mla_attn", "saxpy", "gelu_mul_k",
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

// Correctness of gemm_mtile: run M columns of X through it, and check each output
// column equals the single-column gemv4 result for that same X. Returns worst
// per-column relative error. This must pass before building spec-dec on it.
pub unsafe fn check_mtile(ic: i32, oc: i32, m: i32) -> f64 {
    let c = ctx();
    let np = ic as usize * (oc as usize / 2);
    let packed: Vec<u8> = (0..np).map(|i| ((i * 131 + 7) % 256) as u8).collect();
    let cbk: Vec<f32> = (0..K * oc as usize).map(|i| (((i * 7) % 31) as f32 - 15.0) * 0.02).collect();
    let q = qlinear_create(packed.as_ptr(), cbk.as_ptr(), ic, oc);
    let ql = &*(q as *const QLin);
    // distinct X per column
    let xf: Vec<f32> = (0..(m as usize * ic as usize)).map(|i| (((i * 13 + 1) % 23) as f32 - 11.0) * 0.05).collect();
    let dx = dev_alloc_half(m * ic);
    dev_upload_to_half(dx, xf.as_ptr(), m * ic);
    let dy = dev_alloc_f32(m * oc);
    // run gemm_mtile once
    {
        let cb = cur_cb();
        let blit = cb.new_blit_command_encoder();
        blit.fill_buffer(bufref(dy), NSRange::new(0, (m as u64)*(oc as u64)*4), 0);
        blit.end_encoding();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.pl["gemm_mtile"]);
        enc.set_buffer(0, Some(bufref(dx)), 0);
        enc.set_buffer(1, Some(&ql.packed), 0);
        enc.set_buffer(2, Some(&ql.cb), 0);
        enc.set_buffer(3, Some(bufref(dy)), 0);
        let p: [i32;3] = [ic, oc, m];
        enc.set_bytes(4, 12, p.as_ptr() as *const c_void);
        enc.set_threadgroup_memory_length(0, tg(K*CPB*2));
        enc.dispatch_thread_groups(MTLSize::new(oc as u64/CPB as u64, gs(), 1), MTLSize::new(32, TY as u64, 1));
        enc.end_encoding();
    }
    let mut ygemm = vec![0f32; (m as usize)*(oc as usize)];
    dev_download_f32(ygemm.as_mut_ptr(), dy, m*oc);
    // reference: per-column gemv4
    let mut worst = 0f64;
    let dxc = dev_alloc_half(ic);
    let dyc = dev_alloc_f32(oc);
    for col in 0..m as usize {
        let xcol = &xf[col*ic as usize..(col+1)*ic as usize];
        dev_upload_to_half(dxc, xcol.as_ptr(), ic);
        qlinear_forward_dev(q, dxc, dyc);
        let mut yref = vec![0f32; oc as usize];
        dev_download_f32(yref.as_mut_ptr(), dyc, oc);
        let mut num = 0f64; let mut den = 0f64;
        for o in 0..oc as usize {
            let d = (ygemm[col*oc as usize + o] - yref[o]) as f64;
            num += d*d; den += (yref[o] as f64)*(yref[o] as f64);
        }
        worst = worst.max((num/den.max(1e-30)).sqrt());
    }
    qlinear_free(q); dev_free(dx); dev_free(dy); dev_free(dxc); dev_free(dyc);
    worst
}

// Validate rmsnorm_m (batched, M rows) against the M=1 op_rmsnorm, per row.
// This is the only genuinely new kernel the batched forward needs; the linears
// (gemm_mtile) and elementwise ops are already validated / M-agnostic.
pub unsafe fn check_rmsnorm_m(n: i32, m: i32) -> f64 {
    let c = ctx();
    let eps = 1e-5f32;
    let xf: Vec<f32> = (0..(m as usize * n as usize)).map(|i| (((i * 17 + 3) % 41) as f32 - 20.0) * 0.05).collect();
    let wf: Vec<f32> = (0..n as usize).map(|i| 1.0 + ((i % 7) as f32) * 0.1).collect();
    let dx = dev_alloc_half(m * n);
    dev_upload_to_half(dx, xf.as_ptr(), m * n);
    let dw = dev_alloc_f32(n);
    dev_upload_f32(dw, wf.as_ptr(), n);
    let dout = dev_alloc_half(m * n);
    // batched
    {
        let cb = cur_cb();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.pl["rmsnorm_m"]);
        enc.set_buffer(0, Some(bufref(dx)), 0);
        enc.set_buffer(1, Some(bufref(dw)), 0);
        enc.set_buffer(2, Some(bufref(dout)), 0);
        #[repr(C)] struct P { n: i32, eps: f32 }
        let p = P { n, eps };
        enc.set_bytes(3, 8, &p as *const P as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
    let mut got = vec![0f32; (m*n) as usize];
    dev_download_half(got.as_mut_ptr(), dout, m * n);
    // reference: per-row M=1 op_rmsnorm
    let dxr = dev_alloc_half(n);
    let dor = dev_alloc_half(n);
    let mut worst = 0f64;
    for row in 0..m as usize {
        dev_upload_to_half(dxr, xf[row*n as usize..].as_ptr(), n);
        op_rmsnorm(dxr, dw, dor, n, eps);
        let mut r = vec![0f32; n as usize];
        dev_download_half(r.as_mut_ptr(), dor, n);
        let mut num = 0f64; let mut den = 0f64;
        for i in 0..n as usize { let d = (got[row*n as usize + i] - r[i]) as f64; num += d*d; den += (r[i] as f64).powi(2); }
        worst = worst.max((num/den.max(1e-30)).sqrt());
    }
    dev_free(dx); dev_free(dw); dev_free(dout); dev_free(dxr); dev_free(dor);
    worst
}

// Validate attn_m (batched causal decode attention) against a CPU reference:
// query m attends over base+m+1 keys. Returns worst rel err over the M outputs.
pub unsafe fn check_attn_m(n_heads: i32, n_kv: i32, hd: i32, base: i32, m: i32) -> f64 {
    let c = ctx();
    let qdim = (n_heads * hd) as usize;
    let kvdim = (n_kv * hd) as usize;
    let total = (base + m) as usize; // cache holds base + m rows (m new tokens appended)
    let mut rng = 0x1234_5678u64;
    let mut nx = || { rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; (((rng>>40) as f32/(1u64<<24) as f32)*2.0-1.0)*0.4 };
    let q: Vec<f32> = (0..(m as usize*qdim)).map(|_| nx()).collect();
    let ck: Vec<f32> = (0..(total*kvdim)).map(|_| nx()).collect();
    let cv: Vec<f32> = (0..(total*kvdim)).map(|_| nx()).collect();
    let dq = dev_alloc_half(m*qdim as i32); dev_upload_to_half(dq, q.as_ptr(), m*qdim as i32);
    let dck = dev_alloc_half((total*kvdim) as i32); dev_upload_to_half(dck, ck.as_ptr(), (total*kvdim) as i32);
    let dcv = dev_alloc_half((total*kvdim) as i32); dev_upload_to_half(dcv, cv.as_ptr(), (total*kvdim) as i32);
    let dout = dev_alloc_half(m*qdim as i32);
    {
        let cb = cur_cb();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.pl["attn_m"]);
        enc.set_buffer(0, Some(bufref(dq)), 0);
        enc.set_buffer(1, Some(bufref(dck)), 0);
        enc.set_buffer(2, Some(bufref(dcv)), 0);
        enc.set_buffer(3, Some(bufref(dout)), 0);
        let p: [i32;5] = [n_heads, n_kv, hd, base, m];
        enc.set_bytes(4, 20, p.as_ptr() as *const c_void);
        enc.set_threadgroup_memory_length(0, tg(hd as usize*4));
        enc.set_threadgroup_memory_length(1, tg(total*4));
        enc.dispatch_thread_groups(MTLSize::new(n_heads as u64,1,1), MTLSize::new(hd as u64,1,1));
        enc.end_encoding();
    }
    let mut got = vec![0f32; m as usize*qdim]; dev_download_half(got.as_mut_ptr(), dout, m*qdim as i32);
    // CPU reference (fp16 rounding), per query m, seqlen = base+m+1
    let h16 = |v:f32| half::f16::from_f32(v).to_f32();
    let scale = 1.0/(hd as f32).sqrt();
    let mut worst = 0f64;
    for mm in 0..m as usize {
        let seqlen = base as usize + mm + 1;
        for h in 0..n_heads as usize {
            let kvh = h / (n_heads/n_kv) as usize;
            let mut scores = vec![0f32; seqlen];
            for t in 0..seqlen {
                let mut s=0f32;
                for d in 0..hd as usize { s += h16(q[mm*qdim + h*hd as usize + d]) * h16(ck[t*kvdim + kvh*hd as usize + d]); }
                scores[t]=s*scale;
            }
            let mx=scores.iter().cloned().fold(f32::MIN,f32::max);
            let mut sum=0f32; for s in scores.iter_mut(){*s=(*s-mx).exp();sum+=*s;} for s in scores.iter_mut(){*s/=sum;}
            for d in 0..hd as usize {
                let mut acc=0f32;
                for t in 0..seqlen { acc += scores[t]*h16(cv[t*kvdim + kvh*hd as usize + d]); }
                let r = h16(acc);
                let g = got[mm*qdim + h*hd as usize + d];
                let den = (r as f64).abs().max(1e-4);
                worst = worst.max(((g-r) as f64).abs()/den);
            }
        }
    }
    dev_free(dq); dev_free(dck); dev_free(dcv); dev_free(dout);
    worst
}

// Validate rope_m (batched, per-row position base+row) vs per-row M=1 op_rope.
pub unsafe fn check_rope_m(n_heads: i32, head_dim: i32, base: i32, m: i32) -> f64 {
    let c = ctx();
    let qdim = (n_heads * head_dim) as usize;
    let inv: Vec<f32> = (0..(head_dim/2) as usize).map(|d| 10000f32.powf(-2.0*d as f32/head_dim as f32)).collect();
    let dinv = dev_alloc_f32(inv.len() as i32); dev_upload_f32(dinv, inv.as_ptr(), inv.len() as i32);
    let xf: Vec<f32> = (0..(m as usize*qdim)).map(|i| (((i*13+1)%29) as f32 - 14.0)*0.05).collect();
    // batched
    let dxb = dev_alloc_half(m*qdim as i32); dev_upload_to_half(dxb, xf.as_ptr(), m*qdim as i32);
    {
        let cb = cur_cb(); let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.pl["rope_m"]);
        enc.set_buffer(0, Some(bufref(dxb)), 0);
        enc.set_buffer(1, Some(bufref(dinv)), 0);
        let p: [i32;4] = [base, n_heads, head_dim, m];
        enc.set_bytes(2, 16, p.as_ptr() as *const c_void);
        let n = (m * n_heads * (head_dim/2)) as u64;
        enc.dispatch_thread_groups(MTLSize::new((n+255)/256,1,1), MTLSize::new(256,1,1));
        enc.end_encoding();
    }
    let mut got = vec![0f32; m as usize*qdim]; dev_download_half(got.as_mut_ptr(), dxb, m*qdim as i32);
    // reference: per-row op_rope
    let mut worst = 0f64;
    let dxr = dev_alloc_half(qdim as i32);
    for row in 0..m as usize {
        dev_upload_to_half(dxr, xf[row*qdim..].as_ptr(), qdim as i32);
        op_rope(dxr, base + row as i32, n_heads, head_dim, dinv);
        let mut r = vec![0f32; qdim]; dev_download_half(r.as_mut_ptr(), dxr, qdim as i32);
        for i in 0..qdim { let den=(r[i] as f64).abs().max(1e-4); worst=worst.max(((got[row*qdim+i]-r[i]) as f64).abs()/den); }
    }
    dev_free(dinv); dev_free(dxb); dev_free(dxr);
    worst
}

// ============================================================================
// Batched (M-token) device ops for speculative decoding. Same kernels validated
// by check_mtile / check_rmsnorm_m / check_attn_m / check_rope_m, wired as ops.
// ============================================================================

/// Batched decode GEMM: X[M][ic] fp16 -> Y[M][oc] f32 (one weight read serves all M).
pub unsafe fn qlinear_forward_m(h: *mut c_void, d_x: *const c_void, d_y: *mut c_void, m: i32) {
    let q = &*(h as *const QLin);
    let c = ctx();
    let cb = cur_cb();
    let blit = cb.new_blit_command_encoder();
    blit.fill_buffer(bufref(d_y), NSRange::new(0, (m as u64) * (q.oc as u64) * 4), 0);
    blit.end_encoding();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["gemm_mtile"]);
    enc.set_buffer(0, Some(bufref(d_x)), 0);
    enc.set_buffer(1, Some(&q.packed), 0);
    enc.set_buffer(2, Some(&q.cb), 0);
    enc.set_buffer(3, Some(bufref(d_y)), 0);
    let p: [i32; 3] = [q.ic, q.oc, m];
    enc.set_bytes(4, 12, p.as_ptr() as *const c_void);
    enc.set_threadgroup_memory_length(0, tg(K * CPB * 2));
    enc.dispatch_thread_groups(
        MTLSize::new(q.oc as u64 / CPB as u64, gs(), 1),
        MTLSize::new(32, TY as u64, 1),
    );
    enc.end_encoding();
}

/// Batched RMSNorm: x[M][n] fp16 -> out[M][n] fp16, one row per threadgroup.
pub unsafe fn op_rmsnorm_m(x_half: *const c_void, w_f32: *const c_void, out_half: *mut c_void, n: i32, eps: f32, m: i32) {
    let c = ctx();
    let cb = cur_cb();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["rmsnorm_m"]);
    enc.set_buffer(0, Some(bufref(x_half)), 0);
    enc.set_buffer(1, Some(bufref(w_f32)), 0);
    enc.set_buffer(2, Some(bufref(out_half)), 0);
    #[repr(C)]
    struct P { n: i32, eps: f32 }
    let p = P { n, eps };
    enc.set_bytes(3, 8, &p as *const P as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
}

/// Batched RoPE: rotates M rows, row r at absolute position base+r.
pub unsafe fn op_rope_m(x_half: *mut c_void, base: i32, n_heads: i32, head_dim: i32, inv_freq: *const c_void, m: i32) {
    let c = ctx();
    let cb = cur_cb();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["rope_m"]);
    enc.set_buffer(0, Some(bufref(x_half)), 0);
    enc.set_buffer(1, Some(bufref(inv_freq)), 0);
    let p: [i32; 4] = [base, n_heads, head_dim, m];
    enc.set_bytes(2, 16, p.as_ptr() as *const c_void);
    let n = (m * n_heads * (head_dim / 2)) as u64;
    enc.dispatch_thread_groups(MTLSize::new((n + 255) / 256, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
}

/// Append M contiguous new rows to the KV cache at rows base..base+m (one blit copy).
pub unsafe fn op_cache_append_m(cache_half: *mut c_void, src_half: *const c_void, base: i32, dim: i32, m: i32) {
    let cb = cur_cb();
    let blit = cb.new_blit_command_encoder();
    let nbytes = (m as u64) * (dim as u64) * 2;
    blit.copy_from_buffer(bufref(src_half), 0, bufref(cache_half), (base as u64) * (dim as u64) * 2, nbytes);
    blit.end_encoding();
}

/// Batched causal decode attention: query r attends over base+r+1 keys.
pub unsafe fn op_attn_m(q_half: *const c_void, ck_half: *const c_void, cv_half: *const c_void, out_half: *mut c_void,
                        n_heads: i32, n_kv: i32, head_dim: i32, base: i32, m: i32) {
    let c = ctx();
    let total = (base + m) as usize;
    let cb = cur_cb();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["attn_m"]);
    enc.set_buffer(0, Some(bufref(q_half)), 0);
    enc.set_buffer(1, Some(bufref(ck_half)), 0);
    enc.set_buffer(2, Some(bufref(cv_half)), 0);
    enc.set_buffer(3, Some(bufref(out_half)), 0);
    let p: [i32; 5] = [n_heads, n_kv, head_dim, base, m];
    enc.set_bytes(4, 20, p.as_ptr() as *const c_void);
    enc.set_threadgroup_memory_length(0, tg(head_dim as usize * 4));
    enc.set_threadgroup_memory_length(1, tg(total * 4));
    enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(head_dim as u64, 1, 1));
    enc.end_encoding();
}

// Validate mla_attn (MLA decode attention, absorption form) vs a CPU reference.
// Returns worst rel err over the n_heads*d_c out-latent elements.
pub unsafe fn check_mla_attn(n_heads: i32, d_c: i32, d_rope: i32, seqlen: i32) -> f64 {
    let c = ctx();
    let (nh, dc, dr, sl) = (n_heads as usize, d_c as usize, d_rope as usize, seqlen as usize);
    let mut rng = 0x9E37u64;
    let mut nx = || { rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; (((rng>>40) as f32/(1u64<<24) as f32)*2.0-1.0)*0.4 };
    let aq: Vec<f32> = (0..nh*dc).map(|_| nx()).collect();
    let qr: Vec<f32> = (0..nh*dr).map(|_| nx()).collect();
    let ckv: Vec<f32> = (0..sl*dc).map(|_| nx()).collect();
    let kr: Vec<f32> = (0..sl*dr).map(|_| nx()).collect();
    let daq = dev_alloc_half((nh*dc) as i32); dev_upload_to_half(daq, aq.as_ptr(), (nh*dc) as i32);
    let dqr = dev_alloc_half((nh*dr) as i32); dev_upload_to_half(dqr, qr.as_ptr(), (nh*dr) as i32);
    let dckv = dev_alloc_half((sl*dc) as i32); dev_upload_to_half(dckv, ckv.as_ptr(), (sl*dc) as i32);
    let dkr = dev_alloc_half((sl*dr) as i32); dev_upload_to_half(dkr, kr.as_ptr(), (sl*dr) as i32);
    let dout = dev_alloc_half((nh*dc) as i32);
    let scale = 1.0f32 / ((dc + dr) as f32).sqrt();
    {
        let cb = cur_cb();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.pl["mla_attn"]);
        enc.set_buffer(0, Some(bufref(daq)), 0);
        enc.set_buffer(1, Some(bufref(dqr)), 0);
        enc.set_buffer(2, Some(bufref(dckv)), 0);
        enc.set_buffer(3, Some(bufref(dkr)), 0);
        enc.set_buffer(4, Some(bufref(dout)), 0);
        #[repr(C)]
        struct P { n_heads: i32, d_c: i32, d_rope: i32, seqlen: i32, scale: f32 }
        let p = P { n_heads, d_c, d_rope, seqlen, scale };
        enc.set_bytes(5, 20, &p as *const P as *const c_void);
        enc.set_threadgroup_memory_length(0, tg(dc*4));
        enc.set_threadgroup_memory_length(1, tg(sl*4));
        enc.dispatch_thread_groups(MTLSize::new(nh as u64,1,1), MTLSize::new(dc as u64,1,1));
        enc.end_encoding();
    }
    let mut got = vec![0f32; nh*dc]; dev_download_half(got.as_mut_ptr(), dout, (nh*dc) as i32);
    // CPU reference (fp16 rounding)
    let h16 = |v:f32| half::f16::from_f32(v).to_f32();
    let mut worst = 0f64;
    for h in 0..nh {
        let mut sc = vec![0f32; sl];
        for t in 0..sl {
            let mut cont = 0f32;
            for d in 0..dc { cont += h16(aq[h*dc+d]) * h16(ckv[t*dc+d]); }
            let mut rp = 0f32;
            for r in 0..dr { rp += h16(qr[h*dr+r]) * h16(kr[t*dr+r]); }
            sc[t] = (cont + rp) * scale;
        }
        let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0f32; for s in sc.iter_mut() { *s = (*s-mx).exp(); sum += *s; } for s in sc.iter_mut() { *s /= sum; }
        for d in 0..dc {
            let mut acc = 0f32; for t in 0..sl { acc += sc[t]*h16(ckv[t*dc+d]); }
            let r = h16(acc); let g = got[h*dc+d];
            let den = (r as f64).abs().max(1e-3);
            worst = worst.max(((g-r) as f64).abs()/den);
        }
    }
    dev_free(daq); dev_free(dqr); dev_free(dckv); dev_free(dkr); dev_free(dout);
    worst
}

// acc += alpha * y  (f32 accumulator), for weighted MoE expert combine.
pub unsafe fn op_saxpy(acc_f32: *mut c_void, y_f32: *const c_void, alpha: f32, n: i32) {
    let c = ctx();
    let cb = cur_cb();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["saxpy"]);
    enc.set_buffer(0, Some(bufref(acc_f32)), 0);
    enc.set_buffer(1, Some(bufref(y_f32)), 0);
    #[repr(C)]
    struct P { alpha: f32, n: i32 }
    let p = P { alpha, n };
    enc.set_bytes(2, 8, &p as *const P as *const c_void);
    let tpb = 256u64;
    enc.dispatch_thread_groups(MTLSize::new((n as u64 + tpb - 1)/tpb, 1, 1), MTLSize::new(tpb, 1, 1));
    enc.end_encoding();
}

// Dense fp16 GEMV: y[oc] = W[oc][ic] @ x[ic]. For the MLA projection matrices
// (q_proj, W_DKV, per-head W_UK/W_UV absorption, o_proj) kept dense.
pub unsafe fn op_gemv_fp16(w_half: *const c_void, x_half: *const c_void, y_f32: *mut c_void, ic: i32, oc: i32) {
    let c = ctx();
    let cb = cur_cb();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["gemv_fp16"]);
    enc.set_buffer(0, Some(bufref(w_half)), 0);
    enc.set_buffer(1, Some(bufref(x_half)), 0);
    enc.set_buffer(2, Some(bufref(y_f32)), 0);
    #[repr(C)]
    struct P { ic: i32, oc: i32 }
    let p = P { ic, oc };
    enc.set_bytes(3, 8, &p as *const P as *const c_void);
    let tpb = 256u64;
    enc.dispatch_thread_groups(MTLSize::new((oc as u64 + tpb - 1)/tpb, 1, 1), MTLSize::new(tpb, 1, 1));
    enc.end_encoding();
}

// MLA decode attention op (device buffers), twin of the CUDA op_mla_attn.
pub unsafe fn op_mla_attn(aq: *const c_void, qr: *const c_void, ckv: *const c_void, kr: *const c_void, outl: *mut c_void,
                          n_heads: i32, d_c: i32, d_rope: i32, seqlen: i32, scale: f32) {
    let c = ctx();
    let cb = cur_cb();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.pl["mla_attn"]);
    enc.set_buffer(0, Some(bufref(aq)), 0);
    enc.set_buffer(1, Some(bufref(qr)), 0);
    enc.set_buffer(2, Some(bufref(ckv)), 0);
    enc.set_buffer(3, Some(bufref(kr)), 0);
    enc.set_buffer(4, Some(bufref(outl)), 0);
    #[repr(C)]
    struct P { n_heads: i32, d_c: i32, d_rope: i32, seqlen: i32, scale: f32 }
    let p = P { n_heads, d_c, d_rope, seqlen, scale };
    enc.set_bytes(5, 20, &p as *const P as *const c_void);
    enc.set_threadgroup_memory_length(0, tg(d_c as usize*4));
    enc.set_threadgroup_memory_length(1, tg(seqlen as usize*4));
    enc.dispatch_thread_groups(MTLSize::new(n_heads as u64,1,1), MTLSize::new(d_c as u64,1,1));
    enc.end_encoding();
}

// GeGLU (Gemma): out = gelu_tanh(gate) * up.
pub unsafe fn op_gelu_mul(gate_f32: *const c_void, up_f32: *const c_void, out_half: *mut c_void, n: i32) {
    dispatch1d("gelu_mul_k", &[bufref(gate_f32), bufref(up_f32), bufref(out_half)], &n.to_ne_bytes(), n as usize);
}
