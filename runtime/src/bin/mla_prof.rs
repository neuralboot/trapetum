//! Per-op profiling of one MLA attention layer at the EXACT DeepSeek-R1 dims, on synthetic
//! weights (timing does not depend on values). Answers: which op eats the ~2.3 ms/layer of
//! flat attention time measured on the 671B box (byte traffic only accounts for ~0.2 ms).
//!
//!   mla_prof            -- full-forward timing (respects TRAPETUM_MLA_DEVPREP/HOSTPREP env)
//!                          + per-op sync-timed breakdown of the device-prep path.
//!
//! R1 dims: hidden 7168, 128 heads, kv_lora(d_c) 512, rope 64, nope 128, v_head 128,
//! q_lora_rank 1536 -> qdim 24576, o input 16384. One layer; multiply by 61 for the model.
use std::time::Instant;
use trapetum::{DevHalf, MlaAttn};

fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*state >> 33) as u32) as f32 / u32::MAX as f32 - 0.5) * 0.04
}
fn randv(n: usize, s: &mut u64) -> Vec<f32> { (0..n).map(|_| lcg_f32(s)).collect() }

fn main() {
    let (hidden, nh, dc, dr, nope, vhd, qlr) = (7168usize, 128usize, 512usize, 64usize, 128usize, 128usize, 1536usize);
    let qdim = nh * (nope + dr);      // 24576
    let odim = nh * vhd;              // 16384
    let max_seq = 2048;
    let mut s = 1u64;
    println!("building MlaAttn at R1 dims (hidden={hidden} nh={nh} dc={dc} dr={dr} nope={nope} vhd={vhd} q_lora={qlr})...");
    let q_a_w = randv(qlr * hidden, &mut s);
    let q_a_norm = randv(qlr, &mut s);
    let qb_packed = vec![0u8; qlr * (qdim / 2)];
    let qb_cb = randv(16 * qdim, &mut s);
    let kv_a_w = randv((dc + dr) * hidden, &mut s);
    let kv_a_norm = randv(dc, &mut s);
    let kv_b = randv(nh * (nope + vhd) * dc, &mut s);
    let o_packed = vec![0u8; odim * (hidden / 2)];
    let o_cb = randv(16 * hidden, &mut s);
    let inv_freq: Vec<f32> = (0..dr / 2).map(|d| 1.0 / 10000f32.powf(2.0 * d as f32 / dr as f32)).collect();
    let mut attn = MlaAttn::new_qlora(hidden, nh, dc, dr, nope, vhd, max_seq, 1e-6, 0.135,
        qlr, &q_a_w, &q_a_norm, (&qb_packed, &qb_cb), &kv_a_w, &kv_a_norm, &kv_b,
        (&o_packed, &o_cb), &inv_freq);
    let h = DevHalf::from_host(&randv(hidden, &mut s));

    // full forward timing at growing pos (whatever prep path the env selects)
    for pos in 0..8 { let _ = attn.forward(&h, pos); }          // warmup + prime cache rows
    let n = 50;
    let t0 = Instant::now();
    for i in 0..n {
        let o = attn.forward(&h, 100 + i);
        if i == n - 1 { let _ = o.to_host(); }   // drain once at the end (to_host syncs)
    }
    let per = t0.elapsed().as_micros() as f64 / n as f64;
    println!("full forward: {per:.0} us/layer  -> x61 layers = {:.1} ms/token (env: DEVPREP={} HOSTPREP={})",
        per * 61.0 / 1000.0,
        std::env::var("TRAPETUM_MLA_DEVPREP").unwrap_or_default(),
        std::env::var("TRAPETUM_MLA_HOSTPREP").unwrap_or_default());

    // per-op sync-timed breakdown of the device-prep path (launch overhead included per op)
    let iters = 20;
    let mut agg: Vec<(&'static str, u64)> = Vec::new();
    for i in 0..iters {
        let v = attn.forward_profiled_us(&h, 200 + i);
        if agg.is_empty() { agg = v; } else { for (a, b) in agg.iter_mut().zip(v) { a.1 += b.1; } }
    }
    println!("\nper-op breakdown (sync-timed, mean of {iters}, at pos~200):");
    let total: u64 = agg.iter().map(|(_, t)| *t).sum();
    for (name, t) in &agg {
        let us = *t as f64 / iters as f64;
        println!("  {name:>10}: {us:>8.0} us  ({:.0}% )", 100.0 * *t as f64 / total as f64);
    }
    println!("  {:>10}: {:>8.0} us  -> x61 = {:.1} ms/token (sync-serialized upper bound)",
        "SUM", total as f64 / iters as f64, total as f64 / iters as f64 * 61.0 / 1000.0);
}
