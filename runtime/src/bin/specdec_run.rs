//! Two-model wall-clock speculative decode: real end-to-end timing vs plain greedy decode,
//! for K=1,2,3, with a lossless check. Both models reuse their loaded caches (decode from 0).
//!   specdec_run <target.cbk> <drafter.cbk> <N> <tok0> <tok1> ...
use std::env;
use std::time::Instant;
use trapetum::{Model, spec_decode_two_model};

fn main() {
    let a: Vec<String> = env::args().collect();
    let (tp, dp) = (&a[1], &a[2]);
    let n: usize = a[3].parse().unwrap();
    let prompt: Vec<usize> = a[4..].iter().map(|s| s.parse().unwrap()).collect();
    let mut target = Model::load_cbk(tp, 2048).unwrap();
    let mut drafter = Model::load_cbk(dp, 2048).unwrap();
    println!("target={tp}\ndrafter={dp}\nvocab={} N={n}", target.vocab());

    // plain greedy baseline (target only), timed
    let t0 = Instant::now();
    let reference = target.decode_greedy(&prompt, n);
    let base_ms = t0.elapsed().as_secs_f64() * 1e3;
    let base_tps = n as f64 / (base_ms / 1e3);
    println!("\nplain greedy (target only): {base_ms:.1} ms  ({base_tps:.1} tok/s)");

    println!("\n== two-model speculative, WALL-CLOCK ==");
    println!("{:>3} | {:>10} {:>10} | {:>8} {:>8} | {:>8} | {}", "K", "spec ms", "tok/s", "tgt fwd", "drf fwd", "speedup", "lossless");
    let mut best = (0usize, 0f64);
    for k in [1usize, 2, 3] {
        let t0 = Instant::now();
        let (seq, tf, df) = spec_decode_two_model(&mut target, &mut drafter, &prompt, n, k);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let tps = n as f64 / (ms / 1e3);
        let sp = base_ms / ms;
        let ok = seq == reference;
        if sp > best.1 { best = (k, sp); }
        println!("{:>3} | {:>10.1} {:>10.1} | {:>8} {:>8} | {:>7.2}x | {}", k, ms, tps, tf, df, sp, if ok {"OK"} else {"MISMATCH"});
        if !ok { println!("   !! output diverged from plain greedy at K={k}"); }
    }
    println!("\nbest: K={} -> {:.2}x wall-clock (lossless)", best.0, best.1);
}
