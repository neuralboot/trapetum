//! TWO-MODEL WALL-CLOCK speculative decoding harness. Times real end-to-end decode:
//! plain greedy on the target vs spec-dec with a real drafter model decoding
//! incrementally in its own KV cache. Verifies losslessness (outputs identical).
//!
//!   wallclock_spec <target.cbk> <drafter.cbk> <n_tokens> <tok0> <tok1> ...
//!
//! Prints per-K (1..=3) wall-clock tok/s, speedup vs plain, forward counts and the
//! effective acceptance, plus the lossless check. This replaces the projected
//! S(K) numbers with measured wall-clock.
use std::time::Instant;
use trapetum::{argmax, Model};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: wallclock_spec <target.cbk> <drafter.cbk> <n> <tok...>");
        std::process::exit(1);
    }
    let n: usize = a[3].parse().unwrap();
    let prompt: Vec<usize> = a[4..].iter().map(|s| s.parse().unwrap()).collect();
    let max_seq = (prompt.len() + n + 8).next_power_of_two().max(1024);

    let t0 = Instant::now();
    let mut target = Model::load_cbk(&a[1], max_seq).unwrap();
    let mut drafter = Model::load_cbk(&a[2], max_seq).unwrap();
    println!("loaded target+drafter in {:.1}s  (vocab {} / {})",
        t0.elapsed().as_secs_f64(), target.vocab(), drafter.vocab());
    assert_eq!(target.vocab(), drafter.vocab(), "tokenizer/vocab mismatch: not a valid pair");

    // ---- baseline: plain greedy on the target, timed ----
    let mut plain: Vec<usize> = Vec::with_capacity(n);
    let mut last = vec![];
    for (i, &t) in prompt.iter().enumerate() { last = target.forward(t, i); }
    let tp = Instant::now();
    let mut pos = prompt.len();
    let mut cur = argmax(&last);
    for _ in 0..n {
        plain.push(cur);
        last = target.forward(cur, pos);
        pos += 1;
        cur = argmax(&last);
    }
    let plain_s = tp.elapsed().as_secs_f64();
    let plain_tps = n as f64 / plain_s;
    println!("plain greedy : {:>7.1} tok/s  ({} tokens in {:.2}s)", plain_tps, n, plain_s);

    // ---- spec-dec two-model, K = 1..3, each timed from scratch ----
    for k in 1..=3usize {
        // fresh prefill for a fair wall-clock (positions restart at 0)
        let ts = Instant::now();
        let (out, t_fwds, d_fwds) = target.spec_decode_two_model(&mut drafter, &prompt, n, k);
        let spec_s = ts.elapsed().as_secs_f64();
        let spec_tps = n as f64 / spec_s;
        let lossless = out == plain;
        // acceptance: each target forward verifies k drafts; accepted = n - t_fwds
        // (every forward commits >=1 token; extras are accepted drafts/bonus)
        let acc = (n.saturating_sub(t_fwds)) as f64 / ((t_fwds * k).max(1)) as f64;
        println!(
            "spec  K={}   : {:>7.1} tok/s  speedup {:.2}x  target_fwds {}  drafter_fwds {}  eff_accept {:.3}  lossless {}",
            k, spec_tps, spec_tps / plain_tps, t_fwds, d_fwds, acc,
            if lossless { "YES" } else { "NO  <-- BUG" }
        );
        assert!(lossless, "spec-dec output diverged from plain greedy at K={k}");
    }
    println!("continuation ids: {:?}", &plain[..plain.len().min(24)]);
}
