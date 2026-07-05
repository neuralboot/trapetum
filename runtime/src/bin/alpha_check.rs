//! Measure the REAL speculative-decoding acceptance rate alpha (K=1) between a drafter
//! and a target, plus each model's decode latency, and derive the projected speedup.
//! Backend-agnostic (CUDA or Metal): uses only the M=1 Model::forward, so no batched
//! kernel is needed to get the number. alpha is a property of the two distributions;
//! for K=1 greedy it is the fraction of positions where the drafter's greedy next token
//! equals the target's. Speedup = (1 + alpha) / (1 + t_draft / t_target), since the
//! bandwidth-bound M=2 verify costs ~one target forward and yields 1+alpha tokens.
//!   alpha_check <target.cbk> <drafter.cbk> <N> <tok0> <tok1> ...
use std::env;
use std::time::Instant;
use trapetum::{Model, argmax};

fn main() {
    let a: Vec<String> = env::args().collect();
    let tp = &a[1];
    let dp = &a[2];
    let n: usize = a[3].parse().unwrap();
    let prompt: Vec<usize> = a[4..].iter().map(|s| s.parse().unwrap()).collect();

    let mut target = Model::load_cbk(tp, 2048).unwrap();
    let mut draft = Model::load_cbk(dp, 2048).unwrap();
    assert_eq!(target.vocab(), draft.vocab(), "drafter and target must share a tokenizer/vocab");
    println!("target={tp}\ndrafter={dp}\nvocab={} prompt={} tokens N={n}", target.vocab(), prompt.len());

    // prefill both on the prompt
    let mut t_last = vec![]; let mut d_last = vec![];
    for (i, &tok) in prompt.iter().enumerate() { t_last = target.forward(tok, i); d_last = draft.forward(tok, i); }
    let mut t_tok = argmax(&t_last);      // target's token at the current position
    let mut d_pred = argmax(&d_last);     // drafter's prediction for the same position

    let (mut accepts, mut tt, mut dt) = (0usize, 0f64, 0f64);
    let mut pos = prompt.len();
    for _ in 0..n {
        if d_pred == t_tok { accepts += 1; }               // K=1 greedy acceptance test
        // teacher-force with the TARGET token, advance both models
        let s = Instant::now(); let tl = target.forward(t_tok, pos); tt += s.elapsed().as_secs_f64();
        let s = Instant::now(); let dl = draft.forward(t_tok, pos);  dt += s.elapsed().as_secs_f64();
        t_tok = argmax(&tl);
        d_pred = argmax(&dl);
        pos += 1;
    }
    let alpha = accepts as f64 / n as f64;
    let t_target = tt / n as f64 * 1e3;   // ms/token
    let t_draft = dt / n as f64 * 1e3;
    let r = (dt / n as f64) / (tt / n as f64); // t_draft/t_target
    println!("\n=== speculative decode: measured ===");
    println!("acceptance alpha        : {alpha:.3}  ({accepts}/{n})");
    println!("target latency          : {t_target:.2} ms/token  ({:.1} tok/s)", 1e3/t_target);
    println!("drafter latency         : {t_draft:.2} ms/token  ({:.1} tok/s)", 1e3/t_draft);
    println!("draft/target cost ratio : {r:.3}");
    // K-sweep: expected tokens per target forward = sum_{i=0..K} alpha^i = (1-a^(K+1))/(1-a);
    // per iteration cost = 1 target verify + K drafter forwards. Speedup vs plain decode:
    //   S(K) = tokens_per_forward(K) / (1 + K*ratio). The verify is bandwidth-bound (M<=4),
    //   so the target forward stays ~1 regardless of K.
    println!("\n=== projected speedup by block size K (from measured alpha, ratio) ===");
    let mut best = (1usize, 0f64);
    for k in 1usize..=3 {
        let toks = if (alpha - 1.0).abs() < 1e-9 { (k + 1) as f64 }
                   else { (1.0 - alpha.powi(k as i32 + 1)) / (1.0 - alpha) };
        let speedup = toks / (1.0 + k as f64 * r);
        if speedup > best.1 { best = (k, speedup); }
        println!("  K={k}  tokens/forward={toks:.2}  speedup={speedup:.2}x");
    }
    println!("optimal K = {}  ->  {:.2}x  (lossless; verify validated on CUDA via spec_check)", best.0, best.1);
}
