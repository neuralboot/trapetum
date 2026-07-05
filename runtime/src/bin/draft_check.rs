//! Validate a drafter .cbk loads in the runtime and decodes coherently.
//!   draft_check <model.cbk> <tok0> <tok1> ...   (prompt token ids)
//! Prints the greedy continuation token ids + decode throughput.
use std::env;
use std::time::Instant;
use trapetum::{Model, argmax};

fn main() {
    let a: Vec<String> = env::args().collect();
    let mp = &a[1];
    let prompt: Vec<usize> = a[2..].iter().map(|s| s.parse().unwrap()).collect();
    let t0 = Instant::now();
    let mut model = Model::load_cbk(mp, 1024).unwrap();
    println!("loaded {mp} in {:.1}s  vocab={}  prompt={} tokens", t0.elapsed().as_secs_f64(), model.vocab(), prompt.len());
    // prefill
    let mut last = vec![];
    for (t, &tok) in prompt.iter().enumerate() { last = model.forward(tok, t); }
    // greedy decode 24 tokens
    let n = 24usize;
    let mut tok = argmax(&last);
    let mut out = Vec::new();
    let t0 = Instant::now();
    for i in 0..n {
        out.push(tok);
        let lg = model.forward(tok, prompt.len() + i);
        tok = argmax(&lg);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("continuation ids: {:?}", out);
    println!("decode: {ms:.2} ms/token  ({:.1} tok/s)", 1e3 / ms);
}
