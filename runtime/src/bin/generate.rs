//! End-to-end: load a real codebook-quantized model (`.cbk`, exported by
//! `model/export_runtime.py`), verify the per-position logits against the Python
//! reference (the same quantized weights run through HuggingFace), reproduce HF's greedy
//! continuation, and measure decode tokens/s. Pure Rust, on-device, no Python at runtime.
//!
//! Usage: generate <model.cbk> <prompt.bin> <ref.bin> <cont.bin>
//!   prompt.bin: P  i32 token ids
//!   ref.bin   : P*vocab f32 reference logits (HF on the dequantized weights)
//!   cont.bin  : N  i32 HF greedy continuation token ids
use trapetum::Model;
use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn read_i32s(path: &str) -> Vec<i32> {
    let mut b = Vec::new();
    File::open(path).unwrap().read_to_end(&mut b).unwrap();
    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_f32s(path: &str) -> Vec<f32> {
    let mut b = Vec::new();
    File::open(path).unwrap().read_to_end(&mut b).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn argmax(x: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..x.len() {
        if x[i] > x[bi] {
            bi = i;
        }
    }
    bi
}
fn rel_err(y: &[f32], r: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..y.len() {
        let d = y[i] as f64 - r[i] as f64;
        num += d * d;
        den += r[i] as f64 * r[i] as f64;
    }
    (num / den).sqrt()
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let (mp, pp, rp, cp) = (&a[1], &a[2], &a[3], &a[4]);

    let prompt: Vec<usize> = read_i32s(pp).iter().map(|&t| t as usize).collect();
    let cont: Vec<usize> = read_i32s(cp).iter().map(|&t| t as usize).collect();
    let refs = read_f32s(rp);

    let t0 = Instant::now();
    let mut model = Model::load_cbk(mp, 1024).unwrap();
    let vocab = model.vocab();
    println!("loaded {mp} in {:.1}s, vocab={vocab}, prompt={} tokens", t0.elapsed().as_secs_f64(), prompt.len());

    // verify per-position logits against the HF-on-dequantized reference
    let mut worst = 0f64;
    let mut top1 = 0usize;
    let mut last_logits = Vec::new();
    for (t, &tok) in prompt.iter().enumerate() {
        let lg = model.forward(tok, t);
        let r = &refs[t * vocab..(t + 1) * vocab];
        let e = rel_err(&lg, r);
        worst = worst.max(e);
        if argmax(&lg) == argmax(r) {
            top1 += 1;
        }
        last_logits = lg;
    }
    println!("logits rel err vs HF (worst over prompt): {worst:.2e}");
    println!("top-1 agreement with HF                 : {top1}/{}", prompt.len());

    // reproduce HF greedy continuation + time it. Device argmax: the next token is
    // reduced on the GPU, so the full vocab is never copied to the host per step.
    let mut tok = argmax(&last_logits);
    let mut got = Vec::new();
    let t0 = Instant::now();
    for i in 0..cont.len() {
        got.push(tok);
        tok = model.forward_argmax(tok, prompt.len() + i, vocab) as usize;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / cont.len() as f64;
    let matched = got.iter().zip(&cont).take_while(|(a, b)| a == b).count();

    println!("greedy continuation matches HF          : {matched}/{} tokens", cont.len());
    println!("decode throughput                       : {ms:.3} ms/token  ({:.1} tok/s)", 1e3 / ms);
    println!("real model, pure Rust, on-device, no Python at runtime.");
    // the decisive end-to-end check is reproducing HF's greedy continuation exactly;
    // the logits rel err is informational (HF and this kernel use different fp16 impls).
    if matched == cont.len() {
        println!("OK");
    } else {
        println!("MISMATCH (matched={matched}/{}, worst rel err {worst:.2e})", cont.len());
    }

    // optional energy bench (5th arg = #tokens): a long pure-decode window bracketed by
    // markers so an external power sampler can isolate steady-state decode energy.
    if let Some(nb) = a.get(5).and_then(|s| s.parse::<usize>().ok()) {
        let mut pos = prompt.len() + cont.len();
        let mut t = tok;
        for _ in 0..8 {
            t = model.forward_argmax(t, pos, vocab) as usize;
            pos += 1;
        } // warmup
        println!("READY_DECODE");
        let t0 = Instant::now();
        for _ in 0..nb {
            t = model.forward_argmax(t, pos, vocab) as usize;
            pos += 1;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / nb as f64;
        println!("DONE_DECODE {:.4} {:.2}", ms, 1e3 / ms);
    }
}
