//! Run a compressed Gemma-2 model in the pure-Rust runtime (GeGLU + softcaps + 4-norm).
//!   gemma_run <model.cbk> <prompt.bin>
use std::env; use std::time::Instant;
use trapetum::{GemmaModel, argmax, read_i32s};
fn main() {
    let a: Vec<String> = env::args().collect();
    let prompt: Vec<usize> = read_i32s(&a[2]).iter().map(|&t| t as usize).collect();
    let t0 = Instant::now();
    let mut m = GemmaModel::load_gemma(&a[1], 2048).unwrap();
    println!("loaded in {:.1}s vocab={} prompt={} tokens {:?}", t0.elapsed().as_secs_f64(), m.vocab(), prompt.len(), prompt);
    let mut last = vec![];
    for (t,&tok) in prompt.iter().enumerate() { last = m.forward(tok, t); }
    let n = 24usize; let mut tok = argmax(&last); let mut got = Vec::new();
    let t0 = Instant::now();
    for i in 0..n { got.push(tok); let lg = m.forward(tok, prompt.len()+i); tok = argmax(&lg); }
    let ms = t0.elapsed().as_secs_f64()*1e3/n as f64;
    println!("continuation ids: {:?}", got);
    println!("decode: {ms:.1} ms/token ({:.1} tok/s), pure Rust Gemma-2", 1e3/ms);
}
