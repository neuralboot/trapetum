//! Run a compressed DeepSeek-V2/V3 (MLA+MoE) model in the pure-Rust runtime.
//!   deepseek_run <model.cbk> <prompt.bin> <cont.bin>
use std::env;
use std::time::Instant;
use trapetum::{DeepSeekModel, argmax, read_i32s};
fn main() {
    let a: Vec<String> = env::args().collect();
    let prompt: Vec<usize> = read_i32s(&a[2]).iter().map(|&t| t as usize).collect();
    let cont: Vec<usize> = read_i32s(&a[3]).iter().map(|&t| t as usize).collect();
    let t0 = Instant::now();
    let mut m = DeepSeekModel::load_deepseek(&a[1], 2048).unwrap();
    println!("loaded in {:.1}s, vocab={}, prompt={} tokens", t0.elapsed().as_secs_f64(), m.vocab(), prompt.len());
    let mut last = vec![];
    for (t, &tok) in prompt.iter().enumerate() { last = m.forward(tok, t); }
    let mut tok = argmax(&last); let mut got = Vec::new();
    let t0 = Instant::now();
    for i in 0..cont.len() { got.push(tok); let lg = m.forward(tok, prompt.len()+i); tok = argmax(&lg); }
    let ms = t0.elapsed().as_secs_f64()*1e3/cont.len() as f64;
    let matched = got.iter().zip(&cont).take_while(|(a,b)| a==b).count();
    println!("continuation ids: {:?}", got);
    println!("matches HF fp16 continuation: {matched}/{} (4-bit, so coherence > exactness)", cont.len());
    println!("decode: {ms:.1} ms/token ({:.1} tok/s), pure Rust MLA+MoE, no Python", 1e3/ms);
}
