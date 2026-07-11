//! Run a compressed DeepSeek-V2/V3 (MLA+MoE) model in the pure-Rust runtime.
//!   deepseek_run <model.cbk> <prompt.bin>   (greedy-decodes 24 tokens, prints ids)
use std::env;
use std::time::Instant;
use trapetum::{DeepSeekModel, argmax, dump_logit_margin, read_i32s};
fn main() {
    let a: Vec<String> = env::args().collect();
    let prompt: Vec<usize> = read_i32s(&a[2]).iter().map(|&t| t as usize).collect();
    let t0 = Instant::now();
    let mut m = DeepSeekModel::load_deepseek(&a[1], 2048).unwrap();
    println!("loaded in {:.1}s, vocab={}, prompt={} tokens {:?}", t0.elapsed().as_secs_f64(), m.vocab(), prompt.len(), prompt);
    // TRAPETUM_LAYER_DEBUG=1: dump per-layer hidden-state stats for the LAST prompt position
    // (which predicts token 1) so we can diff layer-by-layer vs an HF reference (model/dump_layers_hf.py).
    let layer_dbg = std::env::var("TRAPETUM_LAYER_DEBUG").map(|v| v == "1").unwrap_or(false);
    let mut last = vec![];
    let last_i = prompt.len().saturating_sub(1);
    for (t, &tok) in prompt.iter().enumerate() {
        last = if layer_dbg && t == last_i { m.forward_dump(tok, t) } else { m.forward(tok, t) };
    }
    // TRAPETUM_NTOK overrides the generation length (default 24); per-token
    // times expose the cold-to-warm transition when experts page in from disk.
    let n: usize = env::var("TRAPETUM_NTOK").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    // TRAPETUM_LOGIT_DEBUG=1: report the top-2/margin of the token chosen from the prompt's last
    // logits and of every generated token, so GPU-vs-hybrid divergences can be classified as
    // near-ties (fp noise) vs real gaps. deepseek_run argmaxes host logits directly, so it dumps here.
    dump_logit_margin(prompt.len().saturating_sub(1), &last);
    let mut tok = argmax(&last); let mut got = Vec::new();
    let mut per_tok = Vec::with_capacity(n);
    let t0 = Instant::now();
    for i in 0..n {
        got.push(tok);
        let ti = Instant::now();
        let lg = m.forward(tok, prompt.len()+i);
        per_tok.push(ti.elapsed().as_secs_f64()*1e3);
        dump_logit_margin(prompt.len()+i, &lg);
        tok = argmax(&lg);
        println!("tok {:>3}: {:>9.1} ms", i, per_tok[i]);
    }
    let ms = t0.elapsed().as_secs_f64()*1e3/n as f64;
    println!("continuation ids: {:?}", got);
    println!("decode: {ms:.1} ms/token ({:.1} tok/s), pure Rust MLA+MoE, no Python", 1e3/ms);
    if n >= 16 {
        let tail = &per_tok[n/2..];
        let tail_ms = tail.iter().sum::<f64>()/tail.len() as f64;
        println!("steady-state (last {} tokens): {:.1} ms/token ({:.2} tok/s)", tail.len(), tail_ms, 1e3/tail_ms);
    }
}
