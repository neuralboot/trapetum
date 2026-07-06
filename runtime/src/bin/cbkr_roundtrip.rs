//! Round-trip check for the streaming CBKR exporter (model/export_deepseek_stream.py
//! --selftest): load the tiny synthetic .cbk it writes and run a few forward steps,
//! asserting finite logits. Catches byte-layout drift between the Python writer and the
//! Rust CBKR loader (DeepSeekModel::load_deepseek_qlora) -- independent of qlora_check.rs,
//! which validates the math in-process without ever touching the file format.
//!   python model/export_deepseek_stream.py --selftest --out /tmp/cbkr_selftest
//!   cargo run --release --no-default-features --features metal --bin cbkr_roundtrip -- \
//!       /tmp/cbkr_selftest/model_selftest.cbk
use std::env;
use trapetum::{argmax, DeepSeekModel};

fn main() {
    let a: Vec<String> = env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "/tmp/cbkr_selftest/model_selftest.cbk".to_string());
    println!("loading {path}");
    let mut m = DeepSeekModel::load_deepseek(&path, 16).expect("load_deepseek failed (byte-layout drift?)");
    println!("loaded: vocab={}", m.vocab());

    let mut tok = 1usize;
    for pos in 0..4usize {
        let logits = m.forward(tok, pos);
        assert_eq!(logits.len(), m.vocab(), "logits length != vocab at pos {pos}");
        assert!(logits.iter().all(|x| x.is_finite()), "non-finite logit at pos {pos}");
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mn = logits.iter().cloned().fold(f32::MAX, f32::min);
        tok = argmax(&logits);
        println!("  pos {pos}: argmax={tok}  min={mn:.3}  max={mx:.3}  finite=OK");
    }
    println!("\nPASS: CBKR round-trip (Python writer -> Rust load_deepseek_qlora) crash-free, finite logits.");
}
