//! Demo: a synthetic codebook-quantized linear layer, decoded on the GPU through the
//! Rust runtime, verified against a CPU reconstruction, and timed. No Python.
use codebook_runtime::{QuantLinear, K};
use std::time::Instant;

fn main() {
    let ic = 4096usize;
    let oc = 4096usize;
    println!("codebook-runtime demo: {ic} x {oc} layer, 4-bit (K={K}), batch 1");

    // deterministic xorshift, so the run is reproducible without external deps
    let mut s: u64 = 0x1234_5678_9abc_def1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    // synthetic quantized weights: a codebook (K, oc) and indices (ic, oc) in [0,K)
    let codebook: Vec<f32> = (0..K * oc)
        .map(|_| (next() % 1000) as f32 / 10000.0 - 0.05)
        .collect();
    let mut idx = vec![0u8; ic * oc];
    for v in idx.iter_mut() {
        *v = (next() % K as u64) as u8;
    }
    // pack two 4-bit indices per byte: (ic, oc/2)
    let mut packed = vec![0u8; ic * (oc / 2)];
    for i in 0..ic {
        for j in 0..oc / 2 {
            packed[i * (oc / 2) + j] = idx[i * oc + 2 * j] | (idx[i * oc + 2 * j + 1] << 4);
        }
    }
    let x: Vec<f32> = (0..ic).map(|_| (next() % 1000) as f32 / 1000.0 - 0.5).collect();

    // run on the GPU through the runtime (weights uploaded once)
    let layer = QuantLinear::new(&packed, &codebook, ic, oc);
    let y = layer.forward(&x); // warmup + the result we verify

    // timing (host<->device copy of x and y is included; a real runtime keeps activations on device)
    let iters = 200;
    let mut sink = 0f32;
    let t0 = Instant::now();
    for _ in 0..iters {
        sink += layer.forward(&x)[0];
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    std::hint::black_box(sink);

    // CPU reference: W[i,j] = codebook[idx[i,j], j]; y[j] = sum_i x[i] * W[i,j]
    let mut yref = vec![0f64; oc];
    for i in 0..ic {
        let xi = x[i] as f64;
        let base = i * oc;
        for j in 0..oc {
            yref[j] += xi * codebook[(idx[base + j] as usize) * oc + j] as f64;
        }
    }
    let (mut num, mut den) = (0f64, 0f64);
    for j in 0..oc {
        let d = y[j] as f64 - yref[j];
        num += d * d;
        den += yref[j] * yref[j];
    }
    let rel = (num / den).sqrt();

    println!("rel err vs CPU reference : {rel:.2e}");
    println!("decode forward           : {ms:.4} ms/call (incl. host<->device copy)");
    println!("weights resident on GPU, no Python in the loop.");
    assert!(rel < 1e-2, "reconstruction error too high: {rel:.2e}");
    println!("OK");
}
