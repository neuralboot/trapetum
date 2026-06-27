//! Demo: two codebook-quantized linear layers chained ON THE GPU, with no host<->device
//! copy between them, verified against a CPU reconstruction that emulates the fp16
//! rounding the GPU does, and timed. No Python.
use codebook_runtime::{sync, DevF32, DevHalf, QuantLinear, K};
use half::f16;
use std::time::Instant;

/// Synthetic quantized layer: a codebook (K, oc), indices (ic, oc), and the packed bytes.
fn make_layer(ic: usize, oc: usize, next: &mut impl FnMut() -> u64) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let codebook: Vec<f32> = (0..K * oc).map(|_| (next() % 1000) as f32 / 10000.0 - 0.05).collect();
    let mut idx = vec![0u8; ic * oc];
    for v in idx.iter_mut() {
        *v = (next() % K as u64) as u8;
    }
    let mut packed = vec![0u8; ic * (oc / 2)];
    for i in 0..ic {
        for j in 0..oc / 2 {
            packed[i * (oc / 2) + j] = idx[i * oc + 2 * j] | (idx[i * oc + 2 * j + 1] << 4);
        }
    }
    (packed, codebook, idx)
}

/// CPU decode: input already rounded to fp16; y[j] = sum_i x[i] * codebook[idx[i,j], j].
fn cpu_decode(x_f16: &[f32], idx: &[u8], cb: &[f32], ic: usize, oc: usize) -> Vec<f32> {
    let mut y = vec![0f64; oc];
    for i in 0..ic {
        let xi = x_f16[i] as f64;
        let b = i * oc;
        for j in 0..oc {
            y[j] += xi * cb[(idx[b + j] as usize) * oc + j] as f64;
        }
    }
    y.iter().map(|&v| v as f32).collect()
}

fn round_f16(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| f16::from_f32(v).to_f32()).collect()
}

fn main() {
    let (ic, oc) = (4096usize, 4096usize); // square so layers chain (oc1 = ic2)
    println!("codebook-runtime: 2 layers {ic}x{oc} chained on-device, 4-bit (K={K})");

    let mut s: u64 = 0x1234_5678_9abc_def1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let (p1, cb1, idx1) = make_layer(ic, oc, &mut next);
    let (p2, cb2, idx2) = make_layer(ic, oc, &mut next);
    let x: Vec<f32> = (0..ic).map(|_| (next() % 1000) as f32 / 1000.0 - 0.5).collect();

    let l1 = QuantLinear::new(&p1, &cb1, ic, oc);
    let l2 = QuantLinear::new(&p2, &cb2, ic, oc);

    // device buffers: input uploaded once, intermediate stays on the GPU
    let dx = DevHalf::from_host(&x);
    let mut dy1 = DevF32::zeros(oc);
    let mut dx2 = DevHalf::zeros(oc); // l1 output cast to fp16, on-device
    let mut dy2 = DevF32::zeros(oc);

    let run = |dx: &DevHalf, dy1: &mut DevF32, dx2: &mut DevHalf, dy2: &mut DevF32| {
        l1.forward_into(dx, dy1);
        dx2.copy_cast_from(dy1); // on-device f32 -> fp16, no host copy
        l2.forward_into(dx2, dy2);
    };

    run(&dx, &mut dy1, &mut dx2, &mut dy2); // warmup
    sync();
    let y = dy2.to_host();

    // timing: the 2-layer chain, no host<->device copy in the loop
    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&dx, &mut dy1, &mut dx2, &mut dy2);
    }
    sync();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // CPU reference, matching the GPU's fp16 rounding at the input and the intermediate
    let xh = round_f16(&x);
    let y1 = cpu_decode(&xh, &idx1, &cb1, ic, oc);
    let y1h = round_f16(&y1);
    let y2 = cpu_decode(&y1h, &idx2, &cb2, ic, oc);

    let (mut num, mut den) = (0f64, 0f64);
    for j in 0..oc {
        let d = y[j] as f64 - y2[j] as f64;
        num += d * d;
        den += y2[j] as f64 * y2[j] as f64;
    }
    let rel = (num / den).sqrt();

    println!("rel err vs CPU reference : {rel:.2e}");
    println!("2-layer chain (on-device): {ms:.4} ms  (no host<->device copy between layers)");
    println!("activations resident on GPU, layers chained, no Python.");
    assert!(rel < 2e-2, "reconstruction error too high: {rel:.2e}");
    println!("OK");
}
