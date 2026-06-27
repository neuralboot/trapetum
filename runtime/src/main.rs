//! Demo: two codebook-quantized linear layers chained ON THE GPU, run both eagerly and
//! as a captured CUDA graph. The graph replays the chain with near-zero CPU launch
//! overhead, the Rust-runtime version of the paper's end-to-end lever. Verified against
//! a CPU reconstruction. No Python.
use codebook_runtime::{sync, DevF32, DevHalf, Graph, QuantLinear, K};
use half::f16;
use std::time::Instant;

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

fn rel_err(y: &[f32], r: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for j in 0..y.len() {
        let d = y[j] as f64 - r[j] as f64;
        num += d * d;
        den += r[j] as f64 * r[j] as f64;
    }
    (num / den).sqrt()
}

fn time_ms(iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    sync(); // warmup
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    sync();
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    let (ic, oc) = (4096usize, 4096usize);
    println!("codebook-runtime: 2 layers {ic}x{oc} on-device, eager vs CUDA graph, 4-bit (K={K})");

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

    let dx = DevHalf::from_host(&x);
    let mut dy1 = DevF32::zeros(oc);
    let mut dx2 = DevHalf::zeros(oc);
    let mut dy2 = DevF32::zeros(oc);

    // one decode step of the 2-layer chain (all on-device)
    macro_rules! chain {
        () => {{
            l1.forward_into(&dx, &mut dy1);
            dx2.copy_cast_from(&dy1);
            l2.forward_into(&dx2, &mut dy2);
        }};
    }

    // CPU reference, matching the GPU's fp16 rounding at input and intermediate
    let xh = round_f16(&x);
    let y1 = cpu_decode(&xh, &idx1, &cb1, ic, oc);
    let y2 = cpu_decode(&round_f16(&y1), &idx2, &cb2, ic, oc);

    // eager
    let ms_eager = time_ms(200, || chain!());
    let err_eager = rel_err(&dy2.to_host(), &y2);

    // CUDA graph: capture the chain once, replay
    let g = Graph::capture(|| chain!());
    let ms_graph = time_ms(200, || g.launch());
    let err_graph = rel_err(&dy2.to_host(), &y2);

    println!("rel err (eager / graph)  : {err_eager:.2e} / {err_graph:.2e}");
    println!("eager 2-layer chain      : {ms_eager:.4} ms");
    println!("CUDA-graph 2-layer chain : {ms_graph:.4} ms   (x{:.2} vs eager)", ms_eager / ms_graph);
    println!("decode chain captured as one CUDA graph, replayed from Rust, no Python.");
    assert!(err_graph < 2e-2, "graph reconstruction error too high: {err_graph:.2e}");
    println!("OK");
}
