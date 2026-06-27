//! Demo: a full Llama-style gated MLP block (RMSNorm -> gate/up codebook GEMVs ->
//! SwiGLU -> down codebook GEMV -> residual), run on the GPU through the Rust runtime,
//! eagerly and as one captured CUDA graph, verified against a CPU reference that
//! emulates the GPU's fp16 rounding at every step. No Python.
use trapetum::{sync, DevHalf, Graph, MlpBlock, K};
use half::f16;
use std::time::Instant;

const EPS: f32 = 1e-5;

fn make_layer(ic: usize, oc: usize, next: &mut impl FnMut() -> u64) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let cb: Vec<f32> = (0..K * oc).map(|_| (next() % 1000) as f32 / 10000.0 - 0.05).collect();
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
    (packed, cb, idx)
}

fn h16(v: f32) -> f32 {
    f16::from_f32(v).to_f32()
}
fn round_f16(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| h16(v)).collect()
}

fn decode(x_f16: &[f32], idx: &[u8], cb: &[f32], ic: usize, oc: usize) -> Vec<f32> {
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

#[allow(clippy::too_many_arguments)]
fn cpu_mlp(
    h: &[f32], nw: &[f32], gi: &[u8], gc: &[f32], ui: &[u8], uc: &[f32], di: &[u8], dc: &[f32],
    hid: usize, int: usize,
) -> Vec<f32> {
    let hf = round_f16(h);
    let ss: f64 = hf.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hid as f64;
    let scale = 1.0 / (ss + EPS as f64).sqrt();
    let norm: Vec<f32> = (0..hid).map(|i| h16((hf[i] as f64 * scale * nw[i] as f64) as f32)).collect();
    let gate = decode(&norm, gi, gc, hid, int);
    let up = decode(&norm, ui, uc, hid, int);
    let act: Vec<f32> = (0..int)
        .map(|i| {
            let g = gate[i];
            h16(g / (1.0 + (-g).exp()) * up[i])
        })
        .collect();
    let mlp = decode(&act, di, dc, int, hid);
    (0..hid).map(|i| h16(hf[i] + mlp[i])).collect()
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
    let (hid, int) = (4096usize, 11008usize); // Llama-2 7B dims
    println!("trapetum: Llama-style MLP block, hidden={hid} inter={int}, 4-bit (K={K})");

    let mut s: u64 = 0xdead_beef_cafe_1234;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let nw: Vec<f32> = (0..hid).map(|_| 1.0 + ((next() % 200) as f32 / 1000.0 - 0.1)).collect();
    let (gp, gc, gi) = make_layer(hid, int, &mut next);
    let (up, uc, ui) = make_layer(hid, int, &mut next);
    let (dp, dc, di) = make_layer(int, hid, &mut next);
    let h0: Vec<f32> = (0..hid).map(|_| (next() % 1000) as f32 / 1000.0 - 0.5).collect();

    let mut block = MlpBlock::new(hid, int, &nw, &gp, &gc, &up, &uc, &dp, &dc, EPS);
    let yref = cpu_mlp(&h0, &nw, &gi, &gc, &ui, &uc, &di, &dc, hid, int);

    let mut h = DevHalf::from_host(&h0);

    // eager correctness
    block.forward(&mut h);
    sync();
    let err_eager = rel_err(&h.to_host(), &yref);

    // eager timing (h accumulates; irrelevant for timing)
    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        block.forward(&mut h);
    }
    sync();
    let ms_eager = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // capture the whole block as one CUDA graph
    h.upload(&h0);
    let g = Graph::capture(|| block.forward(&mut h));
    // graph correctness (reset h, replay once)
    h.upload(&h0);
    g.launch();
    sync();
    let err_graph = rel_err(&h.to_host(), &yref);
    // graph timing
    let t0 = Instant::now();
    for _ in 0..iters {
        g.launch();
    }
    sync();
    let ms_graph = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!("rel err vs CPU (eager/graph): {err_eager:.2e} / {err_graph:.2e}");
    println!("MLP block eager     : {ms_eager:.4} ms");
    println!("MLP block CUDA-graph: {ms_graph:.4} ms   (x{:.2} vs eager)", ms_eager / ms_graph);
    println!("RMSNorm + 3 codebook GEMVs + SwiGLU + residual, on-device, no Python.");
    assert!(err_graph < 3e-2, "block reconstruction error too high: {err_graph:.2e}");
    println!("OK");
}
