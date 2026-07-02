//! Demo: a Llama-style attention block (RMSNorm -> q/k/v codebook projections -> RoPE ->
//! growing KV cache -> batch-1 attention -> o codebook projection -> residual), decoding
//! T tokens on the GPU through the Rust runtime, verified against a CPU reference that
//! replicates RoPE, softmax and the fp16 rounding at every step. No Python.
use trapetum::{sync, AttnBlock, DevHalf, Graph, K};
use half::f16;
use std::time::Instant;

const EPS: f32 = 1e-5;
const BASE: f32 = 10000.0;

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

fn rope_cpu(x: &mut [f32], pos: usize, n_heads: usize, hd: usize) {
    let half = hd / 2;
    for h in 0..n_heads {
        for d in 0..half {
            let angle = pos as f32 * BASE.powf(-2.0 * d as f32 / hd as f32);
            let (c, s) = (angle.cos(), angle.sin());
            let (i, j) = (h * hd + d, h * hd + d + half);
            let (x0, x1) = (x[i], x[j]);
            x[i] = h16(x0 * c - x1 * s);
            x[j] = h16(x1 * c + x0 * s);
        }
    }
}

struct Weights {
    nw: Vec<f32>,
    qi: Vec<u8>, qc: Vec<f32>,
    ki: Vec<u8>, kc: Vec<f32>,
    vi: Vec<u8>, vc: Vec<f32>,
    oi: Vec<u8>, oc: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn cpu_step(
    w: &Weights, h: &[f32], pos: usize, ck: &mut Vec<f32>, cv: &mut Vec<f32>,
    n_heads: usize, hd: usize, hidden: usize,
) -> Vec<f32> {
    let hf: Vec<f32> = h.iter().map(|&v| h16(v)).collect();
    let ss: f64 = hf.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
    let scale = 1.0 / (ss + EPS as f64).sqrt();
    let norm: Vec<f32> = (0..hidden).map(|i| h16((hf[i] as f64 * scale * w.nw[i] as f64) as f32)).collect();
    let mut q: Vec<f32> = decode(&norm, &w.qi, &w.qc, hidden, hidden).iter().map(|&v| h16(v)).collect();
    let mut kk: Vec<f32> = decode(&norm, &w.ki, &w.kc, hidden, hidden).iter().map(|&v| h16(v)).collect();
    let vv: Vec<f32> = decode(&norm, &w.vi, &w.vc, hidden, hidden).iter().map(|&v| h16(v)).collect();
    rope_cpu(&mut q, pos, n_heads, hd);
    rope_cpu(&mut kk, pos, n_heads, hd);
    ck.extend_from_slice(&kk);
    cv.extend_from_slice(&vv);
    let seqlen = pos + 1;
    let inv = 1.0 / (hd as f32).sqrt();
    let mut attn = vec![0f32; hidden];
    for hh in 0..n_heads {
        let mut scores = vec![0f32; seqlen];
        for t in 0..seqlen {
            let mut dot = 0f32;
            for d in 0..hd {
                dot += q[hh * hd + d] * ck[t * hidden + hh * hd + d];
            }
            scores[t] = dot * inv;
        }
        let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }
        for d in 0..hd {
            let mut acc = 0f32;
            for t in 0..seqlen {
                acc += scores[t] * cv[t * hidden + hh * hd + d];
            }
            attn[hh * hd + d] = h16(acc);
        }
    }
    let ob = decode(&attn, &w.oi, &w.oc, hidden, hidden);
    (0..hidden).map(|i| h16(hf[i] + ob[i] as f32)).collect()
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
    let (hidden, n_heads, hd) = (4096usize, 32usize, 128usize); // Llama-2 7B MHA
    let n_kv = n_heads;
    let (tokens, max_seq) = (6usize, 8usize);
    println!("trapetum: Llama attention block, hidden={hidden} heads={n_heads} head_dim={hd}, {tokens} tokens");

    let mut s: u64 = 0x0bad_f00d_1357_9bdf;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let nw: Vec<f32> = (0..hidden).map(|_| 1.0 + ((next() % 200) as f32 / 1000.0 - 0.1)).collect();
    let (qp, qc, qi) = make_layer(hidden, hidden, &mut next);
    let (kp, kc, ki) = make_layer(hidden, hidden, &mut next);
    let (vp, vc, vi) = make_layer(hidden, hidden, &mut next);
    let (op, oc, oi) = make_layer(hidden, hidden, &mut next);
    let xs: Vec<Vec<f32>> = (0..tokens)
        .map(|_| (0..hidden).map(|_| (next() % 1000) as f32 / 1000.0 - 0.5).collect())
        .collect();

    let inv_freq: Vec<f32> = (0..hd / 2).map(|d| BASE.powf(-2.0 * d as f32 / hd as f32)).collect();
    let mut block = AttnBlock::new(
        hidden, n_heads, n_kv, hd, max_seq, &nw,
        (&qp, &qc), (&kp, &kc), (&vp, &vc), (&op, &oc), EPS, &inv_freq, None,
    );
    let w = Weights { nw, qi, qc, ki, kc, vi, vc, oi, oc };

    // GPU: decode the tokens, keep the per-token outputs
    let mut gpu_out = Vec::new();
    for (t, x) in xs.iter().enumerate() {
        let mut h = DevHalf::from_host(x);
        block.forward(&mut h, t);
        sync();
        gpu_out.push(h.to_host());
    }

    // CPU reference, same growing cache
    let (mut ck, mut cv) = (Vec::new(), Vec::new());
    let mut cpu_out = Vec::new();
    for (t, x) in xs.iter().enumerate() {
        cpu_out.push(cpu_step(&w, x, t, &mut ck, &mut cv, n_heads, hd, hidden));
    }

    let flat_g: Vec<f32> = gpu_out.concat();
    let flat_c: Vec<f32> = cpu_out.concat();
    let err = rel_err(&flat_g, &flat_c);

    // timing at the last position (eager vs one captured CUDA graph)
    let pos = tokens - 1;
    let mut h = DevHalf::from_host(&xs[pos]);
    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        block.forward(&mut h, pos);
    }
    sync();
    let ms_eager = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    h.upload(&xs[pos]);
    let g = Graph::capture(|| block.forward(&mut h, pos));
    let t0 = Instant::now();
    for _ in 0..iters {
        g.launch();
    }
    sync();
    let ms_graph = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!("rel err vs CPU ({tokens} tokens): {err:.2e}");
    println!("attention block eager     : {ms_eager:.4} ms");
    println!("attention block CUDA-graph: {ms_graph:.4} ms   (x{:.2} vs eager)", ms_eager / ms_graph);
    println!("RMSNorm + q/k/v + RoPE + KV-cache + attention + o-proj + residual, on-device, no Python.");
    assert!(err < 5e-2, "attention reconstruction error too high: {err:.2e}");
    println!("OK");
}
