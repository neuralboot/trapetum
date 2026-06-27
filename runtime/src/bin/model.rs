//! Demo: a full multi-layer decoder model (token embedding, a stack of attention+MLP
//! layers with a shared KV cache, final RMSNorm, codebook LM head) assembled in the Rust
//! runtime, decoding several tokens on the GPU and verified end-to-end against a CPU
//! reference. Synthetic GQA config (n_kv != n_heads), so it exercises grouped-query
//! attention before the real model. No Python.
use trapetum::{sync, AttnBlock, Layer, MlpBlock, Model, K};
use half::f16;
use std::time::Instant;

const EPS: f32 = 1e-5;
const BASE: f32 = 10000.0;

fn h16(v: f32) -> f32 {
    f16::from_f32(v).to_f32()
}

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

fn rmsnorm(x: &[f32], w: &[f32], n: usize) -> Vec<f32> {
    let xf: Vec<f32> = x.iter().map(|&v| h16(v)).collect();
    let ss: f64 = xf.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
    let scale = 1.0 / (ss + EPS as f64).sqrt();
    (0..n).map(|i| h16((xf[i] as f64 * scale * w[i] as f64) as f32)).collect()
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

// per-layer weights (indices + codebooks), reused for both GPU build and CPU reference
struct LW {
    an: Vec<f32>, pn: Vec<f32>,
    qp: Vec<u8>, qc: Vec<f32>, qi: Vec<u8>,
    kp: Vec<u8>, kc: Vec<f32>, ki: Vec<u8>,
    vp: Vec<u8>, vc: Vec<f32>, vi: Vec<u8>,
    op: Vec<u8>, oc: Vec<f32>, oi: Vec<u8>,
    gp: Vec<u8>, gc: Vec<f32>, gi: Vec<u8>,
    up: Vec<u8>, uc: Vec<f32>, ui: Vec<u8>,
    dp: Vec<u8>, dc: Vec<f32>, di: Vec<u8>,
}

struct Cfg {
    hidden: usize, n_heads: usize, n_kv: usize, hd: usize, inter: usize, vocab: usize,
}

#[allow(clippy::too_many_arguments)]
fn cpu_attn(c: &Cfg, lw: &LW, h: &[f32], pos: usize, ck: &mut Vec<f32>, cv: &mut Vec<f32>) -> Vec<f32> {
    let (hidden, n_heads, n_kv, hd) = (c.hidden, c.n_heads, c.n_kv, c.hd);
    let kv_dim = n_kv * hd;
    let norm = rmsnorm(h, &lw.an, hidden);
    let mut q: Vec<f32> = decode(&norm, &lw.qi, &lw.qc, hidden, hidden).iter().map(|&v| h16(v)).collect();
    let mut kk: Vec<f32> = decode(&norm, &lw.ki, &lw.kc, hidden, kv_dim).iter().map(|&v| h16(v)).collect();
    let vv: Vec<f32> = decode(&norm, &lw.vi, &lw.vc, hidden, kv_dim).iter().map(|&v| h16(v)).collect();
    rope_cpu(&mut q, pos, n_heads, hd);
    rope_cpu(&mut kk, pos, n_kv, hd);
    ck.extend_from_slice(&kk);
    cv.extend_from_slice(&vv);
    let seqlen = pos + 1;
    let inv = 1.0 / (hd as f32).sqrt();
    let group = n_heads / n_kv;
    let mut attn = vec![0f32; hidden];
    for hh in 0..n_heads {
        let kvh = hh / group;
        let mut sc = vec![0f32; seqlen];
        for t in 0..seqlen {
            let mut dot = 0f32;
            for d in 0..hd {
                dot += q[hh * hd + d] * ck[t * kv_dim + kvh * hd + d];
            }
            sc[t] = dot * inv;
        }
        let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0f32;
        for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
        for s in sc.iter_mut() { *s /= sum; }
        for d in 0..hd {
            let mut acc = 0f32;
            for t in 0..seqlen { acc += sc[t] * cv[t * kv_dim + kvh * hd + d]; }
            attn[hh * hd + d] = h16(acc);
        }
    }
    let ob = decode(&attn, &lw.oi, &lw.oc, hidden, hidden);
    (0..hidden).map(|i| h16(h16(h[i]) + ob[i])).collect()
}

fn cpu_mlp(c: &Cfg, lw: &LW, h: &[f32]) -> Vec<f32> {
    let (hidden, inter) = (c.hidden, c.inter);
    let norm = rmsnorm(h, &lw.pn, hidden);
    let gate = decode(&norm, &lw.gi, &lw.gc, hidden, inter);
    let up = decode(&norm, &lw.ui, &lw.uc, hidden, inter);
    let act: Vec<f32> = (0..inter).map(|i| { let g = gate[i]; h16(g / (1.0 + (-g).exp()) * up[i]) }).collect();
    let mlp = decode(&act, &lw.di, &lw.dc, inter, hidden);
    (0..hidden).map(|i| h16(h16(h[i]) + mlp[i])).collect()
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

fn argmax(x: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..x.len() {
        if x[i] > x[bi] { bi = i; }
    }
    bi
}

fn main() {
    let c = Cfg { hidden: 512, n_heads: 8, n_kv: 4, hd: 64, inter: 768, vocab: 512 };
    let (n_layers, max_seq, tokens) = (2usize, 16usize, 5usize);
    println!(
        "trapetum: full model, {n_layers} layers, hidden={} heads={}/{} kv head_dim={} inter={} vocab={}, GQA",
        c.hidden, c.n_heads, c.n_kv, c.hd, c.inter, c.vocab
    );
    let kv_dim = c.n_kv * c.hd;

    let mut s: u64 = 0xa11ce_5eed_1234_9f1;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };

    let mut lws = Vec::new();
    for _ in 0..n_layers {
        let an: Vec<f32> = (0..c.hidden).map(|_| 1.0 + ((next() % 200) as f32 / 1000.0 - 0.1)).collect();
        let pn: Vec<f32> = (0..c.hidden).map(|_| 1.0 + ((next() % 200) as f32 / 1000.0 - 0.1)).collect();
        let (qp, qc, qi) = make_layer(c.hidden, c.hidden, &mut next);
        let (kp, kc, ki) = make_layer(c.hidden, kv_dim, &mut next);
        let (vp, vc, vi) = make_layer(c.hidden, kv_dim, &mut next);
        let (op, oc, oi) = make_layer(c.hidden, c.hidden, &mut next);
        let (gp, gc, gi) = make_layer(c.hidden, c.inter, &mut next);
        let (up, uc, ui) = make_layer(c.hidden, c.inter, &mut next);
        let (dp, dc, di) = make_layer(c.inter, c.hidden, &mut next);
        lws.push(LW { an, pn, qp, qc, qi, kp, kc, ki, vp, vc, vi, op, oc, oi, gp, gc, gi, up, uc, ui, dp, dc, di });
    }
    let final_norm: Vec<f32> = (0..c.hidden).map(|_| 1.0 + ((next() % 200) as f32 / 1000.0 - 0.1)).collect();
    let (lmp, lmc, lmi) = make_layer(c.hidden, c.vocab, &mut next);
    let embedding: Vec<f32> = (0..c.vocab * c.hidden).map(|_| (next() % 1000) as f32 / 1000.0 - 0.5).collect();
    let prompt: Vec<usize> = (0..tokens).map(|_| (next() % c.vocab as u64) as usize).collect();

    // build the GPU model
    let layers: Vec<Layer> = lws.iter().map(|lw| {
        let attn = AttnBlock::new(c.hidden, c.n_heads, c.n_kv, c.hd, max_seq, &lw.an,
            (&lw.qp, &lw.qc), (&lw.kp, &lw.kc), (&lw.vp, &lw.vc), (&lw.op, &lw.oc), EPS, BASE);
        let mlp = MlpBlock::new(c.hidden, c.inter, &lw.pn,
            &lw.gp, &lw.gc, &lw.up, &lw.uc, &lw.dp, &lw.dc, EPS);
        Layer::new(attn, mlp)
    }).collect();
    let mut model = Model::new(c.hidden, c.vocab, embedding.clone(), layers, &final_norm, (&lmp, &lmc), EPS);

    // GPU forward over the prompt
    let mut gpu_logits = Vec::new();
    for (t, &tok) in prompt.iter().enumerate() {
        gpu_logits.push(model.forward(tok, t));
    }
    sync();

    // CPU reference forward (same growing per-layer cache)
    let mut caches: Vec<(Vec<f32>, Vec<f32>)> = (0..n_layers).map(|_| (Vec::new(), Vec::new())).collect();
    let mut cpu_logits = Vec::new();
    for (t, &tok) in prompt.iter().enumerate() {
        let mut h: Vec<f32> = embedding[tok * c.hidden..(tok + 1) * c.hidden].to_vec();
        for li in 0..n_layers {
            let (ck, cv) = &mut caches[li];
            h = cpu_attn(&c, &lws[li], &h, t, ck, cv);
            h = cpu_mlp(&c, &lws[li], &h);
        }
        let hn = rmsnorm(&h, &final_norm, c.hidden);
        cpu_logits.push(decode(&hn, &lmi, &lmc, c.hidden, c.vocab));
    }

    let fg: Vec<f32> = gpu_logits.concat();
    let fc: Vec<f32> = cpu_logits.concat();
    let err = rel_err(&fg, &fc);
    let top1: usize = (0..tokens).filter(|&t| argmax(&gpu_logits[t]) == argmax(&cpu_logits[t])).count();

    // decode throughput (greedy continuation)
    let mut tok = argmax(gpu_logits.last().unwrap());
    let iters = 100;
    let t0 = Instant::now();
    for i in 0..iters {
        let lg = model.forward(tok, tokens + i);
        tok = argmax(&lg);
    }
    sync();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!("rel err logits vs CPU ({tokens} pos): {err:.2e}");
    println!("top-1 next-token agreement     : {top1}/{tokens}");
    println!("decode throughput              : {ms:.4} ms/token  ({:.0} tok/s)", 1e3 / ms);
    println!("embed -> {n_layers}x(attn+mlp) -> norm -> lm_head, on-device, pure Rust.");
    assert!(err < 5e-2 && top1 == tokens, "model mismatch: err={err:.2e} top1={top1}/{tokens}");
    println!("OK");
}
