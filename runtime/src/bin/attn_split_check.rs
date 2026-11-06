//! Batch-1 decode attention check, written for the split-KV kernel (the default; the old
//! sequential kernel with TRAPETUM_ATTN=seq). For several head layouts (MHA, GQA, head_dim 64,
//! 128 and 256), sequence lengths around the 64-position chunk boundaries up to 4096, and both a
//! flat and a peaked softmax, it compares `trapetum::attention` with an f64 CPU reference on the
//! same fp16 inputs and checks that two runs give bitwise identical outputs. It then times one
//! call for Llama-2-7B shapes. Prints one line per case, then OK or FAIL.
use half::f16;
use std::time::Instant;
use trapetum::{attention, sync, DevHalf};

fn cpu_attn(q: &[f32], k: &[f32], v: &[f32], nh: usize, nkv: usize, hd: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; nh * hd];
    let scale = 1.0 / (hd as f64).sqrt();
    for h in 0..nh {
        let kvh = h / (nh / nkv);
        let mut s = vec![0f64; len];
        for (t, st) in s.iter_mut().enumerate() {
            let kb = t * nkv * hd + kvh * hd;
            *st = (0..hd).map(|d| q[h * hd + d] as f64 * k[kb + d] as f64).sum::<f64>() * scale;
        }
        let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
        let sum: f64 = w.iter().sum();
        for d in 0..hd {
            let acc: f64 = (0..len).map(|t| w[t] * v[t * nkv * hd + kvh * hd + d] as f64).sum();
            out[h * hd + d] = (acc / sum) as f32;
        }
    }
    out
}

fn rel_err(y: &[f32], r: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..y.len() {
        let d = y[i] as f64 - r[i] as f64;
        num += d * d;
        den += r[i] as f64 * r[i] as f64;
    }
    (num / den.max(1e-300)).sqrt()
}

fn main() {
    let mode = std::env::var("TRAPETUM_ATTN").unwrap_or_else(|_| "split".into());
    println!("attention kernel: {mode}");
    let max_seq = 4096usize;
    let cases: &[(usize, usize, usize)] = &[(32, 32, 128), (32, 8, 128), (8, 8, 256), (32, 4, 64)];
    let lens: &[usize] = &[1, 2, 63, 64, 65, 127, 129, 200, 1000, 2047, 4096];
    let mut s: u64 = 0x2545_f491_4f6c_dd1d;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };
    let mut ok = true;
    let mut worst = 0f64;
    for &(nh, nkv, hd) in cases {
        let k_h: Vec<f32> = (0..max_seq * nkv * hd).map(|_| f16::from_f32(rnd()).to_f32()).collect();
        let v_h: Vec<f32> = (0..max_seq * nkv * hd).map(|_| f16::from_f32(rnd()).to_f32()).collect();
        let ck = DevHalf::from_host(&k_h);
        let cv = DevHalf::from_host(&v_h);
        let mut out = DevHalf::zeros(nh * hd);
        for &qscale in &[1.0f32, 12.0] {
            // qscale 12 makes the softmax peaked (score spread of several units after 1/sqrt(hd))
            let q_h: Vec<f32> = (0..nh * hd).map(|_| f16::from_f32(rnd() * qscale).to_f32()).collect();
            let q = DevHalf::from_host(&q_h);
            for &len in lens {
                attention(&q, &ck, &cv, &mut out, nh, nkv, hd, len, 0.0);
                sync();
                let a = out.to_host();
                attention(&q, &ck, &cv, &mut out, nh, nkv, hd, len, 0.0);
                sync();
                let b = out.to_host();
                let det = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
                let e = rel_err(&a, &cpu_attn(&q_h, &k_h, &v_h, nh, nkv, hd, len));
                let finite = a.iter().all(|x| x.is_finite());
                let pass = det && finite && e < 2e-3;
                worst = worst.max(e);
                ok &= pass;
                println!("CASE heads={nh} kv={nkv} hd={hd} qscale={qscale} len={len}: rel err {e:.2e}, deterministic {det}, finite {finite} {}",
                         if pass { "ok" } else { "FAIL" });
            }
        }
    }
    // timing, Llama-2-7B shapes (32 heads, MHA, head_dim 128), one attention call per layer
    let (nh, nkv, hd) = (32usize, 32usize, 128usize);
    let q_h: Vec<f32> = (0..nh * hd).map(|_| f16::from_f32(rnd()).to_f32()).collect();
    let k_h: Vec<f32> = (0..max_seq * nkv * hd).map(|_| f16::from_f32(rnd()).to_f32()).collect();
    let (q, ck, cv) = (DevHalf::from_host(&q_h), DevHalf::from_host(&k_h), DevHalf::from_host(&k_h));
    let mut out = DevHalf::zeros(nh * hd);
    for &len in &[64usize, 256, 1024, 2048, 4096] {
        for _ in 0..10 {
            attention(&q, &ck, &cv, &mut out, nh, nkv, hd, len, 0.0);
        }
        sync();
        let iters = 200;
        let t0 = Instant::now();
        for _ in 0..iters {
            attention(&q, &ck, &cv, &mut out, nh, nkv, hd, len, 0.0);
        }
        sync();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("TIME heads={nh} hd={hd} len={len}: {us:.1} us per call ({:.2} ms per token over 32 layers)", us * 32.0 / 1e3);
    }
    println!("worst rel err {worst:.2e}");
    println!("{}", if ok { "OK" } else { "FAIL" });
}
