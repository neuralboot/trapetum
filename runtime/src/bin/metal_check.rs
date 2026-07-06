//! Numerical validation of the GPU backend against a CPU reference (fp16 rounding
//! emulated with the `half` crate, f32 accumulation, same as the kernel).
//!
//! Runs on ANY backend (CUDA or Metal): build with the backend feature you want
//! to validate and compare. Exit code 0 = all checks pass.
//!
//!   cargo run --bin metal_check --no-default-features --features metal
use half::f16;
use trapetum::{attention, silu_mul, sync, rmsnorm, DevF32, DevHalf, QuantLinear, K};

// Raw batch-1 attention reference (fp16 rounding on q/k/v, f32 accumulation),
// matching attn_k exactly. Isolated from rope/projections.
fn attn_ref(q: &[f32], ck: &[f32], cv: &[f32], n_heads: usize, n_kv: usize, hd: usize, seqlen: usize) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; n_heads * hd];
    for h in 0..n_heads {
        let kvh = h / (n_heads / n_kv);
        let mut scores = vec![0f32; seqlen];
        for t in 0..seqlen {
            let mut s = 0f32;
            for d in 0..hd {
                s += f16::from_f32(q[h * hd + d]).to_f32()
                    * f16::from_f32(ck[t * n_kv * hd + kvh * hd + d]).to_f32();
            }
            scores[t] = s * scale;
        }
        let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
        for s in scores.iter_mut() { *s /= sum; }
        for d in 0..hd {
            let mut acc = 0f32;
            for t in 0..seqlen {
                acc += scores[t] * f16::from_f32(cv[t * n_kv * hd + kvh * hd + d]).to_f32();
            }
            out[h * hd + d] = f16::from_f32(acc).to_f32();
        }
    }
    out
}

fn check_attn(rng: &mut Rng, n_heads: usize, n_kv: usize, hd: usize, seqlen: usize) -> bool {
    let q: Vec<f32> = (0..n_heads * hd).map(|_| rng.f32() * 0.5).collect();
    let ck: Vec<f32> = (0..seqlen * n_kv * hd).map(|_| rng.f32() * 0.5).collect();
    let cv: Vec<f32> = (0..seqlen * n_kv * hd).map(|_| rng.f32() * 0.5).collect();
    let r = attn_ref(&q, &ck, &cv, n_heads, n_kv, hd, seqlen);
    let dq = DevHalf::from_host(&q);
    let dck = DevHalf::from_host(&ck);
    let dcv = DevHalf::from_host(&cv);
    let mut dout = DevHalf::zeros(n_heads * hd);
    attention(&dq, &dck, &dcv, &mut dout, n_heads, n_kv, hd, seqlen, 0.0);
    sync();
    let o = dout.to_host();
    let e = rel_err(&o, &r);
    let ok = e < 5e-3;
    println!("attn hd={hd:<3} n_kv={n_kv} seq={seqlen}  rel_err = {e:.2e}  {}", if ok { "OK" } else { "FAIL" });
    ok
}

// deterministic xorshift so CPU and GPU see the exact same data
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x; x
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn nib(&mut self) -> u8 { (self.next() >> 33) as u8 & 0xF }
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64; let mut den = 0f64;
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt() as f32
}

fn main() {
    let mut rng = Rng(0x545241504554554d); // "TRAPETUM"
    let mut fails = 0;

    // ---- 1. fused 4-bit codebook GEMV -------------------------------------
    let (ic, oc) = (384usize, 512usize); // oc multiple of 256 (CPB)
    let x: Vec<f32> = (0..ic).map(|_| rng.f32()).collect();
    let cb: Vec<f32> = (0..K * oc).map(|_| rng.f32() * 0.1).collect();
    // indices packed exactly as the kernel reads them: byte j/2, even j = low nibble
    let mut ids = vec![0u8; ic * oc];
    let mut packed = vec![0u8; ic * oc / 2];
    for i in 0..ic {
        for j in 0..oc {
            let id = rng.nib();
            ids[i * oc + j] = id;
            packed[i * oc / 2 + j / 2] |= id << (4 * (j & 1));
        }
    }
    // CPU reference with the kernel's rounding: x and cb go through fp16
    let xh: Vec<f32> = x.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
    let cbh: Vec<f32> = cb.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
    let mut y_ref = vec![0f32; oc];
    for j in 0..oc {
        let mut acc = 0f32;
        for i in 0..ic {
            acc += xh[i] * cbh[ids[i * oc + j] as usize * oc + j];
        }
        y_ref[j] = acc;
    }
    let q = QuantLinear::new(&packed, &cb, ic, oc);
    let dx = DevHalf::from_host(&x);
    let mut dy = DevF32::zeros(oc);
    q.forward_into(&dx, &mut dy);
    sync();
    let y = dy.to_host();
    let e = rel_err(&y, &y_ref);
    println!("gemv4    rel_err = {e:.2e}  {}", if e < 1e-3 { "OK" } else { "FAIL" });
    if e >= 1e-3 { fails += 1; }

    // ---- 2. rmsnorm ---------------------------------------------------------
    let n = 1024usize;
    let xv: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
    let wv: Vec<f32> = (0..n).map(|_| rng.f32() + 1.5).collect();
    let eps = 1e-5f32;
    let xh: Vec<f32> = xv.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
    let ss: f32 = xh.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    let r_ref: Vec<f32> = (0..n)
        .map(|i| f16::from_f32(xh[i] * scale * wv[i]).to_f32())
        .collect();
    let dxh = DevHalf::from_host(&xv);
    let dw = DevF32::from_host(&wv);
    let mut dout = DevHalf::zeros(n);
    rmsnorm(&dxh, &dw, &mut dout, eps);
    sync();
    let r = dout.to_host();
    let e = rel_err(&r, &r_ref);
    println!("rmsnorm  rel_err = {e:.2e}  {}", if e < 5e-3 { "OK" } else { "FAIL" });
    if e >= 5e-3 { fails += 1; }

    // ---- 3. silu_mul ---------------------------------------------------------
    let g: Vec<f32> = (0..n).map(|_| rng.f32() * 3.0).collect();
    let u: Vec<f32> = (0..n).map(|_| rng.f32() * 3.0).collect();
    let s_ref: Vec<f32> = (0..n)
        .map(|i| {
            let s = g[i] / (1.0 + (-g[i]).exp());
            f16::from_f32(s * u[i]).to_f32()
        })
        .collect();
    let dg = DevF32::from_host(&g);
    let du = DevF32::from_host(&u);
    let mut ds = DevHalf::zeros(n);
    silu_mul(&dg, &du, &mut ds);
    sync();
    let s = ds.to_host();
    let e = rel_err(&s, &s_ref);
    println!("silu_mul rel_err = {e:.2e}  {}", if e < 5e-3 { "OK" } else { "FAIL" });
    if e >= 5e-3 { fails += 1; }

    // ---- 4. raw attention: head_dim 64 (works in 1B) vs 128 (7B, suspect) ----
    if !check_attn(&mut rng, 8, 8, 64, 5) { fails += 1; }   // MHA hd=64
    if !check_attn(&mut rng, 32, 8, 64, 5) { fails += 1; }  // GQA hd=64 (1B-like)
    if !check_attn(&mut rng, 32, 32, 128, 5) { fails += 1; } // MHA hd=128 (7B-like)
    if !check_attn(&mut rng, 32, 32, 128, 33) { fails += 1; } // hd=128 longer seq

    if fails == 0 {
        println!("ALL CHECKS PASS");
    } else {
        println!("{fails} CHECK(S) FAILED");
        std::process::exit(1);
    }
}
