//! Correctness + speed of the two runtime optimizations:
//!   1. device argmax (backend dev_argmax) vs host argmax, including ties and a
//!      real_vocab smaller than the padded buffer.
//!   2. the compile-time-M fused decode GEMM (gemm_mtile_t<M>, wired into
//!      QuantLinear::forward_m for M<=4) vs the per-column M=1 gemv reference.
//! Uses only the public runtime API, so it exercises exactly the wired paths.
use std::time::Instant;
use trapetum::{argmax, DevF32, DevHalf, QuantLinear, K};

// cheap deterministic PRNG (xorshift) so the check is reproducible.
struct Rng(u64);
impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn main() {
    let mut bad = 0;

    // ---- 1. device argmax ---------------------------------------------------
    println!("== device argmax vs host argmax ==");
    let mut rng = Rng(0x1234_5678_9abc_def0);
    // (a) random logits, real_vocab == full length, a few padded sizes / vocabs.
    for &n in &[256usize, 4096, 129_280, 152_064] {
        let v: Vec<f32> = (0..n).map(|_| rng.f32() * 10.0).collect();
        let buf = DevF32::from_host(&v);
        let got = buf.argmax_device(n);
        let cpu = argmax(&v[..n]) as u32;
        let ok = got == cpu;
        println!("  n={n:>7}  dev={got:>7}  cpu={cpu:>7}  {}", if ok { "OK" } else { "FAIL" });
        if !ok { bad += 1; }
    }
    // (b) explicit tie: two positions share the max; both host and device must
    //     pick the SMALLEST index.
    {
        let n = 8192usize;
        let mut v: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        v[100] = 5.0;
        v[7000] = 5.0; // tie with index 100; expect 100
        let buf = DevF32::from_host(&v);
        let got = buf.argmax_device(n);
        let cpu = argmax(&v[..n]) as u32;
        let ok = got == cpu && got == 100;
        println!("  tie@100,7000  dev={got}  cpu={cpu}  {}", if ok { "OK" } else { "FAIL" });
        if !ok { bad += 1; }
    }
    // (c) real_vocab < padded: the global max sits in the padded tail and must be
    //     ignored; the answer is the max within [0, real_vocab).
    {
        let padded = 152_064usize;
        let real = 151_936usize; // Qwen2 real vocab under a 152064 pad
        let mut v: Vec<f32> = (0..padded).map(|_| rng.f32()).collect();
        v[123_456] = 99.0; // inside real vocab: the intended winner
        v[real + 40] = 100.0; // padded tail: bigger, must be ignored
        let buf = DevF32::from_host(&v);
        let got = buf.argmax_device(real);
        let cpu = argmax(&v[..real]) as u32;
        let ok = got == cpu && got == 123_456;
        println!("  real={real} < padded={padded}  dev={got}  cpu={cpu}  {}", if ok { "OK" } else { "FAIL" });
        if !ok { bad += 1; }
    }

    // ---- 2. compile-time-M fused decode GEMM (forward_m, M<=4) ---------------
    println!("== forward_m (gemm_mtile_t<M>) vs per-column gemv reference ==");
    let (ic, oc) = (2048usize, 2048usize);
    let np = ic * (oc / 2);
    let mut rng = Rng(0x0bad_c0de_dead_beef);
    let packed: Vec<u8> = (0..np).map(|i| ((i * 131 + 7) % 256) as u8).collect();
    let cbk: Vec<f32> = (0..K * oc).map(|_| rng.f32() * 0.3).collect();
    let ql = QuantLinear::new(&packed, &cbk, ic, oc);
    for m in [1usize, 2, 3, 4] {
        let xall: Vec<f32> = (0..m * ic).map(|_| rng.f32() * 0.5).collect();
        let dx = DevHalf::from_host(&xall);
        let mut dy = DevF32::zeros(m * oc);
        ql.forward_m(&dx, &mut dy, m);
        let ybatch = dy.to_host();
        // reference: run each column through the single-column forward.
        let mut worst = 0f64;
        for col in 0..m {
            let xcol = &xall[col * ic..(col + 1) * ic];
            let dxc = DevHalf::from_host(xcol);
            let mut dyc = DevF32::zeros(oc);
            ql.forward_into(&dxc, &mut dyc);
            let yref = dyc.to_host();
            let (mut num, mut den) = (0f64, 0f64);
            for o in 0..oc {
                let d = (ybatch[col * oc + o] - yref[o]) as f64;
                num += d * d;
                den += (yref[o] as f64) * (yref[o] as f64);
            }
            worst = worst.max((num / den.max(1e-30)).sqrt());
        }
        let ok = worst < 1e-3;
        println!("  M={m}  rel_err={worst:.2e}  {}", if ok { "OK" } else { "FAIL" });
        if !ok { bad += 1; }
    }

    // ---- 3a. standalone argmax: device reduction vs host download + CPU -------
    // On Apple unified memory the "download" is a memcpy (no PCIe), so an isolated
    // device reduction that pays its own command-buffer submission is not expected
    // to win here; this number is the pessimistic bound.
    println!("== standalone argmax: device vs host-download (152064 vocab) ==");
    let n = 152_064usize;
    let mut rng = Rng(0xfeed_face_cafe_0001);
    let v: Vec<f32> = (0..n).map(|_| rng.f32() * 10.0).collect();
    let buf = DevF32::from_host(&v);
    let iters = 300;
    let mut acc = 0u64;
    for _ in 0..20 { let _ = buf.argmax_device(n); }
    let t = Instant::now();
    for _ in 0..iters { acc = acc.wrapping_add(buf.argmax_device(n) as u64); }
    let dev_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    for _ in 0..20 { let h = buf.to_host(); let _ = argmax(&h); }
    let t = Instant::now();
    for _ in 0..iters { let h = buf.to_host(); acc = acc.wrapping_add(argmax(&h[..n]) as u64); }
    let host_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("  device reduction  : {dev_us:.1} us/call");
    println!("  host download+cpu : {host_us:.1} us/call");
    println!("  speedup           : {:.2}x", host_us / dev_us);

    // ---- 3b. integrated (decode-loop) token selection ------------------------
    // This is how it actually runs: a logits GEMV then the token pick. The GEMV is
    // pending in the command buffer either way; device argmax chains into that same
    // buffer (one drain, 4-byte readback), while the host path must download the full
    // vocab and argmax on the CPU. This is the representative comparison.
    println!("== integrated: logits GEMV + token selection (ic=512, vocab=152064) ==");
    let (ic2, oc2) = (512usize, 152_064usize);
    let mut rng = Rng(0x51de_51de_1234_0001);
    let packed2: Vec<u8> = (0..ic2 * (oc2 / 2)).map(|i| ((i * 97 + 3) % 256) as u8).collect();
    let cbk2: Vec<f32> = (0..K * oc2).map(|_| rng.f32() * 0.2).collect();
    let head = QuantLinear::new(&packed2, &cbk2, ic2, oc2);
    let xin: Vec<f32> = (0..ic2).map(|_| rng.f32() * 0.5).collect();
    let dx = DevHalf::from_host(&xin);
    let mut dlog = DevF32::zeros(oc2);
    for _ in 0..20 { head.forward_into(&dx, &mut dlog); acc = acc.wrapping_add(dlog.argmax_device(oc2) as u64); }
    let t = Instant::now();
    for _ in 0..iters { head.forward_into(&dx, &mut dlog); acc = acc.wrapping_add(dlog.argmax_device(oc2) as u64); }
    let idev_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    for _ in 0..20 { head.forward_into(&dx, &mut dlog); let h = dlog.to_host(); acc = acc.wrapping_add(argmax(&h[..oc2]) as u64); }
    let t = Instant::now();
    for _ in 0..iters { head.forward_into(&dx, &mut dlog); let h = dlog.to_host(); acc = acc.wrapping_add(argmax(&h[..oc2]) as u64); }
    let ihost_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("  GEMV + device argmax : {idev_us:.1} us/token");
    println!("  GEMV + host argmax   : {ihost_us:.1} us/token");
    println!("  speedup              : {:.2}x  (sink {acc})", ihost_us / idev_us);

    if bad == 0 {
        println!("\nALL PASS");
    } else {
        println!("\n{bad} FAIL");
        std::process::exit(1);
    }
}
