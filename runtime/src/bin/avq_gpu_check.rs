//! GPU numeric check for the additive-codebook (CBKA) expert kernel.
//! Builds a deterministic synthetic AVQ matrix (M=2 and M=3), reconstructs it on
//! the CPU, and compares `AvqLinear::forward_into` (device) against the f32
//! reference GEMV. Run on a CUDA machine:
//!   cargo run --release --features cuda --bin avq_gpu_check
use trapetum::{AvqLinear, DevF32, DevHalf, AVQ_D, AVQ_K};

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn check(m: usize) -> f32 {
    let (rows, cols) = (256usize, 512usize); // oc, ic
    let ng = cols / AVQ_D;
    let mut s = 0x5eed_1234u64 + m as u64;
    let codes: Vec<u8> = (0..m * ng * rows).map(|_| (xorshift(&mut s) % AVQ_K as u64) as u8).collect();
    let cb: Vec<f32> = (0..m * AVQ_K * AVQ_D)
        .map(|_| ((xorshift(&mut s) % 2000) as f32 / 1000.0) - 1.0)
        .collect();
    let scale: Vec<f32> = (0..rows).map(|_| 0.5 + ((xorshift(&mut s) % 1000) as f32 / 1000.0)).collect();
    let x: Vec<f32> = (0..cols).map(|_| ((xorshift(&mut s) % 2000) as f32 / 1000.0) - 1.0).collect();

    // CPU reference, bit-faithful to the GPU path: the kernel reads fp16 activations
    // and fp16 codebooks (f32 accumulate), so quantize both the same way here.
    let h16 = |v: f32| half::f16::from_f32(v).to_f32();
    let mut y_ref = vec![0f32; rows];
    for o in 0..rows {
        let mut acc = 0f32;
        for g in 0..ng {
            for e in 0..AVQ_D {
                let mut w = 0f32;
                for mi in 0..m {
                    let code = codes[mi * ng * rows + g * rows + o] as usize;
                    w += h16(cb[mi * AVQ_K * AVQ_D + code * AVQ_D + e]);
                }
                acc += scale[o] * w * h16(x[g * AVQ_D + e]);
            }
        }
        y_ref[o] = acc;
    }

    let lin = AvqLinear::new(&codes, &cb, &scale, m, rows, cols);
    let xd = DevHalf::from_host(&x);
    let mut yd = DevF32::from_host(&vec![0f32; rows]);
    lin.forward_into(&xd, &mut yd);
    let y = yd.to_host();

    let mut max_rel = 0f32;
    for o in 0..rows {
        let denom = y_ref[o].abs().max(1e-3);
        max_rel = max_rel.max((y[o] - y_ref[o]).abs() / denom);
    }
    max_rel
}

fn main() {
    for m in [2usize, 3] {
        let err = check(m);
        let ok = err < 5e-3; // reference emulates the fp16 rounding, so only reduction-order noise remains
        println!("M={m}  max_rel_err={err:.3e}  {}", if ok { "PASS" } else { "FAIL" });
        assert!(ok, "AVQ GPU kernel M={m} exceeded tolerance");
    }
    println!("AVQ_GPU_CHECK ALL PASS");
}
