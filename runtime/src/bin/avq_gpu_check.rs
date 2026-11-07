//! GPU numeric check for the additive-codebook kernels (CBKA Mx8 and CBKH 1x16).
//! Builds deterministic synthetic AVQ matrices (M=2, M=3 and the 1x16 M=1), reconstructs
//! them on the CPU, and compares `AvqLinear::forward_into` (device) against the f32
//! reference GEMV. Run on a CUDA machine:
//!   cargo run --release --features cuda --bin avq_gpu_check
use trapetum::{AvqLinear, DevF32, DevHalf, AVQ16_K, AVQ_D, AVQ_K};

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

/// 1x16 (M=1): u16 codes `[rows][ng]` (passed as little-endian bytes), one codebook of
/// AVQ16_K entries. Returns max |y - y_ref| / max |y_ref| (the kernel and this reference both
/// accumulate in f32 from the same fp16 values, so only summation-order noise, ~1e-7, remains;
/// a layout or indexing bug gives an error of order 1).
fn check16(rows: usize, cols: usize) -> f32 {
    let ng = cols / AVQ_D;
    let mut s = 0x1616_5eedu64 + (rows * 7 + cols) as u64;
    let codes: Vec<u16> = (0..rows * ng).map(|_| (xorshift(&mut s) % AVQ16_K as u64) as u16).collect();
    let bytes: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
    let cb: Vec<f32> = (0..AVQ16_K * AVQ_D).map(|_| ((xorshift(&mut s) % 2000) as f32 / 1000.0) - 1.0).collect();
    let scale: Vec<f32> = (0..rows).map(|_| 0.5 + ((xorshift(&mut s) % 1000) as f32 / 1000.0)).collect();
    let x: Vec<f32> = (0..cols).map(|_| ((xorshift(&mut s) % 2000) as f32 / 1000.0) - 1.0).collect();

    let h16 = |v: f32| half::f16::from_f32(v).to_f32();
    let mut y_ref = vec![0f32; rows];
    for o in 0..rows {
        let mut acc = 0f32;
        for g in 0..ng {
            let code = codes[o * ng + g] as usize;
            for e in 0..AVQ_D {
                acc += h16(cb[code * AVQ_D + e]) * h16(x[g * AVQ_D + e]);
            }
        }
        y_ref[o] = h16(scale[o]) * acc;
    }

    let lin = AvqLinear::new(&bytes, &cb, &scale, 1, rows, cols);
    let xd = DevHalf::from_host(&x);
    let mut yd = DevF32::from_host(&vec![f32::NAN; rows]); // the kernel must write every row
    lin.forward_into(&xd, &mut yd);
    let y = yd.to_host();

    let ymax = y_ref.iter().fold(0f32, |a, v| a.max(v.abs()));
    let mut max_err = 0f32;
    for o in 0..rows {
        let d = (y[o] - y_ref[o]).abs();
        max_err = max_err.max(if d.is_nan() { f32::INFINITY } else { d });
    }
    max_err / ymax
}

/// 1x16 batch-invariant verify GEMV: for every m in VERIFY_M, row r of `forward_v` must be the
/// same bits as `forward_into` on row r alone (what makes speculative decoding exact). Returns the
/// number of output floats whose bits differ.
#[cfg(feature = "cuda")]
fn check16_v(rows: usize, cols: usize) -> usize {
    use trapetum::VERIFY_M;
    let ng = cols / AVQ_D;
    let mut s = 0x7e51_f00du64 + (rows * 13 + cols) as u64;
    let codes: Vec<u8> = (0..rows * ng)
        .flat_map(|_| ((xorshift(&mut s) % AVQ16_K as u64) as u16).to_le_bytes())
        .collect();
    let cb: Vec<f32> = (0..AVQ16_K * AVQ_D).map(|_| ((xorshift(&mut s) % 2000) as f32 / 1000.0) - 1.0).collect();
    let scale: Vec<f32> = (0..rows).map(|_| 0.5 + ((xorshift(&mut s) % 1000) as f32 / 1000.0)).collect();
    let mmax = *VERIFY_M.iter().max().unwrap();
    let x: Vec<f32> = (0..mmax * cols).map(|_| ((xorshift(&mut s) % 2000) as f32 / 1000.0) - 1.0).collect();
    let lin = AvqLinear::new(&codes, &cb, &scale, 1, rows, cols);
    let xd_all = DevHalf::from_host(&x);
    let mut differ = 0usize;
    for &m in VERIFY_M.iter() {
        let mut yv = DevF32::from_host(&vec![f32::NAN; mmax * rows]);
        lin.forward_v(&xd_all, &mut yv, m);
        let yv = yv.to_host();
        for r in 0..m {
            let xd = DevHalf::from_host(&x[r * cols..(r + 1) * cols]);
            let mut yd = DevF32::from_host(&vec![f32::NAN; rows]);
            lin.forward_into(&xd, &mut yd);
            let y1 = yd.to_host();
            differ += (0..rows).filter(|&o| yv[r * rows + o].to_bits() != y1[o].to_bits()).count();
        }
    }
    differ
}

fn main() {
    for m in [2usize, 3] {
        let err = check(m);
        let ok = err < 5e-3; // reference emulates the fp16 rounding, so only reduction-order noise remains
        println!("M={m}  max_rel_err={err:.3e}  {}", if ok { "PASS" } else { "FAIL" });
        assert!(ok, "AVQ GPU kernel M={m} exceeded tolerance");
    }
    // 1x16 shapes: a partial last warp step (ng = 264 = 256 + 8) with rows not a multiple of the
    // block's 8 warps, an exact multiple of the step, and a Llama-2-70B k/v-projection shape.
    for (rows, cols) in [(100usize, 2112usize), (256, 4096), (1024, 8192)] {
        let err = check16(rows, cols);
        let ok = err < 1e-4;
        println!("M=1 (1x16) rows={rows} cols={cols}  max_err/max|y|={err:.3e}  {}", if ok { "PASS" } else { "FAIL" });
        assert!(ok, "AVQ 1x16 GPU kernel exceeded tolerance at rows={rows} cols={cols}");
    }
    // the verify GEMV, bit for bit against the decode GEMV, M = 1, 2, 4, 8, 16 (CUDA only)
    #[cfg(feature = "cuda")]
    for (rows, cols) in [(100usize, 2112usize), (1024, 8192)] {
        let d = check16_v(rows, cols);
        println!("M=1 (1x16) verify rows={rows} cols={cols}  floats with different bits={d}  {}", if d == 0 { "PASS" } else { "FAIL" });
        assert_eq!(d, 0, "the 1x16 verify GEMV is not bit-identical to the decode GEMV at rows={rows} cols={cols}");
    }
    println!("AVQ_GPU_CHECK ALL PASS");
}
