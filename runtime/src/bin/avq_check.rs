//! CPU reference decoder for the CBKA additive-codebook expert record. Reads a .cbka file
//! (written by model/cbka_format.write_cbka_record, the same layout the streaming exporter
//! emits) plus its .ref file (rows*cols little-endian f32, the expected dequantized weight),
//! decodes the record on the CPU, and compares. This proves the on-disk format and the
//! reconstruction math agree end-to-end WITHOUT a GPU (the CUDA numeric check runs later on a
//! pod). It is deliberately self-contained: no backend, no `trapetum` GPU types, just std + half.
//!
//!   cargo run --bin avq_check --no-default-features --features metal -- <file.cbka> <file.ref>
//!
//! Reconstruction (mirrors avq_gemv_t / read_cbka): W[o, g*D+e] = scale[o]*sum_m CB[m][code][e].
use std::io::Read;
use std::process::exit;

fn rd_i32(b: &[u8], o: &mut usize) -> i32 {
    let v = i32::from_le_bytes([b[*o], b[*o + 1], b[*o + 2], b[*o + 3]]);
    *o += 4;
    v
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn rd_f16_vec(b: &[u8], o: &mut usize, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let lo = b[*o + 2 * i];
        let hi = b[*o + 2 * i + 1];
        v.push(f16_to_f32(u16::from_le_bytes([lo, hi])));
    }
    *o += 2 * n;
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: avq_check <file.cbka> <file.ref>");
        exit(2);
    }
    let mut bytes = Vec::new();
    std::fs::File::open(&args[1]).expect("open cbka").read_to_end(&mut bytes).unwrap();
    let mut ref_bytes = Vec::new();
    std::fs::File::open(&args[2]).expect("open ref").read_to_end(&mut ref_bytes).unwrap();

    assert_eq!(&bytes[0..4], b"CBKA", "bad magic (not a CBKA record)");
    let mut o = 4usize;
    let m = rd_i32(&bytes, &mut o) as usize;
    let d = rd_i32(&bytes, &mut o) as usize;
    let k = rd_i32(&bytes, &mut o) as usize;
    let rows = rd_i32(&bytes, &mut o) as usize;
    let cols = rd_i32(&bytes, &mut o) as usize;
    assert!(m == 2 || m == 3, "M must be 2 or 3, got {m}");
    assert_eq!(d, 8, "D must be 8");
    assert_eq!(k, 256, "K must be 256");
    assert_eq!(cols % d, 0, "cols must be a multiple of D");
    let ng = cols / d;

    let cb = rd_f16_vec(&bytes, &mut o, m * k * d); // [M][K][D], flat (m*K+k)*D+e
    let scale = rd_f16_vec(&bytes, &mut o, rows); // [rows]
    let n_idx = m * ng * rows;
    let idx = &bytes[o..o + n_idx]; // [M][ng][rows] u8, flat (m*ng+g)*rows+o
    assert_eq!(bytes.len(), o + n_idx, "trailing bytes / short CBKA record");

    // decode W[o, g*D + e] = scale[o] * sum_m cb[m][code_m[o,g]][e]
    let mut w = vec![0f32; rows * cols];
    for oc in 0..rows {
        for g in 0..ng {
            for e in 0..d {
                let mut acc = 0f32;
                for mm in 0..m {
                    let code = idx[(mm * ng + g) * rows + oc] as usize;
                    acc += cb[(mm * k + code) * d + e];
                }
                w[oc * cols + g * d + e] = scale[oc] * acc;
            }
        }
    }

    assert_eq!(ref_bytes.len(), rows * cols * 4, "ref size mismatch");
    let wref: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    for i in 0..rows * cols {
        let diff = (w[i] - wref[i]).abs() as f64;
        max_abs = max_abs.max(diff);
        let den = (wref[i].abs() as f64).max(1e-6);
        max_rel = max_rel.max(diff / den);
    }
    println!(
        "CBKA M={m} rows={rows} cols={cols} ng={ng}: max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e}",
    );
    // Both sides read the same f16 codebook/scale and accumulate in f32 in the same order, so
    // the match is exact up to f32 rounding of a handful of adds.
    if max_abs < 1e-5 {
        println!("PASS");
    } else {
        println!("FAIL (max_abs_err too large)");
        exit(1);
    }
}
