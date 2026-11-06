//! Batch-invariant verify check (plan step 5, 2026-11-06). A speculative draft is verified in ONE
//! forward (`Model::verify_logits`); row i must be the same bits as the decode step
//! (`Model::forward`) of token i at position base + i.
//!
//! At three bases (the prompt end; rows crossing a 64-position attention chunk boundary; rows
//! near position 1000, whose attention spans 16 chunks) and for M = 1, 2, 4, 8, 16 it compares,
//! with the state scrambled before every checked call (other tokens through the verify path,
//! which rewrites the logits rows and argmax slots, and through decode steps, which rewrite the
//! KV rows): the eager verify logits, a repeat, the next decode step after the verify (it reads
//! the KV rows the verify wrote), the eager argmax, and the CUDA-graph argmax and logits (each
//! graph is captured at the first base and replayed at the others, then replayed at the same
//! base). It prints the bit verdict, then times a CUDA-graph verify step against a CUDA-graph
//! decode step at short context and at about 1024 positions, two rounds.
//!
//! Usage: verify_check <model.cbk> <prompt.bin> <cont.bin> [reps]
//!   cont.bin: at least 16 i32 tokens (any tokens do: the check is about bits, not acceptance)
//! Prints VERIFY lines, VERIFY OK or VERIFY FAIL (or VERIFY UNSUPPORTED), then TIME lines.
use std::env;
use std::fs::File;
use std::io::Read;

fn read_i32s(path: &str) -> Vec<i32> {
    let mut b = Vec::new();
    File::open(path).unwrap().read_to_end(&mut b).unwrap();
    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[cfg(feature = "cuda")]
fn argmax(x: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..x.len() {
        if x[i] > x[bi] {
            bi = i;
        }
    }
    bi
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;
    use trapetum::{Model, VERIFY_M};
    let a: Vec<String> = env::args().collect();
    let (mp, pp, cp) = (&a[1], &a[2], &a[3]);
    let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let prompt: Vec<usize> = read_i32s(pp).iter().map(|&t| t as usize).collect();
    let cont: Vec<usize> = read_i32s(cp).iter().map(|&t| t as usize).collect();
    assert!(cont.len() >= 16, "cont.bin needs at least 16 tokens");
    let p = prompt.len();
    const LONG: usize = 1024;
    // bases b with b % 64 == 56: rows b .. b+15 cross the attention chunk boundary at b + 8
    let next56 = |from: usize| from + (56 + 64 - from % 64) % 64;
    let b1 = next56(p + 17);
    let bases = [p, b1, next56(1000usize.max(b1 + 32))];
    // the timing decodes 2 rounds of reps + 4 positions from p + 17 and from max(LONG, the last
    // checked row), and each verify step there writes 16 more rows
    let long_start = LONG.max(bases[2] + 16);
    // the long phase starts at long_start, or later if the short phase ran past it
    let l0 = long_start.max(p + 17 + 2 * (reps + 4));
    let end = l0 + 2 * (reps + 4) + 16;
    let max_seq = end + 64;
    let mut model = Model::load_cbk(mp, max_seq).unwrap();
    if let Err(e) = model.verify_ok() {
        println!("VERIFY UNSUPPORTED: {e}");
        return;
    }
    println!("VERIFY GEMV mode {}", trapetum::gemv_mode());
    let vocab = model.vocab();
    for (t, &tok) in prompt.iter().enumerate() {
        model.forward(tok, t);
    }
    let toks: Vec<usize> = cont[..16].to_vec();
    let other: Vec<usize> = toks.iter().map(|&t| (t + 1) % vocab).collect();
    let same = |x: &[f32], y: &[f32]| x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
    let mut ok = true;
    let mut filled = p; // KV rows below this hold decode data
    for &base in bases.iter() {
        while filled < base {
            model.forward(cont[filled % cont.len()], filled);
            filled += 1;
        }
        let refl: Vec<Vec<f32>> = (0..16).map(|i| model.forward(toks[i], base + i)).collect();
        filled = filled.max(base + 16);
        let am_ref: Vec<u32> = refl.iter().map(|l| argmax(l) as u32).collect();
        // other tokens through the verify path (logits rows, argmax slots) and decode steps (KV rows)
        let scramble = |model: &mut Model| {
            let _ = model.verify_argmax(&other, base, vocab);
            for (i, &t) in other.iter().enumerate() {
                model.forward(t, base + i);
            }
            trapetum::debug_scramble_scratch(); // and every partial buffer and argmax slot to NaN
        };
        for &m in VERIFY_M.iter() {
            scramble(&mut model);
            let v = model.verify_logits(&toks[..m], base);
            let (mut rows_ok, mut maxd) = (0usize, 0f32);
            for r in 0..m {
                let row = &v[r * vocab..(r + 1) * vocab];
                if same(row, &refl[r]) {
                    rows_ok += 1;
                } else {
                    let d = row.iter().zip(&refl[r]).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
                    maxd = maxd.max(d);
                }
            }
            // the next decode step reads the KV rows this verify wrote
            let kv_ok = m == 16 || same(&model.forward(toks[m], base + m), &refl[m]);
            scramble(&mut model);
            let rep_ok = same(&model.verify_logits(&toks[..m], base), &v);
            scramble(&mut model);
            let am_ok = model.verify_argmax(&toks[..m], base, vocab) == am_ref[..m];
            scramble(&mut model);
            let g1 = model.verify_argmax_graph(&toks[..m], base, vocab) == am_ref[..m]; // capture or replay
            let g1_bits = same(&model.verify_last_logits(m), &v);
            scramble(&mut model);
            let g2 = model.verify_argmax_graph(&toks[..m], base, vocab) == am_ref[..m]; // replay
            let g2_bits = same(&model.verify_last_logits(m), &v);
            let row_ok = rows_ok == m && kv_ok && rep_ok && am_ok && g1 && g1_bits && g2 && g2_bits;
            println!(
                "VERIFY base {base} M={m}: rows {rows_ok}/{m} identical (max abs diff {maxd:.3e}), next decode step {}, repeat {}, \
                 argmax {}, graph {} logits {}, graph replay {} logits {} => {}",
                kv_ok, rep_ok, am_ok, g1, g1_bits, g2, g2_bits, if row_ok { "ok" } else { "FAIL" }
            );
            ok &= row_ok;
        }
    }
    println!("{}", if ok { "VERIFY OK" } else { "VERIFY FAIL" });

    // timing, CUDA graph: `reps` decode steps from the current position, then `reps` verify
    // steps per M at that position (each rewrites the same KV rows); two rounds per context
    let mut pos = p + 17;
    let mut tok = toks[15];
    for (ci, &ctx) in [p + 17, long_start].iter().enumerate() {
        while pos < ctx {
            tok = model.forward_argmax_graph(tok, pos, vocab) as usize;
            pos += 1;
        }
        let label = if ci == 0 { "short" } else { "long" };
        for round in ["a", "b"] {
            for _ in 0..4 {
                tok = model.forward_argmax_graph(tok, pos, vocab) as usize; // warm
                pos += 1;
            }
            let t0 = Instant::now();
            for _ in 0..reps {
                tok = model.forward_argmax_graph(tok, pos, vocab) as usize;
                pos += 1;
            }
            let dec = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            println!("TIME {label} round {round} pos {pos} decode ms {dec:.4}");
            for &m in VERIFY_M.iter() {
                let _ = model.verify_argmax_graph(&toks[..m], pos, vocab); // warm
                let t0 = Instant::now();
                for _ in 0..reps {
                    let _ = model.verify_argmax_graph(&toks[..m], pos, vocab);
                }
                let ver = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
                println!("TIME {label} round {round} pos {pos} verify M={m} ms {ver:.4} ratio {:.3}", ver / dec);
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    let _ = (env::args(), read_i32s as fn(&str) -> Vec<i32>);
    eprintln!("verify_check: the batch-invariant verify path is CUDA-only");
}
