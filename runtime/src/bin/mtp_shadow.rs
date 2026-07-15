//! Shadow-mode MTP accept-length measurement: greedy-decode with the main model exactly as
//! deepseek_run does, and at every step ALSO chain-draft up to DEPTH tokens with the MTP module,
//! then score the drafts against the real tokens the main model produces. The decode itself is
//! untouched (outputs identical to deepseek_run), so this measures the ONE number that decides
//! the speculative-decode ROI: the expected accept length E[L] = 1 + p1 + p1*p2 + ...
//!
//!   mtp_shadow <model.cbk> <mtp.cbk> <prompt.bin>       (TRAPETUM_NTOK tokens, default 64)
//!
//! During prefill the MTP consumes the REAL next token at each position, so its KV cache is
//! built from the true sequence (same shift-by-one stream MTP was trained on). During decode,
//! the depth-1 draft at each position also uses real inputs and stays in the cache; deeper
//! chained drafts write cache rows that later real-token passes overwrite (position-addressed).
use std::env;
use std::time::Instant;
use trapetum::{DeepSeekModel, Mtp, argmax, read_i32s};

const DEPTH: usize = 3;

fn main() {
    let a: Vec<String> = env::args().collect();
    let prompt: Vec<usize> = read_i32s(&a[3]).iter().map(|&t| t as usize).collect();
    let t0 = Instant::now();
    let mut m = DeepSeekModel::load_deepseek(&a[1], 2048).unwrap();
    let mut mtp = Mtp::load_mtp1(&a[2], 2048, m.vocab()).unwrap();
    println!("loaded model+mtp in {:.1}s, vocab={}, prompt={} tokens", t0.elapsed().as_secs_f64(), m.vocab(), prompt.len());

    // Prefill: main forward at every prompt position; MTP pass at every position that has a
    // real next token (builds the MTP KV cache on the true shifted stream).
    let mut last = vec![];
    for (t, &tok) in prompt.iter().enumerate() {
        last = m.forward(tok, t);
        if t + 1 < prompt.len() {
            let _ = mtp.draft(&m, prompt[t + 1], &m.last_hidden(), t);
        }
    }

    let n: usize = env::var("TRAPETUM_NTOK").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let mut tok = argmax(&last);
    let mut got: Vec<usize> = Vec::new();
    // draft_at[s] = the chained drafts made at step s; the depth-d draft targets the token
    // emitted at step s+d and is scored when that token is decided.
    let mut draft_at: Vec<Vec<usize>> = Vec::new();
    let mut hits = vec![0usize; DEPTH];
    let mut totals = vec![0usize; DEPTH];
    let t0 = Instant::now();

    for i in 0..n {
        got.push(tok);
        // chain-draft DEPTH tokens from (tok, main last hidden) BEFORE the main forward,
        // exactly what a real spec loop would have available at this point.
        let mtp_t0 = Instant::now();
        let mut drafts = Vec::with_capacity(DEPTH);
        let mut prev_h = m.last_hidden();
        let mut dtok = tok;
        for d in 0..DEPTH {
            let lg = mtp.draft(&m, dtok, &prev_h, prompt.len() + i + d - 1);
            dtok = argmax(&lg);
            drafts.push(dtok);
            prev_h = mtp.last_hidden();
        }
        let mtp_ms = mtp_t0.elapsed().as_secs_f64() * 1e3;
        draft_at.push(drafts);

        // main greedy step (the reference decode, unchanged)
        let ti = Instant::now();
        let lg = m.forward(tok, prompt.len() + i);
        let main_ms = ti.elapsed().as_secs_f64() * 1e3;
        tok = argmax(&lg);

        // resolve drafts that predicted THIS token: draft made at step s, depth d, targets
        // step s+d's emitted token, which is exactly `tok` when s+d == i+1... the token emitted
        // at step i+1 is `tok` (pushed next loop). Score at push time next iteration instead:
        // simpler -- score all drafts targeting position got.len() (the token just computed).
        for (s, ds) in draft_at.iter().enumerate() {
            for (d, &dt) in ds.iter().enumerate() {
                if s + d + 1 == got.len() {      // this draft targeted the token just decided
                    totals[d] += 1;
                    if dt == tok { hits[d] += 1; }
                }
            }
        }
        if i < 8 || (i + 1) % 16 == 0 {
            println!("step {:>3}: main {:>8.1} ms | mtp x{} {:>7.1} ms | draft1={} real={} {}",
                     i, main_ms, DEPTH, mtp_ms, draft_at[i][0], tok,
                     if draft_at[i][0] == tok { "HIT" } else { "miss" });
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("continuation ids: {:?}", &got[..got.len().min(24)]);
    println!("decode+shadow: {:.1} ms/token ({:.2} tok/s incl. shadow drafting)", dt*1e3/n as f64, n as f64/dt);

    // accept stats: p_d = P(depth-d draft correct | all shallower drafts correct is NOT
    // conditioned here; report both raw per-depth accuracy and the chained expectation).
    let p: Vec<f64> = (0..DEPTH).map(|d| if totals[d] > 0 { hits[d] as f64 / totals[d] as f64 } else { 0.0 }).collect();
    for d in 0..DEPTH {
        println!("depth-{} draft accuracy: {}/{} = {:.3}", d+1, hits[d], totals[d], p[d]);
    }
    // E[L] under the (standard) approximation that acceptance at each depth is the measured
    // per-depth accuracy: E[L] = 1 + p1 + p1*p2 + p1*p2*p3.
    let mut el = 1.0; let mut acc = 1.0;
    for d in 0..DEPTH { acc *= p[d]; el += acc; }
    println!("expected accept length E[L] ~= {:.2}  (speedup bound at zero draft cost: x{:.2})", el, el);
    println!("MTPSHADOW_DONE");
}
