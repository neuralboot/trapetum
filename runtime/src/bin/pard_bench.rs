//! PARD speculative decoding benchmark (plan step 5b, 2026-11-06), target-independent mode: a small
//! drafter (amd/PARD2-Llama-3.1-8B run without target features) proposes K tokens in ONE forward over
//! [the new real tokens, K-1 mask tokens], the target verifies [last token, K drafts] in ONE forward
//! (Model::verify_argmax_graph, every row the decode step's bits), and the longest matching prefix
//! plus the target's next token are emitted. Both caches are addressed by position, so a rejection
//! needs no copy: the next forward simply starts at the right position.
//!
//! Each prompt runs in two arms on the same target file, in the same process: plain greedy decoding
//! (prefill through verify graphs, then CUDA-graph decode steps) and the speculative loop (the same
//! prefill). The outputs must be identical token for token (greedy, EOS included, N tokens at most);
//! the time of each arm runs from the prefill start to its last token. Arms alternate in order from
//! one repetition to the next.
//!
//! Usage: pard_bench <target.cbk> <drafter.cbk> <prompts.jsonl> [--k 7] [--n 256] [--reps 2]
//!        [--eos 128001,128008,128009] [--mask 128020] [--max-seq 1024] [--limit N] [--sets a,b]
//! prompts.jsonl: one {"set": "...", "ids": [...]} per line (chat-templated token ids, BOS included).
//! Prints PARD lines (progress, mismatches, per set and pooled results) and a final RULE line.
use std::collections::BTreeMap;
use std::env;
use std::fs;

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;
    use trapetum::{Model, VERIFY_M};
    let a: Vec<String> = env::args().collect();
    let (tp, dp, pp) = (&a[1], &a[2], &a[3]);
    let (mut k, mut n_max, mut reps, mut mask, mut max_seq, mut limit) = (7usize, 256usize, 2usize, 128020usize, 1024usize, usize::MAX);
    let mut eos: Vec<u32> = vec![128001, 128008, 128009];
    let mut sets_filter: Option<Vec<String>> = None;
    let mut i = 4;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--k" => k = v.parse().unwrap(),
            "--n" => n_max = v.parse().unwrap(),
            "--reps" => reps = v.parse().unwrap(),
            "--mask" => mask = v.parse().unwrap(),
            "--max-seq" => max_seq = v.parse().unwrap(),
            "--limit" => limit = v.parse().unwrap(),
            "--eos" => eos = v.split(',').map(|x| x.trim().parse().unwrap()).collect(),
            "--sets" => sets_filter = Some(v.split(',').map(|s| s.trim().to_string()).collect()),
            o => panic!("unknown option {o}"),
        }
        i += 2;
    }
    assert!(VERIFY_M.contains(&(k + 1)), "the target verify takes K+1 rows, one of {VERIFY_M:?}");
    assert!(2 * k <= 17, "a drafting forward of up to K real tokens plus K-1 masks must fit in 16 rows");

    let mut prompts: Vec<(String, Vec<usize>)> = Vec::new();
    let mut per_set: BTreeMap<String, usize> = BTreeMap::new();
    for line in fs::read_to_string(pp).expect("prompts file").lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).expect("prompt line");
        let set = v["set"].as_str().expect("set").to_string();
        if sets_filter.as_ref().map_or(false, |f| !f.contains(&set)) {
            continue;
        }
        let c = per_set.entry(set.clone()).or_insert(0);
        if *c >= limit {
            continue;
        }
        *c += 1;
        let ids: Vec<usize> = v["ids"].as_array().expect("ids").iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert!(!ids.is_empty(), "empty prompt");
        prompts.push((set, ids));
    }
    assert!(!prompts.is_empty(), "no prompt selected");
    let longest = prompts.iter().map(|p| p.1.len()).max().unwrap();
    assert!(longest + n_max + 2 * k + 16 <= max_seq, "max_seq {max_seq} is too small for {longest} prompt tokens + {n_max}");

    let mut target = Model::load_cbk(tp, max_seq).unwrap();
    let mut drafter = Model::load_cbk(dp, max_seq).unwrap();
    for (name, m) in [("target", &target), ("drafter", &drafter)] {
        if let Err(e) = m.verify_ok() {
            println!("PARD UNSUPPORTED {name}: {e}");
            return;
        }
    }
    let vocab = target.vocab();
    assert_eq!(drafter.vocab(), vocab, "the target and drafter vocabularies differ");
    println!(
        "PARD loaded target {tp} and drafter {dp}: vocab {vocab}, K={k}, N={n_max}, reps {reps}, max_seq {max_seq}, \
         {} prompts {per_set:?}, gemv mode {}",
        prompts.len(), trapetum::gemv_mode()
    );

    // capture every verify graph both models will use (no capture inside a timed region); the KV
    // rows these calls write are overwritten by the next prefill
    for &m in VERIFY_M.iter() {
        let toks = vec![mask; m];
        let _ = target.verify_argmax_graph(&toks, 0, vocab);
        let _ = drafter.verify_argmax_graph(&toks, 0, vocab);
    }
    let _ = plain(&mut target, &prompts[0].1, n_max, &eos, vocab);
    let _ = pard(&mut target, &mut drafter, &prompts[0].1, n_max, &eos, vocab, k, mask);

    #[derive(Default, Clone)]
    struct Acc {
        prompts: usize,
        tokens: usize,
        plain_s: f64,
        pard_s: f64,
        rounds: usize,
        identical: usize,
    }
    let mut acc: BTreeMap<String, Acc> = BTreeMap::new();
    let mut mismatches = 0usize;
    let total = prompts.len() * reps;
    let mut done = 0usize;
    for rep in 0..reps {
        for (pi, (set, ids)) in prompts.iter().enumerate() {
            let run_plain = |t: &mut Model| {
                let t0 = Instant::now();
                let out = plain(t, ids, n_max, &eos, vocab);
                (out, t0.elapsed().as_secs_f64())
            };
            let ((po, ps), (so, ss, rounds)) = if rep % 2 == 0 {
                let p = run_plain(&mut target);
                let t0 = Instant::now();
                let (o, r) = pard(&mut target, &mut drafter, ids, n_max, &eos, vocab, k, mask);
                (p, (o, t0.elapsed().as_secs_f64(), r))
            } else {
                let t0 = Instant::now();
                let (o, r) = pard(&mut target, &mut drafter, ids, n_max, &eos, vocab, k, mask);
                let s = (o, t0.elapsed().as_secs_f64(), r);
                (run_plain(&mut target), s)
            };
            let same = po == so;
            if !same {
                mismatches += 1;
                let at = po.iter().zip(&so).take_while(|(x, y)| x == y).count();
                println!(
                    "PARD MISMATCH rep {rep} set {set} prompt {pi}: plain {} tokens, pard {} tokens, first difference at {at} \
                     (plain {:?}, pard {:?})",
                    po.len(), so.len(), po.get(at), so.get(at)
                );
            }
            let e = acc.entry(set.clone()).or_default();
            e.prompts += 1;
            e.tokens += po.len();
            e.plain_s += ps;
            e.pard_s += ss;
            e.rounds += rounds;
            e.identical += same as usize;
            done += 1;
            if done % 20 == 0 || done == total {
                println!("PARD progress {done}/{total} ({mismatches} mismatches)");
            }
        }
    }
    let mut all = Acc::default();
    for (set, e) in &acc {
        // tau: tokens emitted per verify round, the bonus token included (the prefill's first token excluded)
        let tau = (e.tokens - e.prompts) as f64 / e.rounds.max(1) as f64;
        println!(
            "PARD set {set}: {} runs, {} tokens, plain {:.1} tok/s, pard {:.1} tok/s, speedup {:.3}x, tau {:.2}, identical {}/{}",
            e.prompts, e.tokens, e.tokens as f64 / e.plain_s, e.tokens as f64 / e.pard_s, e.plain_s / e.pard_s, tau,
            e.identical, e.prompts
        );
        all.prompts += e.prompts;
        all.tokens += e.tokens;
        all.plain_s += e.plain_s;
        all.pard_s += e.pard_s;
        all.rounds += e.rounds;
        all.identical += e.identical;
    }
    let speedup = all.plain_s / all.pard_s;
    let tau = (all.tokens - all.prompts) as f64 / all.rounds.max(1) as f64;
    println!(
        "PARD pooled: {} runs, {} tokens, plain {:.1} tok/s, pard {:.1} tok/s, speedup {:.3}x, tau {:.2}, identical {}/{}",
        all.prompts, all.tokens, all.tokens as f64 / all.plain_s, all.tokens as f64 / all.pard_s, speedup, tau,
        all.identical, all.prompts
    );
    let identity = all.identical == all.prompts;
    println!(
        "PARD RULE: identity {} ({}/{}), pooled speedup {:.3}x {} 1.6 => {}",
        if identity { "PASS" } else { "FAIL" }, all.identical, all.prompts, speedup,
        if speedup >= 1.6 { ">=" } else { "<" }, if identity && speedup >= 1.6 { "GO" } else { "NO-GO" }
    );
}

/// Plain greedy: prefill through verify graphs, then one CUDA-graph decode step per token.
#[cfg(feature = "cuda")]
fn plain(target: &mut trapetum::Model, ids: &[usize], n_max: usize, eos: &[u32], vocab: usize) -> Vec<u32> {
    let mut tok = target.prefill_verify(ids, 0, vocab);
    let mut out = vec![tok];
    let mut pos = ids.len();
    while out.len() < n_max && !eos.contains(&tok) {
        tok = target.forward_argmax_graph(tok as usize, pos, vocab);
        pos += 1;
        out.push(tok);
    }
    out
}

/// One drafting forward: `real` tokens at positions base.., then K-1 mask tokens, padded with mask
/// tokens to the next verify size (padding rows come after every used row, so causality keeps them
/// out of the result). The drafts are the argmax of the last real row and of the K-1 mask rows.
#[cfg(feature = "cuda")]
fn draft(drafter: &mut trapetum::Model, real: &[usize], base: usize, vocab: usize, k: usize, mask: usize) -> Vec<u32> {
    let need = real.len() + k - 1;
    let m = *trapetum::VERIFY_M.iter().find(|&&m| m >= need).expect("a drafting forward of at most 16 rows");
    let mut rows = real.to_vec();
    rows.resize(m, mask);
    let am = drafter.verify_argmax_graph(&rows, base, vocab);
    am[real.len() - 1..real.len() - 1 + k].to_vec()
}

/// The speculative loop (greedy). Returns the output and the number of verify rounds. The output is
/// plain greedy's: every verify row is the decode step's bits, an accepted draft equals the target's
/// own choice on the same prefix, and the rows of rejected drafts are overwritten by the next verify
/// before any row reads them.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn pard(target: &mut trapetum::Model, drafter: &mut trapetum::Model, ids: &[usize], n_max: usize, eos: &[u32],
        vocab: usize, k: usize, mask: usize) -> (Vec<u32>, usize) {
    let p = ids.len();
    let g = target.prefill_verify(ids, 0, vocab);
    let mut out = vec![g];
    if out.len() >= n_max || eos.contains(&g) {
        return (out, 0);
    }
    let _ = drafter.prefill_verify(ids, 0, vocab); // the drafter's own cache over the prompt
    let mut drafts = draft(drafter, &[g as usize], p, vocab, k, mask);
    let (mut u, mut n, mut rounds) = (g as usize, p, 0usize); // u sits at position n, not yet in the target's cache
    loop {
        let mut rows = Vec::with_capacity(k + 1);
        rows.push(u);
        rows.extend(drafts.iter().map(|&d| d as usize));
        let gs = target.verify_argmax_graph(&rows, n, vocab);
        rounds += 1;
        let mut acc = 0usize;
        while acc < k && drafts[acc] == gs[acc] {
            acc += 1;
        }
        let bonus = gs[acc];
        for &t in drafts[..acc].iter().chain(std::iter::once(&bonus)) {
            out.push(t);
            if out.len() >= n_max || eos.contains(&t) {
                return (out, rounds);
            }
        }
        let mut real: Vec<usize> = drafts[..acc].iter().map(|&d| d as usize).collect();
        real.push(bonus as usize);
        drafts = draft(drafter, &real, n + 1, vocab, k, mask);
        u = bonus as usize;
        n += acc + 1;
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    let _ = (env::args(), fs::read_to_string::<&str>, BTreeMap::<u8, u8>::new());
    eprintln!("pard_bench: the speculative loop runs on the CUDA backend only");
}
