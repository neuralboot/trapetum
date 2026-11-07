//! Speculative decoding with an autoregressive drafter (2026-11-07): a small model with the same
//! vocabulary (for example Llama-2-7B AQLM 2x8 in front of Llama-2-70B AQLM 1x16) proposes K tokens,
//! one CUDA-graph decode step each; the target verifies [last token, K drafts] in ONE forward
//! (Model::verify_argmax_graph, every row the decode step's bits), and the longest matching prefix
//! plus the target's next token are emitted. Both caches are addressed by position, so a rejection
//! needs no copy: the next step simply writes over the rejected rows before anything reads them.
//!
//! Each prompt runs plain greedy decoding once (prefill through verify graphs, then CUDA-graph
//! decode steps) and the speculative loop once per K (same prefill), on the same target in the
//! same process. Every speculative output must equal the plain output token for token (N tokens at
//! most, EOS included). The order alternates from one prompt to the next (plain first, then last).
//! Each time runs from the prefill start to the last token; a speculative time includes the
//! drafter's prefill (one decode step per prompt token: a drafter with Mx8 projections has no
//! batch-invariant verify, and needs none) and every drafting step.
//!
//! Usage: spec_ar <target.cbk> <drafter.cbk> <prompts.jsonl> [--k 7] [--n 128] [--eos 2]
//!        [--max-seq 1024] [--limit N]
//!   --k takes a comma-separated list (for example 3,7,15): K + 1 must be 2, 4, 8 or 16.
//! prompts.jsonl: one {"set": "...", "ids": [...]} per line (token ids, BOS included).
//! Prints SPEC lines: progress, mismatches, per set and pooled results per K, and a final SPEC
//! IDENTITY line. No em dashes or en dashes anywhere.
use std::collections::BTreeMap;
use std::env;
use std::fs;

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;
    use trapetum::{Model, VERIFY_M};
    let a: Vec<String> = env::args().collect();
    assert!(a.len() >= 4, "usage: spec_ar <target.cbk> <drafter.cbk> <prompts.jsonl> [options]");
    let (tp, dp, pp) = (&a[1], &a[2], &a[3]);
    let (mut ks, mut n_max, mut max_seq, mut limit) = (vec![7usize], 128usize, 1024usize, usize::MAX);
    let mut eos: Vec<u32> = vec![2];
    let mut i = 4;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--k" => ks = v.split(',').map(|x| x.trim().parse().unwrap()).collect(),
            "--n" => n_max = v.parse().unwrap(),
            "--max-seq" => max_seq = v.parse().unwrap(),
            "--limit" => limit = v.parse().unwrap(),
            "--eos" => eos = v.split(',').map(|x| x.trim().parse().unwrap()).collect(),
            o => panic!("unknown option {o}"),
        }
        i += 2;
    }
    for &k in &ks {
        assert!(k >= 1 && VERIFY_M.contains(&(k + 1)), "the target verify takes K+1 rows, one of {VERIFY_M:?}");
    }
    let kmax = *ks.iter().max().unwrap();

    let mut prompts: Vec<(String, Vec<usize>)> = Vec::new();
    for line in fs::read_to_string(pp).expect("prompts file").lines() {
        if line.trim().is_empty() || prompts.len() >= limit {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).expect("prompt line");
        let set = v["set"].as_str().expect("set").to_string();
        let ids: Vec<usize> = v["ids"].as_array().expect("ids").iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert!(!ids.is_empty(), "empty prompt");
        prompts.push((set, ids));
    }
    assert!(!prompts.is_empty(), "no prompt");
    let longest = prompts.iter().map(|p| p.1.len()).max().unwrap();
    // the warm-up below runs every verify size (up to 16 rows) at base 0
    let vmax = *VERIFY_M.iter().max().unwrap();
    assert!(max_seq >= vmax && longest + n_max + kmax + 2 <= max_seq,
            "max_seq {max_seq} is too small for {longest} prompt tokens + {n_max} (and at least {vmax})");

    let mut target = Model::load_cbk(tp, max_seq).unwrap();
    if let Err(e) = target.verify_ok() {
        println!("SPEC UNSUPPORTED target: {e}");
        return;
    }
    let mut drafter = Model::load_cbk(dp, max_seq).unwrap();
    let vocab = target.vocab();
    assert_eq!(drafter.vocab(), vocab, "the target and drafter vocabularies differ");
    println!(
        "SPEC loaded target {tp} and drafter {dp}: vocab {vocab}, K {ks:?}, N {n_max}, max_seq {max_seq}, {} prompts, gemv mode {}",
        prompts.len(), trapetum::gemv_mode()
    );

    // capture every graph the timed runs use (no capture inside a timed region): the target's
    // verify graphs (prefill sizes and K + 1) and both decode graphs; the KV rows these calls
    // write are overwritten by the next prefill
    for &m in VERIFY_M.iter() {
        let _ = target.verify_argmax_graph(&vec![1usize; m], 0, vocab);
    }
    // both decode graphs, even if the first prompt's first token is EOS (row 0 is rewritten by
    // the next prefill and by the drafter's next prompt pass before anything reads it)
    let _ = target.forward_argmax_graph(1, 0, vocab);
    let _ = drafter.forward_argmax_graph(1, 0, vocab);
    let _ = plain(&mut target, &prompts[0].1, n_max.min(8), &eos, vocab);
    for &k in &ks {
        let _ = spec(&mut target, &mut drafter, &prompts[0].1, n_max.min(2 * k + 4), &eos, vocab, k);
    }

    #[derive(Default, Clone)]
    struct Acc {
        prompts: usize,
        tokens: usize,
        plain_s: f64,
        spec_s: f64,
        rounds: usize,
        identical: usize,
    }
    let mut acc: BTreeMap<(usize, String), Acc> = BTreeMap::new();
    let mut mismatches = 0usize;
    for (pi, (set, ids)) in prompts.iter().enumerate() {
        let run_plain = |t: &mut Model| {
            let t0 = Instant::now();
            let out = plain(t, ids, n_max, &eos, vocab);
            (out, t0.elapsed().as_secs_f64())
        };
        let run_specs = |t: &mut Model, d: &mut Model| {
            ks.iter()
                .map(|&k| {
                    let t0 = Instant::now();
                    let (o, r) = spec(t, d, ids, n_max, &eos, vocab, k);
                    (k, o, t0.elapsed().as_secs_f64(), r)
                })
                .collect::<Vec<_>>()
        };
        let ((po, ps), specs) = if pi % 2 == 0 {
            let p = run_plain(&mut target);
            (p, run_specs(&mut target, &mut drafter))
        } else {
            let s = run_specs(&mut target, &mut drafter);
            (run_plain(&mut target), s)
        };
        for (k, so, ss, rounds) in specs {
            let same = po == so;
            if !same {
                mismatches += 1;
                let at = po.iter().zip(&so).take_while(|(x, y)| x == y).count();
                println!(
                    "SPEC MISMATCH K {k} set {set} prompt {pi}: plain {} tokens, spec {} tokens, first difference at {at} \
                     (plain {:?}, spec {:?})",
                    po.len(), so.len(), po.get(at), so.get(at)
                );
            }
            let e = acc.entry((k, set.clone())).or_default();
            e.prompts += 1;
            e.tokens += po.len();
            e.plain_s += ps;
            e.spec_s += ss;
            e.rounds += rounds;
            e.identical += same as usize;
        }
        println!("SPEC progress {}/{} ({mismatches} mismatches)", pi + 1, prompts.len());
    }
    let mut all_identical = true;
    for &k in &ks {
        let mut all = Acc::default();
        for ((kk, set), e) in &acc {
            if *kk != k {
                continue;
            }
            // tau: tokens emitted per verify round, the bonus token included (the prefill's token excluded)
            let tau = (e.tokens - e.prompts) as f64 / e.rounds.max(1) as f64;
            println!(
                "SPEC K {k} set {set}: {} prompts, {} tokens, plain {:.2} tok/s, spec {:.2} tok/s, speedup {:.3}x, tau {:.2}, identical {}/{}",
                e.prompts, e.tokens, e.tokens as f64 / e.plain_s, e.tokens as f64 / e.spec_s, e.plain_s / e.spec_s, tau,
                e.identical, e.prompts
            );
            all.prompts += e.prompts;
            all.tokens += e.tokens;
            all.plain_s += e.plain_s;
            all.spec_s += e.spec_s;
            all.rounds += e.rounds;
            all.identical += e.identical;
        }
        let tau = (all.tokens - all.prompts) as f64 / all.rounds.max(1) as f64;
        println!(
            "SPEC K {k} pooled: {} prompts, {} tokens, plain {:.2} tok/s, spec {:.2} tok/s, speedup {:.3}x, tau {:.2}, identical {}/{}",
            all.prompts, all.tokens, all.tokens as f64 / all.plain_s, all.tokens as f64 / all.spec_s, all.plain_s / all.spec_s,
            tau, all.identical, all.prompts
        );
        all_identical &= all.identical == all.prompts;
    }
    println!("SPEC IDENTITY {} ({mismatches} mismatches)", if all_identical { "PASS" } else { "FAIL" });
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

/// The speculative loop (greedy). Returns the output and the number of verify rounds. The output
/// is plain greedy's: every verify row is the decode step's bits, an accepted draft equals the
/// target's own choice on the same prefix, and the rows of rejected drafts (in both caches) are
/// written over by the next step before any row reads them.
#[cfg(feature = "cuda")]
fn spec(target: &mut trapetum::Model, drafter: &mut trapetum::Model, ids: &[usize], n_max: usize, eos: &[u32],
        vocab: usize, k: usize) -> (Vec<u32>, usize) {
    let p = ids.len();
    let g = target.prefill_verify(ids, 0, vocab);
    let mut out = vec![g];
    if out.len() >= n_max || eos.contains(&g) {
        return (out, 0);
    }
    for (i, &t) in ids.iter().enumerate() {
        let _ = drafter.forward_argmax_graph(t, i, vocab); // the drafter's own cache over the prompt
    }
    // u sits at position n and is in neither cache yet
    let (mut u, mut n, mut rounds) = (g as usize, p, 0usize);
    let mut rows = Vec::with_capacity(k + 1);
    loop {
        // draft: the drafter takes u at n, then each draft; d_K itself is not fed back
        rows.clear();
        rows.push(u);
        let mut t = u;
        for j in 0..k {
            let d = drafter.forward_argmax_graph(t, n + j, vocab);
            rows.push(d as usize);
            t = d as usize;
        }
        let gs = target.verify_argmax_graph(&rows, n, vocab);
        rounds += 1;
        let mut acc = 0usize;
        while acc < k && rows[acc + 1] as u32 == gs[acc] {
            acc += 1;
        }
        let bonus = gs[acc];
        for &t in rows[1..=acc].iter() {
            out.push(t as u32);
            if out.len() >= n_max || eos.contains(&(t as u32)) {
                return (out, rounds);
            }
        }
        out.push(bonus);
        if out.len() >= n_max || eos.contains(&bonus) {
            return (out, rounds);
        }
        if acc == k {
            // every draft accepted: the drafter has not seen d_K (position n + K) yet
            let _ = drafter.forward_argmax_graph(rows[k], n + k, vocab);
        }
        u = bonus as usize;
        n += acc + 1;
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    let _ = (env::args(), fs::read_to_string::<&str>, BTreeMap::<u8, u8>::new());
    eprintln!("spec_ar: the speculative loop runs on the CUDA backend only");
}
