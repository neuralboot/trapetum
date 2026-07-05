//! Micro-profile of the speculative-decode cost model. Isolates per-call cost of:
//! target M=1 forward, target M=2/3/4 batched verify (forward_mk), drafter M=1 forward.
//!   profile_spec <target.cbk> <drafter.cbk>
use std::time::Instant;
use trapetum::Model;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut target = Model::load_cbk(&a[1], 1024).unwrap();
    let mut drafter = Model::load_cbk(&a[2], 1024).unwrap();
    let n = 64usize;

    // warm both models (allocations, caches, JIT)
    for i in 0..8 { let _ = target.forward(1, i); }
    for i in 0..8 { let _ = drafter.forward(1, i); }
    let _ = target.forward_mk(&[1, 2], 8);
    let _ = target.forward_mk(&[1, 2, 3], 8);
    let _ = target.forward_mk(&[1, 2, 3, 4], 8);

    let t = Instant::now();
    for i in 0..n { let _ = target.forward(1, 8 + i); }
    let t1 = t.elapsed().as_secs_f64() / n as f64;
    println!("target  M=1 : {:8.3} ms/call", t1 * 1e3);

    for m in 2..=4usize {
        let toks: Vec<usize> = (1..=m).collect();
        let t = Instant::now();
        for i in 0..n { let _ = target.forward_mk(&toks, 8 + i); }
        let tm = t.elapsed().as_secs_f64() / n as f64;
        println!("target  M={} : {:8.3} ms/call  ({:.2}x of M=1)", m, tm * 1e3, tm / t1);
    }

    let t = Instant::now();
    for i in 0..n { let _ = drafter.forward(1, 8 + i); }
    let td = t.elapsed().as_secs_f64() / n as f64;
    println!("drafter M=1 : {:8.3} ms/call  (ratio {:.3} of target M=1)", td * 1e3, td / t1);

    // spec cost model at K=2, alpha=0.83: per round = 1 verify(M=3) + ~3 drafter fwds
    println!("cost model K=2: round = M=3 verify + 3 drafts; tokens/round ~2.7");
}
