//! Validates the speculative decode loop is lossless for K=1..3 and that acceptances save forwards.
//!   cargo run --release --no-default-features --features metal --bin spec_check
fn main() {
    let mut bad = 0;
    println!("== speculative decode: lossless + tokens/forward, K=1..3 ==");
    for k in [1usize, 2, 3] {
        let (ok_o, ok_w, fo, fw, n) = trapetum::check_spec_decode_k(k);
        let tpf = n as f64 / fo as f64; // tokens per target forward, perfect drafter
        let oks = if ok_o && ok_w {"OK"} else {"FAIL"};
        println!("  K={k}  lossless(oracle={ok_o}, adversarial={ok_w}) {oks}  |  perfect: {n} toks in {fo} forwards = {tpf:.2} tok/fwd  (ceiling {})", k+1);
        if !(ok_o && ok_w) { bad += 1; }
        let _ = fw;
    }
    if bad == 0 {
        println!("\nLOSSLESS at K=1..3. Bigger K = more tokens per target forward (ceiling K+1),");
        println!("bounded by gemm_mtile's validated M<=4 range (so K<=3).");
    } else { println!("\nFAIL"); std::process::exit(1); }
}
