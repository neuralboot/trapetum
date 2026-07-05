//! Validates the K=1 speculative decode loop is lossless and that acceptances save forwards.
//!   cargo run --release --no-default-features --features metal --bin spec_check
fn main() {
    let (ok_o, ok_w, fwds_o, fwds_w, n) = trapetum::check_spec_decode();
    println!("== speculative decode K=1 (lossless check) ==");
    println!("  perfect drafter : seq==greedy {}  forwards={}/{}  (accepts save forwards)", if ok_o {"OK"} else {"FAIL"}, fwds_o, n);
    println!("  adversarial     : seq==greedy {}  forwards={}/{}  (rejects still correct)", if ok_w {"OK"} else {"FAIL"}, fwds_w, n);
    if ok_o && ok_w {
        let speedup = n as f64 / fwds_o as f64;
        println!("\nLOSSLESS: spec-dec output == plain greedy for both drafters.");
        println!("Perfect-drafter speedup (tokens/forward): {:.2}x  ({} tokens in {} target forwards)", speedup, n, fwds_o);
    } else {
        println!("\nFAIL: speculative output diverged from greedy.");
        std::process::exit(1);
    }
}
