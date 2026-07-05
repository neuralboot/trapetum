//! Validate the DeepSeek MLA attention block (full projections + absorption + latent cache)
//! vs a full-reconstruction CPU reference. Phase 1 of DeepSeek support (attention wiring).
//!   cargo run --release --no-default-features --features metal --bin deepseek_check
fn main() {
    println!("== DeepSeek MLA attention block (absorbed) vs full-reconstruction reference ==");
    let e = trapetum::check_mla_block();
    let ok = e < 4e-2;
    println!("  rel_err = {e:.2e}  {}", if ok {"OK"} else {"FAIL"});
    if ok { println!("\nMLA attention block CORRECT (q/kv proj + absorption + decoupled rope + latent cache)."); }
    else { println!("\nFAIL"); std::process::exit(1); }
}
