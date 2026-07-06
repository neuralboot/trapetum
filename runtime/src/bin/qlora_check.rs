//! Validate the DeepSeek-V3/R1 additions: q_lora MLA (`MlaAttn::new_qlora`, quantized q_b/o)
//! and the V3 sigmoid+bias grouped ("noaux_tc") router, both on tiny synthetic configs (no
//! download, no Python).
//!   cargo run --release --no-default-features --features metal --bin qlora_check
fn main() {
    println!("== DeepSeek-V3/R1 q_lora MLA (quantized q_b/o) vs CPU reference ==");
    let e = trapetum::check_qlora_mla();
    let ok1 = e < 2e-2;
    println!("  rel_err = {e:.2e}  {}", if ok1 { "OK" } else { "FAIL" });

    println!("\n== V3 sigmoid+bias grouped router (noaux_tc) vs independent CPU reference ==");
    let ok2 = trapetum::check_moe_route_v3();
    println!("  {}", if ok2 { "OK" } else { "FAIL" });

    if ok1 && ok2 {
        println!("\nPASS: q_lora MLA (dense q_a + quantized q_b/o) and V3 router both correct.");
    } else {
        println!("\nFAIL");
        std::process::exit(1);
    }
}
