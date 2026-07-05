//! Validate the MLA (Multi-head Latent Attention, DeepSeek-V2/V3) decode attention kernel
//! against a CPU reference. This is a kernel building block: full DeepSeek also needs MoE
//! routing and does not fit a single 24GB GPU, so this is a proof-of-concept, not a runnable model.
//!   cargo run --release --no-default-features --features metal --bin mla_check
fn main() {
    println!("== MLA decode attention (DeepSeek-V2/V3, absorption form) vs CPU ref ==");
    let mut bad = 0;
    // (n_heads, d_c=kv_lora_rank, d_rope=qk_rope_head_dim, seqlen)
    for (nh, dc, dr, sl) in [(16usize,512usize,64usize,7usize),(8,512,64,20),(32,256,64,12),(4,128,32,6)] {
        let e = trapetum::check_mla_attn(nh, dc, dr, sl);
        let ok = e < 2e-2;
        println!("  mla nh={nh} d_c={dc} d_rope={dr} seqlen={sl}  rel_err={e:.2e}  {}", if ok {"OK"} else {"FAIL"});
        if !ok { bad += 1; }
    }
    if bad == 0 { println!("\nMLA decode attention CORRECT. Kernel only; full DeepSeek needs MoE + >>24GB."); }
    else { println!("\nFAIL"); std::process::exit(1); }
}
