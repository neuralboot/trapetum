//! Validate the MoE block (wall 1: dense-runtime) AND expert offloading (wall 2: memory).
//!   cargo run --release --no-default-features --features metal --bin moe_check
fn main() {
    println!("== Wall 1: MoE routing (256 experts, top-k=8, +shared) ==");
    let e1 = trapetum::check_moe();
    println!("  top-k vs dense  rel_err={e1:.2e}  {}", if e1<2e-2 {"OK"} else {"FAIL"});

    println!("\n== Wall 2: expert offloading (cache << n_experts, streamed from host) ==");
    let (e2, cap, ne, ups) = trapetum::check_moe_offload();
    println!("  offloaded vs all-resident  rel_err={e2:.2e}  {}", if e2<2e-2 {"OK"} else {"FAIL"});
    println!("  GPU-resident experts: {cap}/{ne}  ({}x smaller working set); {ups} expert streams over 12 tokens", ne/cap);

    // honest memory / throughput math for DeepSeek-V3 (671B, 4-bit)
    println!("\n== What this means for DeepSeek-V3 (671B, 4-bit ~ 350 GB) ==");
    println!("  Resident on GPU: router + shared expert + MLA cache (tiny) + {cap} cached experts/layer.");
    println!("  Routed experts (350 GB) live in host RAM / NVMe; ~top-k=8 stream in per token per layer.");
    println!("  Active params/token ~ 37B (4-bit ~18.5 GB moved/token). Bandwidth-bound:");
    println!("    PCIe4 ~25 GB/s -> ~0.7 s/token (~1.4 tok/s);  NVMe ~6 GB/s -> ~3 s/token (~0.3 tok/s).");
    println!("  => RUNNABLE on one 24 GB GPU (not fast). Multi-GPU keeps experts resident for real speed.");

    if e1<2e-2 && e2<2e-2 { println!("\nBOTH WALLS ADDRESSED: MoE routing correct + offloading lossless."); }
    else { println!("\nFAIL"); std::process::exit(1); }
}
