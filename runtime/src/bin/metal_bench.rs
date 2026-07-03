//! Apple-GPU microbenchmark: fused 4-bit codebook decode GEMV vs dense fp16 GEMV.
//! The Metal analogue of the paper's cuBLAS comparison, at the same 4096x4096 shape.
//!   cargo run --release --no-default-features --features metal --bin metal_bench
fn main() {
    let shapes = [(4096usize, 4096usize), (4096, 11008), (11008, 4096)];
    let iters = 200;
    println!("Trapetum Metal microbenchmark (batch-1 GEMV), {iters} iters/shape\n");
    println!("{:<14} {:>10} {:>10} {:>9} {:>12}", "shape ICxOC", "4-bit ms", "fp16 ms", "speedup", "4b GB/s");
    for (ic, oc) in shapes {
        let (ms4, msf) = trapetum::bench_gemv(ic, oc, iters);
        // 4-bit reads: packed (ic*oc/2 bytes) + codebook (16*oc*2) + x (ic*2)
        let bytes4 = (ic * oc / 2 + 16 * oc * 2 + ic * 2) as f64;
        let gbps = bytes4 / (ms4 * 1e-3) / 1e9;
        println!("{:<14} {:>10.4} {:>10.4} {:>8.2}x {:>11.1}",
            format!("{ic}x{oc}"), ms4, msf, msf / ms4, gbps);
    }
}
