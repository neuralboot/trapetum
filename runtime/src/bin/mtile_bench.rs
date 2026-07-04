//! M0 go/no-go: does the small-M fused 4-bit decode GEMM stay bandwidth-bound?
//! If verifying M=6 columns costs about the same wall-clock as M=1, then a
//! speculative-decoding verification of K+1 draft tokens is ~free (one weight
//! read serves all M), and the projected 1.3-2.5x speedup is unlocked.
//!   cargo run --release --no-default-features --features metal --bin mtile_bench
fn main() {
    let (ic, oc) = (4096usize, 4096usize); // a Llama-2 7B attention/proj shape
    let iters = 200;
    println!("Trapetum M0: small-M fused decode GEMM, {ic}x{oc}, {iters} iters\n");
    let b1 = trapetum::bench_mtile(ic, oc, 1, iters);
    let b2 = trapetum::bench_mtile2(ic, oc, 1, iters);
    println!("v1 (naive acc[8][8], atomics)      vs   v2 (acc[2][M], no atomics)");
    println!("{:>3} | {:>9} {:>13} | {:>9} {:>13}", "M", "v1 ms", "v1 ms/token", "v2 ms", "v2 ms/token");
    for m in [1usize, 2, 3, 4, 5, 6, 8] {
        let ms1 = trapetum::bench_mtile(ic, oc, m, iters);
        let ms2 = trapetum::bench_mtile2(ic, oc, m, iters);
        println!("{:>3} | {:>9.4} {:>13.4} | {:>9.4} {:>13.4}", m, ms1, ms1 / m as f64, ms2, ms2 / m as f64);
    }
    let _ = (b1, b2);
    println!("\nVERDICT: v2 stays bandwidth-bound if its ms/token keeps dropping to M=6-8.");
}
