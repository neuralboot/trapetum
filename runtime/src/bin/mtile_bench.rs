//! M0 go/no-go: does the small-M fused 4-bit decode GEMM stay bandwidth-bound?
//! If verifying M=6 columns costs about the same wall-clock as M=1, then a
//! speculative-decoding verification of K+1 draft tokens is ~free (one weight
//! read serves all M), and the projected 1.3-2.5x speedup is unlocked.
//!   cargo run --release --no-default-features --features metal --bin mtile_bench
fn main() {
    let (ic, oc) = (4096usize, 4096usize); // a Llama-2 7B attention/proj shape
    let iters = 200;
    println!("Trapetum M0: small-M fused decode GEMM, {ic}x{oc}, {iters} iters\n");
    let base = trapetum::bench_mtile(ic, oc, 1, iters);
    println!("{:>3}  {:>10}  {:>9}  {:>14}", "M", "ms", "vs M=1", "ms/token");
    for m in [1usize, 2, 3, 4, 5, 6, 8] {
        let ms = trapetum::bench_mtile(ic, oc, m, iters);
        println!("{:>3}  {:>10.4}  {:>8.2}x  {:>13.4}", m, ms, ms / base, ms / m as f64);
    }
    println!("\nVERDICT: bandwidth-bound if ms(M=6) stays close to ms(M=1)");
    println!("(ms/token should DROP sharply as M grows: that is the free verification).");
}
