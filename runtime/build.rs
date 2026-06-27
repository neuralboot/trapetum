// Compiles the CUDA wrapper with nvcc into a static lib and links the CUDA runtime.
// Override the GPU arch with CUDA_ARCH (default sm_80); CUDA_PATH for a non-standard
// toolkit location.
use std::env;
use std::process::Command;

fn main() {
    let out = env::var("OUT_DIR").unwrap();
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
    let cuda = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let obj = format!("{out}/codebook_gemv.o");
    let ok = Command::new("nvcc")
        .args([
            "-O3",
            &format!("-arch={arch}"),
            "--compiler-options",
            "-fPIC",
            "-c",
            "cuda/codebook_gemv.cu",
            "-o",
            &obj,
        ])
        .status()
        .expect("failed to spawn nvcc (is the CUDA toolkit on PATH?)")
        .success();
    assert!(ok, "nvcc failed to compile cuda/codebook_gemv.cu");

    let lib = format!("{out}/libcodebook_gemv.a");
    let _ = std::fs::remove_file(&lib);
    let ok = Command::new("ar")
        .args(["crus", &lib, &obj])
        .status()
        .expect("failed to spawn ar")
        .success();
    assert!(ok, "ar failed to archive the object");

    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=codebook_gemv");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=cuda/codebook_gemv.cu");
    println!("cargo:rerun-if-changed=build.rs");
}
