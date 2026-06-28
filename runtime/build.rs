// Compiles the CUDA wrapper with nvcc into a static lib and links the CUDA runtime.
// Cross-platform: GNU/ar on Linux, MSVC/lib.exe on Windows.
// Override the GPU arch with CUDA_ARCH (default sm_80); CUDA_PATH for a non-standard
// toolkit location.
use std::env;
use std::process::Command;

fn main() {
    let out = env::var("OUT_DIR").unwrap();
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
    let windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");

    // SKIP_CUDA lets you `cargo check` the Rust on a machine without nvcc (no linking).
    if env::var("SKIP_CUDA").is_ok() {
        println!("cargo:warning=SKIP_CUDA set: not compiling the CUDA kernel");
        println!("cargo:rerun-if-changed=cuda/codebook_gemv.cu");
        return;
    }

    let cuda = env::var("CUDA_PATH").unwrap_or_else(|_| {
        if windows {
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6".to_string()
        } else {
            "/usr/local/cuda".to_string()
        }
    });

    // 1. compile the .cu with nvcc
    let obj = format!("{out}/codebook_gemv.{}", if windows { "obj" } else { "o" });
    let mut nvcc = Command::new("nvcc");
    nvcc.arg("-O3").arg(format!("-arch={arch}"));
    if !windows {
        nvcc.args(["--compiler-options", "-fPIC"]);
    }
    nvcc.args(["-c", "cuda/codebook_gemv.cu", "-o", &obj]);
    let ok = nvcc
        .status()
        .expect("failed to spawn nvcc (is the CUDA toolkit on PATH?)")
        .success();
    assert!(ok, "nvcc failed to compile cuda/codebook_gemv.cu");

    // 2. archive into a static lib + link
    if windows {
        let lib = format!("{out}/codebook_gemv.lib");
        let _ = std::fs::remove_file(&lib);
        let ok = Command::new("lib.exe")
            .args([format!("/OUT:{lib}"), obj.clone()])
            .status()
            .expect("failed to spawn lib.exe (is the MSVC librarian on PATH? run from a VS dev shell)")
            .success();
        assert!(ok, "lib.exe failed to archive the object");
        println!("cargo:rustc-link-search=native={out}");
        println!("cargo:rustc-link-lib=static=codebook_gemv");
        println!("cargo:rustc-link-search=native={cuda}/lib/x64");
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
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
    }
    println!("cargo:rerun-if-changed=cuda/codebook_gemv.cu");
    println!("cargo:rerun-if-changed=build.rs");
}
