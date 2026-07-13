// Compiles the CUDA wrapper with nvcc into a static lib and links the CUDA runtime.
// Cross-platform: GNU/ar on Linux, MSVC/lib.exe on Windows.
// Override the GPU arch with CUDA_ARCH (default sm_80); CUDA_PATH for a non-standard
// toolkit location.
use std::env;
use std::process::Command;

fn main() {
    let out = env::var("OUT_DIR").unwrap();
    let windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");

    // Backend features: `cuda` compiles/links the nvcc kernel; `metal` compiles
    // metal/kernels.metal into a metallib that backend_metal.rs embeds.
    if env::var("CARGO_FEATURE_METAL").is_ok() && env::var("CARGO_FEATURE_CUDA").is_err() {
        let macos = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
            || env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ios");
        assert!(macos, "the `metal` feature targets Apple platforms");
        let air = format!("{out}/kernels.air");
        let lib = format!("{out}/kernels.metallib");
        // Compile the shaders for the SDK matching the Rust target: iphoneos for
        // a device build, iphonesimulator for the sim, macosx otherwise. A Metal
        // library is target-family specific, so this must track the target OS.
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let sim = env::var("CARGO_CFG_TARGET_ABI").as_deref() == Ok("sim")
            || env::var("TARGET").unwrap_or_default().contains("ios-sim");
        let sdk = match target_os.as_str() {
            "ios" if sim => "iphonesimulator",
            "ios" => "iphoneos",
            _ => "macosx",
        };
        let ok = Command::new("xcrun")
            .args(["-sdk", sdk, "metal", "-std=metal3.0", "-O2",
                   "-c", "metal/kernels.metal", "-o", &air])
            .status().expect("failed to spawn xcrun metal (Xcode command line tools?)")
            .success();
        assert!(ok, "metal shader compilation failed");
        let ok = Command::new("xcrun")
            .args(["-sdk", sdk, "metallib", &air, "-o", &lib])
            .status().expect("failed to spawn xcrun metallib")
            .success();
        assert!(ok, "metallib link failed");
        println!("cargo:warning=trapetum: built Metal shaders for SDK {sdk}");
        println!("cargo:rerun-if-changed=metal/kernels.metal");
        println!("cargo:rerun-if-changed=build.rs");
        return;
    }
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        println!("cargo:warning=no GPU backend feature enabled");
        println!("cargo:rerun-if-changed=cuda/codebook_gemv.cu");
        return;
    }

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

    // 1. compile the .cu with nvcc.
    // GPU code generation: if CUDA_ARCH is set explicitly, honor it as a single
    // target (dev/bench boxes that want one arch). Otherwise emit a portable
    // fatbin with real SASS for Turing..Hopper plus compute_90 PTX, so a shipped
    // release runs natively on those GPUs and JIT-compiles on anything newer.
    // The kernel is plain CUDA C (no arch-specific intrinsics), so every arch
    // below compiles unchanged.
    let obj = format!("{out}/codebook_gemv.{}", if windows { "obj" } else { "o" });
    let mut nvcc = Command::new("nvcc");
    nvcc.arg("-O3");
    match env::var("CUDA_ARCH") {
        Ok(a) => {
            nvcc.arg(format!("-arch={a}"));
        }
        Err(_) => {
            for a in ["70", "75", "80", "86", "89", "90"] {
                nvcc.args(["-gencode", &format!("arch=compute_{a},code=sm_{a}")]);
            }
            // PTX for forward compatibility (JIT on Blackwell and later).
            nvcc.args(["-gencode", "arch=compute_90,code=compute_90"]);
        }
    }
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
