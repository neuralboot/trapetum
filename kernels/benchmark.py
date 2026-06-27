#!/usr/bin/env python3
"""
Reproducible benchmark harness for the trapetum kernels.

Compiles the CUDA sources with a fixed configuration, runs them, captures the
environment (GPU, driver, CUDA toolkit) and the raw kernel output, then writes
results.json + results.md.

Determinism: all randomness in the kernels comes from std::mt19937(0) in the C++
sources (fixed SEED = 0). Given a fixed GPU / driver / toolkit, runs are
reproducible. The raw kernel logs are stored verbatim in results.json and are the
source of truth; the parsed metrics are best-effort convenience only.

Requirements: an NVIDIA GPU, the CUDA toolkit (nvcc) on PATH, Python 3.8+ (stdlib
only, no pip dependencies).

Usage:
    python benchmark.py                 # auto-detect arch
    python benchmark.py --arch sm_90    # H100 / H200
    python benchmark.py --out run.json
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# work from this script's directory so the relative .cu paths resolve from anywhere
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 0  # fixed in the .cu sources via std::mt19937(0); kept here for the record

# (source, extra -D defines, needs -lcublas, human label)
BENCHES = [
    ("bench.cu",  [],        True,  "decode GEMV (uint8) + standalone dequant, vs cuBLAS"),
    ("prof7.cu",  ["GS=20"], False, "decode GEMV, 4-bit packed (K=16)"),
    ("prof10.cu", [],        True,  "prefill GEMM, Tensor Cores (pipelined dequant) vs cuBLAS"),
]

# best-effort metric extraction; raw logs remain authoritative
METRIC_RES = {
    "latency_ms": re.compile(r"([0-9]*\.?[0-9]+)\s*ms"),
    "gbps":       re.compile(r"([0-9]*\.?[0-9]+)\s*GB/s"),
    "tflops":     re.compile(r"([0-9]*\.?[0-9]+)\s*TFLOP/s"),
    "speedup_x":  re.compile(r"x([0-9]*\.?[0-9]+)"),
    "rel_err":    re.compile(r"(?:rel)?err\s*=?\s*([0-9.eE+-]+)"),
}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def detect_env(arch):
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "seed": SEED,
        "arch": arch,
    }
    smi = run(["nvidia-smi",
               "--query-gpu=name,driver_version,memory.total,compute_cap",
               "--format=csv,noheader"])
    env["gpu"] = smi.stdout.strip() if smi.returncode == 0 else "unknown (nvidia-smi failed)"
    nv = run(["nvcc", "--version"])
    env["nvcc"] = nv.stdout.strip().splitlines()[-1] if nv.returncode == 0 else "unknown"
    return env


def detect_arch():
    q = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if q.returncode == 0 and q.stdout.strip():
        cc = q.stdout.strip().splitlines()[0].strip().replace(".", "")
        return "sm_" + cc
    print("warning: could not detect compute capability, defaulting to sm_80")
    return "sm_80"


def compile_and_run(src, defines, cublas, arch):
    out_bin = "./_bench_" + os.path.splitext(src)[0]
    cmd = ["nvcc", "-O3", "-arch=" + arch, src, "-o", out_bin]
    cmd += ["-D" + d for d in defines]
    if cublas:
        cmd.append("-lcublas")
    c = run(cmd)
    if c.returncode != 0:
        return {"compiled": False, "compile_cmd": " ".join(cmd),
                "compile_error": c.stderr[-2000:]}
    try:
        r = run([out_bin], timeout=600)
        rc, out, err = r.returncode, r.stdout, r.stderr[-1000:]
    except subprocess.TimeoutExpired:
        rc, out, err = -1, "", "timeout (600s)"
    finally:
        try:
            os.remove(out_bin)
        except OSError:
            pass
    return {"compiled": True, "compile_cmd": " ".join(cmd),
            "returncode": rc, "stdout": out, "stderr": err}


def extract_metrics(stdout):
    rows = []
    for ln in stdout.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        found = {k: [float(x) for x in rx.findall(ln)] for k, rx in METRIC_RES.items()}
        found = {k: v for k, v in found.items() if v}
        if found:
            rows.append({"line": ln, **found})
    return rows


def write_markdown(report, path):
    env = report["env"]
    out = ["# Benchmark results", "",
           "| field | value |", "|---|---|",
           "| GPU | `%s` |" % env.get("gpu"),
           "| arch | `%s` |" % env.get("arch"),
           "| CUDA | `%s` |" % env.get("nvcc"),
           "| seed | `%s` (fixed in sources) |" % env.get("seed"),
           "| date (UTC) | %s |" % env.get("timestamp_utc"), "",
           "Raw kernel logs below are the source of truth; see `results.json` for the",
           "parsed metrics.", ""]
    for b in report["benchmarks"]:
        out.append("## `%s` - %s" % (b["source"], b["label"]))
        if b.get("status") == "missing":
            out.append("_source file not found_\n")
        elif not b.get("compiled"):
            out += ["_compilation failed:_", "```", (b.get("compile_error") or "")[:1500], "```\n"]
        else:
            out += ["```", (b.get("stdout") or "").strip(), "```\n"]
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description="Reproducible trapetum benchmark harness")
    ap.add_argument("--arch", default=None,
                    help="CUDA arch (sm_80/sm_86/sm_89/sm_90). Auto-detected if omitted.")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if shutil.which("nvcc") is None:
        sys.exit("error: nvcc not found on PATH. Install the CUDA toolkit.")

    arch = args.arch or detect_arch()
    env = detect_env(arch)
    print("GPU : %s" % env["gpu"])
    print("arch: %s | CUDA: %s | seed: %d\n" % (arch, env["nvcc"], SEED))

    benchmarks = []
    for src, defines, cublas, label in BENCHES:
        entry = {"source": src, "label": label, "defines": defines}
        if not os.path.exists(src):
            entry["status"] = "missing"
            print("==> %-12s MISSING" % src)
            benchmarks.append(entry)
            continue
        print("==> %-12s (%s) [arch=%s defines=%s]" % (src, label, arch, defines or "-"))
        entry.update(compile_and_run(src, defines, cublas, arch))
        if entry.get("compiled") and entry.get("returncode") == 0:
            entry["metrics"] = extract_metrics(entry["stdout"])
            print(entry["stdout"].strip(), "\n")
        else:
            print("    FAILED:", (entry.get("compile_error") or entry.get("stderr") or "")[:300], "\n")
        benchmarks.append(entry)

    report = {"env": env, "benchmarks": benchmarks}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    md = os.path.splitext(args.out)[0] + ".md"
    write_markdown(report, md)
    print("Wrote %s and %s" % (args.out, md))


if __name__ == "__main__":
    main()
