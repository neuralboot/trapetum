#!/usr/bin/env python3
"""
Speed-vs-accuracy benchmark of LLM weight-quantization methods, measured on one
GPU under a single fair harness.

For each (method, model) it measures:
  - decode throughput  : greedy generation, batch=1, warmup + median over repeats
  - prefill latency    : single forward over a long prompt
  - peak VRAM
  - wikitext-2 perplexity (GPTQ-style, the de-facto standard most papers report)

Methodology choices (stated so the result is not debunkable):
  * Every method is loaded through the SAME HuggingFace forward path and timed by
    the SAME loop, so we measure the quantization KERNEL's cost, not a serving
    engine. A vLLM/TensorRT-LLM serving comparison is a SEPARATE study.
  * Greedy decoding (do_sample=False), fixed seq lengths, fixed seed, warmup, and
    median-of-N. Versions are captured in the output.
  * Each method uses its own intended kernel (AQLM/AWQ/GPTQ-Marlin/QTIP); none is
    crippled. The installed library + version is recorded so the kernel is known.

Honest caveats:
  * Kernels differ in maturity; the HF forward path is not the fastest possible
    deployment for any of them, but it is the same for all -> fair relative numbers.
  * QTIP is not integrated in transformers; it is loaded best-effort via its own
    package and skipped (with a logged reason) if unavailable.
  * Exact HuggingFace checkpoint ids must be verified at runtime; configs marked
    verify=True are placeholders to confirm against the method's own repo.

Usage:
  huggingface-cli login            # gated Llama checkpoints
  python bench_quant.py --only fp16,aqlm,gptq,awq --out results.json
"""
import argparse
import gc
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn

SEED = 0
torch.manual_seed(SEED)

# ----------------------------------------------------------------------------
# Edit these. Start with one model family that has checkpoints across all methods
# (Llama-2-7B does). loader: "hf" works for fp16/AQLM/GPTQ/AWQ when the matching
# library is installed (transformers auto-detects the quantization_config).
# ----------------------------------------------------------------------------
CONFIGS = [
    dict(name="fp16",        method="fp16",  bits=16.0,
         model="NousResearch/Llama-2-7b-hf",                     loader="hf"),  # ungated mirror
    dict(name="aqlm-2bit",   method="AQLM",  bits=2.0,
         model="ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf",       loader="hf"),
    dict(name="gptq-4bit",   method="GPTQ",  bits=4.0,
         model="TheBloke/Llama-2-7B-GPTQ",                       loader="gptqmodel"),
    dict(name="awq-4bit",    method="AWQ",   bits=4.0,
         model="TheBloke/Llama-2-7B-AWQ",                        loader="hf"),
    # QTIP: verify the exact id against https://github.com/Cornell-RelaxML/qtip
    dict(name="qtip-2bit",   method="QTIP",  bits=2.0,
         model="relaxml/Llama-2-7b-QTIP-2Bit",  loader="qtip",   verify=True),

    # 70B: fp16 needs ~140 GB (2+ GPUs); quantized fits on ONE 80 GB H100.
    dict(name="awq-4bit-70b",  method="AWQ",  bits=4.0,
         model="TheBloke/Llama-2-70B-AWQ",                       loader="hf"),
    dict(name="aqlm-2bit-70b", method="AQLM", bits=2.0,
         model="ISTA-DASLab/Llama-2-70b-AQLM-2Bit-1x16-hf",      loader="hf", verify=True),
]

PROMPT = ("The history of computing is a long and winding road that begins with "
          "mechanical calculators and leads, eventually, to the large language "
          "models of today. In this essay we will trace ")


class SkipConfig(Exception):
    pass


def env_info():
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": SEED,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    for pkg in ("transformers", "aqlm", "auto_gptq", "gptqmodel", "awq", "autoawq", "datasets"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            pass
    return info


def load_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if cfg["loader"] == "qtip":
        # QTIP is not on PyPI: clone https://github.com/Cornell-RelaxML/qtip to
        # /root/qtip and `pip install -e .` first. Best-effort use of their loader.
        import os
        sys.path.insert(0, os.environ.get("QTIP_DIR", "/root/qtip"))
        try:
            from lib.utils.unsafe_import import model_from_hf_path
        except Exception as e:
            raise SkipConfig("QTIP repo not importable (clone + pip install -e it): %r" % e)
        model, _ = model_from_hf_path(cfg["model"], use_cuda_graph=False)
        model.eval()
        return model, tok
    if cfg["loader"] == "gptqmodel":
        # Load GPTQ directly via gptqmodel (its own kernel, Marlin on Hopper when
        # the checkpoint allows it), bypassing the transformers-GPTQ backend version
        # dance. gm.model is a normal HF model -> works with the rest of the harness.
        from gptqmodel import GPTQModel
        gm = GPTQModel.load(cfg["model"], device="cuda:0")
        model = getattr(gm, "model", gm)
        model.eval()
        return model, tok
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


@torch.no_grad()
def measure_decode(model, tok, n_new=256, warmup=2, repeats=3):
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
    gen = dict(max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
               use_cache=True, pad_token_id=tok.eos_token_id)
    for _ in range(warmup):
        model.generate(ids, **dict(gen, max_new_tokens=16, min_new_tokens=16))
    torch.cuda.synchronize()
    secs = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.generate(ids, **gen)
        torch.cuda.synchronize(); secs.append(time.perf_counter() - t0)
    med = statistics.median(secs)
    return {"decode_tok_per_s": n_new / med, "decode_ms_per_tok": med / n_new * 1e3,
            "decode_secs_runs": secs}


@torch.no_grad()
def measure_prefill(model, tok, seqlen=2048, repeats=3):
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
    ids = ids[:, :1].repeat(1, seqlen)  # synthetic long prompt of fixed length
    for _ in range(2):
        model(ids)
    torch.cuda.synchronize()
    secs = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model(ids)
        torch.cuda.synchronize(); secs.append(time.perf_counter() - t0)
    return {"prefill_ms": statistics.median(secs) * 1e3, "prefill_seqlen": seqlen}


@torch.no_grad()
def eval_ppl_wikitext2(model, tok, seqlen=2048, max_samples=None):
    from datasets import load_dataset
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    n = enc.numel() // seqlen
    if max_samples:
        n = min(n, max_samples)
    loss = nn.CrossEntropyLoss(reduction="sum")
    nlls = []
    for i in range(n):
        batch = enc[:, i * seqlen:(i + 1) * seqlen].to(model.device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch[:, 1:]
        nll = loss(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        nlls.append(nll)
    ppl = torch.exp(torch.stack(nlls).sum() / (n * seqlen))
    return {"wikitext2_ppl": ppl.item(), "ppl_seqlen": seqlen, "ppl_samples": n}


def run_one(cfg, args):
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model, tok = load_model(cfg)
    load_s = time.perf_counter() - t_load
    out = {"name": cfg["name"], "method": cfg["method"], "bits": cfg["bits"],
           "model": cfg["model"], "load_s": round(load_s, 1)}
    out.update(measure_prefill(model, tok))
    out.update(measure_decode(model, tok, n_new=args.decode_tokens))
    out.update(eval_ppl_wikitext2(model, tok, seqlen=args.ppl_seqlen,
                                  max_samples=args.ppl_samples))
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of config names to run")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--ppl-samples", type=int, default=None, help="cap PPL windows (speed)")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required.")
    only = set(args.only.split(",")) if args.only else None
    env = env_info()
    print(json.dumps(env, indent=2), flush=True)

    results = []
    for cfg in CONFIGS:
        if only and cfg["name"] not in only:
            continue
        print("\n==> %s  (%s, %s)" % (cfg["name"], cfg["method"], cfg["model"]), flush=True)
        try:
            r = run_one(cfg, args)
            print("    decode %.1f tok/s | prefill %.1f ms | PPL %.3f | VRAM %.2f GB"
                  % (r["decode_tok_per_s"], r["prefill_ms"], r["wikitext2_ppl"],
                     r["peak_vram_gb"]), flush=True)
            results.append(r)
        except SkipConfig as e:
            print("    SKIP:", e, flush=True)
            results.append({"name": cfg["name"], "method": cfg["method"], "skipped": str(e)})
        except Exception as e:
            print("    FAILED:", repr(e)[:300], flush=True)
            results.append({"name": cfg["name"], "method": cfg["method"], "error": repr(e)[:500]})
        gc.collect(); torch.cuda.empty_cache()

    json.dump({"env": env, "results": results}, open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
