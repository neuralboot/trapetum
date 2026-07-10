#!/usr/bin/env python3
"""Stage 1 probe: do the routed-expert FFN weights of a DeepSeek MoE model survive
2-bit ADDITIVE codebook quantization (AQLM-style 2x8) with acceptable perplexity, while
attention / dense-MLP / shared-expert / router / embeddings / lm_head stay untouched?

"2x8" additive VQ = M=2 additive codebooks of K=256 vectors over groups of D=8 weights,
plus a per-output-row scale:
    W[out, group] = scale[out] * sum_m C_m[ code_m[out, group] ]
Each group of D=8 weights costs M indices of log2(K)=8 bits = M*8 bits, so bits/weight
= M*8/D = M. Hence M=2 gives 2 bits/weight. This is exactly the scheme the repo's avq
kernel decodes and the same beam-search + LSQ trainer used for the paper's AQLM-style
Llama results (model/llama_aqlm_train.py), reused here on the DeepSeek experts.

What the script does on a CUDA GPU:
  1. Load deepseek-ai/DeepSeek-V2-Lite (bf16, trust_remote_code) with device_map=auto and
     a GPU memory cap so the 16B model spills to host RAM (no disk offload).
  2. Measure wikitext-2 perplexity of the untouched bf16 model (2048 ctx).
  3. Quantize every routed expert's gate_proj / up_proj / down_proj to 2-bit additive
     (2x8, group 8) expert by expert, freeing as it goes, and write the DEQUANTIZED
     tensor back into the live model (simulation, no custom kernel needed for a PPL probe).
     Attention, dense MLP, shared expert, router, embeddings and lm_head stay as loaded.
  4. Measure wikitext-2 perplexity again, print a 60-token greedy continuation of
     "The capital of France is" for both models, and print a compact byte / size summary.

Invoke (cheap first signal, 4 MoE layers + ~10k PPL tokens):
    python probe_avq2bit_moe.py --fast
Full run (all MoE layers + ~61k PPL tokens):
    python probe_avq2bit_moe.py
Higher-quality quantizer (slower beam width + more LSQ rounds):
    python probe_avq2bit_moe.py --beam 4 --rounds 3

Pure Python / PyTorch (+ transformers, datasets). Targets one RunPod 4090 (24GB) with
32GB+ RAM. No em dashes or en dashes anywhere.
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-V2-Lite"
K = 256           # codebook size (8-bit indices)
D = 8             # group size
SEQLEN = 2048     # PPL context, matches the repo's wikitext-2 protocol

# 671B projection constants (DeepSeek-R1 / V3 full architecture). Routed experts are the
# overwhelming majority of the parameters: 58 MoE layers x 256 experts x 3 matrices of
# (7168 x 2048) = about 653B params, i.e. essentially the whole 671B. The repo's measured
# 4-bit CBKR artifact for R1 is 326 GB (bench/RESULTS_deepseek.md), which matches routed
# experts at 4 bits, confirming experts dominate. We project the 2-bit expert size from
# that same param count so the numbers stay self-consistent with the repo.
R1_ROUTED_EXPERT_PARAMS = 653e9


# ------------------------------------------------------------------------------------
# Additive (residual) VQ trainer: beam-search code assignment + LSQ codebook update.
# Adapted (self-contained) from model/llama_aqlm_train.py, dropped the diagonal-Hessian
# calibration branch (uniform weights = the winning no-calibration path in that script).
# ------------------------------------------------------------------------------------
@torch.no_grad()
def kmeans(P, k=K, iters=5, chunk=200_000):
    N = P.shape[0]
    C = P[torch.randperm(N, device=P.device)[:k]].clone()
    asg = torch.zeros(N, dtype=torch.long, device=P.device)
    for _ in range(iters):
        Cn2 = (C * C).sum(1)
        for s in range(0, N, chunk):
            p = P[s:s + chunk]
            asg[s:s + chunk] = ((p * p).sum(1, keepdim=True) - 2 * (p @ C.t()) + Cn2).argmin(1)
        Cnew = torch.zeros_like(C)
        cnt = torch.zeros(k, device=P.device)
        Cnew.index_add_(0, asg, P)
        cnt.index_add_(0, asg, torch.ones(N, device=P.device))
        C = Cnew / cnt.clamp(min=1).unsqueeze(1)
    return C


@torch.no_grad()
def beam_search(W, C, B, chunk=8000):
    # W (N,d), C (M,K,d). Assign M codes per row minimizing sum over groups of (w - recon)^2.
    # B is the beam width: B=1 is fast sequential greedy residual matching, B>1 searches wider.
    N, d = W.shape
    Mc = C.shape[0]
    codes = torch.empty(N, Mc, dtype=torch.long, device=W.device)
    for s in range(0, N, chunk):
        w = W[s:s + chunk]
        n = w.shape[0]
        d0 = ((w[:, None, :] - C[0][None, :, :]) ** 2).sum(-1)
        _, idx = d0.topk(B, dim=1, largest=False)
        bc = idx.unsqueeze(-1)
        br = C[0][idx]
        for m in range(1, Mc):
            cand = br[:, :, None, :] + C[m][None, None, :, :]
            sc = ((w[:, None, None, :] - cand) ** 2).sum(-1).reshape(n, B * K)
            _, flat = sc.topk(B, dim=1, largest=False)
            bsel = flat // K
            ksel = flat % K
            bc = torch.cat([torch.gather(bc, 1, bsel.unsqueeze(-1).expand(-1, -1, m)),
                            ksel.unsqueeze(-1)], -1)
            br = torch.gather(br, 1, bsel.unsqueeze(-1).expand(-1, -1, d)) + C[m][ksel]
        codes[s:s + chunk] = bc[:, 0, :]
    return codes


@torch.no_grad()
def lsq_update(W, codes, Mc, reg=1e-2):
    # Least-squares codebook refit given fixed codes: one solve for all D dims at once.
    N, d = W.shape
    P = Mc * K
    feat = codes + (torch.arange(Mc, device=W.device) * K)[None, :]
    AtW = torch.zeros(P, d, device=W.device)
    for m in range(Mc):
        AtW.index_add_(0, feat[:, m], W)
    AtA = torch.zeros(P, P, device=W.device)
    ones = torch.ones(N, device=W.device)
    for m in range(Mc):
        for mp in range(Mc):
            AtA.view(-1).index_add_(0, feat[:, m] * P + feat[:, mp], ones)
    AtA += reg * torch.eye(P, device=W.device)
    return torch.linalg.solve(AtA, AtW).reshape(Mc, K, d)


@torch.no_grad()
def aqlm_quantize(W, Mc, beam, rounds):
    # W (OC, IC) float32 on GPU. Returns the DEQUANTIZED (OC, IC) tensor (2-bit additive
    # simulation). Per-row abs-max scale, grouped into D-vectors, M additive codebooks.
    OC, IC = W.shape
    assert IC % D == 0, f"in-features {IC} must be divisible by group size {D}"
    s = W.abs().amax(1, keepdim=True).clamp(min=1e-8)
    P = (W / s).reshape(OC, IC // D, D).reshape(-1, D).contiguous()
    # greedy residual init: kmeans per codebook on the running residual
    R = P.clone()
    cb = []
    for _ in range(Mc):
        cm = kmeans(R, K)
        codes_m = beam_search(R, cm.unsqueeze(0), B=beam)[:, 0]
        R = R - cm[codes_m]
        cb.append(cm)
    C = torch.stack(cb)
    # alternate beam-search assignment and LSQ codebook refit
    for _ in range(rounds):
        codes = beam_search(P, C, B=beam)
        C = lsq_update(P, codes, Mc)
    codes = beam_search(P, C, B=beam)
    recon = sum(C[m][codes[:, m]] for m in range(Mc))
    return (recon.reshape(OC, IC // D, D).reshape(OC, IC)) * s


# ------------------------------------------------------------------------------------
# Model helpers.
# ------------------------------------------------------------------------------------
def moe_layers(model):
    # Every decoder layer whose mlp carries a routed-expert list (dense layers do not).
    out = []
    for li, L in enumerate(model.model.layers):
        mlp = getattr(L, "mlp", None)
        if mlp is not None and hasattr(mlp, "experts"):
            out.append((li, L))
    return out


@torch.no_grad()
def quantize_experts(model, layers, mc, beam, rounds):
    # Quantize each routed expert's gate/up/down to 2-bit additive, expert by expert,
    # writing the dequantized weight back in place. Returns byte accounting.
    n_matrices = 0
    expert_params = 0
    scale_elems = 0
    t0 = time.time()
    for pos, (li, L) in enumerate(layers):
        experts = L.mlp.experts
        for e in experts:
            for lin in (e.gate_proj, e.up_proj, e.down_proj):
                W = lin.weight
                dev, dt = W.device, W.dtype
                Wc = W.detach().to(device="cuda", dtype=torch.float32)
                Wq = aqlm_quantize(Wc, mc, beam, rounds)
                lin.weight.data = Wq.to(device=dev, dtype=dt)
                n_matrices += 1
                expert_params += W.numel()
                scale_elems += W.shape[0]
                del Wc, Wq
        torch.cuda.empty_cache()
        print(f"  quantized MoE layer {li} ({pos + 1}/{len(layers)}, "
              f"{len(experts)} experts) at {time.time() - t0:.0f}s", flush=True)
    # byte accounting for the quantized experts
    bytes_fp16 = expert_params * 2
    idx_bytes = expert_params * mc / D                 # mc indices of 8 bits per D weights
    cb_bytes = n_matrices * mc * K * D * 4             # f32 codebooks, per matrix
    scale_bytes = scale_elems * 4                      # f32 per-row scales
    bytes_2bit = idx_bytes + cb_bytes + scale_bytes
    return {
        "n_matrices": n_matrices,
        "expert_params": expert_params,
        "bytes_fp16": bytes_fp16,
        "bytes_2bit": bytes_2bit,
        "idx_bytes": idx_bytes,
        "cb_bytes": cb_bytes,
        "scale_bytes": scale_bytes,
    }


@torch.no_grad()
def ppl(model, enc, n_seq):
    lf = nn.CrossEntropyLoss(reduction="sum")
    tot = 0.0
    nt = 0
    nseq = min(n_seq, enc.size(1) // SEQLEN)
    for i in range(nseq):
        ids = enc[:, i * SEQLEN:(i + 1) * SEQLEN].to("cuda")
        o = model(ids).logits
        tot += lf(o[:, :-1, :].reshape(-1, o.size(-1)).float(), ids[:, 1:].reshape(-1)).item()
        nt += ids[:, 1:].numel()
        del o
    return float(torch.tensor(tot / nt).exp()), nt


@torch.no_grad()
def greedy_continue(model, tok, prompt, n_new=60):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    try:
        out = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
        return tok.decode(out[0], skip_special_tokens=True)
    except Exception as ex:
        # fallback: manual greedy without cache (slower, but robust to generate() quirks)
        print("  generate() fell back to manual greedy:", str(ex)[:120], flush=True)
        cur = ids
        for _ in range(n_new):
            logits = model(cur).logits[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
            if nxt.item() == tok.eos_token_id:
                break
        return tok.decode(cur[0], skip_special_tokens=True)


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fast", action="store_true",
                    help="4 MoE layers + ~10k PPL tokens for a cheap first signal")
    ap.add_argument("--skip-first", type=int, default=0,
                    help="leave the first N MoE layers unquantized (dynamic-mix probe)")
    ap.add_argument("--layers", type=int, default=0,
                    help="cap the number of MoE layers to quantize (0 = all)")
    ap.add_argument("--ppl-seqs", type=int, default=30,
                    help="number of 2048-token sequences for PPL (30 = ~61k tokens)")
    ap.add_argument("--beam", type=int, default=1,
                    help="beam width for code assignment (1 = fast greedy, 4 = paper quality)")
    ap.add_argument("--rounds", type=int, default=1,
                    help="LSQ codebook refit rounds after init")
    ap.add_argument("--mc", type=int, default=2, choices=[2, 3],
                    help="additive codebooks M: 2 = 2-bit/weight, 3 = 3-bit/weight")
    ap.add_argument("--gpu-mem", default="20GiB",
                    help="per-GPU memory cap for device_map=auto (lower this if it OOMs)")
    ap.add_argument("--cpu-mem", default="60GiB", help="host RAM cap for CPU offload")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--cpu-load", action="store_true",
                    help="load ALL weights as real CPU tensors and quantize BEFORE dispatching "
                         "to the GPU; avoids accelerate putting cpu-offloaded weights on the "
                         "meta device (which breaks in-place quantization)")
    ap.add_argument("--skip-bf16", action="store_true",
                    help="skip the bf16 baseline pass (use with --bf16-ppl to reuse a prior run)")
    ap.add_argument("--bf16-ppl", type=float, default=0.0,
                    help="externally measured bf16 baseline PPL, used for the ratio with --skip-bf16")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this probe needs a CUDA GPU"
    n_ppl_seqs = 5 if args.fast else args.ppl_seqs
    cap_layers = 4 if args.fast else args.layers

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.cpu_load:
        print(f"loading {args.model} (bf16, ALL real tensors on CPU)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
    else:
        print(f"loading {args.model} (bf16, device_map=auto, gpu<={args.gpu_mem}, "
              f"cpu<={args.cpu_mem})", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={0: args.gpu_mem, "cpu": args.cpu_mem},
        ).eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids

    layers = moe_layers(model)
    if args.skip_first:
        # "dynamic" mix: leave the first N MoE layers at full precision (they are
        # the most quantization-sensitive; Unsloth's 1.58-bit R1 uses the same idea)
        layers = layers[args.skip_first:]
    if cap_layers:
        layers = layers[:cap_layers]
    total_moe = len(moe_layers(model))
    print(f"MoE layers: {total_moe} total, quantizing {len(layers)} "
          f"(beam={args.beam}, rounds={args.rounds}, config 2x8 = 2 bits/weight)", flush=True)

    if args.skip_bf16:
        ppl_before, nt = (args.bf16_ppl if args.bf16_ppl > 0 else float("nan")), 0
        gen_before = "(baseline skipped, PPL supplied externally)"
        print(f"bf16 baseline SKIPPED, using supplied PPL = {ppl_before}", flush=True)
    else:
        ppl_before, nt = ppl(model, enc, n_ppl_seqs)
        print(f"bf16 wikitext-2 PPL = {ppl_before:.4f}  ({nt} tokens)", flush=True)
        gen_before = greedy_continue(model, tok, args.prompt)
        print(f"bf16 continuation: {gen_before!r}", flush=True)

    print(f"quantizing routed experts to {args.mc}-bit additive ({args.mc}x8, group 8)...", flush=True)
    acct = quantize_experts(model, layers, mc=args.mc, beam=args.beam, rounds=args.rounds)

    if args.cpu_load:
        # everything is a real tensor now; hand placement over to accelerate for the eval
        from accelerate import dispatch_model, infer_auto_device_map
        dmap = infer_auto_device_map(model, max_memory={0: args.gpu_mem, "cpu": args.cpu_mem})
        model = dispatch_model(model, device_map=dmap)
        print("dispatched quantized model for evaluation", flush=True)

    ppl_after, _ = ppl(model, enc, n_ppl_seqs)
    print(f"experts-{args.mc}bit wikitext-2 PPL = {ppl_after:.4f}  ({nt} tokens)", flush=True)
    gen_after = greedy_continue(model, tok, args.prompt)
    print(f"experts-{args.mc}bit continuation: {gen_after!r}", flush=True)

    ratio = ppl_after / ppl_before if ppl_before > 0 else float("nan")
    frac_671 = R1_ROUTED_EXPERT_PARAMS   # routed-expert param count
    proj_fp16 = frac_671 * 2             # 16 bits/weight -> 2 bytes
    proj_4bit = frac_671 * 4 / 8         # 4 bits/weight  -> 0.5 bytes
    proj_2bit = frac_671 * 2 / 8         # 2 bits/weight  -> 0.25 bytes
    proj_3bit = frac_671 * 3 / 8         # 3 bits/weight  -> 0.375 bytes
    proj_mc = frac_671 * args.mc / 8     # this run's bits/weight

    print("\n================= SUMMARY =================", flush=True)
    print(f"model                : {args.model}", flush=True)
    print(f"MoE layers quantized : {len(layers)} / {total_moe}", flush=True)
    print(f"expert matrices      : {acct['n_matrices']} "
          f"(3 per routed expert)", flush=True)
    print(f"expert params        : {acct['expert_params'] / 1e9:.3f} B", flush=True)
    print(f"PPL bf16             : {ppl_before:.4f}", flush=True)
    print(f"PPL experts-{args.mc}bit     : {ppl_after:.4f}", flush=True)
    print(f"PPL ratio (after/bf16): {ratio:.4f}", flush=True)
    print(f"expert bytes fp16    : {human(acct['bytes_fp16'])}", flush=True)
    print(f"expert bytes {args.mc}-bit   : {human(acct['bytes_2bit'])}  "
          f"(indices {human(acct['idx_bytes'])} + codebooks {human(acct['cb_bytes'])} "
          f"+ scales {human(acct['scale_bytes'])})", flush=True)
    print(f"expert compression   : {acct['bytes_fp16'] / acct['bytes_2bit']:.2f}x "
          f"vs fp16", flush=True)
    print("--- projected 671B artifact (routed experts ~653B params) ---", flush=True)
    print(f"  experts @ fp16     : {human(proj_fp16)}", flush=True)
    print(f"  experts @ 4-bit    : {human(proj_4bit)}   "
          f"(repo CBKR artifact = 326 GB, matches)", flush=True)
    print(f"  experts @ 3-bit    : {human(proj_3bit)}", flush=True)
    print(f"  experts @ 2-bit    : {human(proj_2bit)}", flush=True)
    print("==========================================", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
