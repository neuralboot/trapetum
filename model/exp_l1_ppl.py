# exp_l1_ppl.py -- end-to-end PPL confirmation of Lever 1 (Hadamard rotation) on the FULL model.
#
# Quantizes EVERY decoder Linear (q,k,v,o,gate,up,down) at K=16 (4-bit) and measures
# real wikitext-2 perplexity for three configs:
#   fp32   : unquantized reference
#   L0     : per-column k-means (current method)
#   L1     : randomized Hadamard on the input dim, k-means in the rotated basis, folded back.
#
# Folding  W <- Q^T @ quant(Q @ Wt)  into the weight and running a normal forward is
# ALGEBRAICALLY IDENTICAL to the QuaRot deployment (online Hadamard on activations + weight
# stored in the rotated basis), because  y = x Wt = (x Q^T)(Q Wt).  So this PPL is the true
# deployment PPL, not a proxy.
#
# Fast 1-D k-means (searchsorted assignment + scatter_add centroid update) makes full-model
# CPU quantization tractable.  Run:  /tmp/lev_venv/bin/python model/exp_l1_ppl.py

import os, json, math, time
import torch, torch.nn as nn
torch.manual_seed(0)
DEV = "cpu"
MODEL = os.environ.get("LEV_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
K = 16
ITERS = int(os.environ.get("ITERS", 10))
SEQLEN = int(os.environ.get("SEQLEN", 2048))
PPL_SAMPLES = int(os.environ.get("PPL_SAMPLES", 20))

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ---------------- Hadamard (randomized, block for non-pow2) ----------------
def hadamard_pow2(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H
def rand_hadamard(n, gen):
    p2 = 1
    while (p2 * 2) <= n and (n % (p2 * 2) == 0): p2 *= 2
    Hb = hadamard_pow2(p2) / math.sqrt(p2)
    odd = n // p2
    Q = torch.zeros(n, n)
    for b in range(odd): Q[b*p2:(b+1)*p2, b*p2:(b+1)*p2] = Hb
    d = (torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1)
    return Q * d.unsqueeze(0)

# ---------------- fast 1-D per-column k-means ----------------
def kmeans_cols_fast(Wt, k=K, iters=ITERS):
    # Wt: (IC, OC). Cluster each output column over its IC values (1-D), plain L2.
    IC, OC = Wt.shape
    lo = Wt.min(0).values; hi = (Wt.max(0).values - lo).clamp(min=1e-9)
    ar = torch.arange(k).float().unsqueeze(1)                     # (k,1)
    cent = lo.unsqueeze(0) + hi.unsqueeze(0) * ar / (k - 1)       # (k, OC) ascending per col
    WtT = Wt.t().contiguous()                                    # (OC, IC)
    for _ in range(iters):
        bnd = ((cent[:-1] + cent[1:]) / 2).t().contiguous()      # (OC, k-1) ascending
        idx = torch.searchsorted(bnd, WtT).t().contiguous()      # (IC, OC) in 0..k-1
        sums = torch.zeros(k, OC); cnts = torch.zeros(k, OC)
        sums.scatter_add_(0, idx, Wt)
        cnts.scatter_add_(0, idx, torch.ones_like(Wt))
        new = sums / cnts.clamp(min=1e-9)
        empty = cnts < 0.5
        cent = torch.where(empty, cent, new)
        cent, _ = cent.sort(0)                                   # keep monotone for searchsorted
    bnd = ((cent[:-1] + cent[1:]) / 2).t().contiguous()
    idx = torch.searchsorted(bnd, WtT).t().contiguous()
    return torch.gather(cent, 0, idx)                            # (IC, OC) dequantized

# ---------------- load model + wikitext ----------------
from transformers import AutoModelForCausalLM, AutoTokenizer
log("loading", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()

def get_text():
    try:
        from huggingface_hub import hf_hub_download; import pandas as pd
        p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
        return "\n\n".join(t for t in pd.read_parquet(p)["text"].tolist() if t.strip())
    except Exception as e:
        log("wikitext dl failed:", repr(e)[:100]); return " ".join(["The study of science spans many centuries of inquiry."]*4000)
ids = tok(get_text(), return_tensors="pt").input_ids                 # (1, T)
log("tokens", ids.shape[1])

def ppl():
    nll = 0.0; cnt = 0
    with torch.no_grad():
        for i in range(PPL_SAMPLES):
            b = ids[:, i*SEQLEN:(i+1)*SEQLEN]
            if b.shape[1] < 2: break
            loss = model(b, labels=b).loss.item()
            nll += loss * (b.shape[1]-1); cnt += (b.shape[1]-1)
    return math.exp(nll / cnt)

# ---------------- targets + originals ----------------
targets = [(n, m) for n, m in model.named_modules()
           if isinstance(m, nn.Linear) and n.startswith("model.layers")]
log("decoder Linear layers:", len(targets))
orig = {n: m.weight.data.clone() for n, m in targets}

def restore():
    for n, m in targets: m.weight.data.copy_(orig[n])

def quantize(mode):
    gen = torch.Generator().manual_seed(0)
    for i, (n, m) in enumerate(targets):
        Wt = orig[n].t().float().contiguous()                   # (IC, OC)
        if mode == "L0":
            deq = kmeans_cols_fast(Wt, K)
        else:  # L1 rotation, folded back
            Q = rand_hadamard(Wt.shape[0], gen)
            deq = Q.t() @ kmeans_cols_fast(Q @ Wt, K)
        m.weight.data.copy_(deq.t().to(m.weight.dtype))
        if (i+1) % 40 == 0: log(f"  {mode} quantized {i+1}/{len(targets)}")

# ---------------- run ----------------
log("PPL fp32 ...");           p_fp = ppl(); log("fp32 PPL =", round(p_fp, 4))
log("quantize L0 ...");        quantize("L0"); p_l0 = ppl(); log("L0 PPL =", round(p_l0, 4)); restore()
log("quantize L1 rotation ..."); quantize("L1"); p_l1 = ppl(); log("L1 PPL =", round(p_l1, 4)); restore()

res = dict(model=MODEL, K=K, bits=4, ppl_samples=PPL_SAMPLES, seqlen=SEQLEN,
           ppl_fp32=p_fp, ppl_L0=p_l0, ppl_L1=p_l1,
           delta_L0=p_l0 - p_fp, delta_L1=p_l1 - p_fp,
           L1_vs_L0_ppl_gap_reduction=(p_l0 - p_l1) / max(p_l0 - p_fp, 1e-9))
open(os.path.join(os.path.dirname(__file__), "exp_l1_ppl_results.json"), "w").write(json.dumps(res, indent=2))
log("RESULT"); print(json.dumps(res, indent=2))
