# exp_levers.py  --  local pre-screen for compressor Levers 1, 2, 3 on a REAL model.
#
# Model: TinyLlama-1.1B (Llama architecture, same nn.Linear layout as the 671B target).
# We quantize REAL weight matrices and score each lever by the metric that actually
# drives perplexity: relative OUTPUT error on REAL calibration activations, at CONSTANT bits.
#
#     err(method) = || X W^T  -  X Wq^T ||_F  /  || X W^T ||_F
#
# X = real activations captured by forward hooks on wikitext calibration text.
# Because the metric uses real X and real W, it is exact for every method: it does not
# depend on how the codebook was fit, only on the final dequantized weight.
#
# Levers:
#   L0  baseline    : per-output-column k-means, K=16 (4-bit), linear init, plain L2  (current method)
#   L1  rotation    : randomized (block-)Hadamard on the input dim before k-means, folded back (QuaRot-style)
#   L2  weighted    : activation-weighted (diagonal-Hessian) per-column k-means  (AWQ/SqueezeLLM-style)
#   L1+L2           : both (h recomputed in the rotated basis, exactly, from the same X)
#   L3  avq-2bit    : additive VQ M=2 (2-bit) -- greedy init  vs  OA-EM init (output-aware EM, 2604.08118)
#
# All of L0/L1/L2 use the SAME 4 bits/weight and the SAME per-column codebook budget, so the
# comparison is bit-for-bit fair. Rotation adds zero inference cost (Hadamard fused offline).
#
# Run:  /tmp/lev_venv/bin/python model/exp_levers.py   (writes model/exp_levers_results.json)

import os, json, math, time, sys
import torch

torch.manual_seed(0)
DEV = "cpu"
MODEL = os.environ.get("LEV_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
K = 16                    # scalar codebook size -> 4-bit
CALIB_SEQS = int(os.environ.get("CALIB_SEQS", 6))
SEQLEN = int(os.environ.get("SEQLEN", 512))
LAYER_IDX = [int(x) for x in os.environ.get("LAYERS", "0,5,11,16,21").split(",")]
PROJS = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"]
RUN_L3 = os.environ.get("RUN_L3", "1") == "1"

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ---------------------------------------------------------------- Hadamard
def hadamard_pow2(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H

def rand_hadamard(n, gen):
    # randomized (block-)Hadamard, orthonormal, size n. n = p2 * odd -> block-diag of p2 Hadamards.
    p2 = 1
    while (p2 * 2) <= n and (n % (p2 * 2) == 0):
        p2 *= 2
    Hb = hadamard_pow2(p2) / math.sqrt(p2)           # orthonormal p2 x p2
    odd = n // p2
    # block diagonal: odd blocks of Hb  (kron(I_odd, Hb))
    Q = torch.zeros(n, n)
    for b in range(odd):
        Q[b*p2:(b+1)*p2, b*p2:(b+1)*p2] = Hb
    d = (torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1)   # random sign flip
    Q = Q * d.unsqueeze(0)
    return Q                                          # orthonormal n x n

# ---------------------------------------------------------------- k-means (per output column)
def kmeans_cols(Wt, k=K, iters=12, w=None):
    # Wt: (IC, OC). Cluster each output column over its IC values (1-D). Optional per-row weight w (IC,).
    IC, OC = Wt.shape
    lo = Wt.min(0).values; hi = Wt.max(0).values
    js = torch.arange(k).float().unsqueeze(1)                     # (k,1)
    cent = lo.unsqueeze(0) + (hi - lo).unsqueeze(0) * js / (k - 1)  # (k, OC) linear init
    wv = None if w is None else w.unsqueeze(1)                    # (IC,1)
    for _ in range(iters):
        d = (Wt.unsqueeze(2) - cent.t().unsqueeze(0)) ** 2        # (IC, OC, k)
        idx = d.argmin(2)                                         # (IC, OC)
        for c in range(k):
            m = (idx == c)                                        # (IC, OC)
            if wv is None:
                cnt = m.sum(0).clamp(min=1)
                cent[c] = (Wt * m).sum(0) / cnt
            else:
                num = (Wt * m * wv).sum(0)
                den = (m * wv).sum(0).clamp(min=1e-9)
                cent[c] = num / den
    d = (Wt.unsqueeze(2) - cent.t().unsqueeze(0)) ** 2
    idx = d.argmin(2)
    deq = torch.gather(cent.t(), 1, idx.t()).t()                 # (IC, OC) dequantized
    return deq

def rel_out_err(X, W, Wq):
    # X:(N,IC)  W,Wq:(OC,IC)   output = X @ W.t()
    num = (X @ (W - Wq).t()).norm()
    den = (X @ W.t()).norm().clamp(min=1e-12)
    return (num / den).item()

# ---------------------------------------------------------------- additive VQ (2-bit) for L3
FIT_CAP = int(os.environ.get("FIT_CAP", 40000))
def _assign(P, C, chunk=200000):
    out = torch.empty(P.shape[0], dtype=torch.long)
    for i in range(0, P.shape[0], chunk):
        out[i:i+chunk] = torch.cdist(P[i:i+chunk], C).argmin(1)
    return out
def kmeans_vec(P, k=256, iters=4, w=None, gen=None):
    # P:(N,D) vectors. optional row weight w:(N,). returns centroids (k,D). CPU-friendly:
    # fit centroids on a random subsample (<= FIT_CAP), then assign all vectors in chunks.
    N, D = P.shape
    fit_idx = torch.randperm(N, generator=gen)[:min(N, FIT_CAP)]
    Pf = P[fit_idx]; wf = None if w is None else w[fit_idx]
    C = Pf[torch.randperm(Pf.shape[0], generator=gen)[:k]].clone()
    for _ in range(iters):
        a = _assign(Pf, C)
        for c in range(k):
            m = (a == c)
            if m.any():
                C[c] = (Pf[m] * (wf[m].unsqueeze(1) if wf is not None else 1)).sum(0) / \
                       ((wf[m].sum() if wf is not None else m.sum()).clamp(min=1e-9))
    return C, _assign(P, C)

def avq_quantize(Wt, M=2, D=8, k=256, rounds=2, oa=False, h=None, gen=None):
    # Wt:(IC,OC) -> group input dim into D-chunks. additive M codebooks. returns deq (IC,OC).
    IC, OC = Wt.shape
    pad = (-IC) % D
    Wp = torch.cat([Wt, torch.zeros(pad, OC)], 0) if pad else Wt
    ICp = Wp.shape[0]
    G = ICp // D
    # reshape to vectors: (G*OC, D)  -- each group of D input rows, per output col
    V = Wp.reshape(G, D, OC).permute(0, 2, 1).reshape(G * OC, D)   # (G*OC, D)
    # per-vector weight from diagonal Hessian h (IC,) -> mean over the D rows of the group
    wv = None
    if oa and h is not None:
        hp = torch.cat([h, torch.zeros(pad)]) if pad else h
        hg = hp.reshape(G, D).mean(1)                              # (G,)
        wv = hg.unsqueeze(1).expand(G, OC).reshape(-1)             # (G*OC,)
    R = V.clone()
    codes = []; cbs = []
    for m in range(M):
        # OA-EM init: weight the k-means by h (output-aware); greedy: plain
        C, a = kmeans_vec(R, k=k, iters=6, w=(wv if oa else None), gen=gen)
        cbs.append(C); codes.append(a)
        R = R - C[a]                                              # residual for next codebook
    # LSQ-style refit could go here; keep init-only to isolate the init effect (L3 is about init)
    recon = sum(cb[a] for cb, a in zip(cbs, codes))              # (G*OC, D)
    Wr = recon.reshape(G, OC, D).permute(0, 2, 1).reshape(ICp, OC)[:IC]
    return Wr

# ---------------------------------------------------------------- load model + calib activations
from transformers import AutoModelForCausalLM, AutoTokenizer
log("loading", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()

# calibration text: real wikitext-2 test; fallback to a bundled sample
def get_text():
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(p)
        txt = "\n\n".join(t for t in df["text"].tolist() if t.strip())
        log("wikitext-2 test loaded", len(txt), "chars")
        return txt
    except Exception as e:
        log("wikitext download failed, using bundled text:", repr(e)[:120])
        return (" ".join(["The history of science is the study of the development of "
                "knowledge across natural philosophy, mathematics and engineering over many centuries."]*400))

text = get_text()
ids = tok(text, return_tensors="pt").input_ids[0]
seqs = [ids[i*SEQLEN:(i+1)*SEQLEN].unsqueeze(0) for i in range(CALIB_SEQS)]

# capture inputs (X) to the target Linear layers
targets = {}
for li in LAYER_IDX:
    for pj in PROJS:
        mod = model.model.layers[li]
        for part in pj.split("."):
            mod = getattr(mod, part)
        targets[f"L{li}.{pj}"] = mod
Xcap = {name: [] for name in targets}
hooks = []
def mk(name):
    def hook(m, inp, out): Xcap[name].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float())
    return hook
for name, mod in targets.items():
    hooks.append(mod.register_forward_hook(mk(name)))
log("forward calibration:", CALIB_SEQS, "seqs x", SEQLEN)
with torch.no_grad():
    for s in seqs:
        model(s.to(DEV))
for h in hooks: h.remove()
X = {name: torch.cat(v, 0) for name, v in Xcap.items()}          # (N, IC) per layer
log("activations captured for", len(X), "layers")

# ---------------------------------------------------------------- run levers per layer
gen = torch.Generator().manual_seed(0)
rows = []
for name, mod in targets.items():
    W = mod.weight.data.float()                                  # (OC, IC)
    Wt = W.t().contiguous()                                      # (IC, OC)
    Xl = X[name]                                                 # (N, IC)
    h = (Xl ** 2).mean(0)                                        # diagonal Hessian (IC,)
    IC, OC = Wt.shape

    # L0 baseline
    deq0 = kmeans_cols(Wt, K)
    e0 = rel_out_err(Xl, W, deq0.t())

    # L2 weighted
    deq2 = kmeans_cols(Wt, K, w=h)
    e2 = rel_out_err(Xl, W, deq2.t())

    # L1 rotation (fold back exactly): quantize in rotated basis Wt_r = Q @ Wt, deq back Q^T @ .
    Q = rand_hadamard(IC, gen)                                   # (IC,IC) orthonormal
    Wt_r = Q @ Wt
    deq1_r = kmeans_cols(Wt_r, K)
    deq1 = Q.t() @ deq1_r
    e1 = rel_out_err(Xl, W, deq1.t())

    # L1+L2 : weight in rotated basis. x_rot = Xl @ Q^T -> h_rot = mean(x_rot^2)
    h_rot = ((Xl @ Q.t()) ** 2).mean(0)
    deq12_r = kmeans_cols(Wt_r, K, w=h_rot)
    deq12 = Q.t() @ deq12_r
    e12 = rel_out_err(Xl, W, deq12.t())

    row = dict(layer=name, IC=IC, OC=OC, L0=e0, L1=e1, L2=e2, L1L2=e12)

    li_num = int(name.split(".")[0][1:])
    if RUN_L3 and li_num in {LAYER_IDX[0], LAYER_IDX[len(LAYER_IDX)//2], LAYER_IDX[-1]}:
        # 2-bit additive VQ: greedy init vs OA-EM init
        dqg = avq_quantize(Wt, M=2, oa=False, gen=torch.Generator().manual_seed(0))
        dqo = avq_quantize(Wt, M=2, oa=True, h=h, gen=torch.Generator().manual_seed(0))
        row["L3_greedy2b"] = rel_out_err(Xl, W, dqg.t())
        row["L3_oaem2b"]   = rel_out_err(Xl, W, dqo.t())
    rows.append(row)
    log(name, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})

# ---------------------------------------------------------------- aggregate
def gmean_ratio(rows, num, den):
    import math
    rs = [r[num] / r[den] for r in rows if r.get(den, 0) > 0]
    return math.exp(sum(math.log(x) for x in rs) / len(rs))

summary = {
    "model": MODEL, "K": K, "bits": 4, "n_layers": len(rows),
    "mean_L0": sum(r["L0"] for r in rows) / len(rows),
    "mean_L1": sum(r["L1"] for r in rows) / len(rows),
    "mean_L2": sum(r["L2"] for r in rows) / len(rows),
    "mean_L1L2": sum(r["L1L2"] for r in rows) / len(rows),
    "L1_vs_L0_gmean": gmean_ratio(rows, "L1", "L0"),
    "L2_vs_L0_gmean": gmean_ratio(rows, "L2", "L0"),
    "L1L2_vs_L0_gmean": gmean_ratio(rows, "L1L2", "L0"),
}
l3 = [r for r in rows if "L3_greedy2b" in r]
if RUN_L3 and l3:
    summary["L3_oaem_vs_greedy_gmean"] = gmean_ratio(l3, "L3_oaem2b", "L3_greedy2b")
    summary["mean_L3_greedy2b"] = sum(r["L3_greedy2b"] for r in l3) / len(l3)
    summary["mean_L3_oaem2b"] = sum(r["L3_oaem2b"] for r in l3) / len(l3)
    summary["n_layers_L3"] = len(l3)

out = dict(summary=summary, rows=rows)
open(os.path.join(os.path.dirname(__file__), "exp_levers_results.json"), "w").write(json.dumps(out, indent=2))
log("SUMMARY")
print(json.dumps(summary, indent=2))
