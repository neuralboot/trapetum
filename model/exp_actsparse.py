# exp_actsparse.py -- Phase-1 go/no-go for ACTIVATION SPARSITY on a SwiGLU FFN (the R1 experts'
# architecture), measured locally on TinyLlama-1.1B. CATS-style: the SwiGLU intermediate
# act = silu(gate(x)) * up(x); if silu(gate(x))[j] ~ 0 then act[j] ~ 0 regardless of up[j], so
# neuron j can be skipped -- we never read up's row j nor down's column j (2/3 of the FFN bytes,
# which are the Trapetum CPU-expert-decode bottleneck). This measures the QUALITY question:
# how much of silu(gate) can be zeroed before wikitext-2 PPL degrades.
#
# We use a per-token ORACLE magnitude threshold (keep the top (1-S) fraction of |silu(gate)| per
# token) = the BEST case a threshold policy could do. If even the oracle degrades badly at S=0.5,
# a fixed-threshold CATS policy is worse -> clear no-go for SwiGLU sparsity. SwiGLU is smooth
# (no hard ReLU zeros), so the risk is that it is much less sparsifiable than ReLU FFNs.
#
# Run:  /tmp/lev_venv/bin/python model/exp_actsparse.py

import os, math, json, time
import torch, torch.nn as nn
torch.manual_seed(0)
MODEL = os.environ.get("LEV_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SEQLEN = 2048; PPL_SAMPLES = 20
LEVELS = [0.0, 0.3, 0.5, 0.7, 0.9]

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
log("loading", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

def get_text():
    from huggingface_hub import hf_hub_download; import pandas as pd
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    return "\n\n".join(t for t in pd.read_parquet(p)["text"].tolist() if t.strip())
ids = tok(get_text(), return_tensors="pt").input_ids
log("tokens", ids.shape[1])

# ---- monkeypatch every LlamaMLP.forward with a CATS-sparsified SwiGLU ----
STATE = {"S": 0.0, "kept": 0, "total": 0, "natural": None}
mlps = [m for m in model.modules() if m.__class__.__name__.endswith("MLP")]
log(f"{len(mlps)} MLP blocks")

def sparse_forward(self, x):
    g = self.act_fn(self.gate_proj(x))                      # silu(gate(x)) : (B,T,inter)
    S = STATE["S"]
    if STATE["natural"] is not None:                        # accumulate |silu(gate)| histogram once
        a = g.detach().abs().reshape(-1)
        STATE["natural"][0] += (a < 1e-3).sum().item()
        STATE["natural"][1] += (a < 1e-2).sum().item()
        STATE["natural"][2] += (a < 5e-2).sum().item()
        STATE["natural"][3] += a.numel()
    if S > 0.0:
        inter = g.shape[-1]
        k = max(1, int(round(inter * (1.0 - S))))           # keep top-k by |silu(gate)| per token
        idx = g.abs().topk(k, dim=-1).indices
        mask = torch.zeros_like(g); mask.scatter_(-1, idx, 1.0)
        g = g * mask
        STATE["kept"] += k * (g.shape[0]*g.shape[1]); STATE["total"] += inter * (g.shape[0]*g.shape[1])
    return self.down_proj(g * self.up_proj(x))

import types
for m in mlps: m.forward = types.MethodType(sparse_forward, m)

def ppl():
    nll = 0.0; cnt = 0
    with torch.no_grad():
        for i in range(PPL_SAMPLES):
            b = ids[:, i*SEQLEN:(i+1)*SEQLEN]
            if b.shape[1] < 2: break
            loss = model(b, labels=b).loss.item()
            nll += loss*(b.shape[1]-1); cnt += (b.shape[1]-1)
    return math.exp(nll/cnt)

# baseline pass also measures the natural sparsity of |silu(gate)|
STATE["S"] = 0.0; STATE["natural"] = [0,0,0,0]
p_fp = ppl()
nat = STATE["natural"]; STATE["natural"] = None
log(f"fp32 PPL = {p_fp:.4f}")
log(f"natural |silu(gate)| sparsity: <1e-3 {nat[0]/nat[3]:.3f}, <1e-2 {nat[1]/nat[3]:.3f}, <5e-2 {nat[2]/nat[3]:.3f}")

res = {"model": MODEL, "ppl_fp32": p_fp,
       "natural_sparsity": {"lt_1e-3": nat[0]/nat[3], "lt_1e-2": nat[1]/nat[3], "lt_5e-2": nat[2]/nat[3]},
       "levels": []}
for S in LEVELS[1:]:
    STATE["S"] = S; STATE["kept"] = 0; STATE["total"] = 0
    p = ppl()
    ach = 1.0 - STATE["kept"]/max(STATE["total"],1)
    # bytes: gate always read; up+down (2/3 of FFN) scale with (1-S). total FFN bytes ~ (1 + 2*(1-S))/3
    byte_frac = (1 + 2*(1-S))/3
    res["levels"].append({"target_S": S, "achieved_S": ach, "ppl": p, "dppl": p-p_fp, "ffn_byte_frac": byte_frac})
    log(f"S={S:.1f}: PPL {p:.4f} (dPPL +{p-p_fp:.4f}), FFN bytes -> {byte_frac:.2f}x")

open(os.path.join(os.path.dirname(__file__), "exp_actsparse_results.json"), "w").write(json.dumps(res, indent=2))
log("RESULT"); print(json.dumps(res, indent=2))
