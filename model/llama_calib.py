#!/usr/bin/env python3
"""Activation-aware calibration for the codebook scheme, AWQ-style, to close the
PPL gap. Naive per-column k-means minimizes raw weight error; this weights each
input channel by its activation importance s_i = mean|x_i| over calibration data,
so the k-means minimizes the activation-weighted error (a proxy for output error),
which is the core of activation-aware quantization.

Reports fp16 PPL and calibrated codebook-4bit PPL (compare to the naive 6.34 from
llama_quant.py). Run:  python llama_calib.py
"""
import torch, torch.nn as nn, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
K = 16; SEQLEN = 2048; PPL_SAMPLES = 30; CALIB_SEQS = 8
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def quantize_weighted(W, s, k=16, iters=10, chunk=2048):
    # W (IC, OC), s (IC,) activation importance. Returns dequantized (IC, OC).
    IC, OC = W.shape; deq = torch.empty_like(W); sw = s.unsqueeze(1)
    for c0 in range(0, OC, chunk):
        c1 = min(OC, c0 + chunk); Wc = W[:, c0:c1]; cw = c1 - c0
        cb = torch.zeros(k, cw, device=W.device); lo = Wc.min(0).values; hi = Wc.max(0).values
        for c in range(k): cb[c] = lo + (hi - lo) * (c + 0.5) / k
        ii = None
        for _ in range(iters):
            d = (Wc.unsqueeze(-1) - cb.t().unsqueeze(0)).abs(); ii = d.argmin(-1); del d
            for c in range(k):
                m = (ii == c).float() * sw                       # weighted membership
                cb[c] = (Wc * m).sum(0) / m.sum(0).clamp(min=1e-6)
        deq[:, c0:c1] = cb[ii, torch.arange(cw, device=W.device)]
    return deq

@torch.no_grad()
def ppl(model, tok, enc):
    nseq = min(PPL_SAMPLES, enc.size(1) // SEQLEN)
    lf = nn.CrossEntropyLoss(reduction="sum"); tot = 0.0; nt = 0
    for i in range(nseq):
        ids = enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device)
        out = model(ids).logits
        tot += lf(out[:, :-1, :].reshape(-1, out.size(-1)).float(), ids[:, 1:].reshape(-1)).item()
        nt += ids[:, 1:].numel()
    return float(torch.tensor(tot/nt).exp())

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    print("fp16            : PPL = %.4f" % ppl(model, tok, enc))

    # collect activation importance per target linear
    acts = {}
    def mk(name):
        def hook(mod, inp, out):
            a = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).sum(0).float()
            acts[name] = acts.get(name, 0) + a
        return hook
    hooks = [m.register_forward_hook(mk(n)) for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and any(t in n for t in TARGETS)]
    print("collecting activation stats over %d calib sequences..." % CALIB_SEQS)
    with torch.no_grad():
        for i in range(CALIB_SEQS):
            model(enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device))
    for h in hooks: h.remove()

    # calibrated quantize, in place
    print("calibrated quantize (activation-weighted k-means)...")
    t0 = time.time()
    with torch.no_grad():
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
                Wt = mod.weight.data.t().float().contiguous()       # (IC, OC)
                s = acts[name].to(Wt.device) / acts[name].sum().clamp(min=1) * Wt.size(0)  # normalized importance
                deq = quantize_weighted(Wt, s, K)
                mod.weight.data = deq.t().contiguous().to(mod.weight.dtype)
                del Wt, deq, s; torch.cuda.empty_cache()
    print("  calib quantize time %.0fs" % (time.time()-t0))
    ppl_c = ppl(model, tok, enc)
    print("codebook 4-bit (calibrated): PPL = %.4f" % ppl_c)
    print("\n(naive codebook 4-bit was 6.34; fp16 5.83)")

if __name__ == "__main__":
    main()
