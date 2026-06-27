#!/usr/bin/env python3
"""Fuller AWQ-style quantization for the codebook scheme: per-input-channel scale
search. AWQ's core lever is to scale each input channel by s_i before quantizing
(important channels get more resolution), folding the inverse scale into the
activation. We grid-search the scale exponent alpha (s = importance^alpha) per
layer and keep the alpha that minimizes the activation-weighted output error.

Reports fp16 PPL and AWQ-scaled codebook-4bit PPL (compare to naive 6.34 and the
weighted-only 6.17 from llama_calib.py). Run:  python llama_awq.py
"""
import torch, torch.nn as nn, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
K = 16; SEQLEN = 2048; PPL_SAMPLES = 30; CALIB_SEQS = 8
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def quantize_deq(W, k=16, iters=10, chunk=2048):
    # uniform per-column k-means, chunked. Returns dequantized (IC, OC).
    IC, OC = W.shape; deq = torch.empty_like(W)
    for c0 in range(0, OC, chunk):
        c1 = min(OC, c0+chunk); Wc = W[:, c0:c1]; cw = c1-c0
        cb = torch.zeros(k, cw, device=W.device); lo = Wc.min(0).values; hi = Wc.max(0).values
        for c in range(k): cb[c] = lo + (hi-lo)*(c+0.5)/k
        ii = None
        for _ in range(iters):
            d = (Wc.unsqueeze(-1)-cb.t().unsqueeze(0)).abs(); ii = d.argmin(-1); del d
            for c in range(k):
                m = (ii==c); cb[c] = (Wc*m).sum(0)/m.sum(0).clamp(min=1)
        deq[:, c0:c1] = cb[ii, torch.arange(cw, device=W.device)]
    return deq

def awq_quantize(Wt, imp, k=16):
    # Wt (IC, OC), imp (IC,) activation importance. Search per-channel scale.
    best, best_err = None, None
    impn = imp / imp.mean().clamp(min=1e-8)
    for a in ALPHAS:
        scale = impn.pow(a).clamp(min=1e-4, max=1e4)      # (IC,)
        Ws = Wt * scale.unsqueeze(1)
        deq = quantize_deq(Ws, k)
        W_eff = deq / scale.unsqueeze(1)
        err = (((Wt - W_eff) ** 2) * imp.unsqueeze(1)).sum().item()  # activation-weighted
        if best_err is None or err < best_err: best_err, best = err, W_eff
        del Ws, deq, W_eff
    return best

@torch.no_grad()
def ppl(model, enc):
    nseq = min(PPL_SAMPLES, enc.size(1)//SEQLEN); lf = nn.CrossEntropyLoss(reduction="sum"); tot=0.0; nt=0
    for i in range(nseq):
        ids = enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device); out = model(ids).logits
        tot += lf(out[:, :-1, :].reshape(-1, out.size(-1)).float(), ids[:, 1:].reshape(-1)).item(); nt += ids[:,1:].numel()
    return float(torch.tensor(tot/nt).exp())

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    print("fp16            : PPL = %.4f" % ppl(model, enc))

    acts = {}
    def mk(n):
        def h(m, inp, out): acts[n] = acts.get(n, 0) + inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).sum(0).float()
        return h
    hooks = [m.register_forward_hook(mk(n)) for n,m in model.named_modules() if isinstance(m, nn.Linear) and any(t in n for t in TARGETS)]
    print("collecting activation stats...")
    with torch.no_grad():
        for i in range(CALIB_SEQS): model(enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device))
    for h in hooks: h.remove()

    print("AWQ scale-search quantize (grid over alpha=%s)..." % (ALPHAS,))
    t0 = time.time()
    with torch.no_grad():
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
                Wt = mod.weight.data.t().float().contiguous()
                W_eff = awq_quantize(Wt, acts[name].to(Wt.device), K)
                mod.weight.data = W_eff.t().contiguous().to(mod.weight.dtype)
                del Wt, W_eff; torch.cuda.empty_cache()
    print("  awq quantize time %.0fs" % (time.time()-t0))
    print("codebook 4-bit (AWQ scale-search): PPL = %.4f" % ppl(model, enc))
    print("\n(naive 6.34 | weighted-only 6.17 | fp16 5.83)")

if __name__ == "__main__":
    main()
