#!/usr/bin/env python3
"""Full AWQ pipeline for the codebook scheme. Two levers, both selected by the REAL
output error on cached calibration activations (not the weight-error proxy of
llama_awq.py): (1) per-input-channel scale search, (2) per-output-channel weight
clipping. This is the faithful AWQ objective minimize ||x W - x W_q|| on real x.

Reports fp16 PPL and full-AWQ codebook-4bit PPL. Compare: naive 6.34, simple
calibration 6.17, scale-search (weight-error) 6.18, fp16 5.83.
Run:  python llama_awq_full.py
"""
import torch, torch.nn as nn, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
K = 16; SEQLEN = 2048; PPL_SAMPLES = 30; CALIB_SEQS = 8; N_CACHE = 128
ALPHAS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
CLIPS = (1.0, 0.92, 0.85, 0.78, 0.70)
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def quantize_deq(W, k=16, iters=10, chunk=2048):
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

def out_err(xc, W, Weff, ref):
    return ((xc @ Weff - ref) ** 2).sum().item()

def awq_full(Wt, xc, imp, k=16):
    # Wt (IC,OC), xc (N,IC) cached activations, imp (IC,) importance.
    ref = xc @ Wt                                  # (N, OC) true output
    impn = imp / imp.mean().clamp(min=1e-8)
    best_err, best_W, best_a = None, None, 0.0
    for a in ALPHAS:                               # scale search, output-error selected
        scale = impn.pow(a).clamp(min=1e-4, max=1e4)
        deq = quantize_deq(Wt * scale.unsqueeze(1), k, iters=5)
        Weff = deq / scale.unsqueeze(1)
        e = out_err(xc, Wt, Weff, ref)
        if best_err is None or e < best_err: best_err, best_W, best_a = e, Weff, a
        del deq, Weff
    scale = impn.pow(best_a).clamp(min=1e-4, max=1e4)
    Ws = Wt * scale.unsqueeze(1); mx = Ws.abs().amax(0, keepdim=True)
    for cl in CLIPS:                               # weight clipping at best alpha
        Wc = Ws if cl >= 1.0 else Ws.clamp(-cl*mx, cl*mx)
        deq = quantize_deq(Wc, k, iters=10)
        Weff = deq / scale.unsqueeze(1)
        e = out_err(xc, Wt, Weff, ref)
        if e < best_err: best_err, best_W = e, Weff
        del deq, Weff
    return best_W

@torch.no_grad()
def ppl(model, enc):
    nseq = min(PPL_SAMPLES, enc.size(1)//SEQLEN); lf = nn.CrossEntropyLoss(reduction="sum"); tot=0.0; nt=0
    for i in range(nseq):
        ids = enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device); o = model(ids).logits
        tot += lf(o[:, :-1, :].reshape(-1, o.size(-1)).float(), ids[:, 1:].reshape(-1)).item(); nt += ids[:,1:].numel()
    return float(torch.tensor(tot/nt).exp())

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    print("fp16            : PPL = %.4f" % ppl(model, enc), flush=True)

    cache, imp = {}, {}
    def mk(n):
        def h(m, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            imp[n] = imp.get(n, 0) + x.abs().sum(0).float()
            cur = cache.get(n)
            if cur is None or cur.shape[0] < N_CACHE:
                row = x[:N_CACHE].float()
                cache[n] = row if cur is None else torch.cat([cur, row])[:N_CACHE]
        return h
    hooks = [m.register_forward_hook(mk(n)) for n,m in model.named_modules()
             if isinstance(m, nn.Linear) and any(t in n for t in TARGETS)]
    print("collecting activations + caching %d rows/layer..." % N_CACHE, flush=True)
    with torch.no_grad():
        for i in range(CALIB_SEQS): model(enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device))
    for h in hooks: h.remove()

    print("full AWQ (scale search + clipping, output-error selected)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
                Wt = mod.weight.data.t().float().contiguous()
                xc = cache[name][:N_CACHE].to(Wt.device)
                W_eff = awq_full(Wt, xc, imp[name].to(Wt.device), K)
                mod.weight.data = W_eff.t().contiguous().to(mod.weight.dtype)
                del Wt, xc, W_eff; torch.cuda.empty_cache()
    print("  full-AWQ quantize time %.0fs" % (time.time()-t0), flush=True)
    print("codebook 4-bit (FULL AWQ): PPL = %.4f" % ppl(model, enc), flush=True)
    print("(naive 6.34 | simple-calib 6.17 | scale-search 6.18 | fp16 5.83)", flush=True)

if __name__ == "__main__":
    main()
