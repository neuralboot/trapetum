#!/usr/bin/env python3
"""Additive (residual) vector quantization with a per-output scale, sweeping the
number of codebooks M. This is exactly the scheme the avq kernel decodes (M shared
K=256 codebooks over D=8 groups, plus a per-output scale). The question: does adding
codebooks (M=2 -> M=4, i.e. 2-bit -> 4-bit) reach near-fp16 accuracy while staying
LUT/kernel friendly (small K=256), so it beats the scalar 4-bit codebook (6.34)?

Greedy residual fit (not full AQLM calibration), so this is a floor on what the scheme
can do. Run:  python llama_advq.py
"""
import torch, torch.nn as nn, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
K = 256; D = 8; SEQLEN = 2048; PPL_SAMPLES = 30
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def kmeans(P, k=K, iters=6, chunk=1_000_000):
    N = P.shape[0]
    C = P[torch.randperm(N, device=P.device)[:k]].clone()
    assign = torch.zeros(N, dtype=torch.long, device=P.device)
    for _ in range(iters):
        Cn2 = (C*C).sum(1)
        for s in range(0, N, chunk):
            p = P[s:s+chunk]
            assign[s:s+chunk] = ((p*p).sum(1,keepdim=True) - 2*(p@C.t()) + Cn2).argmin(1)
        Cnew = torch.zeros_like(C); cnt = torch.zeros(k, device=P.device)
        Cnew.index_add_(0, assign, P); cnt.index_add_(0, assign, torch.ones(N, device=P.device))
        C = Cnew / cnt.clamp(min=1).unsqueeze(1)
    return C, assign

@torch.no_grad()
def residual_vq(W, Mc):
    OC, IC = W.shape
    s = W.abs().amax(1, keepdim=True).clamp(min=1e-8)
    R = (W / s).reshape(OC, IC//D, D).reshape(-1, D).contiguous()
    recon = torch.zeros_like(R)
    for m in range(Mc):
        C, idx = kmeans(R)
        q = C[idx]; recon += q; R = R - q
    return (recon.reshape(OC, IC//D, D).reshape(OC, IC)) * s

@torch.no_grad()
def quantize_model(model, Mc):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
            W = mod.weight.data.float()
            mod.weight.data = residual_vq(W, Mc).to(mod.weight.dtype)
            del W; torch.cuda.empty_cache()

@torch.no_grad()
def ppl(model, enc):
    nseq = min(PPL_SAMPLES, enc.size(1)//SEQLEN); lf = nn.CrossEntropyLoss(reduction="sum"); tot=0.0; nt=0
    for i in range(nseq):
        ids = enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device); o = model(ids).logits
        tot += lf(o[:, :-1, :].reshape(-1, o.size(-1)).float(), ids[:, 1:].reshape(-1)).item(); nt += ids[:,1:].numel()
    return float(torch.tensor(tot/nt).exp())

def load(): return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    m = load(); print("fp16            : PPL = %.4f" % ppl(m, enc), flush=True); del m; torch.cuda.empty_cache()
    for Mc in (2, 4):
        m = load(); t0 = time.time(); quantize_model(m, Mc)
        bits = Mc * 8 / D
        print("additive-VQ M=%d (%.0f-bit): PPL = %.4f  (%.0fs)" % (Mc, bits, ppl(m, enc), time.time()-t0), flush=True)
        del m; torch.cuda.empty_cache()
    print("(refs: fp16 5.83 | scalar 4-bit 6.34 | scalar 2-bit diverges | AQLM-2x8 7.63)", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
