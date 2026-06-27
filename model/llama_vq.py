#!/usr/bin/env python3
"""Vector quantization for the codebook scheme, the real accuracy lever. Instead of
quantizing each weight alone (scalar K=16, 4 bits), we quantize GROUPS of d adjacent
weights as one vector against a K-entry vector codebook. At d=2, K=256 the rate is
log2(256)/2 = 4 bits/weight, iso-bit with the scalar codebook, but the vectors
capture pairwise weight correlation, the idea behind AQLM and QuIP#.

Decisive test: does VQ d=2,K=256 (4 bits) beat the scalar codebook (6.34) at the
same bit budget? Run:  python llama_vq.py
"""
import torch, torch.nn as nn, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
D = 2; K = 256; SEQLEN = 2048; PPL_SAMPLES = 30
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def vq_quantize(W, d=D, k=K, iters=12, chunk=2_000_000):
    # W (OC, IC). Per-output-channel normalize (so one shared codebook can span all
    # channels despite scale heterogeneity), then group adjacent d-tuples and k-means.
    OC, IC = W.shape
    if IC % d: return W                                  # skip if not divisible
    s = W.abs().amax(1, keepdim=True).clamp(min=1e-8)    # (OC,1) per-channel scale
    Wn = W / s
    P = Wn.reshape(OC, IC // d, d).reshape(-1, d).contiguous()   # (M, d)
    M = P.shape[0]
    C = P[torch.randperm(M, device=W.device)[:k]].clone()       # (k, d) init
    assign = torch.zeros(M, dtype=torch.long, device=W.device)
    for _ in range(iters):
        Cn2 = (C * C).sum(1)                              # (k,)
        for s in range(0, M, chunk):
            p = P[s:s+chunk]
            d2 = (p*p).sum(1, keepdim=True) - 2.0 * (p @ C.t()) + Cn2   # (c, k)
            assign[s:s+chunk] = d2.argmin(1)
            del d2
        Cnew = torch.zeros_like(C); cnt = torch.zeros(k, device=W.device)
        Cnew.index_add_(0, assign, P)
        cnt.index_add_(0, assign, torch.ones(M, device=W.device))
        C = Cnew / cnt.clamp(min=1).unsqueeze(1)
    C = torch.nan_to_num(C)                               # guard degenerate centroids
    Pq = C[assign]                                        # (M, d) reconstructed (normalized)
    Wq = Pq.reshape(OC, IC // d, d).reshape(OC, IC) * s   # un-normalize per channel
    Wq = torch.nan_to_num(Wq).clamp(W.min(), W.max())     # nan guard + keep original range
    return Wq

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
    bits = (torch.log2(torch.tensor(float(K))) / D).item()
    print("fp16            : PPL = %.4f" % ppl(model, enc), flush=True)
    print("vector quantize d=%d K=%d (%.2f bits/weight)..." % (D, K, bits), flush=True)
    t0 = time.time()
    with torch.no_grad():
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
                W = mod.weight.data.float()
                Wq = vq_quantize(W)
                mod.weight.data = Wq.to(mod.weight.dtype)
                del W, Wq; torch.cuda.empty_cache()
    print("  VQ quantize time %.0fs" % (time.time()-t0), flush=True)
    print("codebook VQ d=%d K=%d (%.0f-bit): PPL = %.4f" % (D, K, bits, ppl(model, enc)), flush=True)
    print("(scalar 4-bit was 6.34 | simple-calib 6.17 | fp16 5.83)", flush=True)

if __name__ == "__main__":
    main()
