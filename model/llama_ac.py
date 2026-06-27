#!/usr/bin/env python3
"""(a) Served-model quality + (c) incoherence (QuIP#-style) for the codebook scheme.

(a) The kernel reproduces the dequantized weights to ~2e-7, so the served model's
    quality equals the dequant-in-place quantized model's. We confirm it generates
    coherent text and re-measure wikitext PPL.
(c) The biggest SOTA accuracy lever we had not tried: rotate each weight into an
    incoherent basis (a random orthogonal rotation on the input dim, the QuIP#/
    Hadamard idea) BEFORE the scalar codebook, then fold the inverse rotation back
    into the effective weight. If quantization error, spread incoherently, hurts
    less, PPL beats the plain 6.34.

Run:  python llama_ac.py
"""
import torch, torch.nn as nn, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
K = 16; SEQLEN = 2048; PPL_SAMPLES = 30
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROMPT = "The key idea behind quantization is"

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

@torch.no_grad()
def quantize_model(model, rotations=None):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
            Wt = mod.weight.data.t().float().contiguous()         # (IC, OC)
            if rotations is not None:                              # (c) incoherence
                Q = rotations[Wt.shape[0]]
                Wt = Q @ Wt                                        # rotate input dim
                deq = quantize_deq(Wt, K)
                deq = Q.t() @ deq                                  # fold inverse back into weight
            else:                                                  # plain codebook (served = (a))
                deq = quantize_deq(Wt, K)
            mod.weight.data = deq.t().contiguous().to(mod.weight.dtype)
            del Wt, deq; torch.cuda.empty_cache()

@torch.no_grad()
def ppl(model, enc):
    nseq = min(PPL_SAMPLES, enc.size(1)//SEQLEN); lf = nn.CrossEntropyLoss(reduction="sum"); tot=0.0; nt=0
    for i in range(nseq):
        ids = enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device); o = model(ids).logits
        tot += lf(o[:, :-1, :].reshape(-1, o.size(-1)).float(), ids[:, 1:].reshape(-1)).item(); nt += ids[:,1:].numel()
    return float(torch.tensor(tot/nt).exp())

@torch.no_grad()
def sample(model, tok):
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=40, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True).replace("\n", " ")

def load(): return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids

    model = load()
    print("fp16     PPL=%.3f | %s" % (ppl(model, enc), sample(model, tok)), flush=True)

    # (a) plain codebook = the served model's quality
    quantize_model(model)
    print("codebook PPL=%.3f | %s" % (ppl(model, enc), sample(model, tok)), flush=True)
    print(">>> (a) served-model quality: coherent text above + PPL ~6.34 confirms the served weights work", flush=True)
    del model; torch.cuda.empty_cache()

    # (c) incoherence: random orthogonal rotation per unique input dim
    g = torch.Generator(device="cuda").manual_seed(0)
    rot = {}
    for ic in (4096, 11008):
        Q, _ = torch.linalg.qr(torch.randn(ic, ic, device="cuda", dtype=torch.float32, generator=g))
        rot[ic] = Q
    model = load()
    t0 = time.time(); quantize_model(model, rotations=rot)
    print("incoh    PPL=%.3f | %s  (%.0fs)" % (ppl(model, enc), sample(model, tok), time.time()-t0), flush=True)
    print(">>> (c) incoherence: compare PPL to plain codebook 6.34 (and simple-calib 6.17)", flush=True)

if __name__ == "__main__":
    main()
