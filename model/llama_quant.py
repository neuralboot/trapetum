#!/usr/bin/env python3
"""Model-level accuracy of the per-output-channel 4-bit codebook scheme on a real
LLM. Quantizes every projection Linear of Llama-2-7B with per-output-channel
k-means (K=16, 4 bits), dequantizes in place, and measures wikitext-2 perplexity
against the fp16 baseline. This closes the "no model-level evaluation" gap.

(PPL is determined by the quantized WEIGHTS, not the kernel, so dequant-in-place
gives the exact accuracy of the scheme. The kernel reproduces these weights to
~2e-7, so kernel-served PPL is identical.)

Run:  python llama_quant.py
"""
import torch, time, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
K = 16
SEQLEN = 2048
PPL_SAMPLES = 30          # capped for a first number; raise for the final figure
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def quantize_per_column(W, k=16, iters=10):
    # W: (IC, OC) on cuda. Per output column: k-means over the IC values. Returns the
    # dequantized matrix (same shape, fp16-representable values).
    IC, OC = W.shape
    cb = torch.zeros(k, OC, device=W.device, dtype=torch.float32)
    lo = W.min(0).values; hi = W.max(0).values
    for c in range(k):
        cb[c] = lo + (hi - lo) * (c + 0.5) / k
    idx = torch.zeros(IC, OC, device=W.device, dtype=torch.long)
    for _ in range(iters):
        d = (W.unsqueeze(-1) - cb.t().unsqueeze(0)).abs()   # (IC, OC, K)
        idx = d.argmin(-1)
        del d
        for c in range(k):
            mask = (idx == c)
            cnt = mask.sum(0).clamp(min=1)
            cb[c] = (W * mask).sum(0) / cnt
    deq = cb[idx, torch.arange(OC, device=W.device)]        # (IC, OC) reconstructed
    return deq

@torch.no_grad()
def quantize_model(model):
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and any(t in name for t in TARGETS):
            W = mod.weight.data                              # (OC, IC) fp16
            Wt = W.t().float().contiguous()                  # (IC, OC) = per-output-channel cols
            deq = quantize_per_column(Wt, K)                 # (IC, OC)
            mod.weight.data = deq.t().contiguous().to(W.dtype)
            n += 1
            del W, Wt, deq; torch.cuda.empty_cache()
    print("quantized %d linear layers (%s) to %d-bit codebook" % (n, "/".join(TARGETS), (K).bit_length()-1))

@torch.no_grad()
def ppl_wikitext2(model, tok):
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    nseq = min(PPL_SAMPLES, enc.size(1) // SEQLEN)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total, ntok = 0.0, 0
    for i in range(nseq):
        ids = enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device)
        out = model(ids).logits
        sl = out[:, :-1, :].reshape(-1, out.size(-1)).float()
        tg = ids[:, 1:].reshape(-1)
        total += loss_fn(sl, tg).item(); ntok += tg.numel()
    return float(torch.tensor(total / ntok).exp())

def vram_gb(): return torch.cuda.max_memory_allocated() / 1e9

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("loading fp16 model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"": 0})
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    ppl_fp16 = ppl_wikitext2(model, tok)
    print("fp16            : wikitext-2 PPL = %.4f | peak VRAM %.2f GB" % (ppl_fp16, vram_gb()))

    print("quantizing (per-output-channel k-means, this takes a few minutes)...")
    t0 = time.time(); quantize_model(model); print("  quantize time %.0fs" % (time.time()-t0))
    torch.cuda.reset_peak_memory_stats()
    ppl_q = ppl_wikitext2(model, tok)
    eff_bits = (K).bit_length() - 1
    print("codebook %d-bit  : wikitext-2 PPL = %.4f | peak VRAM %.2f GB" % (eff_bits, ppl_q, vram_gb()))
    print("\nGAP: fp16 %.4f -> codebook-%dbit %.4f  (delta %.4f PPL)" % (ppl_fp16, eff_bits, ppl_q, ppl_q - ppl_fp16))

if __name__ == "__main__":
    main()
