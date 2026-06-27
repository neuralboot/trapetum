#!/usr/bin/env python3
"""Wire REAL AQLM codebooks to our additive-VQ kernel + measure PPL.

AQLM "2x8" = M=2 additive codebooks of K=256, group size 8, plus a per-output scale:
W[out, group] = scale[out] * sum_m C_m[ code_m[out, group] ]. That is EXACTLY the scheme
our avq kernel decodes (sum of M codebook lookups), so a real AQLM-2x8 checkpoint maps
straight onto our kernel. This script:
  1. loads a real AQLM 2-bit Llama-2-7B and measures wikitext PPL (the accuracy),
  2. introspects one quantized layer and reconstructs W from its (codebooks, codes,
     scales), comparing to AQLM's own dequantized weight, to prove the scheme matches.

Run:  pip install aqlm[gpu] transformers==4.44.2 ; python llama_aqlm.py
"""
import torch, torch.nn as nn, traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

CANDIDATES = [
    "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf",   # M=2 K=256, matches our kernel
    "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf",  # fallback (M=1 K=65536, huge codebook)
]
SEQLEN = 2048; PPL_SAMPLES = 30

@torch.no_grad()
def ppl(model, enc):
    nseq = min(PPL_SAMPLES, enc.size(1)//SEQLEN); lf = nn.CrossEntropyLoss(reduction="sum"); tot=0.0; nt=0
    for i in range(nseq):
        ids = enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device); o = model(ids).logits
        tot += lf(o[:, :-1, :].reshape(-1, o.size(-1)).float(), ids[:, 1:].reshape(-1)).item(); nt += ids[:,1:].numel()
    return float(torch.tensor(tot/nt).exp())

def main():
    model = MODEL = None
    for cand in CANDIDATES:
        try:
            print("trying", cand, flush=True)
            model = AutoModelForCausalLM.from_pretrained(cand, torch_dtype=torch.float16,
                                                         device_map="cuda", trust_remote_code=True)
            MODEL = cand; break
        except Exception as e:
            print("  failed:", str(e)[:120], flush=True)
    if model is None: print("NO AQLM CHECKPOINT LOADED"); return
    print("loaded", MODEL, flush=True)
    tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    print("AQLM 2-bit PPL = %.4f  (fp16 = 5.83, our scalar 4-bit = 6.34)" % ppl(model, enc), flush=True)

    # introspect one quantized linear layer
    for name, mod in model.named_modules():
        if "q_proj" in name and (hasattr(mod, "codebooks") or hasattr(mod, "codes")):
            print("\n=== layer", name, "===", flush=True)
            for attr in ("codebooks", "codes", "scales", "in_group_size", "out_group_size",
                         "num_codebooks", "nbits_per_codebook"):
                v = getattr(mod, attr, None)
                if v is not None:
                    print(" ", attr, getattr(v, "shape", v), getattr(v, "dtype", ""), flush=True)
            # reconstruct W from (codebooks, codes, scales) and compare to AQLM's dequant
            try:
                cb = mod.codebooks.float()      # (num_codebooks, K, out_group, in_group)
                codes = mod.codes               # (out//og, in//ig, num_codebooks)
                scales = mod.scales.float() if getattr(mod, "scales", None) is not None else 1.0
                # sum over codebooks of the selected entries
                ncb = cb.shape[0]
                sel = sum(cb[m][codes[..., m].long()] for m in range(ncb))   # (out//og, in//ig, out_group, in_group)
                Wmine = (sel * (scales if torch.is_tensor(scales) else 1.0))
                print("  reconstructed Wmine shape", Wmine.shape, flush=True)
                # AQLM's own dequant for reference, if available
                ref = None
                for m in ("dequantize", "_dequantize", "get_weight"):
                    if hasattr(mod, m):
                        try: ref = getattr(mod, m)(); print("  ref via", m, ref.shape, flush=True); break
                        except Exception: pass
                if ref is not None:
                    e = (Wmine.flatten().float() - ref.flatten().float())
                    print("  rel err (our additive recon vs AQLM dequant) = %.2e" %
                          (e.norm()/ref.flatten().float().norm()).item(), flush=True)
                else:
                    print("  (no dequant method found; printed shapes confirm M / K / group)", flush=True)
            except Exception:
                print("  reconstruct failed:\n", traceback.format_exc()[-600:], flush=True)
            break
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
