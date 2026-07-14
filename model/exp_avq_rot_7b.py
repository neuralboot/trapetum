# exp_avq_rot_7b.py -- FAST A/B of Lever 1 (Hadamard rotation) on the REAL AQLM/beam+LSQ
# compressor, on a handful of real Llama-2-7B layers. Metric = relative OUTPUT error on real
# activations (the deployment-faithful proxy validated locally), ROT=0 vs ROT=1, at the SAME
# M=4 4-bit beam+LSQ config. Answers "does rotation help on top of the strong AVQ baseline?"
# in ~20 min instead of a multi-hour full-model PPL run. Reuses aqlm_quantize verbatim.
#
# Run (on the pod, weights cached):  python model/exp_avq_rot_7b.py
import os, math, json, torch
os.environ["ROUNDS"] = "1"; os.environ["CALIB"] = "0"
import llama_aqlm_train as A                      # reuse aqlm_quantize / beam_search / lsq_update
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
LAYERS = [0, 15, 31]
PROJS  = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj"]
CALIB_SEQS = 4; SEQLEN = 2048

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()
enc = tok("\n\n".join(load_dataset("wikitext","wikitext-2-raw-v1",split="test")["text"]), return_tensors="pt").input_ids

targets = {}
for li in LAYERS:
    for pj in PROJS:
        m = model.model.layers[li]
        for p in pj.split("."): m = getattr(m, p)
        targets[f"L{li}.{pj}"] = m
cap = {n: [] for n in targets}; hooks = []
def mk(n):
    def h(mod, i, o): cap[n].append(i[0].detach().reshape(-1, i[0].shape[-1]).float().cpu())
    return h
for n, m in targets.items(): hooks.append(m.register_forward_hook(mk(n)))
with torch.no_grad():
    for i in range(CALIB_SEQS): model(enc[:, i*SEQLEN:(i+1)*SEQLEN].cuda())
for h in hooks: h.remove()
X = {n: torch.cat(v, 0) for n, v in cap.items()}
print("activations captured for", len(X), "layers", flush=True)

@torch.no_grad()
def rel_err(Xl, W, deq):
    num = (Xl @ (W - deq).t()).norm(); den = (Xl @ W.t()).norm().clamp(min=1e-12)
    return (num / den).item()

rows = []
for n, m in targets.items():
    W = m.weight.data.float()                       # (OC, IC) cuda
    Xl = X[n].cuda()
    hdiag = (Xl ** 2).mean(0)                        # unused at CALIB=0, passed for signature
    e0 = rel_err(Xl, W, A.aqlm_quantize(W.clone(), hdiag, 4, rotate=False))
    e1 = rel_err(Xl, W, A.aqlm_quantize(W.clone(), hdiag, 4, rotate=True))
    rows.append((n, e0, e1)); del Xl; torch.cuda.empty_cache()
    print(f"{n}  ROT0={e0:.4f}  ROT1={e1:.4f}  ratio={e1/e0:.3f}", flush=True)

g = math.exp(sum(math.log(r[2]/r[1]) for r in rows) / len(rows))
me0 = sum(r[1] for r in rows)/len(rows); me1 = sum(r[2] for r in rows)/len(rows)
res = dict(model=MODEL, config="AVQ M=4 4-bit beam+LSQ ROUNDS=1", n_layers=len(rows),
           mean_ROT0=me0, mean_ROT1=me1, ROT1_vs_ROT0_gmean=g,
           rows=[dict(layer=r[0], rot0=r[1], rot1=r[2]) for r in rows])
open("/work/exp_avq_rot_7b_results.json", "w").write(json.dumps(res, indent=2))
print("AVQ-ROT gmean(ROT1/ROT0) =", round(g,4), "| mean ROT0", round(me0,4), "ROT1", round(me1,4), flush=True)
print("AVQROT_DONE", flush=True)
