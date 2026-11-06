#!/usr/bin/env python3
"""
exp_kmeans_fix: does fixing the per-column k-means lower the perplexity of the 4-bit
(K=16) Llama-2-7B, at exactly the same bits?

The codebook audit (model/cbk_audit.py) found that the shipped exporter leaves dead
centroids: it starts every column on a linear grid between min and max, and a centroid
that gets no weight never moves, so columns with a few outliers spend most of their 16
entries on nothing (11% of Llama-2-7B columns have at least one dead entry). This script
measures, on one GPU, what fixing that is worth, alone and combined with sensitivity
weighting, against the exact baselines already published:

  A0m  naive as MEASURED (llama_quant.py before c9d8b72): mid-grid init, empty -> 0, 10 iters,
       no weights, whole matrix                                         (published 6.34)
  A0   current llama_quant.py (linear init after c9d8b72), otherwise as A0m (never measured)
  A0e  export_runtime.py k-means on the 224 projections: linear init, empty kept, 12 iters
       (LM head left in fp16)
  A0eH A0e plus the LM head quantized the same way (the shipped .cbk recipe end to end)
  A1   calibrated (llama_calib.py)   mid-grid init, empty -> 0, 10 iters, mean|x| on the first
                                     8 windows of the TEST split                     (published 6.17)
  A1t  A1 calibrated on 8 windows of the TRAIN split (the honest version of A1)
  A2   quantile init + empty-centroid reseeding, no weights
  A3   quantile init + reseeding, mean|x| on the same 8 train windows as A1t
  A4   quantile init + reseeding, E[x^2] on CALIB_N train windows
  A5   mid-grid init, empty -> 0, 10 iters, E[x^2] on CALIB_N train windows (weighting alone)

Every arm quantizes all 224 projections to 4 bits in place, then measures wikitext-2 PPL in
both repo harnesses from one pass: custom = first 30 windows of 2048, exp(NLL / (30*2047))
(fp16 5.83); full = all windows, exp(NLL / (n*2048)) (bench_quant.py, fp16 5.47). It also
reports the audit metrics of the resulting codebooks (dead entries, index entropy).
Reseeding moves an empty centroid onto the weight of its column with the largest (weighted)
quantization error. Results: OUT (json), rewritten after every arm. No em dashes anywhere.
"""
import json, math, os, sys, time
import torch, torch.nn as nn

MODEL = os.environ.get("MODEL", "NousResearch/Llama-2-7b-hf")
OUT = os.environ.get("OUT", "/work/kmeans_results.json")
K, SEQLEN, CUSTOM = 16, 2048, 30
CALIB_N = int(os.environ.get("CALIB_N", 128))
ARMS_ENV = os.environ.get("ARMS", "A0m,A0,A0e,A0eH,A1,A1t,A2,A3,A4,A5")
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ARMS = {
    "A0m": dict(init="mid", empty="zero", iters=10, final="stale", s=None, chunk=0, desc="naive as measured for 6.34 (mid-grid init)"),
    "A0":  dict(init="linear", empty="zero", iters=10, final="stale", s=None, chunk=0, desc="current llama_quant.py (linear init)"),
    "A0e": dict(init="linear", empty="keep", iters=12, final="stale", s=None, chunk=0, desc="export_runtime k-means, projections (head fp16)"),
    "A0eH": dict(init="linear", empty="keep", iters=12, final="stale", s=None, chunk=0, head=True, desc="export_runtime k-means, projections + head"),
    "A1":  dict(init="mid", empty="zero", iters=10, final="stale", s="abs_test8", desc="calibrated as llama_calib.py (test windows)"),
    "A1t": dict(init="mid", empty="zero", iters=10, final="stale", s="abs_train8", desc="A1 calibrated on train windows"),
    "A2":  dict(init="quantile", empty="reseed", iters=12, final="reassign", s=None, desc="quantile init + reseed"),
    "A3":  dict(init="quantile", empty="reseed", iters=12, final="reassign", s="abs_train8", desc="quantile + reseed + mean|x| (8 train)"),
    "A4":  dict(init="quantile", empty="reseed", iters=12, final="reassign", s="sq_train", desc="quantile + reseed + E[x^2] (%d train)" % CALIB_N),
    "A5":  dict(init="mid", empty="zero", iters=10, final="stale", s="sq_train", desc="E[x^2] (%d train) with the A1 k-means" % CALIB_N),
}
T0 = time.time()
RES = {"meta": dict(model=MODEL, K=K, calib_n=CALIB_N, seqlen=SEQLEN, custom_windows=CUSTOM), "arms": {}, "errors": []}


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def save():
    tmp = OUT + ".tmp"
    json.dump(RES, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)


@torch.no_grad()
def kmeans_cols(Wt, s=None, init="linear", empty="keep", iters=12, final="stale", k=K, chunk=2048):
    """Per-output-column 1-D k-means. Wt (IC, OC) fp32 cuda; s (IC,) row weights or None.
    final="stale" keeps the last assignment made BEFORE the last centroid update, exactly as
    llama_quant.py, llama_calib.py and export_runtime.py do; final="reassign" assigns every
    weight to its nearest final centroid. Returns the dequantized (IC, OC) and the per-column
    usage counts (k, OC) of the indices actually used."""
    IC, OC = Wt.shape
    deq = torch.empty_like(Wt)
    counts_all = torch.empty(k, OC, dtype=torch.long, device=Wt.device)
    ar = torch.arange(k, device=Wt.device, dtype=torch.float32)
    sw = None if s is None else s.view(IC, 1)
    for c0 in range(0, OC, chunk):
        W = Wt[:, c0:c0 + chunk]
        n = W.shape[1]
        cols = torch.arange(n, device=W.device)
        if init == "quantile":
            srt = W.sort(0).values
            pos = ((ar + 0.5) / k * (IC - 1)).round().long()
            cent = srt[pos].clone()                                  # (k, n)
            del srt
        else:
            lo, hi = W.min(0).values, W.max(0).values
            frac = (ar / (k - 1)) if init == "linear" else ((ar + 0.5) / k)
            cent = lo.unsqueeze(0) + (hi - lo).unsqueeze(0) * frac.unsqueeze(1)

        def assign(cent):
            best = torch.full_like(W, float("inf"))
            idx = torch.zeros(W.shape, dtype=torch.long, device=W.device)
            for j in range(k):
                d = (W - cent[j:j + 1]) ** 2
                better = d < best
                idx = torch.where(better, torch.full_like(idx, j), idx)
                best = torch.where(better, d, best)
            return idx, best

        idx = None
        for _ in range(iters):
            idx, best = assign(cent)
            for j in range(k):
                m = (idx == j).float()
                mw = m if sw is None else m * sw
                wsum = mw.sum(0)
                num = (W * mw).sum(0)
                if empty == "zero":
                    cent[j] = num / wsum.clamp(min=1e-6 if sw is not None else 1.0)
                else:
                    cent[j] = torch.where(wsum > 0, num / wsum.clamp(min=1e-30), cent[j])
            if empty == "reseed":
                cnt = torch.stack([(idx == j).sum(0) for j in range(k)])     # (k, n)
                emp = cnt == 0
                has = emp.any(0)
                if bool(has.any()):
                    err = best if sw is None else best * sw
                    r = err.argmax(0)                                         # worst-quantized row per column
                    val = W[r, cols]
                    j0 = emp.float().argmax(0)                                # first empty entry per column
                    cc = has.nonzero().squeeze(1)
                    cent[j0[cc], cc] = val[cc]
        if final == "reassign" or idx is None:
            idx, _ = assign(cent)
        deq[:, c0:c0 + n] = torch.gather(cent, 0, idx)
        counts_all[:, c0:c0 + n] = torch.stack([(idx == j).sum(0) for j in range(k)])
        del idx, W
    return deq, counts_all


def codebook_metrics(counts, IC):
    p = counts.double() / IC
    used = counts > 0
    ent = -(torch.where(used, p * torch.log2(torch.where(used, p, torch.ones_like(p))), torch.zeros_like(p))).sum(0)
    dead = (~used).sum(0)
    return dict(dead_slots=int(dead.sum()), slots=int(dead.numel() * counts.shape[0]),
                cols_any_dead=int((dead > 0).sum()), cols=int(dead.numel()), entropy_sum=float(ent.sum()))


@torch.no_grad()
def window_nlls(model, enc):
    n = enc.numel() // SEQLEN
    lf = nn.CrossEntropyLoss(reduction="sum")
    out = []
    for i in range(n):
        ids = enc[:, i * SEQLEN:(i + 1) * SEQLEN].to(model.device)
        lg = model(ids).logits
        out.append(lf(lg[:, :-1, :].reshape(-1, lg.size(-1)).float(), ids[:, 1:].reshape(-1)).item())
    return out


def ppl_both(nlls):
    n = len(nlls)
    return math.exp(sum(nlls[:CUSTOM]) / (CUSTOM * (SEQLEN - 1))), math.exp(sum(nlls) / (n * SEQLEN)), n


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda").eval()
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    enc_tr = tok("\n\n".join(train["text"]), return_tensors="pt").input_ids
    RES["meta"].update(test_tokens=int(enc.numel()), train_tokens=int(enc_tr.numel()), torch=torch.__version__,
                       gpu=torch.cuda.get_device_name(0))
    log("test tokens", enc.numel(), "train tokens", enc_tr.numel())

    lins = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and any(t in n for t in TARGETS)}
    assert len(lins) == 224, len(lins)

    # calibration statistics of the fp16 model's linear inputs
    stats = {key: {} for key in ("abs_test8", "abs_train8", "sq_train")}
    mode = {"keys": ()}

    def mk(name):
        def hook(mod, inp, out):
            x16 = inp[0].detach().reshape(-1, inp[0].shape[-1])
            for key in mode["keys"]:
                if key.startswith("abs"):
                    v = x16.abs().sum(0).float()        # fp16 per-window sum, exactly as llama_calib.py
                else:
                    xf = x16.float()
                    v = (xf * xf).sum(0)                # x^2 in fp32 (fp16 can overflow)
                stats[key][name] = stats[key].get(name, 0) + v
        return hook
    hooks = [m.register_forward_hook(mk(n)) for n, m in lins.items()]
    with torch.no_grad():
        mode["keys"] = ("abs_test8",)
        for i in range(8):
            model(enc[:, i * SEQLEN:(i + 1) * SEQLEN].cuda())
        for i in range(max(8, CALIB_N)):
            mode["keys"] = (("abs_train8", "sq_train") if i < 8 else ("sq_train",)) if i < CALIB_N else ("abs_train8",)
            model(enc_tr[:, i * SEQLEN:(i + 1) * SEQLEN].cuda())
    for h in hooks:
        h.remove()
    log("calibration statistics collected (8 test windows, %d train windows)" % CALIB_N)

    orig = {n: m.weight.data.detach().to("cpu", copy=True) for n, m in lins.items()}
    head_orig = model.lm_head.weight.data.detach().to("cpu", copy=True)
    log("original weights saved on the host")

    t = time.time()
    nl = window_nlls(model, enc)
    c, f, n = ppl_both(nl)
    RES["fp16"] = dict(ppl_custom=c, ppl_full=f, n_windows=n, s=round(time.time() - t, 1))
    log("fp16: custom %.4f  full %.4f" % (c, f))
    save()

    for key in [a.strip() for a in ARMS_ENV.split(",") if a.strip()]:
        cfg = ARMS[key]
        try:
            t = time.time()
            agg = dict(dead_slots=0, slots=0, cols_any_dead=0, cols=0, entropy_sum=0.0)
            with torch.no_grad():
                for name, mod in lins.items():
                    Wt = orig[name].to("cuda").t().float().contiguous()       # (IC, OC)
                    s = None
                    if cfg["s"] is not None:
                        v = stats[cfg["s"]][name].to(Wt.device)
                        s = v / v.sum().clamp(min=1e-30) * Wt.shape[0]
                    ch = cfg.get("chunk", 2048) or Wt.shape[1]
                    deq, counts = kmeans_cols(Wt, s, cfg["init"], cfg["empty"], cfg["iters"], cfg["final"], chunk=ch)
                    mod.weight.data = deq.t().contiguous().to(torch.float16)
                    mm = codebook_metrics(counts, Wt.shape[0])
                    for kk in agg:
                        agg[kk] += mm[kk]
                    del Wt, deq, counts
                if cfg.get("head"):
                    # the exporter quantizes the LM head too, in chunks of 16384 output columns
                    Wt = head_orig.to("cuda").t().float().contiguous()
                    parts = []
                    for c0 in range(0, Wt.shape[1], 16384):
                        d, _ = kmeans_cols(Wt[:, c0:c0 + 16384].contiguous(), None, cfg["init"], cfg["empty"], cfg["iters"], cfg["final"], chunk=16384)
                        parts.append(d)
                    model.lm_head.weight.data = torch.cat(parts, 1).t().contiguous().to(torch.float16)
                    del Wt, parts
            torch.cuda.empty_cache()
            tq = time.time() - t
            nl = window_nlls(model, enc)
            c, f, n = ppl_both(nl)
            RES["arms"][key] = dict(cfg, ppl_custom=c, ppl_full=f, quantize_s=round(tq, 1), s_total=round(time.time() - t, 1),
                                    dead_pct=100.0 * agg["dead_slots"] / agg["slots"],
                                    cols_any_dead_pct=100.0 * agg["cols_any_dead"] / agg["cols"],
                                    index_entropy=agg["entropy_sum"] / agg["cols"])
            log("%-4s %-46s custom %.4f  full %.4f  dead %.3f%%  cols_dead %.2f%%  H %.3f" % (
                key, cfg["desc"], c, f, RES["arms"][key]["dead_pct"], RES["arms"][key]["cols_any_dead_pct"], RES["arms"][key]["index_entropy"]))
        except Exception as e:
            import traceback
            RES["errors"].append("%s: %s" % (key, traceback.format_exc()[-1200:]))
            log("ERROR", key, repr(e))
        finally:
            with torch.no_grad():
                for name, mod in lins.items():
                    mod.weight.data = orig[name].to("cuda")
                model.lm_head.weight.data = head_orig.to("cuda")
            save()

    arms = RES["arms"]
    need = ["A2", "A3", "A4", "A5"]
    missing = [a for a in need if a not in arms]
    new = {k: arms[k]["ppl_custom"] for k in need if k in arms}
    best = min(new, key=new.get) if new else None
    fp = RES.get("fp16", {}).get("ppl_custom")
    checks = {"fp16_5.83": fp, "A0m_6.34": arms.get("A0m", {}).get("ppl_custom"), "A1_6.17": arms.get("A1", {}).get("ppl_custom")}
    targets = {"fp16_5.83": 5.83, "A0m_6.34": 6.34, "A1_6.17": 6.17}
    repro = {k: (v is not None and abs(v - targets[k]) < 0.006) for k, v in checks.items()}
    repro_ok = all(repro.values())
    if missing:
        verdict = "INCOMPLETE"
    else:
        verdict = "GO" if new[best] <= 6.07 else ("NO-GO" if all(v > 6.14 for v in new.values()) else "BETWEEN")
        if not repro_ok:
            verdict += " (UNVERIFIED BASELINE)"
    base_t = arms.get("A1t", {}).get("ppl_custom")
    RES["verdict"] = dict(verdict=verdict, best_arm=best, best_ppl_custom=new.get(best) if best else None,
                          missing_arms=missing, errors=len(RES["errors"]), reproduction=checks, reproduction_ok=repro,
                          delta_best_vs_A1t=(new[best] - base_t) if (best and base_t is not None) else None,
                          rule="GO: a new arm (A2-A5) at or below 6.07 in the 30-window harness at 4 bits; NO-GO: all above 6.14; "
                               "INCOMPLETE unless A2-A5 all ran; UNVERIFIED if fp16, A0m or A1 miss 5.83/6.34/6.17 by 0.006 or more")
    RES["meta"]["wall_s"] = round(time.time() - T0, 1)
    save()
    log("VERDICT", json.dumps(RES["verdict"]))


if __name__ == "__main__":
    main()
