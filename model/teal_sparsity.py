#!/usr/bin/env python3
"""
teal_sparsity: activation-sparsity thresholds (TEAL-style) for a 4-bit .cbk Llama model, and what
they cost in perplexity, for Trapetum's opt-in sparse decode (TRAPETUM_SPARSITY=<file>).

At decode time the runtime zeroes, in every layer, the inputs of four sites whose magnitude is
below a per-site threshold, and its 4-bit GEMVs skip the index rows of zero inputs, so fewer
weight bytes are read per token:
  qkv   the q/k/v input (the attention RMSNorm output)
  o     the o_proj input (the attention output)
  gu    the gate/up input (the MLP RMSNorm output)
  down  the down_proj input, SiLU(gate) * up
This script loads the .cbk FILE into the Hugging Face model (model/cbk_ppl.py's loader, file LM
head included), collects log-histograms of |x| at the four sites over wikitext-2 train windows,
and for each configuration writes the thresholds that zero the target fraction of each site's
inputs, rounded to fp16 (the precision of the runtime's comparison), to sparsity_<name>.bin. It
then measures, with the thresholds applied: the wikitext-2 perplexity in the repository's two
harnesses (as cbk_ppl.py), the realized sparsity on test windows, with --text-len the mean
negative log-likelihood of the first wikitext-2 test tokens (text.bin, which the runtime's
TRAPETUM_NLL=1 mode measures on its side), and with --ref-dir the prompt logits.

The first --dense-first positions of every sequence stay dense (default 1, the attention sink:
its normalized state is a few massive values and thousands of tiny ones that any threshold
zeroes; the runtime's SPARSE_DENSE_FIRST must be the same).

Configurations: --sparsities 0.25,0.4 gives uniform ones (every site at s, named by the
percentage); --configs "mixA:qkv=0.2,o=0.5,gu=0.2,down=0.5;mixAL:qkv=0.2,o=0.5,gu=0.2,down=0.5,dense=0+1+31"
gives per-site targets (a missing site stays dense) and optional layers kept dense.

Threshold file (little-endian): b"SPT1", i32 layers, i32 sites (4), f32 target sparsity (the
weight-byte-weighted mean over sites and layers), then per layer the f32 thresholds of qkv, o,
gu and down (0 = dense).

Usage:
  python model/teal_sparsity.py --cbk /root/cbk/model.cbk --model NousResearch/Llama-2-7b-hf \
      --out /root/sparsity --sparsities 0.25 [--configs ...] [--calib-windows 8] [--text-len 2048]
No em dashes or en dashes anywhere.
"""
import argparse, json, os, struct, sys, time
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbk_ppl import load_file_weights, window_nlls, ppl_custom, ppl_full, SEQLEN, CUSTOM, log  # noqa: E402

SITES = ("qkv", "o", "gu", "down")
NB, LO, HI = 6800, -12.0, 5.0   # log10 |x| histogram, 0.0025-decade bins (0.58% relative width)


def site_modules(layer):
    a, m = layer.self_attn, layer.mlp
    return {"qkv": [a.q_proj, a.k_proj, a.v_proj], "o": [a.o_proj], "gu": [m.gate_proj, m.up_proj], "down": [m.down_proj]}


@torch.no_grad()
def histograms(model, enc, n_windows):
    """Counts [layers, sites, NB] of log10 |x| at each site's input (the first module of the site)."""
    layers = model.model.layers
    H = torch.zeros(len(layers), len(SITES), NB, dtype=torch.float64, device=model.device)
    hooks = []
    for li, L in enumerate(layers):
        mods = site_modules(L)
        for si, name in enumerate(SITES):
            def pre(mod, args, li=li, si=si):
                lg = torch.log10(args[0].detach().abs().float().flatten().clamp_min(1e-12))
                H[li, si] += torch.histc(lg, bins=NB, min=LO, max=HI).double()
            hooks.append(mods[name][0].register_forward_pre_hook(pre))
    try:
        for i in range(n_windows):
            model(enc[:, i * SEQLEN:(i + 1) * SEQLEN].to(model.device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return H


def thresholds(H, s):
    """Per (layer, site): the upper edge of the first bin where the cumulative fraction reaches s,
    rounded to fp16. Inputs strictly below it are zeroed, a fraction of about s (one bin more)."""
    c = torch.cumsum(H, dim=-1)
    frac = c / c[..., -1:].clamp_min(1.0)
    target = torch.full(frac.shape[:-1] + (1,), float(s), dtype=frac.dtype, device=frac.device)
    idx = torch.searchsorted(frac.contiguous(), target).clamp(max=NB - 1).squeeze(-1)
    upper = LO + (HI - LO) * (idx.double() + 1.0) / NB
    return torch.pow(10.0, upper).float().half().float().cpu().numpy()       # [layers, sites]


def apply_thresholds(model, T, dense_first=1):
    """Forward pre-hooks that zero |x| < threshold at every module of every site (q, k, v share the
    qkv threshold; gate and up share gu), as the runtime does on the producing kernel. The first
    dense_first positions of the sequence are left untouched (the runtime runs them dense)."""
    hooks = []
    for li, L in enumerate(model.model.layers):
        mods = site_modules(L)
        for si, name in enumerate(SITES):
            thr = float(T[li, si])
            if thr <= 0.0:
                continue
            for m in mods[name]:
                def pre(mod, args, thr=thr):
                    x = args[0]
                    keep = x.abs() >= thr                                    # compared in fp16, like the runtime
                    if dense_first > 0:
                        keep[:, :dense_first] = True
                    return (x * keep,) + tuple(args[1:])
                hooks.append(m.register_forward_pre_hook(pre))
    return hooks


def parse_configs(sparsities, configs):
    """[(name, {site: target}, dense layers)] from --sparsities (uniform) and --configs."""
    out = []
    for s in [float(x) for x in sparsities.split(",") if x.strip()]:
        assert 0.0 < s < 1.0, "sparsities must be in (0, 1)"
        out.append(("%02d" % round(s * 100), {n: s for n in SITES}, []))
    for c in [c.strip() for c in (configs or "").split(";") if c.strip()]:
        name, _, spec = c.partition(":")
        targets, dense = {}, []
        for kv in [x.strip() for x in spec.split(",") if x.strip()]:
            k, _, v = kv.partition("=")
            if k == "dense":
                dense = [int(x) for x in v.split("+") if x.strip()]
            else:
                assert k in SITES, "unknown site %r in %r" % (k, c)
                targets[k] = float(v)
                assert 0.0 <= targets[k] < 1.0, "site sparsity must be in [0, 1)"
        assert name and name.replace("_", "").isalnum(), "bad configuration name %r" % name
        out.append((name, targets, dense))
    names = [n for n, _, _ in out]
    assert len(set(names)) == len(names), "duplicate configuration names"
    return out


def config_thresholds(Hst, targets, dense):
    """[layers, sites] thresholds: each site at its own target, 0 for missing sites and dense layers."""
    T = np.zeros((Hst.shape[0], len(SITES)), dtype=np.float32)
    for si, name in enumerate(SITES):
        if targets.get(name, 0.0) > 0.0:
            T[:, si] = thresholds(Hst, targets[name])[:, si]
    for li in dense:
        T[li, :] = 0.0
    return T


def site_weights(cfg):
    """Weight count of each site's projections in one layer (what the skipped rows save)."""
    hd = cfg.hidden_size // cfg.num_attention_heads
    q, kv = cfg.num_attention_heads * hd, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * hd
    h, i = cfg.hidden_size, cfg.intermediate_size
    return np.array([h * (q + 2 * kv), q * h, h * 2 * i, i * h], dtype=np.float64)


@torch.no_grad()
def realized(model, enc, T, n_windows=4):
    """Mean fraction of zeroed inputs per site over the first test windows (thresholds applied)."""
    zeros = np.zeros(len(SITES)); total = np.zeros(len(SITES))
    hooks = []
    for li, L in enumerate(model.model.layers):
        mods = site_modules(L)
        for si, name in enumerate(SITES):
            def pre(mod, args, si=si):
                x = args[0]
                zeros[si] += float((x == 0).sum()); total[si] += x.numel()
            hooks.append(mods[name][0].register_forward_pre_hook(pre))   # runs after the threshold hooks: sees the zeroed input
    try:
        for i in range(n_windows):
            model(enc[:, i * SEQLEN:(i + 1) * SEQLEN].to(model.device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {n: float(zeros[i] / max(total[i], 1.0)) for i, n in enumerate(SITES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbk", required=True)
    ap.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sparsities", default="0.25", help="uniform configurations (every site at s)")
    ap.add_argument("--configs", default="", help='per-site configurations, "name:qkv=0.2,o=0.5,gu=0.2,down=0.5,dense=0+1+31;..."')
    ap.add_argument("--dense-first", type=int, default=1, help="positions kept dense (the runtime's SPARSE_DENSE_FIRST)")
    ap.add_argument("--calib-windows", type=int, default=8, help="wikitext-2 train windows of 2048 tokens")
    ap.add_argument("--text-len", type=int, default=0, help="write text.bin (first wikitext-2 test tokens) and its mean NLL per configuration")
    ap.add_argument("--ref-dir", default=None, help="exporter output dir with prompt.bin: write sparse prompt logits")
    ap.add_argument("--no-full", action="store_true", help="30-window harness only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    configs = parse_configs(args.sparsities, args.configs)
    assert configs, "no configuration: give --sparsities and/or --configs"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda").eval()
    dev = model.device
    H, head_file, checks = load_file_weights(model, args.cbk, dev)
    model.lm_head.weight = nn.Parameter(head_file, requires_grad=False)   # the file's LM head throughout
    ok = (checks["embedding_max_abs_diff"] == 0.0 and checks["norms_max_abs_diff"] == 0.0 and checks["trailing_bytes_none"])
    if not ok:
        raise SystemExit("integrity check failed: %r" % (checks,))
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    enc_tr = tok("\n\n".join(train["text"]), return_tensors="pt").input_ids
    ids = None
    if args.ref_dir:
        ids = torch.from_numpy(np.fromfile(os.path.join(args.ref_dir, "prompt.bin"), dtype="<i4").astype(np.int64)).view(1, -1).to(dev)
    text = None
    if args.text_len > 0:
        text = enc[:, :args.text_len].to(dev)                  # starts with BOS, like a runtime prompt
        text[0].cpu().numpy().astype("<i4").tofile(os.path.join(args.out, "text.bin"))
    wsite = site_weights(model.config)

    R = dict(cbk=os.path.abspath(args.cbk), model=args.model, calib_windows=args.calib_windows,
             dense_first=args.dense_first, text_len=args.text_len, results={})

    def evaluate(tag, T):
        hooks = apply_thresholds(model, T, args.dense_first) if T is not None else []
        try:
            t = time.time()
            nl = window_nlls(model, enc, CUSTOM if args.no_full else None)
            r = dict(ppl_custom=ppl_custom(nl), n_windows=len(nl), s=round(time.time() - t, 1))
            if not args.no_full:
                r["ppl_full"] = ppl_full(nl)
            if T is not None:
                r["realized_sparsity"] = realized(model, enc, T)
            if text is not None:
                with torch.no_grad():
                    lg = model(text).logits[0].float()
                r["text_nll"] = float(F.cross_entropy(lg[:-1], text[0, 1:], reduction="mean"))
            if ids is not None:
                with torch.no_grad():
                    lg = model(ids).logits[0].float().cpu().numpy().astype("<f4")
                name = "ref_dense.bin" if T is None else "ref_sparse_%s.bin" % tag
                lg.tofile(os.path.join(args.out, name))
        finally:
            for h in hooks:
                h.remove()
        R["results"][tag] = r
        log("TEAL %-6s custom %.4f  full %s  text_nll %s  realized %s" % (
            tag, r["ppl_custom"], ("%.4f" % r["ppl_full"]) if "ppl_full" in r else "n/a",
            ("%.6f" % r["text_nll"]) if "text_nll" in r else "n/a", json.dumps(r.get("realized_sparsity"))))
        json.dump(R, open(os.path.join(args.out, "teal.json"), "w"), indent=1)

    evaluate("dense", None)
    t = time.time()
    Hst = histograms(model, enc_tr, args.calib_windows)
    log("histograms of %d layers x %d sites over %d train windows (%.0fs)" % (Hst.shape[0], Hst.shape[1], args.calib_windows, time.time() - t))
    for name, targets, dense in configs:
        T = config_thresholds(Hst, targets, dense)
        nl = T.shape[0]
        per_layer = np.array([[targets.get(n, 0.0) for n in SITES]] * nl)
        per_layer[dense, :] = 0.0
        target = float((per_layer * wsite).sum() / (nl * wsite.sum()))     # weight-byte-weighted
        with open(os.path.join(args.out, "sparsity_%s.bin" % name), "wb") as f:
            f.write(b"SPT1" + struct.pack("<iif", nl, T.shape[1], target))
            f.write(T.astype("<f4").tobytes())
        R.setdefault("configs", {})[name] = dict(targets=targets, dense_layers=dense, byte_weighted_target=target,
                                                 thresholds={n: [float(T[:, i].min()), float(np.median(T[:, i])), float(T[:, i].max())]
                                                             for i, n in enumerate(SITES)})
        log("config %s: targets %s dense layers %s, weight-byte-weighted sparsity %.4f" % (name, json.dumps(targets), dense, target))
        evaluate(name, T)
    log("wrote %s" % os.path.join(args.out, "teal.json"))


if __name__ == "__main__":
    main()
