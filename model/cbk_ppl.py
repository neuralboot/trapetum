#!/usr/bin/env python3
"""
cbk_ppl: measure the wikitext-2 perplexity of a .cbk FILE, not of an exporter's in-memory
weights. It reads the 4-bit scalar projections and the scalar LM head stored in a CBK1, CBK2
or CBK3 file, or a CBK4 file with avq_m = 0 (scalar projections, 8-bit head), dequantizes
them (the centroid of every stored index, cast to fp16, exactly as the exporters build their
reference), loads them into the Hugging Face model and evaluates the two perplexity harnesses
of the repository in one pass (the harness of model/exp_kmeans_fix.py, unchanged):
  custom  first 30 windows of 2048 tokens, exp(NLL / (30 * 2047))   (Llama-2-7B fp16: 5.833)
  full    every window, exp(NLL / (n * 2048)), as bench_quant.py     (Llama-2-7B fp16: 5.468)

Integrity checks on the way: the file's config matches the model's, its embedding and norms
equal the model's, every padded vocabulary row is zero, nothing trails the LM head, and with
--ref-dir the dequantized model reproduces the exporter's reference logits (ref.bin: relative
L2 error at most 2e-3 and the same top-1 at every prompt position; not exact, because the
exporter's reference rounds the RoPE frequencies to fp16).
Additive (CBKA) projections are not handled: model/export_aqlm.py --ppl measures those.

Usage:
  python model/cbk_ppl.py --cbk /root/cbk/model.cbk --model NousResearch/Llama-2-7b-hf \
      [--ref-dir /root/cbk] [--heads file,fp16] [--fp16-control] [--skip-ppl] [--json out.json]
--heads: "file" = the file's LM head, "fp16" = the model's own head (isolates the head's cost).
The first listed head is measured on both harnesses, the others on the 30-window one only
(--full-all for both). Lines starting with "CBKPPL" summarize the results.
No em dashes or en dashes anywhere.
"""
import argparse, json, math, os, struct, time
import numpy as np
import torch, torch.nn as nn

SEQLEN, CUSTOM = 2048, 30
PROJS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"), ("self_attn", "o_proj"),
         ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def rd(f, dtype, count):
    a = np.fromfile(f, dtype=dtype, count=count)
    if a.size != count:
        raise SystemExit("truncated .cbk: wanted %d items of %s, got %d" % (count, dtype, a.size))
    return a


def read_header(f):
    magic = f.read(4)
    if magic not in (b"CBK1", b"CBK2", b"CBK3", b"CBK4"):
        raise SystemExit("not a .cbk file (magic %r)" % (magic,))
    v2, v4 = magic == b"CBK2", magic == b"CBK4"
    v3 = magic == b"CBK3" or v4
    n_layers, hidden, n_heads, n_kv, head_dim, inter, vocab = struct.unpack("<7i", f.read(28))
    eps, base = struct.unpack("<2f", f.read(8))
    scale = struct.unpack("<f", f.read(4))[0] if (v2 or v3) else 1.0
    has_bias = (struct.unpack("<i", f.read(4))[0] != 0) if v3 else False
    inv_freq = rd(f, "<f4", head_dim // 2) if v3 else None
    avq_m, lm_k = struct.unpack("<2i", f.read(8)) if v4 else (0, 16)
    return dict(format=magic.decode(), n_layers=n_layers, hidden=hidden, n_heads=n_heads, n_kv=n_kv,
                head_dim=head_dim, inter=inter, vocab=vocab, eps=eps, rope_base=base, rope_scale=scale,
                has_bias=has_bias, avq_m=avq_m, lm_k=lm_k, inv_freq=inv_freq)


def read_scalar(f, ic, oc, k, dev):
    """One scalar codebook matrix as the exporters write it: indices ([ic][oc/2] bytes, low nibble
    = even column, for k=16; [ic][oc] bytes for k=256), then the [k][oc] f32 codebook.
    Returns W [oc, ic] fp16 on dev with W[o, i] = codebook[index[i, o], o]."""
    if k == 16:
        p = torch.from_numpy(rd(f, np.uint8, ic * (oc // 2)).reshape(ic, oc // 2)).to(dev)
        idx = torch.empty(ic, oc, dtype=torch.long, device=dev)
        idx[:, 0::2] = (p & 0x0F).long()
        idx[:, 1::2] = (p >> 4).long()
        del p
    elif k == 256:
        idx = torch.from_numpy(rd(f, np.uint8, ic * oc).reshape(ic, oc)).to(dev).long()
    else:
        raise SystemExit("unsupported codebook size %d" % k)
    cb = torch.from_numpy(rd(f, "<f4", k * oc).reshape(k, oc)).to(dev)
    return torch.gather(cb, 0, idx).t().contiguous().half()


def maxdiff(a_np, t):
    return float(np.abs(a_np.astype(np.float32) - t.detach().float().cpu().numpy()).max())


@torch.no_grad()
def window_nlls(model, enc, n_max=None):
    n = enc.numel() // SEQLEN
    if n_max is not None:
        n = min(n, n_max)
    lf = nn.CrossEntropyLoss(reduction="sum")
    out = []
    for i in range(n):
        ids = enc[:, i * SEQLEN:(i + 1) * SEQLEN].to(model.device)
        lg = model(ids).logits
        out.append(lf(lg[:, :-1, :].reshape(-1, lg.size(-1)).float(), ids[:, 1:].reshape(-1)).item())
    return out


def ppl_custom(nlls):
    return math.exp(sum(nlls[:CUSTOM]) / (CUSTOM * (SEQLEN - 1)))


def ppl_full(nlls):
    return math.exp(sum(nlls) / (len(nlls) * SEQLEN))


def load_file_weights(model, path, dev):
    """Load the scalar projections of a .cbk FILE into a Hugging Face Llama-style model, in place,
    and return (header, the file's LM head as fp16 [real vocab, hidden] on dev, integrity checks).
    The model's embedding and norms are compared with the file's, not replaced."""
    cfg = model.config
    real_vocab = cfg.vocab_size
    checks = {}
    with open(path, "rb") as f:
        H = read_header(f)
        log("%s %s: %d layers, hidden %d, inter %d, vocab %d, avq_m %d, lm_k %d" % (
            os.path.basename(path), H["format"], H["n_layers"], H["hidden"], H["inter"], H["vocab"], H["avq_m"], H["lm_k"]))
        if H["avq_m"]:
            raise SystemExit("additive (CBKA) projections: measure them with model/export_aqlm.py --ppl")
        hidden, vocab = H["hidden"], H["vocab"]
        want = dict(n_layers=cfg.num_hidden_layers, hidden=cfg.hidden_size, n_heads=cfg.num_attention_heads,
                    n_kv=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
                    head_dim=cfg.hidden_size // cfg.num_attention_heads, inter=cfg.intermediate_size,
                    vocab=((real_vocab + 255) // 256) * 256)
        bad = {k: (H[k], v) for k, v in want.items() if H[k] != v}
        if bad:
            raise SystemExit("file config does not match the model (file, model): %r" % (bad,))

        emb = rd(f, "<f2", vocab * hidden).reshape(vocab, hidden)
        checks["embedding_max_abs_diff"] = maxdiff(emb[:real_vocab], model.model.embed_tokens.weight)
        checks["embedding_pad_rows_zero"] = bool(not np.any(emb[real_vocab:]))
        del emb
        norm_d, bias_d = 0.0, 0.0
        for li in range(H["n_layers"]):
            L = model.model.layers[li]
            norm_d = max(norm_d, maxdiff(rd(f, "<f4", hidden), L.input_layernorm.weight))
            for blk, pn in PROJS:
                if pn == "gate_proj":
                    norm_d = max(norm_d, maxdiff(rd(f, "<f4", hidden), L.post_attention_layernorm.weight))
                mod = getattr(getattr(L, blk), pn)
                mod.weight.data = read_scalar(f, mod.in_features, mod.out_features, 16, dev)
                if H["has_bias"] and pn in ("q_proj", "k_proj", "v_proj"):
                    b = rd(f, "<f4", mod.out_features)
                    ref_b = mod.bias if getattr(mod, "bias", None) is not None else torch.zeros(mod.out_features)
                    bias_d = max(bias_d, maxdiff(b, ref_b))
        norm_d = max(norm_d, maxdiff(rd(f, "<f4", hidden), model.model.norm.weight))
        head_file = read_scalar(f, hidden, vocab, H["lm_k"], dev)        # [vocab_padded, hidden]
        checks["head_pad_rows_zero"] = bool(not head_file[real_vocab:].any())
        head_file = head_file[:real_vocab].contiguous()
        checks["trailing_bytes_none"] = (f.read(1) == b"")
    checks["norms_max_abs_diff"] = norm_d
    if H["has_bias"]:
        checks["bias_max_abs_diff"] = bias_d
    return H, head_file, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbk", required=True)
    ap.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--ref-dir", default=None, help="exporter output dir with prompt.bin and ref.bin")
    ap.add_argument("--heads", default="file", help="comma list of file,fp16")
    ap.add_argument("--fp16-control", action="store_true", help="also measure the unquantized model first")
    ap.add_argument("--full-all", action="store_true", help="full harness for every head, not only the first")
    ap.add_argument("--skip-ppl", action="store_true", help="integrity checks only")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    assert heads and all(h in ("file", "fp16") for h in heads), "--heads takes file and/or fp16"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    R = dict(cbk=os.path.abspath(args.cbk), bytes=os.path.getsize(args.cbk), model=args.model, checks={}, ppl={})
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda").eval()
    dev = model.device
    cfg = model.config
    real_vocab = cfg.vocab_size

    enc = None
    if not args.skip_ppl:
        from datasets import load_dataset
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
        R["test_tokens"] = int(enc.numel())
        if args.fp16_control:
            nl = window_nlls(model, enc)
            R["ppl"]["fp16_model"] = dict(ppl_custom=ppl_custom(nl), ppl_full=ppl_full(nl), n_windows=len(nl))
            log("CBKPPL fp16 model: custom %.4f  full %.4f" % (R["ppl"]["fp16_model"]["ppl_custom"], R["ppl"]["fp16_model"]["ppl_full"]))

    head_fp16 = model.lm_head.weight.detach().clone()
    H, head_file, checks = load_file_weights(model, args.cbk, dev)
    R["header"] = {k: v for k, v in H.items() if k != "inv_freq"}
    R["checks"].update(checks)
    log("file read: projections and head dequantized (%.0fs)" % (time.time() - T0))

    def use_head(which):
        # a new Parameter, never .data: with tied embeddings .data would also change the input embedding
        model.lm_head.weight = nn.Parameter(head_file if which == "file" else head_fp16, requires_grad=False)

    if args.ref_dir:
        use_head("file")
        ids = np.fromfile(os.path.join(args.ref_dir, "prompt.bin"), dtype="<i4")
        ref = np.fromfile(os.path.join(args.ref_dir, "ref.bin"), dtype="<f4").reshape(ids.size, -1)
        with torch.no_grad():
            lg = model(torch.from_numpy(ids.astype(np.int64)).view(1, -1).to(dev)).logits[0].float().cpu().numpy()
        R["checks"]["ref_logits_max_abs_diff"] = float(np.abs(lg - ref).max())
        R["checks"]["ref_logits_rel_l2"] = float(np.linalg.norm(lg - ref) / max(np.linalg.norm(ref), 1e-30))
        R["checks"]["ref_top1_agree"] = "%d/%d" % (int((lg.argmax(-1) == ref.argmax(-1)).sum()), ids.size)
    c = R["checks"]
    ok = (c["embedding_max_abs_diff"] == 0.0 and c["embedding_pad_rows_zero"] and c["head_pad_rows_zero"]
          and c["trailing_bytes_none"] and c["norms_max_abs_diff"] == 0.0 and c.get("bias_max_abs_diff", 0.0) == 0.0)
    if args.ref_dir:
        # Not exact: the exporter builds its reference after model.half(), which also rounds the RoPE
        # frequency buffer to fp16 (relative logit difference about 5e-4 on Llama-2-7B). A layout error
        # gives errors of order 1, so a 2e-3 bound and full top-1 agreement separate the two cases.
        agree, total = (int(x) for x in c["ref_top1_agree"].split("/"))
        ok = ok and c["ref_logits_rel_l2"] <= 2e-3 and agree == total
    R["checks"]["integrity_ok"] = bool(ok)
    log("CBKPPL checks %s %s" % ("OK" if ok else "FAILED", json.dumps(c)))

    if not args.skip_ppl:
        for i, which in enumerate(heads):
            use_head(which)
            t = time.time()
            full = i == 0 or args.full_all
            nl = window_nlls(model, enc, None if full else CUSTOM)
            r = dict(ppl_custom=ppl_custom(nl), n_windows=len(nl), s=round(time.time() - t, 1))
            if full:
                r["ppl_full"] = ppl_full(nl)
            R["ppl"]["head_" + which] = r
            log("CBKPPL head %-4s custom %.4f  full %s" % (which, r["ppl_custom"], ("%.4f" % r["ppl_full"]) if full else "n/a"))
    R["seconds"] = round(time.time() - T0, 1)
    if args.json:
        json.dump(R, open(args.json, "w"), indent=1)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
