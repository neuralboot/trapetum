#!/usr/bin/env python3
"""
Export a HuggingFace Llama model to the Rust runtime's .cbk format: the seven projections
of every layer are quantized to a 4-bit per-output-column codebook (K=16), the LM head to
an 8-bit (K=256, default) or 4-bit codebook, embedding and norms stay dense.
Also produces a reference for the Rust runtime to check against: the same quantized
weights are dequantized back into the HF model, run on a prompt, and the per-position
logits + greedy continuation are saved.

k-means recipes (--recipe):
  weighted (default)  mid-grid start, an empty centroid moves to 0, 10 iterations, and every
                      centroid update weighted by E[x^2] of its input channel, measured on the
                      fp16 model over --calib-n windows of 2048 tokens of the wikitext-2 train
                      split. Arm A5 of model/exp_kmeans_fix.py: Llama-2-7B wikitext-2 PPL 6.098
                      in the 30-window harness (fp16 5.833), LM head left in fp16.
  legacy              min-max linear start, an empty centroid stays where it is, 12 iterations,
                      unweighted: the recipe of every file exported before 2026-11-05 (same
                      harness: 6.494 with the LM head in fp16, 6.792 with its 4-bit head).

LM head (--lm-k): auto (default) picks 256, an 8-bit head, when this machine's GPU can run the
runtime's 8-bit CUDA kernel (69,632 B of shared memory per block: compute capability 7.0 or 8.0
and up), and 16, a 4-bit head, otherwise (Turing, Pascal). On Llama-2-7B the 4-bit head costs
+0.30 PPL; the 8-bit one +0.0004. A file with an 8-bit head refuses to load on such GPUs.

File format: CBK3 when the LM head is 4-bit (--lm-k 16). With the 8-bit head (--lm-k 256) the
file is CBK4 with avq_m = 0: the CBK3 header plus two i32 (avq_m = 0, lm_k = 256) after the
RoPE frequencies, then the CBK3 body unchanged except the LM head, stored as one uint8 index
per weight ([hidden][vocab] bytes) and a [256][vocab] f32 codebook. Runtimes older than this
exporter reject that file (CBK4 used to mean additive projections only). Batched (speculative)
decoding of an 8-bit head re-reads it once per row on CUDA and is not implemented on Metal:
export with --lm-k 16 for speculative decoding on Metal.
Without the `datasets` package or network access the weighted recipe runs without its row
weights (a printed WARNING says so). For models with tied embeddings, the reference (ref.bin,
cont.bin) now keeps the input embedding in fp16, as the file and the runtime do; older
references of such models used the quantized head as their input embedding.

Outputs (in --out dir): model.cbk, prompt.bin (i32), ref.bin (f32 P*vocab), cont.bin (i32).

Usage:
  python export_runtime.py --model NousResearch/Llama-2-7b-hf --out /root/cbk \
      --prompt "The capital of France is" --gen 16
  # the recipe and format of the files exported before 2026-11-05 (CBK3, 4-bit head):
  python export_runtime.py --model NousResearch/Llama-2-7b-hf --out /root/cbk_legacy \
      --recipe legacy --lm-k 16
"""
import argparse, os, struct, sys
import numpy as np
import torch

K = 16
# projections that get an E[x^2] row weight (with the LM head) under the weighted recipe
CALIB_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@torch.no_grad()
def kmeans_cols(Wt, k=K, iters=12):
    # Wt: [in, out] float on cuda. Cluster each column into `k` centroids (1-D k-means).
    inn, out = Wt.shape
    lo = Wt.min(0).values
    hi = Wt.max(0).values
    centroids = torch.stack([lo + (hi - lo) * (j / (k - 1)) for j in range(k)], 0)  # [k, out]
    best_k = torch.zeros(inn, out, dtype=torch.long, device=Wt.device)
    for _ in range(iters):
        best_d = torch.full((inn, out), float("inf"), device=Wt.device)
        for j in range(k):
            d = (Wt - centroids[j:j + 1, :]) ** 2
            better = d < best_d
            best_k = torch.where(better, torch.full_like(best_k, j), best_k)
            best_d = torch.where(better, d, best_d)
        for j in range(k):
            msk = (best_k == j).float()
            cnt = msk.sum(0)
            newc = (Wt * msk).sum(0) / cnt.clamp_min(1.0)
            centroids[j] = torch.where(cnt > 0, newc, centroids[j])
    return centroids, best_k  # [k,out], [in,out]


@torch.no_grad()
def kmeans_cols_weighted(Wt, row_w=None, k=K, iters=10, chunk=2048):
    # The "weighted" recipe: per-output-column 1-D k-means, operation for operation the arm A5
    # of model/exp_kmeans_fix.py (init "mid", empty "zero", final "stale", chunks of 2048):
    #  - start on a mid-grid between the column's min and max, lo + (hi - lo) * (j + 0.5) / k, so a
    #    centroid stays near each extreme, where outliers such as super weights sit;
    #  - update every centroid to the mean of its weights weighted by row_w (one weight per input
    #    row, E[x^2] of that input channel; None = plain mean); a centroid that received no weight
    #    moves to exactly 0, back into the bulk;
    #  - keep the indices of the last assignment, made before the last update, as A5 did.
    # Wt: [in, out] f32 cuda; row_w: [in] f32 cuda, normalized to mean 1, or None. Columns are
    # independent, so chunking over output columns only bounds memory.
    # Returns centroids [k, out] f32 and indices [in, out] long, like kmeans_cols.
    assert iters >= 1
    inn, out = Wt.shape
    cb_all = torch.empty(k, out, dtype=Wt.dtype, device=Wt.device)
    idx_all = torch.empty(inn, out, dtype=torch.long, device=Wt.device)
    ar = torch.arange(k, device=Wt.device, dtype=torch.float32)
    sw = None if row_w is None else row_w.view(inn, 1)
    floor = 1e-6 if sw is not None else 1.0
    for c0 in range(0, out, chunk):
        W = Wt[:, c0:c0 + chunk]
        n = W.shape[1]
        lo, hi = W.min(0).values, W.max(0).values
        cent = lo.unsqueeze(0) + (hi - lo).unsqueeze(0) * ((ar + 0.5) / k).unsqueeze(1)  # [k, n]
        idx = None
        for _ in range(iters):
            best = torch.full_like(W, float("inf"))
            idx = torch.zeros(W.shape, dtype=torch.long, device=W.device)
            for j in range(k):
                d = (W - cent[j:j + 1]) ** 2
                better = d < best
                idx = torch.where(better, torch.full_like(idx, j), idx)
                best = torch.where(better, d, best)
            for j in range(k):
                m = (idx == j).float()
                mw = m if sw is None else m * sw
                cent[j] = (W * mw).sum(0) / mw.sum(0).clamp(min=floor)
        cb_all[:, c0:c0 + n] = cent
        idx_all[:, c0:c0 + n] = idx
        del W, best, idx, cent
    return cb_all, idx_all  # [k,out], [in,out]


def _quant_block(Wt, k=K, recipe="legacy", row_w=None):
    # Wt: [in, c] f32 on cuda. Returns packed u8, cb [k,c] f32, Wt_dq [c,in] half (cpu).
    # k=16 -> nibble-packed [in, c/2] (two indices/byte). k=256 -> uint8 indices [in, c] (one/byte).
    if recipe == "legacy":
        cb, idx = kmeans_cols(Wt, k)                        # [k,c], [in,c]
    elif recipe == "weighted":
        cb, idx = kmeans_cols_weighted(Wt, row_w, k)
    else:
        raise ValueError("unknown k-means recipe %r" % (recipe,))
    Wt_dq = torch.gather(cb, 0, idx)                        # [in,c]
    idxu = idx.to(torch.uint8)
    if k == 256:
        packed = idxu.contiguous()                         # [in, c] one uint8 index per element
    else:
        packed = (idxu[:, 0::2] | (idxu[:, 1::2] << 4)).contiguous()  # [in, c/2] nibble pairs
    return packed.cpu().numpy(), cb.cpu().numpy().astype(np.float32), Wt_dq.t().contiguous().half().cpu()


@torch.no_grad()
def quantize(weight, chunk=16384, k=K, recipe="legacy", row_w=None):
    # weight: nn.Linear.weight [out, in]. Returns packed u8, cb [k,out] f32, W_dq [out,in].
    # k=16: packed [in,out/2] (gemv4). k=256: packed [in,out] uint8 indices (gemv8, S19 mixed precision).
    # recipe: "legacy" by default here, so the exporters that import this function (export_aqlm,
    # export_deepseek, export_gemma, dump_layers_hf) keep their exact output; main() below passes
    # --recipe. row_w: [in] row weights for the "weighted" recipe (None = unweighted).
    assert k in (16, 256), "quantize supports K=16 (4-bit) and K=256 (8-bit)"
    if row_w is not None:
        assert recipe == "weighted", "row weights only apply to the weighted recipe"
        assert row_w.numel() == weight.shape[1], "one row weight per input feature"
        row_w = row_w.detach().float().cuda().contiguous()
    out_dim = weight.shape[0]
    if out_dim <= 20000:                                   # transformer layers: fast single-shot GPU path
        return _quant_block(weight.t().contiguous().float().cuda(), k, recipe, row_w)
    # large LM head (big vocab): chunk over output columns so the k-means workspace stays bounded.
    # chunk must be even so the K=16 nibble-pairing (cols 2j, 2j+1) never straddles a boundary.
    # The row weights belong to the input rows, so every column chunk uses all of them.
    # [in, out] f32 on CPU, fed to the GPU per chunk; moved to the CPU first, so the transposed f32
    # copy (4.2 GB for a 128,256 x 8,192 head) never sits on the GPU (same values: both steps are exact)
    Wt_full = weight.detach().cpu().t().contiguous().float()
    packs, cbs, dqs = [], [], []
    for s in range(0, out_dim, chunk):
        p, c, d = _quant_block(Wt_full[:, s:s + chunk].cuda(), k, recipe, row_w)
        packs.append(p); cbs.append(c); dqs.append(d)
        torch.cuda.empty_cache()
    return (np.concatenate(packs, axis=1), np.concatenate(cbs, axis=1), torch.cat(dqs, 0))


def w_f32(f, t):
    f.write(t.detach().float().cpu().numpy().astype("<f4").tobytes())


def w_f16(f, t):
    f.write(t.detach().half().cpu().numpy().astype("<f2").tobytes())


def gemv8_supported(device=0):
    # The runtime's 8-bit (K=256) CUDA kernel stages a 256 x 128 fp16 codebook and an 8 x 128 f32
    # reduction in shared memory: 69,632 B per block. Turing (compute capability 7.5: 65,536 B) and
    # older GPUs cannot run it (the runtime refuses such a model at load time); Volta 7.0 and every
    # GPU from 8.0 on can. Metal needs only 17 KB and is always fine.
    p = torch.cuda.get_device_properties(device)
    optin = getattr(p, "shared_memory_per_block_optin", None)
    if optin:
        return optin >= 69632
    return (p.major, p.minor) == (7, 0) or p.major >= 8


@torch.no_grad()
def sq_stats(model, tok, n_windows, seqlen=2048):
    # Sum over tokens of x^2 (fp32) of the input of every projection and of the LM head, from the
    # fp16 model, on the first n_windows windows of seqlen tokens of the wikitext-2 train split:
    # the text, windows and arithmetic of the "sq_train" statistics of model/exp_kmeans_fix.py.
    # Returns {module name: [in] f32 cuda}. Must run before any weight is replaced.
    from datasets import load_dataset
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(train["text"]), return_tensors="pt").input_ids
    if enc.numel() < n_windows * seqlen:
        raise SystemExit("calibration: %d train tokens, %d windows of %d need more" % (enc.numel(), n_windows, seqlen))
    mods = {n: m for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and (n == "lm_head" or any(t in n for t in CALIB_TARGETS))}
    stats = {}

    def mk(name):
        def hook(mod, inp, out):
            xf = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            v = (xf * xf).sum(0)
            stats[name] = stats[name] + v if name in stats else v
        return hook
    hooks = [m.register_forward_hook(mk(n)) for n, m in mods.items()]
    try:
        for i in range(n_windows):
            model(enc[:, i * seqlen:(i + 1) * seqlen].to(model.device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--out", default="/root/cbk")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--recipe", choices=("weighted", "legacy"), default="weighted",
                    help="k-means recipe (see the module docstring); legacy = files exported before 2026-11-05")
    ap.add_argument("--calib-n", type=int, default=128,
                    help="wikitext-2 train windows of 2048 tokens for the E[x^2] row weights (weighted recipe; 0 = unweighted)")
    ap.add_argument("--lm-k", choices=("auto", "16", "256"), default="auto",
                    help="LM head codebook size: 256 (8-bit head, CBK4 file with avq_m = 0), 16 (4-bit head, CBK3 "
                         "file) or auto (default): 256 when this machine's GPU can run the runtime's 8-bit kernel, else 16")
    args = ap.parse_args()
    if args.lm_k == "auto":
        args.lm_k = 256 if gemv8_supported() else 16
        print(f"LM head: --lm-k auto -> {args.lm_k} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        args.lm_k = int(args.lm_k)
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading", args.model, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda"
    )  # load straight onto the GPU (24 GB) to avoid CPU-RAM OOM on small boxes
    cfg = model.config
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = hidden // n_heads
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size
    pad_vocab = ((vocab + 255) // 256) * 256  # kernel tiles quantized output in blocks of 256
    n_layers = cfg.num_hidden_layers
    eps = cfg.rms_norm_eps
    base = float(getattr(cfg, "rope_theta", 10000.0))
    print(f"config: L={n_layers} hidden={hidden} heads={n_heads}/{n_kv} hd={head_dim} inter={inter} vocab={vocab}", flush=True)

    # E[x^2] row weights, measured on the fp16 model before any weight is quantized
    stats = {}
    if args.recipe == "weighted" and args.calib_n > 0:
        import time as _time
        t_cal = _time.time()
        try:
            stats = sq_stats(model, tok, args.calib_n)
            print(f"calibration: E[x^2] of {len(stats)} linear inputs over {args.calib_n} wikitext-2 train windows "
                  f"of 2048 tokens ({_time.time() - t_cal:.0f}s)", flush=True)
        except Exception as e:  # no `datasets` package or no network: export anyway, without row weights
            stats = {}
            print(f"WARNING: no calibration data ({type(e).__name__}: {e}). The weighted recipe runs without its "
                  f"E[x^2] row weights (Llama-2-7B: about 6.34 instead of 6.10 PPL). Install `datasets` and allow "
                  f"network access to get them.", flush=True)
    names = {m: n for n, m in model.named_modules()}

    def row_weights(lin):
        if not stats:
            return None
        v = stats[names[lin]]                              # KeyError = a linear the calibration missed
        return v / v.sum().clamp(min=1e-30) * v.numel()    # normalized to mean 1, as exp_kmeans_fix.py
    print(f"recipe: {args.recipe}{' (E[x^2] row weights)' if stats else ''}, LM head K={args.lm_k}", flush=True)

    path = os.path.join(args.out, "model.cbk")
    f = open(path, "wb")
    import torch as _torch
    rs = getattr(cfg, "rope_scaling", None)
    rope_scale = 1.0
    if isinstance(rs, dict) and (rs.get("type") or rs.get("rope_type")) == "linear":
        rope_scale = float(rs.get("factor", 1.0))
    kvdim = n_kv * head_dim
    if kvdim % 256 != 0:
        raise SystemExit(f"INCOMPATIBLE: KV dim {kvdim} is not a multiple of 256 (GQA too small for the kernel)")
    has_bias = 1 if getattr(model.model.layers[0].self_attn.q_proj, "bias", None) is not None else 0
    # compute RoPE frequencies ourselves (robust across versions): default / linear / llama3
    import math
    rope_theta = 10000.0
    if isinstance(rs, dict) and rs.get("rope_theta"):
        rope_theta = float(rs["rope_theta"])
    else:
        try:
            rope_theta = float(cfg.rope_theta)
        except Exception:
            rope_theta = 10000.0
    inv_freq = 1.0 / (rope_theta ** (_torch.arange(0, head_dim, 2).float() / head_dim))
    rtype = (rs.get("rope_type") or rs.get("type")) if isinstance(rs, dict) else None
    if rtype == "llama3":
        factor = float(rs["factor"]); lo = float(rs["low_freq_factor"]); hi = float(rs["high_freq_factor"])
        old_len = float(rs["original_max_position_embeddings"])
        low_wl, high_wl = old_len / lo, old_len / hi
        wavelen = 2 * math.pi / inv_freq
        inv_l = _torch.where(wavelen > low_wl, inv_freq / factor, inv_freq)
        smooth = (old_len / wavelen - lo) / (hi - lo)
        smoothed = (1 - smooth) * inv_l / factor + smooth * inv_l
        is_med = (~(wavelen < high_wl)) & (~(wavelen > low_wl))
        inv_freq = _torch.where(is_med, smoothed, inv_l)
    elif rtype == "linear":
        inv_freq = inv_freq / float(rs.get("factor", 1.0))
    inv_freq = inv_freq.float().cpu().contiguous()
    base = rope_theta
    print(f"rope: theta={rope_theta} type={rtype} has_bias={has_bias} inv_freq[{inv_freq.numel()}]", flush=True)
    f.write(b"CBK3" if args.lm_k == 16 else b"CBK4")
    f.write(struct.pack("<7i", n_layers, hidden, n_heads, n_kv, head_dim, inter, pad_vocab))
    f.write(struct.pack("<3f", eps, base, rope_scale))
    f.write(struct.pack("<i", has_bias))
    f.write(inv_freq.numpy().astype("<f4").tobytes())
    if args.lm_k != 16:
        f.write(struct.pack("<2i", 0, args.lm_k))  # CBK4: avq_m = 0 (scalar 4-bit projections), lm_k
    def _pad_rows(w):
        if w.shape[0] >= pad_vocab:
            return w
        z = _torch.zeros(pad_vocab - w.shape[0], w.shape[1], dtype=w.dtype, device=w.device)
        return _torch.cat([w, z], 0)

    w_f16(f, _pad_rows(model.model.embed_tokens.weight))  # [pad_vocab, hidden]

    def quant_write(lin, bias=False):
        packed, cb, w_dq = quantize(lin.weight, recipe=args.recipe, row_w=row_weights(lin))
        f.write(packed.tobytes())
        f.write(cb.tobytes())
        if bias:
            b = lin.bias if getattr(lin, "bias", None) is not None else _torch.zeros(lin.weight.shape[0], device=lin.weight.device)
            f.write(b.detach().float().cpu().numpy().astype("<f4").tobytes())  # [out]
        lin.weight.data = w_dq.to(lin.weight.device)  # replace for the reference forward

    for li in range(n_layers):
        L = model.model.layers[li]
        w_f32(f, L.input_layernorm.weight)
        quant_write(L.self_attn.q_proj, bias=bool(has_bias))
        quant_write(L.self_attn.k_proj, bias=bool(has_bias))
        quant_write(L.self_attn.v_proj, bias=bool(has_bias))
        quant_write(L.self_attn.o_proj)
        w_f32(f, L.post_attention_layernorm.weight)
        quant_write(L.mlp.gate_proj)
        quant_write(L.mlp.up_proj)
        quant_write(L.mlp.down_proj)
        print(f"  layer {li+1}/{n_layers} quantized", flush=True)
    w_f32(f, model.model.norm.weight)
    # lm_head: pad output rows to pad_vocab for the kernel; keep the real vocab for the reference
    _packed, _cb, _wdq = quantize(_pad_rows(model.lm_head.weight), k=args.lm_k,
                                  recipe=args.recipe, row_w=row_weights(model.lm_head))
    f.write(_packed.tobytes())
    f.write(_cb.tobytes())
    # A new Parameter, not .data: with tied embeddings (e.g. Llama-3.2-1B) writing .data would also
    # quantize the input embedding of the reference, while the file and the runtime keep it fp16.
    model.lm_head.weight = torch.nn.Parameter(_wdq[:vocab].to(model.lm_head.weight.device), requires_grad=False)
    f.close()
    print("wrote", path, os.path.getsize(path) // (1024 * 1024), "MB", flush=True)

    # reference: run the dequantized model on the prompt, save logits + greedy continuation
    model = model.half().cuda().eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    print("prompt:", repr(args.prompt), "->", ids.tolist(), flush=True)
    with torch.no_grad():
        out = model(ids)
        logits = out.logits[0].detach().float().cpu().numpy().astype("<f4")  # [P, vocab]
        gen = model.generate(ids, max_new_tokens=args.gen, do_sample=False)
    cont = gen[0, ids.shape[1]:].cpu().numpy().astype("<i4")
    print("continuation:", repr(tok.decode(cont)), flush=True)

    ids.cpu().numpy().astype("<i4").tofile(os.path.join(args.out, "prompt.bin"))
    logits.tofile(os.path.join(args.out, "ref.bin"))
    cont.tofile(os.path.join(args.out, "cont.bin"))
    print("wrote prompt.bin ref.bin cont.bin", flush=True)


if __name__ == "__main__":
    main()
