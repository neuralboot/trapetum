#!/usr/bin/env python3
"""
Export a HuggingFace Llama model to the Rust runtime's .cbk format: every linear is
quantized to a 4-bit per-output-column codebook (K=16), embedding and norms stay dense.
Also produces a reference for the Rust runtime to check against: the same quantized
weights are dequantized back into the HF model, run on a prompt, and the per-position
logits + greedy continuation are saved.

Outputs (in --out dir): model.cbk, prompt.bin (i32), ref.bin (f32 P*vocab), cont.bin (i32).

Usage:
  python export_runtime.py --model NousResearch/Llama-2-7b-hf --out /root/cbk \
      --prompt "The capital of France is" --gen 16
"""
import argparse, os, struct, sys
import numpy as np
import torch

K = 16


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


def _quant_block(Wt, k=K):
    # Wt: [in, c] f32 on cuda. Returns packed u8, cb [k,c] f32, Wt_dq [c,in] half (cpu).
    # k=16 -> nibble-packed [in, c/2] (two indices/byte). k=256 -> uint8 indices [in, c] (one/byte).
    cb, idx = kmeans_cols(Wt, k)                            # [k,c], [in,c]
    Wt_dq = torch.gather(cb, 0, idx)                        # [in,c]
    idxu = idx.to(torch.uint8)
    if k == 256:
        packed = idxu.contiguous()                         # [in, c] one uint8 index per element
    else:
        packed = (idxu[:, 0::2] | (idxu[:, 1::2] << 4)).contiguous()  # [in, c/2] nibble pairs
    return packed.cpu().numpy(), cb.cpu().numpy().astype(np.float32), Wt_dq.t().contiguous().half().cpu()


@torch.no_grad()
def quantize(weight, chunk=16384, k=K):
    # weight: nn.Linear.weight [out, in]. Returns packed u8, cb [k,out] f32, W_dq [out,in].
    # k=16: packed [in,out/2] (gemv4). k=256: packed [in,out] uint8 indices (gemv8, S19 mixed precision).
    assert k in (16, 256), "quantize supports K=16 (4-bit) and K=256 (8-bit)"
    out_dim = weight.shape[0]
    if out_dim <= 20000:                                   # transformer layers: fast single-shot GPU path
        return _quant_block(weight.t().contiguous().float().cuda(), k)
    # large LM head (big vocab): chunk over output columns so the k-means workspace stays bounded.
    # chunk must be even so the K=16 nibble-pairing (cols 2j, 2j+1) never straddles a boundary.
    Wt_full = weight.t().contiguous().float().cpu()        # [in, out] on CPU, fed to GPU per chunk
    packs, cbs, dqs = [], [], []
    for s in range(0, out_dim, chunk):
        p, c, d = _quant_block(Wt_full[:, s:s + chunk].cuda(), k)
        packs.append(p); cbs.append(c); dqs.append(d)
        torch.cuda.empty_cache()
    return (np.concatenate(packs, axis=1), np.concatenate(cbs, axis=1), torch.cat(dqs, 0))


def w_f32(f, t):
    f.write(t.detach().float().cpu().numpy().astype("<f4").tobytes())


def w_f16(f, t):
    f.write(t.detach().half().cpu().numpy().astype("<f2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--out", default="/root/cbk")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=16)
    args = ap.parse_args()
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
    f.write(b"CBK3")
    f.write(struct.pack("<7i", n_layers, hidden, n_heads, n_kv, head_dim, inter, pad_vocab))
    f.write(struct.pack("<3f", eps, base, rope_scale))
    f.write(struct.pack("<i", has_bias))
    f.write(inv_freq.numpy().astype("<f4").tobytes())
    def _pad_rows(w):
        if w.shape[0] >= pad_vocab:
            return w
        z = _torch.zeros(pad_vocab - w.shape[0], w.shape[1], dtype=w.dtype, device=w.device)
        return _torch.cat([w, z], 0)

    w_f16(f, _pad_rows(model.model.embed_tokens.weight))  # [pad_vocab, hidden]

    def quant_write(lin, bias=False):
        packed, cb, w_dq = quantize(lin.weight)
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
    _packed, _cb, _wdq = quantize(_pad_rows(model.lm_head.weight))
    f.write(_packed.tobytes())
    f.write(_cb.tobytes())
    model.lm_head.weight.data = _wdq[:vocab].to(model.lm_head.weight.device)
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
