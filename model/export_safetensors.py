#!/usr/bin/env python3
"""
TORCH-FREE export: compress a Llama-architecture model to the runtime's CBK3 .cbk by reading
the .safetensors weights + config.json directly, with the k-means done in numpy. No torch, no
transformers -- the runtime is pure Rust, and now the compression pipeline needs neither either.
Kills the transformers/torch dependency-hell (tokenizers/protobuf/torchvision versions).

  python export_safetensors.py --dir /path/to/model_snapshot --out /path/out
The snapshot dir must contain config.json + model*.safetensors (+ index for sharded models).
"""
import argparse, glob, json, math, os, struct
import numpy as np

K = 16

# Minimal torch-free safetensors reader. numpy has no bfloat16, so we parse the
# container ourselves (8-byte header len + JSON header + raw bytes) and upcast
# BF16/F16 -> F32 by hand. No torch, no safetensors dep either.
_ST_DT = {"F32": np.float32, "F16": np.float16, "F64": np.float64,
          "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
          "U8": np.uint8, "BOOL": np.bool_}


def _read_safetensors(path):
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            shape = meta["shape"]; dt = meta["dtype"]
            if dt == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                arr = (u16 << 16).view(np.float32)
                # EXPORT_LOWMEM: keep weights in fp16 to halve host RAM (needed for 14B+ on
                # small-RAM pods). k-means upcasts per-matrix; weight precision impact is
                # negligible at 4-bit. quantize() casts back to fp32.
                if os.environ.get("EXPORT_LOWMEM") == "1":
                    arr = arr.astype(np.float16)
            else:
                arr = np.frombuffer(raw, dtype=_ST_DT[dt])
            out[name] = arr.reshape(shape) if shape else arr.reshape(())
    return out


def load_weights(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    w = {}
    if os.path.exists(idx):
        for shard in sorted(set(json.load(open(idx))["weight_map"].values())):
            w.update(_read_safetensors(os.path.join(d, shard)))
    else:
        for st in glob.glob(os.path.join(d, "*.safetensors")):
            w.update(_read_safetensors(st))
    return w


def _assign(Wt, centroids):
    # Wt [in,out], centroids [K,out] -> idx [in,out] uint8 (nearest per column).
    # Vectorized argmin over K via a single broadcast; column-chunked to bound memory.
    inn, out = Wt.shape
    idx = np.empty((inn, out), dtype=np.uint8)
    blk = max(1, (32 * 1024 * 1024) // (K * inn * 4))  # ~32MB working set per block
    for s in range(0, out, blk):
        W = Wt[:, s:s + blk]                       # [in, b]
        C = centroids[:, s:s + blk]                # [K, b]
        d = (W[None, :, :] - C[:, None, :]) ** 2    # [K, in, b]
        idx[:, s:s + blk] = d.argmin(0).astype(np.uint8)
    return idx


def kmeans_cols(Wt, iters=8):
    # Wt: [in, out] float32. Per-column 1-D k-means into K centroids. cb[K,out], idx[in,out].
    inn, out = Wt.shape
    lo, hi = Wt.min(0), Wt.max(0)
    centroids = np.stack([lo + (hi - lo) * (k / (K - 1)) for k in range(K)], 0)  # [K,out]
    idx = _assign(Wt, centroids)
    for _ in range(iters):
        for k in range(K):
            msk = (idx == k)
            cnt = msk.sum(0)
            newc = np.where(cnt > 0, (Wt * msk).sum(0) / np.maximum(cnt, 1.0), centroids[k])
            centroids[k] = newc
        idx = _assign(Wt, centroids)
    return centroids.astype(np.float32), idx


def kmeans_cols_torch(Wt_np, iters=8):
    # GPU k-means via torch ONLY (no transformers/tokenizers/safetensors) for large models.
    # Enabled with EXPORT_DEV=cuda; ~100x faster than CPU numpy. Same shape as kmeans_cols.
    # K-loop running-min assignment (no cdist), column-chunked to bound memory on big matrices.
    import torch
    dev = os.environ.get("EXPORT_DEV", "cuda")
    Wt = torch.from_numpy(Wt_np).to(dev)              # [in,out]
    inn, out = Wt.shape
    lo, hi = Wt.min(0).values, Wt.max(0).values
    C = torch.stack([lo + (hi - lo) * (k / (K - 1)) for k in range(K)], 0)  # [K,out]
    blk = max(1, (256 * 1024 * 1024) // (inn * 4))     # ~256MB/[in,blk] working set

    def assign():
        idx = torch.empty((inn, out), dtype=torch.uint8, device=dev)
        for s in range(0, out, blk):
            W = Wt[:, s:s + blk]
            best_d = torch.full_like(W, float("inf"))
            best_k = torch.zeros_like(W, dtype=torch.uint8)
            for k in range(K):
                d = (W - C[k, s:s + blk]) ** 2
                m = d < best_d
                best_k = torch.where(m, torch.full_like(best_k, k), best_k)
                best_d = torch.where(m, d, best_d)
            idx[:, s:s + blk] = best_k
        return idx

    idx = assign()
    for _ in range(iters):
        for k in range(K):
            msk = idx == k
            cnt = msk.sum(0)
            C[k] = torch.where(cnt > 0, (Wt * msk).sum(0) / cnt.clamp(min=1), C[k])
        idx = assign()
    return C.float().cpu().numpy(), idx.cpu().numpy()


def quantize(weight, chunk=16384):
    # weight [out,in] -> packed [in,out/2] u8, cb [K,out] f32
    Wt = np.ascontiguousarray(weight.T.astype(np.float32))  # [in,out]
    if os.environ.get("EXPORT_DEV", "").startswith("cuda"):
        cb, idx = kmeans_cols_torch(Wt)
    else:
        cb, idx = kmeans_cols(Wt)
    idxu = idx.astype(np.uint8)
    packed = (idxu[:, 0::2] | (idxu[:, 1::2] << 4))
    return np.ascontiguousarray(packed), np.ascontiguousarray(cb)


def wf16(f, a): f.write(a.astype("<f2").tobytes())
def wf32(f, a): f.write(a.astype("<f4").tobytes())
def qwrite(f, w):
    p, cb = quantize(w); f.write(np.ascontiguousarray(p, dtype=np.uint8).tobytes()); wf32(f, cb)


def split_fused(W, n_layers, n_heads, n_kv, head_dim):
    # Phi-3/Phi-4 (Phi3ForCausalLM) fuse qkv_proj and gate_up_proj into single matrices.
    # Split them into the standard q/k/v and gate/up keys the CBK3 writer expects. Lossless.
    if "model.layers.0.self_attn.qkv_proj.weight" not in W:
        return W
    for li in range(n_layers):
        P = f"model.layers.{li}."
        qkv = W.pop(P + "self_attn.qkv_proj.weight")
        q_r, kv_r = n_heads * head_dim, n_kv * head_dim
        W[P + "self_attn.q_proj.weight"] = qkv[:q_r]
        W[P + "self_attn.k_proj.weight"] = qkv[q_r:q_r + kv_r]
        W[P + "self_attn.v_proj.weight"] = qkv[q_r + kv_r:q_r + 2 * kv_r]
        gu = W.pop(P + "mlp.gate_up_proj.weight")
        inter = gu.shape[0] // 2
        W[P + "mlp.gate_proj.weight"] = gu[:inter]
        W[P + "mlp.up_proj.weight"] = gu[inter:]
    return W


def pad_rows(w, n):
    if w.shape[0] >= n: return w
    return np.concatenate([w, np.zeros((n - w.shape[0], w.shape[1]), w.dtype)], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    c = json.load(open(os.path.join(a.dir, "config.json")))
    hidden = c["hidden_size"]; n_heads = c["num_attention_heads"]
    n_kv = c.get("num_key_value_heads", n_heads); head_dim = hidden // n_heads
    inter = c["intermediate_size"]; vocab = c["vocab_size"]; n_layers = c["num_hidden_layers"]
    eps = c["rms_norm_eps"]; base = float(c.get("rope_theta", 10000.0))
    pad_vocab = ((vocab + 255) // 256) * 256
    W = load_weights(a.dir)
    W = split_fused(W, n_layers, n_heads, n_kv, head_dim)  # Phi-3/Phi-4 fused qkv/gate_up
    has_bias = 1 if "model.layers.0.self_attn.q_proj.bias" in W else 0
    # rope inv_freq (default / llama3 scaling)
    inv = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    rs = c.get("rope_scaling"); rope_scale = 1.0
    if isinstance(rs, dict) and (rs.get("rope_type") or rs.get("type")) == "llama3":
        factor = rs["factor"]; lo = rs["low_freq_factor"]; hi = rs["high_freq_factor"]
        old = rs["original_max_position_embeddings"]
        low_wl, high_wl = old / lo, old / hi; wl = 2 * math.pi / inv
        inv_l = np.where(wl > low_wl, inv / factor, inv)
        sm = (old / wl - lo) / (hi - lo); smoothed = (1 - sm) * inv_l / factor + sm * inv_l
        is_med = (~(wl < high_wl)) & (~(wl > low_wl)); inv = np.where(is_med, smoothed, inv_l)
    print(f"cfg L={n_layers} hidden={hidden} heads={n_heads}/{n_kv} inter={inter} vocab={vocab} has_bias={has_bias}", flush=True)

    f = open(os.path.join(a.out, "model.cbk"), "wb")
    f.write(b"CBK3")
    f.write(struct.pack("<7i", n_layers, hidden, n_heads, n_kv, head_dim, inter, pad_vocab))
    f.write(struct.pack("<3f", eps, base, rope_scale)); f.write(struct.pack("<i", has_bias))
    wf32(f, inv.astype(np.float32))
    wf16(f, pad_rows(W["model.embed_tokens.weight"], pad_vocab))
    for li in range(n_layers):
        P = f"model.layers.{li}."
        wf32(f, W[P + "input_layernorm.weight"])
        for proj in ("q", "k", "v", "o"):
            qwrite(f, W[P + f"self_attn.{proj}_proj.weight"])
            # CBK3 stores biases for q/k/v ONLY (o_proj has none in Qwen and the loader
            # reads none) — writing an o bias shifts every later byte and corrupts the model.
            if has_bias and proj != "o":
                b = W.get(P + f"self_attn.{proj}_proj.bias", np.zeros(W[P + f"self_attn.{proj}_proj.weight"].shape[0], np.float32))
                wf32(f, b)
        wf32(f, W[P + "post_attention_layernorm.weight"])
        qwrite(f, W[P + "mlp.gate_proj.weight"]); qwrite(f, W[P + "mlp.up_proj.weight"]); qwrite(f, W[P + "mlp.down_proj.weight"])
        f.flush(); print(f"  layer {li+1}/{n_layers}", flush=True)
    wf32(f, W["model.norm.weight"])
    lm = W.get("lm_head.weight", W["model.embed_tokens.weight"])  # tied fallback
    qwrite(f, pad_rows(lm, pad_vocab))
    f.close()
    print("wrote", os.path.join(a.out, "model.cbk"), os.path.getsize(os.path.join(a.out, "model.cbk")) // (1024*1024), "MB", flush=True)


if __name__ == "__main__":
    main()
