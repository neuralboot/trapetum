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
def kmeans_cols(Wt, iters=12):
    # Wt: [in, out] float on cuda. Cluster each column into K centroids (1-D k-means).
    inn, out = Wt.shape
    lo = Wt.min(0).values
    hi = Wt.max(0).values
    centroids = torch.stack([lo + (hi - lo) * (k / (K - 1)) for k in range(K)], 0)  # [K, out]
    best_k = torch.zeros(inn, out, dtype=torch.long, device=Wt.device)
    for _ in range(iters):
        best_d = torch.full((inn, out), float("inf"), device=Wt.device)
        for k in range(K):
            d = (Wt - centroids[k:k + 1, :]) ** 2
            better = d < best_d
            best_k = torch.where(better, torch.full_like(best_k, k), best_k)
            best_d = torch.where(better, d, best_d)
        for k in range(K):
            msk = (best_k == k).float()
            cnt = msk.sum(0)
            newc = (Wt * msk).sum(0) / cnt.clamp_min(1.0)
            centroids[k] = torch.where(cnt > 0, newc, centroids[k])
    return centroids, best_k  # [K,out], [in,out]


@torch.no_grad()
def quantize(weight):
    # weight: nn.Linear.weight [out, in]. Returns packed [in,out/2] u8, cb [K,out] f32, W_dq [out,in].
    Wt = weight.t().contiguous().float().cuda()             # [in, out]
    cb, idx = kmeans_cols(Wt)                               # [K,out], [in,out]
    Wt_dq = torch.gather(cb, 0, idx)                        # [in,out]
    idxu = idx.to(torch.uint8)
    packed = (idxu[:, 0::2] | (idxu[:, 1::2] << 4)).contiguous()  # [in, out/2]
    return (packed.cpu().numpy(), cb.cpu().numpy().astype(np.float32), Wt_dq.t().contiguous().half().cpu())


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
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    cfg = model.config
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = hidden // n_heads
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size
    n_layers = cfg.num_hidden_layers
    eps = cfg.rms_norm_eps
    base = float(getattr(cfg, "rope_theta", 10000.0))
    print(f"config: L={n_layers} hidden={hidden} heads={n_heads}/{n_kv} hd={head_dim} inter={inter} vocab={vocab}", flush=True)

    path = os.path.join(args.out, "model.cbk")
    f = open(path, "wb")
    f.write(b"CBK1")
    f.write(struct.pack("<7i", n_layers, hidden, n_heads, n_kv, head_dim, inter, vocab))
    f.write(struct.pack("<2f", eps, base))
    w_f16(f, model.model.embed_tokens.weight)  # [vocab, hidden]

    def quant_write(lin):
        packed, cb, w_dq = quantize(lin.weight)
        f.write(packed.tobytes())
        f.write(cb.tobytes())
        lin.weight.data = w_dq.to(lin.weight.device)  # replace for the reference forward

    for li in range(n_layers):
        L = model.model.layers[li]
        w_f32(f, L.input_layernorm.weight)
        quant_write(L.self_attn.q_proj)
        quant_write(L.self_attn.k_proj)
        quant_write(L.self_attn.v_proj)
        quant_write(L.self_attn.o_proj)
        w_f32(f, L.post_attention_layernorm.weight)
        quant_write(L.mlp.gate_proj)
        quant_write(L.mlp.up_proj)
        quant_write(L.mlp.down_proj)
        print(f"  layer {li+1}/{n_layers} quantized", flush=True)
    w_f32(f, model.model.norm.weight)
    quant_write(model.lm_head)
    f.close()
    print("wrote", path, os.path.getsize(path) // (1024 * 1024), "MB", flush=True)

    # reference: run the dequantized model on the prompt, save logits + greedy continuation
    model = model.half().cuda().eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    print("prompt:", repr(args.prompt), "->", ids.tolist(), flush=True)
    out = model(ids)
    logits = out.logits[0].float().cpu().numpy().astype("<f4")  # [P, vocab]
    gen = model.generate(ids, max_new_tokens=args.gen, do_sample=False)
    cont = gen[0, ids.shape[1]:].cpu().numpy().astype("<i4")
    print("continuation:", repr(tok.decode(cont)), flush=True)

    ids.cpu().numpy().astype("<i4").tofile(os.path.join(args.out, "prompt.bin"))
    logits.tofile(os.path.join(args.out, "ref.bin"))
    cont.tofile(os.path.join(args.out, "cont.bin"))
    print("wrote prompt.bin ref.bin cont.bin", flush=True)


if __name__ == "__main__":
    main()
