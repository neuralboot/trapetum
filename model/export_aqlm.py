#!/usr/bin/env python3
"""
Export a HuggingFace AQLM checkpoint in the 2x8 (or 3x8) additive-codebook format, for example
ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf, to the Rust runtime's CBK4 container.

The seven per-layer projections are written as CBKA records (model/cbka_format.py): the AQLM
codes, codebooks and per-output scales are copied bit for bit, nothing is requantized, and the
runtime decodes them with AvqLinear (avq_gemv_t). Embedding and norms stay dense. AQLM keeps the
LM head in fp16; the runtime's LM head is a scalar codebook, so it is quantized here with
--lm-k 256 (8-bit, default, near lossless) or 16 (4-bit).

Reference for the runtime (same contract as export_runtime.py): the AQLM model itself
(transformers + aqlm) runs the prompt with its LM head replaced by the exact quantized head the
.cbk carries; the per-position logits and the greedy continuation are saved. --ppl N also prints
the wikitext-2 perplexity of the exported model over the first N windows of 2048 tokens
(N=30 is the repo's custom harness, see llama_aqlm.py).

Outputs (in --out dir): model.cbk, prompt.bin (i32), ref.bin (f32 P*pad_vocab), cont.bin (i32).

CBK4 layout (little-endian):
  magic "CBK4"
  7 x i32   n_layers, hidden, n_heads, n_kv, head_dim, inter, pad_vocab
  3 x f32   eps, rope_theta, rope_scale (informational; the reader uses inv_freq)
  i32       has_bias
  f32 x head_dim/2   inv_freq (RoPE frequencies, any scaling baked in, as in CBK3)
  i32       avq_m  (additive codebooks per projection: 2 for "2x8")
  i32       lm_k   (LM head scalar codebook size: 16 or 256)
  f16 x pad_vocab*hidden   embedding
  per layer: input_layernorm f32[hidden]; CBKA q [+ f32 q bias]; CBKA k [+ bias]; CBKA v [+ bias];
             CBKA o; post_attention_layernorm f32[hidden]; CBKA gate; CBKA up; CBKA down
  final norm f32[hidden]
  LM head: lm_k=16  -> u8[hidden*pad_vocab/2] nibble-packed [in][out/2], f32[16*pad_vocab]
           lm_k=256 -> u8[hidden*pad_vocab] [in][out], f32[256*pad_vocab]
The reader is Model::load_cbk / load_cbk4_body in runtime/src/lib.rs; keep the two in sync.

Usage:
  python model/export_aqlm.py --model ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf --out /root/cbk4 \
      --prompt "The capital of France is" --gen 16 --lm-k 256 --ppl 30
No em dashes or en dashes anywhere.
"""
import argparse, math, os, struct, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cbka_format import AVQ_D, AVQ_K, write_cbka_record  # noqa: E402  single source of truth for CBKA
from export_runtime import quantize  # noqa: E402  scalar codebook (LM head), same packing as CBK3


def w_f32(f, t):
    f.write(t.detach().float().cpu().numpy().astype("<f4").tobytes())


def w_f16(f, t):
    f.write(t.detach().half().cpu().numpy().astype("<f2").tobytes())


def pad_rows(w, n):
    if w.shape[0] >= n:
        return w
    z = torch.zeros(n - w.shape[0], w.shape[1], dtype=w.dtype, device=w.device)
    return torch.cat([w, z], 0)


def rope(cfg, head_dim):
    """RoPE inverse frequencies with the scaling baked in (default, linear, llama3), as in
    export_runtime.py. Returns (inv_freq [head_dim/2] f32 cpu, rope_theta, linear_scale, type)."""
    rs = getattr(cfg, "rope_scaling", None)
    theta = float(getattr(cfg, "rope_theta", 10000.0) or 10000.0)
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    rtype = (rs.get("rope_type") or rs.get("type")) if isinstance(rs, dict) else None
    lin = 1.0
    if rtype == "llama3":
        factor = float(rs["factor"]); lo = float(rs["low_freq_factor"]); hi = float(rs["high_freq_factor"])
        old_len = float(rs["original_max_position_embeddings"])
        low_wl, high_wl = old_len / lo, old_len / hi
        wavelen = 2 * math.pi / inv
        inv_l = torch.where(wavelen > low_wl, inv / factor, inv)
        smooth = (old_len / wavelen - lo) / (hi - lo)
        smoothed = (1 - smooth) * inv_l / factor + smooth * inv_l
        is_med = (~(wavelen < high_wl)) & (~(wavelen > low_wl))
        inv = torch.where(is_med, smoothed, inv_l)
    elif rtype == "linear":
        lin = float(rs.get("factor", 1.0))
        inv = inv / lin
    elif rtype not in (None, "default"):
        raise SystemExit(f"unsupported rope_scaling type {rtype!r}")
    return inv.float().cpu().contiguous(), theta, lin, rtype


def aqlm_parts(mod):
    """(codes [OC, ng, M] uint8, C [M, K, D] f32, scale [OC] f32, M) of an aqlm QuantizedLinear."""
    if type(mod).__name__ != "QuantizedLinear":
        raise SystemExit(f"expected an aqlm QuantizedLinear, got {type(mod).__name__} (not an AQLM checkpoint?)")
    cb = mod.codebooks.detach()
    M, K, og, ig = cb.shape
    if not (K == AVQ_K and og == 1 and ig == AVQ_D and M in (2, 3)):
        raise SystemExit(f"unsupported AQLM layout {tuple(cb.shape)}: the runtime decodes Mx8 codebooks of "
                         f"{AVQ_K} entries with in_group {AVQ_D}, out_group 1, M in (2, 3) (e.g. a 2x8 checkpoint)")
    if getattr(mod, "bias", None) is not None:
        raise SystemExit("AQLM linear with a bias is not supported by this exporter")
    codes = (mod.codes.detach().to(torch.int64) % 256).to(torch.uint8)  # aqlm stores 8-bit codes as int8
    OC, ng, Mc = codes.shape
    assert Mc == M, f"codes M {Mc} != codebooks M {M}"
    scale = mod.scales.detach().reshape(-1)
    assert scale.numel() == OC, f"scales {scale.numel()} != OC {OC}"
    return codes.cpu().numpy(), cb[:, :, 0, :].float().cpu().numpy(), scale.float().cpu().numpy(), M


def walk_cbk4(path):
    """Mirror of the Rust reader: walk the file record by record and require it to end exactly
    after the LM head. Catches writer bugs before the runtime ever loads the file."""
    with open(path, "rb") as f:
        assert f.read(4) == b"CBK4"
        L, hid, nh, nkv, hd, inter, pv = struct.unpack("<7i", f.read(28))
        f.read(12)
        has_bias = struct.unpack("<i", f.read(4))[0]
        f.read(4 * (hd // 2))
        m, lm_k = struct.unpack("<2i", f.read(8))
        f.seek(2 * pv * hid, 1)
        qd, kvd = nh * hd, nkv * hd

        def rec(rows, cols):
            assert f.read(4) == b"CBKA", "missing CBKA magic"
            mm, d, k, r, c = struct.unpack("<5i", f.read(20))
            assert (mm, d, k, r, c) == (m, AVQ_D, AVQ_K, rows, cols), ((mm, d, k, r, c), (m, rows, cols))
            f.seek(2 * mm * k * d + 2 * r + mm * (c // d) * r, 1)

        for _ in range(L):
            f.seek(4 * hid, 1)
            rec(qd, hid); f.seek(4 * qd * has_bias, 1)
            rec(kvd, hid); f.seek(4 * kvd * has_bias, 1)
            rec(kvd, hid); f.seek(4 * kvd * has_bias, 1)
            rec(hid, qd)
            f.seek(4 * hid, 1)
            rec(inter, hid); rec(inter, hid); rec(hid, inter)
        f.seek(4 * hid, 1)
        f.seek((hid * pv if lm_k == 256 else hid * pv // 2) + 4 * lm_k * pv, 1)
        end = f.tell()
        size = os.path.getsize(path)
        assert end == size, f"CBK4 walk ended at {end}, file is {size} bytes"
    return L, m, lm_k


@torch.no_grad()
def ppl_windows(model, tok, n_windows, seqlen=2048, window0_out=None):
    """wikitext-2 test PPL over the first n_windows windows of seqlen tokens. Also prints the
    running PPL after windows 1 and 30 (window 1 alone is what the runtime's TRAPETUM_NLL check
    compares against) and, with window0_out, writes window 0's token ids (i32) for that check."""
    from datasets import load_dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    n = min(n_windows, enc.size(1) // seqlen)
    if window0_out:
        enc[0, :seqlen].numpy().astype("<i4").tofile(window0_out)
        print(f"wrote {window0_out} ({seqlen} tokens of wikitext-2 test window 0)", flush=True)
    lf = torch.nn.CrossEntropyLoss(reduction="sum")
    tot, nt = 0.0, 0
    for i in range(n):
        ids = enc[:, i * seqlen:(i + 1) * seqlen].to(model.device)
        o = model(ids).logits
        tot += lf(o[:, :-1, :].reshape(-1, o.size(-1)).float(), ids[:, 1:].reshape(-1)).item()
        nt += ids[:, 1:].numel()
        if i + 1 in (1, 30) or (i + 1) % 10 == 0:
            print(f"  PPL_RUNNING windows={i + 1} mean_nll={tot / nt:.6f} ppl={math.exp(tot / nt):.4f}", flush=True)
    return math.exp(tot / nt), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf")
    ap.add_argument("--out", default="/root/cbk4")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--lm-k", type=int, default=256, choices=(16, 256))
    ap.add_argument("--ppl", type=int, default=0, help="also report wikitext-2 PPL over N windows of 2048")
    ap.add_argument("--ppl-tokenizer", default=None, help="tokenizer for --ppl (default: the model's)")
    ap.add_argument("--ppl-window0-out", default=None, help="with --ppl: write window 0's token ids (i32) here")
    ap.add_argument("--lm-chunk", type=int, default=16384,
                    help="LM-head quantization: output columns per GPU chunk (smaller = less GPU memory, same result)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading", args.model, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda").eval()
    cfg = model.config
    hidden, n_heads = cfg.hidden_size, cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_heads
    inter, vocab, n_layers, eps = cfg.intermediate_size, cfg.vocab_size, cfg.num_hidden_layers, cfg.rms_norm_eps
    pad_vocab = ((vocab + 255) // 256) * 256  # scalar LM-head kernels tile the output in blocks of 256
    inv_freq, theta, lin, rtype = rope(cfg, head_dim)
    for sub in ("model.rotary_emb", "model.layers.0.self_attn.rotary_emb"):  # cross-check against HF
        try:
            hf_inv = model.get_submodule(sub).inv_freq.detach().float().cpu()
        except AttributeError:
            continue
        if hf_inv.numel() == inv_freq.numel():
            d = ((hf_inv - inv_freq).abs().max() / hf_inv.abs().max()).item()
            print(f"rope inv_freq vs HF {sub}: max rel diff {d:.2e}", flush=True)
            if d > 1e-4:
                raise SystemExit("RoPE frequencies disagree with the HF model (rope_scaling handling?)")
        break
    M = aqlm_parts(model.model.layers[0].self_attn.q_proj)[3]
    print(f"config: L={n_layers} hidden={hidden} heads={n_heads}/{n_kv} hd={head_dim} inter={inter} "
          f"vocab={vocab} (pad {pad_vocab}) rope={rtype or 'default'} theta={theta} avq_m={M} lm_k={args.lm_k}", flush=True)

    path = os.path.join(args.out, "model.cbk")
    with open(path, "wb") as f:
        f.write(b"CBK4")
        f.write(struct.pack("<7i", n_layers, hidden, n_heads, n_kv, head_dim, inter, pad_vocab))
        f.write(struct.pack("<3f", eps, theta, lin))
        f.write(struct.pack("<i", 0))  # has_bias: Llama-family AQLM checkpoints have none (aqlm_parts refuses biases)
        f.write(inv_freq.numpy().astype("<f4").tobytes())
        f.write(struct.pack("<2i", M, args.lm_k))
        w_f16(f, pad_rows(model.model.embed_tokens.weight, pad_vocab))

        def rec(lin_mod):
            codes, C, scale, m = aqlm_parts(lin_mod)
            assert m == M, f"mixed codebook counts ({m} vs {M}) are not supported"
            write_cbka_record(f, codes, C, scale, M)

        for li, L in enumerate(model.model.layers):
            w_f32(f, L.input_layernorm.weight)
            for p in (L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj, L.self_attn.o_proj):
                rec(p)
            w_f32(f, L.post_attention_layernorm.weight)
            for p in (L.mlp.gate_proj, L.mlp.up_proj, L.mlp.down_proj):
                rec(p)
            if (li + 1) % 8 == 0 or li + 1 == n_layers:
                print(f"  layer {li + 1}/{n_layers} written", flush=True)
        w_f32(f, model.model.norm.weight)
        assert args.lm_chunk > 0 and args.lm_chunk % 2 == 0, "--lm-chunk must be even (nibble pairs)"
        packed, cb, w_dq = quantize(pad_rows(model.lm_head.weight, pad_vocab), chunk=args.lm_chunk, k=args.lm_k)
        f.write(packed.tobytes())
        f.write(cb.tobytes())
    L_, m_, k_ = walk_cbk4(path)
    print(f"wrote {path} {os.path.getsize(path) / 1e9:.3f} GB (walk OK: {L_} layers, M={m_}, lm_k={k_})", flush=True)

    # The reference runs the AQLM model with the SAME quantized LM head as the .cbk. A fresh
    # Parameter (not .data=) so a tied embedding (e.g. Llama-3.2) keeps its dense weights.
    dev = model.lm_head.weight.device
    model.lm_head.weight = torch.nn.Parameter(w_dq[:vocab].to(dev, torch.float16), requires_grad=False)

    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    print("prompt:", repr(args.prompt), "->", ids.tolist(), flush=True)
    with torch.no_grad():
        logits = model(ids).logits[0].float().cpu()  # [P, vocab]
        gen = model.generate(ids, max_new_tokens=args.gen, do_sample=False, pad_token_id=tok.eos_token_id)
    ref = torch.zeros(logits.shape[0], pad_vocab)
    ref[:, :vocab] = logits  # padded vocab rows are zero weights in the .cbk, so zero logits
    cont = gen[0, ids.shape[1]:].cpu().numpy().astype("<i4")
    print("continuation:", repr(tok.decode(cont)), flush=True)
    ids.cpu().numpy().astype("<i4").tofile(os.path.join(args.out, "prompt.bin"))
    ref.numpy().astype("<f4").tofile(os.path.join(args.out, "ref.bin"))
    cont.tofile(os.path.join(args.out, "cont.bin"))
    print("wrote prompt.bin ref.bin cont.bin", flush=True)

    if args.ppl > 0:
        ptok = AutoTokenizer.from_pretrained(args.ppl_tokenizer) if args.ppl_tokenizer else tok
        p, n = ppl_windows(model, ptok, args.ppl, window0_out=args.ppl_window0_out)
        print(f"PPL of the exported model (AQLM codes + {args.lm_k}-entry LM head), wikitext-2 {n}x2048 = {p:.4f}", flush=True)


if __name__ == "__main__":
    main()
