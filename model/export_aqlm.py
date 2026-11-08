#!/usr/bin/env python3
"""
Export a HuggingFace AQLM checkpoint in the 2x8 (or 3x8) additive-codebook format, for example
ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf, or in the 1x16 format (one codebook of 65,536
entries), for example ISTA-DASLab/Llama-2-70b-AQLM-PV-2Bit-1x16-hf, to the Rust runtime's CBK4
container.

The seven per-layer projections are written as CBKA records, or CBKH records for 1x16
(model/cbka_format.py): the AQLM codes, codebooks and per-output scales are copied bit for bit,
nothing is requantized, and the runtime decodes them with AvqLinear (avq_gemv_t for Mx8,
avq16_gemv for 1x16). Embedding and norms stay dense. AQLM keeps the
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
  i32       avq_m  (additive codebooks per projection: 2 for "2x8", 1 for "1x16")
  i32       lm_k   (LM head scalar codebook size: 16 or 256)
  f16 x pad_vocab*hidden   embedding
  per layer: input_layernorm f32[hidden]; CBKA q [+ f32 q bias]; CBKA k [+ bias]; CBKA v [+ bias];
             CBKA o; post_attention_layernorm f32[hidden]; CBKA gate; CBKA up; CBKA down
             (CBKH records instead of CBKA when avq_m = 1)
  final norm f32[hidden]
  LM head: lm_k=16  -> u8[hidden*pad_vocab/2] nibble-packed [in][out/2], f32[16*pad_vocab]
           lm_k=256 -> u8[hidden*pad_vocab] [in][out], f32[256*pad_vocab]
The reader is Model::load_cbk / load_cbk4_body in runtime/src/lib.rs; keep the two in sync.

Usage:
  python model/export_aqlm.py --model ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf --out /root/cbk4 \
      --prompt "The capital of France is" --gen 16 --lm-k 256 --ppl 30
Llama-3.1-70B-Instruct AQLM 1x16 (ISTA-DASLab) fit the export on one 24 GB RTX 4090 (highest memory
sample 23,091 MiB) with --embed-on-cpu --lm-chunk 4096 --ppl-chunk 256 and
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
No em dashes or en dashes anywhere.
"""
import argparse, math, os, struct, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cbka_format import AVQ_D, AVQ_K, AVQ16_K, write_cbka_record, write_cbkh_record  # noqa: E402  single source of truth for CBKA/CBKH
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
    """(codes, C, scale [OC] f32, M) of an aqlm QuantizedLinear. Mx8 (M in 2, 3): codes [OC, ng, M]
    uint8 and C [M, 256, D] f32. 1x16 (M = 1): codes [OC, ng] uint16 and C [65536, D] f32."""
    if type(mod).__name__ != "QuantizedLinear":
        raise SystemExit(f"expected an aqlm QuantizedLinear, got {type(mod).__name__} (not an AQLM checkpoint?)")
    cb = mod.codebooks.detach()
    M, K, og, ig = cb.shape
    x8 = K == AVQ_K and M in (2, 3)
    x16 = K == AVQ16_K and M == 1
    if not ((x8 or x16) and og == 1 and ig == AVQ_D):
        raise SystemExit(f"unsupported AQLM layout {tuple(cb.shape)}: the runtime decodes Mx8 codebooks of "
                         f"{AVQ_K} entries (M in (2, 3), e.g. a 2x8 checkpoint) or one codebook of {AVQ16_K} "
                         f"entries (1x16), with in_group {AVQ_D} and out_group 1")
    if getattr(mod, "bias", None) is not None:
        raise SystemExit("AQLM linear with a bias is not supported by this exporter")
    # aqlm stores b-bit codes as signed ints (int8 for 8 bits, int16 for 16): reduce mod 2^b
    codes = mod.codes.detach().to(torch.int64) % K
    OC, ng, Mc = codes.shape
    assert Mc == M, f"codes M {Mc} != codebooks M {M}"
    scale = mod.scales.detach().reshape(-1)
    assert scale.numel() == OC, f"scales {scale.numel()} != OC {OC}"
    if x16:
        return (codes[:, :, 0].to(torch.int32).cpu().numpy().astype(np.uint16), cb[0, :, 0, :].float().cpu().numpy(),
                scale.float().cpu().numpy(), M)
    return codes.to(torch.uint8).cpu().numpy(), cb[:, :, 0, :].float().cpu().numpy(), scale.float().cpu().numpy(), M


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
            magic = f.read(4)
            assert magic == (b"CBKH" if m == 1 else b"CBKA"), f"missing {'CBKH' if m == 1 else 'CBKA'} magic"
            mm, d, k, r, c = struct.unpack("<5i", f.read(20))
            assert (mm, d, k, r, c) == (m, AVQ_D, AVQ16_K if m == 1 else AVQ_K, rows, cols), ((mm, d, k, r, c), (m, rows, cols))
            idx_bytes = 2 * r * (c // d) if m == 1 else mm * (c // d) * r   # u16 [rows][ng] or u8 [M][ng][rows]
            f.seek(2 * mm * k * d + 2 * r + idx_bytes, 1)

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
def ppl_windows(model, tok, n_windows, seqlen=2048, window0_out=None, chunk=0):
    """wikitext-2 test PPL over the first n_windows windows of seqlen tokens. Also prints the
    running PPL after windows 1 and 30 (window 1 alone is what the runtime's TRAPETUM_NLL check
    compares against) and, with window0_out, writes window 0's token ids (i32) for that check.
    chunk > 0 computes each window's loss over chunks of that many positions (less GPU memory for a
    large vocabulary: the f32 logits of 2,048 positions of Llama-3 alone are 1.05 GB)."""
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
        ids = enc[:, i * seqlen:(i + 1) * seqlen].cuda()
        o = model(ids).logits
        if chunk > 0:
            lo, tg = o[0, :-1, :], ids[0, 1:]
            for s in range(0, lo.size(0), chunk):
                tot += lf(lo[s:s + chunk].float(), tg[s:s + chunk]).item()
        else:
            tot += lf(o[:, :-1, :].reshape(-1, o.size(-1)).float(), ids[:, 1:].reshape(-1)).item()
        nt += ids[:, 1:].numel()
        del o
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
    ap.add_argument("--ppl-chunk", type=int, default=0,
                    help="with --ppl: compute the loss over chunks of this many positions (less GPU memory for "
                         "a large vocabulary; 0 = one pass)")
    ap.add_argument("--embed-on-cpu", action="store_true",
                    help="keep the input embedding table on the CPU during the export (each forward moves its "
                         "rows to the GPU): frees vocab*hidden fp16 bytes of GPU memory, 2.1 GB on Llama-3.1-70B")
    ap.add_argument("--ref-gemm", action="store_true",
                    help="also write ref_gemm.bin: the prompt logits with every AQLM layer forced onto its "
                         "dequantize-then-matmul path (aqlm 1.1.7 runs prompts of 6 tokens or fewer through its "
                         "GEMV kernel, which adds the 8 products of each group in fp16), and print how far apart "
                         "the two references are")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading", args.model, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda").eval()
    if args.embed_on_cpu:
        emb = model.model.embed_tokens.to("cpu")
        emb.register_forward_pre_hook(lambda m, inp: tuple(t.to("cpu") if torch.is_tensor(t) else t for t in inp))
        emb.register_forward_hook(lambda m, inp, out: out.to("cuda"))
        torch.cuda.empty_cache()
        print(f"input embedding kept on the CPU ({emb.weight.numel() * 2 / 1e9:.2f} GB of fp16 off the GPU)", flush=True)
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

        def rec(lin_mod, name=None):
            codes, C, scale, m = aqlm_parts(lin_mod)
            assert m == M, f"mixed codebook counts ({m} vs {M}) are not supported"
            if M == 1:
                write_cbkh_record(f, codes, C, scale)
                if name:  # information only: how concentrated the 1x16 code usage is
                    cnt = np.sort(np.bincount(codes.ravel(), minlength=AVQ16_K))[::-1]
                    print(f"  code usage {name}: {(cnt > 0).sum()} of {AVQ16_K} entries used, the 6144 most used "
                          f"cover {cnt[:6144].sum() / cnt.sum():.1%}", flush=True)
            else:
                write_cbka_record(f, codes, C, scale, M)

        for li, L in enumerate(model.model.layers):
            nm = (lambda s: f"layer {li} {s}") if li == 0 else (lambda s: None)
            w_f32(f, L.input_layernorm.weight)
            for s, p in (("q", L.self_attn.q_proj), ("k", L.self_attn.k_proj), ("v", L.self_attn.v_proj), ("o", L.self_attn.o_proj)):
                rec(p, nm(s))
            w_f32(f, L.post_attention_layernorm.weight)
            for s, p in (("gate", L.mlp.gate_proj), ("up", L.mlp.up_proj), ("down", L.mlp.down_proj)):
                rec(p, nm(s))
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
    model.lm_head.weight = None  # free the dense head before its dequantized copy lands on the GPU
    torch.cuda.empty_cache()
    model.lm_head.weight = torch.nn.Parameter(w_dq[:vocab].to("cuda", torch.float16), requires_grad=False)

    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    print("prompt:", repr(args.prompt), "->", ids.tolist(), flush=True)
    with torch.no_grad():
        logits = model(ids).logits[0].float().cpu()  # [P, vocab]
        logits_gemm = None
        if args.ref_gemm:
            qls = [m for m in model.modules() if type(m).__name__ == "QuantizedLinear"]
            if qls and all(hasattr(m, "use_gemv_rule") for m in qls):
                saved = [m.use_gemv_rule for m in qls]
                for m in qls:
                    m.use_gemv_rule = lambda inp: False  # dequantize-then-matmul, whatever the token count
                logits_gemm = model(ids).logits[0].float().cpu()
                for m, s in zip(qls, saved):
                    m.use_gemv_rule = s
                d = ((logits.double() - logits_gemm.double()).norm(dim=1) / logits_gemm.double().norm(dim=1)).max().item()
                print(f"HF prompt logits, aqlm GEMV path vs dequantized path: worst rel err {d:.2e} "
                      f"(top-1 agreement {(logits.argmax(1) == logits_gemm.argmax(1)).sum().item()}/{logits.shape[0]})", flush=True)
            else:
                print("--ref-gemm: this aqlm version has no use_gemv_rule, ref_gemm.bin not written", flush=True)
        gen = model.generate(ids, max_new_tokens=args.gen, do_sample=False, pad_token_id=tok.eos_token_id)
    ref = torch.zeros(logits.shape[0], pad_vocab)
    ref[:, :vocab] = logits  # padded vocab rows are zero weights in the .cbk, so zero logits
    if logits_gemm is not None:
        ref_g = torch.zeros(logits_gemm.shape[0], pad_vocab)
        ref_g[:, :vocab] = logits_gemm
        ref_g.numpy().astype("<f4").tofile(os.path.join(args.out, "ref_gemm.bin"))
        print("wrote ref_gemm.bin", flush=True)
    cont = gen[0, ids.shape[1]:].cpu().numpy().astype("<i4")
    print("continuation:", repr(tok.decode(cont)), flush=True)
    ids.cpu().numpy().astype("<i4").tofile(os.path.join(args.out, "prompt.bin"))
    ref.numpy().astype("<f4").tofile(os.path.join(args.out, "ref.bin"))
    cont.tofile(os.path.join(args.out, "cont.bin"))
    print("wrote prompt.bin ref.bin cont.bin", flush=True)

    if args.ppl > 0:
        ptok = AutoTokenizer.from_pretrained(args.ppl_tokenizer) if args.ppl_tokenizer else tok
        p, n = ppl_windows(model, ptok, args.ppl, window0_out=args.ppl_window0_out, chunk=args.ppl_chunk)
        print(f"PPL of the exported model (AQLM codes + {args.lm_k}-entry LM head), wikitext-2 {n}x2048 = {p:.4f}", flush=True)


if __name__ == "__main__":
    main()
