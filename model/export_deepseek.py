#!/usr/bin/env python3
"""
Export a DeepSeek-V2/V3 (MLA + MoE) model to the Rust runtime's DeepSeek .cbk format.
Two variants, picked automatically from the model config:

- "CBKD" (no q_lora, e.g. DeepSeek-V2-Lite): MLA projections (q_proj, kv_a_proj_with_mqa,
  kv_b_proj, o_proj) stay DENSE fp16 (they are small and the absorption folds W_UK/W_UV into
  the query/output); the MoE experts, router, dense-MLP and LM head are 4-bit
  codebook-quantized (K=16). RMSNorm weights + the kv_a_layernorm stay dense.
- "CBKR" (q_lora_rank set, e.g. DeepSeek-V3/R1, 671B): q_a_proj/kv_a_proj/kv_b_proj stay
  dense fp16 (small), but q_b_proj and o_proj are 4-bit codebook-quantized (big at 671B
  scale); the V3 sigmoid+bias grouped router's `e_score_correction_bias` is written before
  the router weight.

Reuses the codebook k-means from export_runtime.py.

CUDA (24GB+) recommended; DeepSeek-V2-Lite (16B) is the tractable target for CBKD. CBKR
(671B) needs a machine with enough host RAM/disk to hold the fp16 checkpoint plus the
quantized output; the runtime streams routed experts from host RAM (see MoeBlockOffload).
  python export_deepseek.py --model deepseek-ai/DeepSeek-V2-Lite --out /workspace/ds \
      --prompt "The capital of France is" --gen 16
  python export_deepseek.py --model deepseek-ai/DeepSeek-V3 --out /workspace/ds3
"""
import argparse, math, os, struct
import numpy as np
import torch
from export_runtime import quantize, w_f32, w_f16, K   # shared codebook helpers


def _yarn_get_mscale(scale, mscale=1.0):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def mla_scale_and_inv_freq(c, q_head_dim, rope_dim, rope_theta):
    """softmax_scale + rope inv_freq for the runtime MLA, EXACTLY as modeling_deepseek
    (revision 604d566): DeepseekV2Attention uses softmax_scale = q_head_dim**-0.5, times
    mscale**2 when rope_scaling.mscale_all_dim is set; and DeepseekV2YarnRotaryEmbedding blends
    freq_inter/freq_extra by a correction-range ramp. The prior export wrote rope_theta into the
    softmax_scale slot (~10000 vs ~0.11) and a PLAIN inv_freq -- the block-0 attention bug the
    layer bisect found. NOTE: assumes the rope cos/sin _mscale == 1 (i.e. config mscale ==
    mscale_all_dim, true for V2-Lite/V3); asserted below, since the runtime bakes only inv_freq."""
    # the runtime writes only projection weights: a model with attention biases
    # would silently lose them (q_a/kv_a/o_proj use bias=config.attention_bias)
    assert not getattr(c, "attention_bias", False), \
        "attention_bias=True is not representable by the runtime export"
    softmax_scale = q_head_dim ** (-0.5)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))  # plain default
    rs = getattr(c, "rope_scaling", None)
    if rs is not None:
        factor = float(rs.get("factor", 1.0))
        mscale_all_dim = rs.get("mscale_all_dim", 0)
        if mscale_all_dim:
            m = _yarn_get_mscale(factor, mscale_all_dim)
            softmax_scale = softmax_scale * m * m
        if rs.get("type") == "yarn":
            mscale = rs.get("mscale", 0)
            _mscale = _yarn_get_mscale(factor, mscale) / _yarn_get_mscale(factor, mscale_all_dim)
            assert abs(_mscale - 1.0) < 1e-6, (
                f"rope cos/sin _mscale={_mscale} != 1 (config mscale {mscale} != mscale_all_dim "
                f"{mscale_all_dim}); the runtime bakes only inv_freq and cannot represent it")
            dim, base = rope_dim, rope_theta
            beta_fast = float(rs.get("beta_fast", 32)); beta_slow = float(rs.get("beta_slow", 1))
            orig_max = float(rs.get("original_max_position_embeddings", 4096))
            freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            freq_inter = 1.0 / (factor * base ** (torch.arange(0, dim, 2).float() / dim))
            corr = lambda nr: (dim * math.log(orig_max / (nr * 2 * math.pi))) / (2 * math.log(base))
            low = max(math.floor(corr(beta_fast)), 0)
            high = min(math.ceil(corr(beta_slow)), dim - 1)
            if low == high: high += 0.001
            ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
            inv_freq_mask = 1.0 - ramp
            inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    return float(softmax_scale), inv_freq.float().contiguous()

DEV = os.environ.get("EXPORT_DEV", "cuda")


def pad256(n): return ((n + 255) // 256) * 256

def _pad_out(w, oc):  # pad output rows (dim 0) with zeros to oc (kernel tiles oc in 256s)
    if w.shape[0] >= oc: return w
    return torch.cat([w, torch.zeros(oc - w.shape[0], w.shape[1], dtype=w.dtype, device=w.device)], 0)

def _pad_in(w, ic):   # pad input cols (dim 1) with zeros to ic (down-proj input = padded inter)
    if w.shape[1] >= ic: return w
    return torch.cat([w, torch.zeros(w.shape[0], ic - w.shape[1], dtype=w.dtype, device=w.device)], 1)

def qw(f, w):         # quantize a raw [out][in] weight tensor
    packed, cb, _ = quantize(w)
    f.write(packed.tobytes()); f.write(cb.tobytes())

def qffn(f, gate, up, down, inter_pad):   # gate/up padded on output, down padded on input; zeros are lossless (silu(0)=0)
    qw(f, _pad_out(gate.weight, inter_pad)); qw(f, _pad_out(up.weight, inter_pad)); qw(f, _pad_in(down.weight, inter_pad))


def _write_ffn(f, L, li, first_k_dense, inter_dense_pad, moe_inter_pad, shared_inter_pad, n_group, topk_group, sigmoid_flag):
    # MLP: dense (first_k_dense layers) or MoE. Shared by CBKD and CBKR.
    if li < first_k_dense:
        M = L.mlp
        qffn(f, M.gate_proj, M.up_proj, M.down_proj, inter_dense_pad)
    else:
        M = L.mlp
        if sigmoid_flag:  # CBKR: V3 router bias comes BEFORE the router weight
            w_f32(f, M.gate.e_score_correction_bias)         # [n_routed] f32
        w_f16(f, M.gate.weight)                              # router [n_routed][hidden] DENSE fp16
        for e in M.experts:
            qffn(f, e.gate_proj, e.up_proj, e.down_proj, moe_inter_pad)
        S = M.shared_experts
        qffn(f, S.gate_proj, S.up_proj, S.down_proj, shared_inter_pad)
    return 'dense' if li < first_k_dense else 'moe'


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--out", default="/workspace/ds")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading", args.model, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        trust_remote_code=True)  # stays on CPU; quantize() moves each weight to GPU one at a time
    c = model.config
    hidden = c.hidden_size
    n_heads = c.num_attention_heads
    kv_lora = c.kv_lora_rank
    nope = c.qk_nope_head_dim
    rope = c.qk_rope_head_dim
    vhd = c.v_head_dim
    inter_dense = c.intermediate_size
    moe_inter = c.moe_intermediate_size
    n_routed = c.n_routed_experts
    n_shared = c.n_shared_experts
    top_k = c.num_experts_per_tok
    vocab = c.vocab_size
    first_k_dense = getattr(c, "first_k_dense_replace", 0)
    n_layers = c.num_hidden_layers
    inter_dense_pad = pad256(inter_dense)
    moe_inter_pad = pad256(moe_inter)
    shared_inter_pad = pad256(n_shared * moe_inter_pad)
    eps = c.rms_norm_eps
    rope_theta = float(getattr(c, "rope_theta", 10000.0))
    rscale = float(getattr(c, "routed_scaling_factor", 1.0))
    q_lora_rank = getattr(c, "q_lora_rank", None) or 0
    n_group = getattr(c, "n_group", 1) or 1
    topk_group = getattr(c, "topk_group", 1) or 1
    sigmoid_flag = 1 if getattr(c, "scoring_func", "softmax") == "sigmoid" else 0
    assert kv_lora % 256 == 0, f"kv_lora_rank {kv_lora} must be %256 (kv_b/absorption is dense so ok, but experts need it)"
    assert vocab % 256 == 0, f"vocab {vocab} must be %256 for the quantized LM head"
    print(f"cfg: L={n_layers} hidden={hidden} heads={n_heads} kv_lora={kv_lora} nope={nope} rope={rope} "
          f"vhd={vhd} inter={inter_dense} moe_inter={moe_inter} n_routed={n_routed} n_shared={n_shared} "
          f"top_k={top_k} vocab={vocab} first_k_dense={first_k_dense} q_lora_rank={q_lora_rank}", flush=True)

    # softmax_scale (NOT rope_theta) + yarn-corrected inv_freq, matching modeling_deepseek.
    softmax_scale, inv_freq = mla_scale_and_inv_freq(c, nope + rope, rope, rope_theta)
    print(f"mla: q_head_dim={nope + rope} softmax_scale={softmax_scale:.6f} (was rope_theta={rope_theta}); "
          f"inv_freq[{inv_freq.numel()}] yarn={getattr(c, 'rope_scaling', None) is not None}", flush=True)
    path = os.path.join(args.out, "model.cbk")
    f = open(path, "wb")

    if q_lora_rank:
        # CBKR: DeepSeek-V3/R1 q_lora MLA + (optionally) V3 sigmoid/grouped router.
        qdim = n_heads * (nope + rope)
        assert qdim % 256 == 0, f"n_heads*(nope+rope)={qdim} must be %256 for the quantized q_b"
        assert hidden % 256 == 0, f"hidden={hidden} must be %256 for the quantized o_proj"
        f.write(b"CBKR")
        f.write(struct.pack("<18i", n_layers, hidden, n_heads, kv_lora, nope, rope, vhd,
                            inter_dense_pad, moe_inter_pad, n_routed, n_shared, top_k, vocab, first_k_dense,
                            q_lora_rank, n_group, topk_group, sigmoid_flag))
        f.write(struct.pack("<3f", eps, softmax_scale, rscale))
        f.write(inv_freq.numpy().astype("<f4").tobytes())
        w_f16(f, model.model.embed_tokens.weight)   # [vocab][hidden] fp16

        for li in range(n_layers):
            L = model.model.layers[li]
            A = L.self_attn
            w_f32(f, L.input_layernorm.weight)
            w_f16(f, A.q_a_proj.weight)                      # [q_lora_rank][hidden]
            w_f32(f, A.q_a_layernorm.weight)                 # [q_lora_rank]
            qw(f, A.q_b_proj.weight)                         # [qdim][q_lora_rank] 4-bit
            w_f16(f, A.kv_a_proj_with_mqa.weight)             # [kv_lora+rope][hidden]
            w_f32(f, A.kv_a_layernorm.weight)                 # [kv_lora]
            w_f16(f, A.kv_b_proj.weight)                      # [n_heads*(nope+vhd)][kv_lora]
            qw(f, A.o_proj.weight)                            # [hidden][n_heads*vhd] 4-bit
            w_f32(f, L.post_attention_layernorm.weight)
            kind = _write_ffn(f, L, li, first_k_dense, inter_dense_pad, moe_inter_pad, shared_inter_pad,
                               n_group, topk_group, sigmoid_flag)
            print(f"  layer {li+1}/{n_layers} ({kind}) written", flush=True)
    else:
        # CBKD: no q_lora (DeepSeek-V2-Lite style); MLA projections stay dense fp16.
        f.write(b"CBKD")
        f.write(struct.pack("<14i", n_layers, hidden, n_heads, kv_lora, nope, rope, vhd,
                            inter_dense_pad, moe_inter_pad, n_routed, n_shared, top_k, vocab, first_k_dense))
        f.write(struct.pack("<3f", eps, softmax_scale, rscale))
        f.write(inv_freq.numpy().astype("<f4").tobytes())
        w_f16(f, model.model.embed_tokens.weight)   # [vocab][hidden] fp16

        for li in range(n_layers):
            L = model.model.layers[li]
            A = L.self_attn
            w_f32(f, L.input_layernorm.weight)
            assert hasattr(A, "q_proj"), "expected plain q_proj (no q_lora_rank in config); use the CBKR branch otherwise"
            w_f16(f, A.q_proj.weight)                        # [n_heads*(nope+rope)][hidden]
            w_f16(f, A.kv_a_proj_with_mqa.weight)             # [kv_lora+rope][hidden]
            w_f32(f, A.kv_a_layernorm.weight)                 # [kv_lora]
            w_f16(f, A.kv_b_proj.weight)                      # [n_heads*(nope+vhd)][kv_lora]
            w_f16(f, A.o_proj.weight)                         # [hidden][n_heads*vhd]
            w_f32(f, L.post_attention_layernorm.weight)
            kind = _write_ffn(f, L, li, first_k_dense, inter_dense_pad, moe_inter_pad, shared_inter_pad,
                               n_group, topk_group, sigmoid_flag)
            print(f"  layer {li+1}/{n_layers} ({kind}) written", flush=True)

    w_f32(f, model.model.norm.weight)
    qw(f, model.lm_head.weight)
    f.close()
    print("wrote", path, os.path.getsize(path)//(1024*1024), "MB", flush=True)

    # save the tokenized prompt (the 16B fp16 reference forward does not fit the GPU here;
    # the runtime output is checked for coherence by decoding + detokenizing separately).
    ids = tok(args.prompt, return_tensors="pt").input_ids
    ids.numpy().astype("<i4").tofile(os.path.join(args.out, "prompt.bin"))
    print("prompt tokens:", ids.tolist(), flush=True)
    print("wrote prompt.bin", flush=True)


if __name__ == "__main__":
    main()
