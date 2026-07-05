#!/usr/bin/env python3
"""
Export a Gemma-2 model to the Rust runtime's Gemma .cbk format ("CBKG"). q/k/v/o + MLP +
LM head are 4-bit codebook-quantized; norms stay dense. Gemma specifics baked at export:
  - embedding scaled by sqrt(hidden) (the input lookup; lm_head is tied but written unscaled),
  - RMSNorm (1+w): +1 added to every norm weight (runtime uses standard RMSNorm),
  - attention + final logit softcapping stored in the header,
  - GeGLU / softcap handled by the runtime kernels.
q_head_dim (n_heads*head_dim) may differ from hidden. Loads on CPU; per-linear k-means on GPU.
  python export_gemma.py --model google/gemma-2-9b-it --out /workspace/gm --prompt "The capital of France is"
"""
import argparse, math, os, struct
import numpy as np
import torch
from export_runtime import quantize, w_f16, K

DEV = os.environ.get("EXPORT_DEV", "cuda")


def qw(f, w):
    packed, cb, _ = quantize(w)
    f.write(packed.tobytes()); f.write(cb.tobytes())


def norm1(f, t):  # Gemma RMSNorm is output*(1+w); bake +1 so the runtime uses standard RMSNorm
    f.write((t.detach().float().cpu().numpy().astype("<f4") + 1.0).tobytes())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--out", default="/workspace/gm")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    c = model.config
    hidden = c.hidden_size; n_heads = c.num_attention_heads; n_kv = c.num_key_value_heads
    head_dim = c.head_dim; inter = c.intermediate_size; vocab = c.vocab_size; n_layers = c.num_hidden_layers
    eps = c.rms_norm_eps; base = float(getattr(c, "rope_theta", 10000.0))
    attn_softcap = float(getattr(c, "attn_logit_softcapping", 0.0) or 0.0)
    final_softcap = float(getattr(c, "final_logit_softcapping", 0.0) or 0.0)
    qscalar = float(getattr(c, "query_pre_attn_scalar", head_dim))
    assert abs(qscalar - head_dim) < 1e-6, f"query_pre_attn_scalar {qscalar} != head_dim {head_dim} (bake a q-scale)"
    for d in (hidden, n_heads*head_dim, n_kv*head_dim, inter, vocab):
        assert d % 256 == 0, f"dim {d} not %256"
    print(f"cfg: L={n_layers} hidden={hidden} heads={n_heads}/{n_kv} hd={head_dim} inter={inter} "
          f"vocab={vocab} attn_sc={attn_softcap} final_sc={final_softcap}", flush=True)

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = open(os.path.join(args.out, "model.cbk"), "wb")
    f.write(b"CBKG")
    f.write(struct.pack("<7i", n_layers, hidden, n_heads, n_kv, head_dim, inter, vocab))
    f.write(struct.pack("<4f", eps, base, attn_softcap, final_softcap))
    f.write(inv_freq.numpy().astype("<f4").tobytes())
    # embedding scaled by sqrt(hidden) (Gemma input-embedding normalizer)
    emb = model.model.embed_tokens.weight
    w_f16(f, emb * math.sqrt(hidden))

    for li in range(n_layers):
        L = model.model.layers[li]; A = L.self_attn
        norm1(f, L.input_layernorm.weight)
        qw(f, A.q_proj.weight); qw(f, A.k_proj.weight); qw(f, A.v_proj.weight); qw(f, A.o_proj.weight)
        norm1(f, L.post_attention_layernorm.weight)
        norm1(f, L.pre_feedforward_layernorm.weight)
        qw(f, L.mlp.gate_proj.weight); qw(f, L.mlp.up_proj.weight); qw(f, L.mlp.down_proj.weight)
        norm1(f, L.post_feedforward_layernorm.weight)
        print(f"  layer {li+1}/{n_layers}", flush=True)
    norm1(f, model.model.norm.weight)
    qw(f, model.get_output_embeddings().weight)   # lm_head (tied to embed, written UNscaled)
    f.close()
    print("wrote", os.path.join(args.out, "model.cbk"), flush=True)

    ids = tok(args.prompt, return_tensors="pt").input_ids
    ids.numpy().astype("<i4").tofile(os.path.join(args.out, "prompt.bin"))
    print("prompt tokens:", ids.tolist(), flush=True)


if __name__ == "__main__":
    main()
