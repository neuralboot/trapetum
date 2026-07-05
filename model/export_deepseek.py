#!/usr/bin/env python3
"""
Export a DeepSeek-V2/V3 (MLA + MoE) model to the Rust runtime's DeepSeek .cbk format
("CBKD"). MLA projections (q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj) stay DENSE fp16
(they are small and the absorption folds W_UK/W_UV into the query/output); the MoE experts,
router, dense-MLP and LM head are 4-bit codebook-quantized (K=16). RMSNorm weights + the
kv_a_layernorm stay dense. Reuses the codebook k-means from export_runtime.py.

CUDA (24GB+) recommended; DeepSeek-V2-Lite (16B) is the tractable target.
  python export_deepseek.py --model deepseek-ai/DeepSeek-V2-Lite --out /workspace/ds \
      --prompt "The capital of France is" --gen 16
"""
import argparse, os, struct
import numpy as np
import torch
from export_runtime import quantize, w_f32, w_f16, K   # shared codebook helpers

DEV = os.environ.get("EXPORT_DEV", "cuda")


def qwrite(f, lin):
    packed, cb, _ = quantize(lin.weight)
    f.write(packed.tobytes()); f.write(cb.tobytes())


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
        device_map=DEV, trust_remote_code=True)
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
    eps = c.rms_norm_eps
    rope_theta = float(getattr(c, "rope_theta", 10000.0))
    rscale = float(getattr(c, "routed_scaling_factor", 1.0))
    assert kv_lora % 256 == 0, f"kv_lora_rank {kv_lora} must be %256 (kv_b/absorption is dense so ok, but experts need it)"
    assert vocab % 256 == 0, f"vocab {vocab} must be %256 for the quantized LM head"
    assert n_routed % 256 == 0, f"n_routed_experts {n_routed} must be %256 for the quantized router"
    print(f"cfg: L={n_layers} hidden={hidden} heads={n_heads} kv_lora={kv_lora} nope={nope} rope={rope} "
          f"vhd={vhd} inter={inter_dense} moe_inter={moe_inter} n_routed={n_routed} n_shared={n_shared} "
          f"top_k={top_k} vocab={vocab} first_k_dense={first_k_dense}", flush=True)

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope, 2).float() / rope))
    path = os.path.join(args.out, "model.cbk")
    f = open(path, "wb")
    f.write(b"CBKD")
    f.write(struct.pack("<14i", n_layers, hidden, n_heads, kv_lora, nope, rope, vhd,
                        inter_dense, moe_inter, n_routed, n_shared, top_k, vocab, first_k_dense))
    f.write(struct.pack("<3f", eps, rope_theta, rscale))
    f.write(inv_freq.numpy().astype("<f4").tobytes())
    w_f16(f, model.model.embed_tokens.weight)   # [vocab][hidden] fp16

    for li in range(n_layers):
        L = model.model.layers[li]
        A = L.self_attn
        w_f32(f, L.input_layernorm.weight)
        # MLA projections: dense fp16
        if hasattr(A, "q_proj"):
            w_f16(f, A.q_proj.weight)                       # [n_heads*(nope+rope)][hidden]
        else:  # q_lora variant (V2 full / V3): fold q_a->q_b at export? keep dense q_b only if no lora
            raise SystemExit("q_lora_rank models: export q_a_proj/q_b_proj not yet supported (V2-Lite has plain q_proj)")
        w_f16(f, A.kv_a_proj_with_mqa.weight)               # [kv_lora+rope][hidden]
        w_f32(f, A.kv_a_layernorm.weight)                   # [kv_lora]
        w_f16(f, A.kv_b_proj.weight)                        # [n_heads*(nope+vhd)][kv_lora]
        w_f16(f, A.o_proj.weight)                           # [hidden][n_heads*vhd]
        w_f32(f, L.post_attention_layernorm.weight)
        # MLP: dense (first_k_dense layers) or MoE
        if li < first_k_dense:
            M = L.mlp
            qwrite(f, M.gate_proj); qwrite(f, M.up_proj); qwrite(f, M.down_proj)
        else:
            M = L.mlp
            qwrite(f, M.gate)                               # router [n_routed][hidden]
            for e in M.experts:
                qwrite(f, e.gate_proj); qwrite(f, e.up_proj); qwrite(f, e.down_proj)
            S = M.shared_experts
            qwrite(f, S.gate_proj); qwrite(f, S.up_proj); qwrite(f, S.down_proj)
        print(f"  layer {li+1}/{n_layers} ({'dense' if li<first_k_dense else 'moe'}) written", flush=True)

    w_f32(f, model.model.norm.weight)
    qwrite(f, model.lm_head)
    f.close()
    print("wrote", path, os.path.getsize(path)//(1024*1024), "MB", flush=True)

    # reference: HF fp16 greedy continuation (approximate target; the runtime is 4-bit so it
    # should be coherent + high top-1 agreement, not bit-identical).
    model = model.eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(DEV)
    am = torch.ones_like(ids)
    out = model(ids)
    logits = out.logits[0].detach().float().cpu().numpy().astype("<f4")
    gen = model.generate(ids, attention_mask=am, max_new_tokens=args.gen, do_sample=False)
    cont = gen[0, ids.shape[1]:].cpu().numpy().astype("<i4")
    print("continuation:", repr(tok.decode(cont)), flush=True)
    ids.cpu().numpy().astype("<i4").tofile(os.path.join(args.out, "prompt.bin"))
    logits.tofile(os.path.join(args.out, "ref.bin"))
    cont.tofile(os.path.join(args.out, "cont.bin"))
    print("wrote prompt.bin ref.bin cont.bin", flush=True)


if __name__ == "__main__":
    main()
