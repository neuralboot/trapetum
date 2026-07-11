#!/usr/bin/env python3
"""Per-layer hidden-state dump for DeepSeek-V2-Lite via HuggingFace, to diff LAYER-BY-LAYER
against the Rust runtime (`deepseek_run` with TRAPETUM_LAYER_DEBUG=1). Same `[ldump]` line
format on both sides.

THE POINT (--quantize): comparing Rust-4bit vs HF-fp16 would only show gradual quantization
drift and be useless for bisection. With --quantize this script REPLACES exactly the tensors
`model/export_deepseek.py` quantizes (FFN gate/up/down of dense + routed + shared experts, and
lm_head) with their DEQUANTIZED 4-bit versions -- reusing `export_runtime.quantize` (the SAME
k-means the export uses, not a reimplementation) -- while keeping embed / MLA projections
(q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj) / the router in fp16 and the norms in f32, i.e.
the CBKD tensor split. Then:
  - HF-dequant4bit answers Paris-like but Rust-4bit rambles  -> the bug is OURS (export writer or
    Rust forward); the layer diff is meaningful, apples to apples.
  - HF-dequant4bit ALSO rambles -> 4-bit quantization itself is the wall for V2-Lite.

NOTE on padding: the export pads gate/up on output and down on input to a 256 multiple before
quantizing (zeros, lossless by silu(0)=0). This mirror quantizes the UNPADDED weights (the
padding is a runtime-tiling detail, designed lossless); the tensor SELECTION is exact. If pass 1
is borderline we can add the exact input-padding for down-proj.

Usage:
  python model/dump_layers_hf.py --model deepseek-ai/DeepSeek-V2-Lite \
      --prompt "The capital of France is" [--quantize] > hf.ldump 2>&1
  python model/dump_layers_hf.py --dry-run    # CPU-only syntax/plumbing smoke test, no download
"""
import argparse
import os
import sys

import torch


def stats(x):
    x = x.detach().float().reshape(-1)
    return x.norm(2).item(), x.abs().max().item(), x.mean().item()


def e6(x):
    # Match Rust's `{:.6e}` exactly (e.g. 1.234560e2, 1.234560e-3, 0.000000e0) so the two dumps
    # are BYTE-identical -- Python's native %.6e uses e+02/e-03, so strip the sign-padding.
    m, e = f"{x:.6e}".split("e")
    return f"{m}e{int(e)}"


def ldump(stage, layer, x):
    l2, am, mn = stats(x)
    print(f"[ldump] L={layer} stage={stage} l2={e6(l2)} absmax={e6(am)} mean={e6(mn)}", flush=True)


def top5(logits):
    v = logits.detach().float().reshape(-1)
    vals, idx = torch.topk(v, min(5, v.numel()))
    print("[ldump] top5= " + ",".join(f"{int(i)}:{float(x):.4f}" for i, x in zip(idx, vals)), flush=True)


def _quantized_linears(model):
    """Yield the nn.Linear modules the CBKD export 4-bit quantizes: every FFN gate/up/down
    (dense layers + routed experts + shared experts) plus lm_head. The router (mlp.gate), the
    MLA projections, and the embedding stay fp16; norms stay f32 -- none are yielded."""
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "experts"):  # MoE layer
            for e in mlp.experts:
                yield e.gate_proj; yield e.up_proj; yield e.down_proj
            if getattr(mlp, "shared_experts", None) is not None:
                s = mlp.shared_experts
                yield s.gate_proj; yield s.up_proj; yield s.down_proj
            # mlp.gate is the router -> stays fp16 (NOT yielded)
        elif hasattr(mlp, "gate_proj"):  # dense FFN (first_k_dense layers)
            yield mlp.gate_proj; yield mlp.up_proj; yield mlp.down_proj
    yield model.lm_head


def apply_quant(model, quantize):
    """Replace each export-quantized weight with quantize(weight) -> W_dq (dequantized 4-bit),
    in place. `quantize` returns (packed, cb, W_dq [out,in]); we load W_dq back. Uses the GPU
    (quantize moves each weight to cuda), so this path needs a CUDA device (the pod)."""
    n = 0
    for lin in _quantized_linears(model):
        _packed, _cb, w_dq = quantize(lin.weight)
        lin.weight.data.copy_(w_dq.to(lin.weight.dtype).to(lin.weight.device))
        n += 1
    print(f"[quant] replaced {n} FFN/lm_head weights with dequantized 4-bit (export CBKD set)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--quantize", action="store_true", help="replace export-quantized tensors with dequant 4-bit (needs CUDA)")
    ap.add_argument("--split", type=int, default=None, help="also emit intra-layer sub-stages for this layer index (mirror of TRAPETUM_LAYER_DEBUG_SPLIT)")
    ap.add_argument("--dry-run", action="store_true", help="CPU-only smoke test: exercise stats/ldump + import quantize, no model download")
    args = ap.parse_args()

    # export_runtime lives next to this file; reuse its quantizer (do not reimplement).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from export_runtime import quantize  # noqa: E402

    if args.dry_run:
        x = torch.randn(4096)
        ldump("embed", 0, x)
        ldump("block", 3, x * 2.0)
        for st in ("norm1_w", "norm2_w", "attn_out", "post_attn", "ffn_out", "post_ffn"):
            ldump(st, args.split if args.split is not None else 0, x)
        ldump("final", 0, x)
        top5(torch.randn(32))
        assert callable(quantize), "export_runtime.quantize import failed"
        print("[dry-run] OK: dump format (incl split sub-stages) + quantize import validated on CPU", flush=True)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    if args.quantize:
        apply_quant(model, quantize)

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(model.device)
    last = ids.shape[1] - 1
    print(f"prompt tokens: {ids[0].tolist()}  (dumping position {last})", flush=True)

    # embed (before layer 0)
    with torch.no_grad():
        emb = model.model.embed_tokens(ids)
    ldump("embed", 0, emb[0, last])

    # after each full decoder layer (post both residual adds), and after the final norm
    def mk_block_hook(i):
        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out
            ldump("block", i, hs[0, last])
        return hook
    handles = [layer.register_forward_hook(mk_block_hook(i)) for i, layer in enumerate(model.model.layers)]

    def final_hook(_m, _inp, out):
        ldump("final", 0, out[0, last])
    handles.append(model.model.norm.register_forward_hook(final_hook))

    # intra-layer sub-stages for the split layer: attn_out (self_attn output, pre-residual),
    # post_attn (input to post_attention_layernorm = after first residual), ffn_out (mlp output,
    # pre-residual). post_ffn is the block hook above. Plus the two norm WEIGHT checksums.
    if args.split is not None:
        si = args.split
        L = model.model.layers[si]
        ldump("norm1_w", si, L.input_layernorm.weight)
        ldump("norm2_w", si, L.post_attention_layernorm.weight)

        def attn_hook(_m, _i, o):
            hs = o[0] if isinstance(o, (tuple, list)) else o
            ldump("attn_out", si, hs[0, last])
        handles.append(L.self_attn.register_forward_hook(attn_hook))

        def post_attn_pre(_m, inp):
            ldump("post_attn", si, inp[0][0, last])
        handles.append(L.post_attention_layernorm.register_forward_pre_hook(post_attn_pre))

        def mlp_hook(_m, _i, o):
            hs = o[0] if isinstance(o, (tuple, list)) else o
            ldump("ffn_out", si, hs[0, last])
        handles.append(L.mlp.register_forward_hook(mlp_hook))

    with torch.no_grad():
        out = model(ids)
    for h in handles:
        h.remove()

    logits = out.logits[0, last]
    ldump("logits", 0, logits)
    top5(logits)


if __name__ == "__main__":
    main()
