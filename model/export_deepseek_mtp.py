#!/usr/bin/env python3
"""
MTP (multi-token-prediction) exporter for DeepSeek-V3/R1: writes the MTP module
(model.layers.<n_layers>, i.e. layer 61 on R1, num_nextn_predict_layers=1) to a
standalone `mtp.cbk` so the already-exported 350 GB model.cbk never has to be
regenerated. The MTP module is a FULL transformer block (MLA attention + MoE FFN
with its own 256 routed experts) plus the MTP glue: enorm/hnorm (RMSNorm on the
token embedding / previous hidden state), eh_proj (2*hidden -> hidden concat
projection) and shared_head (norm + head).

Same streaming discipline and byte primitives as export_deepseek_stream.py
(LazySafetensors headers-only index, qwrite/wf16/wf32 4-bit packing), so the two
files cannot drift on how a tensor is packed. Quantization matches the main
export: q_b/o_proj/FFN experts 4-bit scalar codebook, small glue tensors f32/f16,
eh_proj 4-bit (both dims are %256 on R1).

The checkpoint stores its own copies of layers.61.embed_tokens.weight and
layers.61.shared_head.head.weight; on V3/R1 these are TIED to the main model's
embedding / lm_head (already inside model.cbk). We verify the tie by sampled
comparison and skip them when tied (a `tied` flag lands in the header); pass
--force-own-head to write them anyway.

  python export_deepseek_mtp.py --dir /path/to/DeepSeek-R1-bf16 --out /workspace/ds3
  python export_deepseek_mtp.py --selftest --out /tmp/mtp_selftest      # no download

MTP1 layout (all little-endian, written in this exact order):
  magic "MTP1"
  <16i>  hidden n_heads kv_lora nope rope vhd q_lora_rank moe_inter_pad
         n_routed n_shared top_k n_group topk_group sigmoid tied_embed tied_head
  <3f>   eps softmax_scale rscale
  f32    inv_freq [rope/2]
  [f16   embed_tokens [vocab][hidden]]      -- only if tied_embed == 0
  f32    enorm [hidden]
  f32    hnorm [hidden]
  4bit   eh_proj [hidden][2*hidden]
  ... then the standard CBKR MoE layer block, byte-identical to write_cbkr's:
  f32    input_layernorm ; f16 q_a ; f32 q_a_norm ; 4bit q_b ; f16 kv_a ;
  f32    kv_a_norm ; f16 kv_b ; 4bit o_proj ; f32 post_attention_layernorm ;
  f32    e_score_correction_bias ; f16 router ; 256x 4bit expert FFN ;
  4bit   shared-expert FFN
  f32    shared_head.norm [hidden]
  [4bit  shared_head.head [vocab][hidden]]  -- only if tied_head == 0
"""
import argparse, json, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_safetensors import qwrite, wf16, wf32                       # noqa: E402
from export_deepseek_stream import (LazySafetensors, pad256, compute_rope,   # noqa: E402
                                    build_cfg_from_json, SELFTEST_CFG,
                                    make_selftest_weights)


def _sampled_equal(a, b, n=4096, seed=0):
    # cheap tie check: same shape + n random elements equal (bf16-roundtrip tolerant).
    if a.shape != b.shape:
        return False
    rng = np.random.RandomState(seed)
    idx = (rng.randint(0, a.shape[0], n), rng.randint(0, a.shape[1], n))
    return bool(np.allclose(a[idx], b[idx], rtol=0, atol=1e-6))


def write_mtp(out_path, cfg, get, has, force_own_head=False):
    hidden, n_heads = cfg["hidden"], cfg["n_heads"]
    kv_lora, nope, rope, vhd = cfg["kv_lora"], cfg["nope"], cfg["rope"], cfg["vhd"]
    q_lora_rank, moe_inter = cfg["q_lora_rank"], cfg["moe_inter"]
    n_routed, n_shared, top_k = cfg["n_routed"], cfg["n_shared"], cfg["top_k"]
    n_group, topk_group, sigmoid = cfg["n_group"], cfg["topk_group"], cfg["sigmoid"]
    eps, rscale = cfg["eps"], cfg["rscale"]
    li = cfg["n_layers"]                                   # MTP module lives at layers.<n_layers>
    P = f"model.layers.{li}"
    assert has(f"{P}.eh_proj.weight"), \
        f"{P}.eh_proj.weight not found -- this snapshot has no MTP module (num_nextn_predict_layers=0?)"

    moe_inter_pad = pad256(moe_inter)
    shared_inter_pad = pad256(n_shared * moe_inter_pad)
    assert hidden % 256 == 0 and (2 * hidden) % 256 == 0, "eh_proj dims must be %256 for 4-bit"
    inv_freq, softmax_scale = compute_rope(cfg)

    # tie checks against the main model's embedding / lm_head (sampled, cheap).
    tied_embed = 0 if force_own_head else int(_sampled_equal(
        get(f"{P}.embed_tokens.weight"), get("model.embed_tokens.weight")))
    tied_head = 0 if force_own_head else int(_sampled_equal(
        get(f"{P}.shared_head.head.weight"), get("lm_head.weight")))
    print(f"  tie check: embed_tokens {'TIED (skipped)' if tied_embed else 'own copy (written)'}, "
          f"shared_head {'TIED (skipped)' if tied_head else 'own copy (written)'}", flush=True)

    def _pad_out(w, oc):
        if w.shape[0] >= oc:
            return w
        return np.concatenate([w, np.zeros((oc - w.shape[0], w.shape[1]), w.dtype)], 0)

    def _pad_in(w, ic):
        if w.shape[1] >= ic:
            return w
        return np.concatenate([w, np.zeros((w.shape[0], ic - w.shape[1]), w.dtype)], 1)

    def qwrite_ffn(prefix, inter_pad):
        qwrite(f, _pad_out(get(f"{prefix}.gate_proj.weight"), inter_pad))
        qwrite(f, _pad_out(get(f"{prefix}.up_proj.weight"), inter_pad))
        qwrite(f, _pad_in(get(f"{prefix}.down_proj.weight"), inter_pad))

    f = open(out_path, "wb")
    f.write(b"MTP1")
    f.write(struct.pack("<16i", hidden, n_heads, kv_lora, nope, rope, vhd, q_lora_rank,
                        moe_inter_pad, n_routed, n_shared, top_k, n_group, topk_group,
                        sigmoid, tied_embed, tied_head))
    f.write(struct.pack("<3f", eps, softmax_scale, rscale))
    wf32(f, inv_freq)
    if not tied_embed:
        wf16(f, get(f"{P}.embed_tokens.weight"))
    # MTP glue
    wf32(f, get(f"{P}.enorm.weight"))
    wf32(f, get(f"{P}.hnorm.weight"))
    qwrite(f, get(f"{P}.eh_proj.weight"))                              # [hidden][2*hidden] 4-bit
    # standard MoE layer block (same order as write_cbkr)
    wf32(f, get(f"{P}.input_layernorm.weight"))
    wf16(f, get(f"{P}.self_attn.q_a_proj.weight"))
    wf32(f, get(f"{P}.self_attn.q_a_layernorm.weight"))
    qwrite(f, get(f"{P}.self_attn.q_b_proj.weight"))
    wf16(f, get(f"{P}.self_attn.kv_a_proj_with_mqa.weight"))
    wf32(f, get(f"{P}.self_attn.kv_a_layernorm.weight"))
    wf16(f, get(f"{P}.self_attn.kv_b_proj.weight"))
    qwrite(f, get(f"{P}.self_attn.o_proj.weight"))
    wf32(f, get(f"{P}.post_attention_layernorm.weight"))
    wf32(f, get(f"{P}.mlp.gate.e_score_correction_bias"))
    wf16(f, get(f"{P}.mlp.gate.weight"))
    import time
    te0 = time.time()
    for ei in range(n_routed):
        qwrite_ffn(f"{P}.mlp.experts.{ei}", moe_inter_pad)
        if (ei + 1) % 32 == 0 or ei + 1 == n_routed:
            per_e = (time.time() - te0) / (ei + 1)
            print(f"    expert {ei+1}/{n_routed}  {per_e:.2f}s/expert  "
                  f"ETA={(per_e*(n_routed-ei-1))/60:.1f}min", flush=True)
    qwrite_ffn(f"{P}.mlp.shared_experts", shared_inter_pad)
    # output side
    wf32(f, get(f"{P}.shared_head.norm.weight"))
    if not tied_head:
        qwrite(f, get(f"{P}.shared_head.head.weight"))
    f.close()
    print(f"wrote {out_path}  {os.path.getsize(out_path)//(1024*1024)} MB "
          f"(tied_embed={tied_embed} tied_head={tied_head})", flush=True)


def make_selftest_mtp_weights(cfg, seed=1):
    # extend the parent selftest with a synthetic MTP module at layers.<n_layers>.
    rng = np.random.RandomState(seed)
    hidden, n_heads = cfg["hidden"], cfg["n_heads"]
    kv_lora, nope, rope, vhd = cfg["kv_lora"], cfg["nope"], cfg["rope"], cfg["vhd"]
    q_lora_rank, moe_inter = cfg["q_lora_rank"], cfg["moe_inter"]
    n_routed, n_shared, vocab = cfg["n_routed"], cfg["n_shared"], cfg["vocab"]

    def rnd(*shape):
        return (rng.randn(*shape) * 0.02).astype(np.float32)

    def onesish(n):
        return (1.0 + rng.randn(n) * 0.02).astype(np.float32)

    W = make_selftest_weights(cfg)
    P = f"model.layers.{cfg['n_layers']}"
    W[f"{P}.embed_tokens.weight"] = W["model.embed_tokens.weight"]      # tied, like R1
    W[f"{P}.shared_head.head.weight"] = W["lm_head.weight"]             # tied, like R1
    W[f"{P}.enorm.weight"] = onesish(hidden)
    W[f"{P}.hnorm.weight"] = onesish(hidden)
    W[f"{P}.eh_proj.weight"] = rnd(hidden, 2 * hidden)
    W[f"{P}.input_layernorm.weight"] = onesish(hidden)
    W[f"{P}.self_attn.q_a_proj.weight"] = rnd(q_lora_rank, hidden)
    W[f"{P}.self_attn.q_a_layernorm.weight"] = onesish(q_lora_rank)
    W[f"{P}.self_attn.q_b_proj.weight"] = rnd(n_heads * (nope + rope), q_lora_rank)
    W[f"{P}.self_attn.kv_a_proj_with_mqa.weight"] = rnd(kv_lora + rope, hidden)
    W[f"{P}.self_attn.kv_a_layernorm.weight"] = onesish(kv_lora)
    W[f"{P}.self_attn.kv_b_proj.weight"] = rnd(n_heads * (nope + vhd), kv_lora)
    W[f"{P}.self_attn.o_proj.weight"] = rnd(hidden, n_heads * vhd)
    W[f"{P}.post_attention_layernorm.weight"] = onesish(hidden)
    W[f"{P}.mlp.gate.e_score_correction_bias"] = rnd(n_routed)
    W[f"{P}.mlp.gate.weight"] = rnd(n_routed, hidden)
    for ei in range(n_routed):
        EP = f"{P}.mlp.experts.{ei}"
        W[f"{EP}.gate_proj.weight"] = rnd(moe_inter, hidden)
        W[f"{EP}.up_proj.weight"] = rnd(moe_inter, hidden)
        W[f"{EP}.down_proj.weight"] = rnd(hidden, moe_inter)
    SP = f"{P}.mlp.shared_experts"
    si = n_shared * moe_inter
    W[f"{SP}.gate_proj.weight"] = rnd(si, hidden)
    W[f"{SP}.up_proj.weight"] = rnd(si, hidden)
    W[f"{SP}.down_proj.weight"] = rnd(hidden, si)
    W[f"{P}.shared_head.norm.weight"] = onesish(hidden)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="HF snapshot dir (config.json + safetensors shards)")
    ap.add_argument("--out", default="/workspace/ds3")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force-own-head", action="store_true",
                    help="write embed/head even if tied to the main model's")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.selftest:
        cfg = dict(SELFTEST_CFG)
        W = make_selftest_mtp_weights(cfg)
        out_path = os.path.join(args.out, "mtp_selftest.cbk")
        if os.path.exists(out_path):
            os.remove(out_path)
        write_mtp(out_path, cfg, W.__getitem__, W.__contains__, args.force_own_head)
        print(f"\nselftest export OK: {out_path}")
        return

    assert args.dir, "--dir is required (or pass --selftest)"
    c = json.load(open(os.path.join(args.dir, "config.json")))
    assert c.get("num_nextn_predict_layers", 0) >= 1, "config has no MTP module"
    cfg = build_cfg_from_json(c)
    idx = LazySafetensors(args.dir)
    write_mtp(os.path.join(args.out, "mtp.cbk"), cfg, idx.get, idx.has, args.force_own_head)


if __name__ == "__main__":
    main()
