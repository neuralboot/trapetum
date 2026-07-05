#!/usr/bin/env python3
"""
Patch a DeepSeek CBKD .cbk IN PLACE with the correct YaRN RoPE frequencies + the DeepSeek
softmax scale (mscale) + routed_scaling_factor, WITHOUT recompressing the weights. The
initial export wrote a plain inv_freq; DeepSeek-V2 uses YaRN rope scaling and an mscale-
adjusted attention scale, and scales the routed-expert sum by routed_scaling_factor. Only
the header floats + inv_freq change (same length), so this is a small in-place byte patch.

Header floats are stored as [eps, softmax_scale, routed_scaling_factor].
  python patch_deepseek_rope.py --model deepseek-ai/DeepSeek-V2-Lite --cbk /workspace/ds/model.cbk
"""
import argparse, math, struct
import numpy as np


def yarn_find_correction_dim(num_rot, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base, max_pos):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(mn, mx, dim):
    if mn == mx:
        mx += 0.001
    f = (np.arange(dim, dtype=np.float32) - mn) / (mx - mn)
    return np.clip(f, 0, 1)


def yarn_get_mscale(scale, mscale):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--cbk", default="/workspace/ds/model.cbk")
    args = ap.parse_args()
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    eps = c.rms_norm_eps
    base = float(getattr(c, "rope_theta", 10000.0))
    dim = c.qk_rope_head_dim
    q_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
    rscale = float(getattr(c, "routed_scaling_factor", 1.0))
    rs = getattr(c, "rope_scaling", None)

    softmax_scale = q_head_dim ** (-0.5)
    if rs and rs.get("type") == "yarn":
        factor = float(rs["factor"])
        beta_fast = float(rs.get("beta_fast", 32))
        beta_slow = float(rs.get("beta_slow", 1))
        orig_max = float(rs.get("original_max_position_embeddings", 4096))
        mscale_all = float(rs.get("mscale_all_dim", 0))
        idx = np.arange(0, dim, 2, dtype=np.float32)
        freq_extra = 1.0 / (base ** (idx / dim))
        freq_inter = 1.0 / (factor * base ** (idx / dim))
        low, high = yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max)
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        if mscale_all:
            m = yarn_get_mscale(factor, mscale_all)
            softmax_scale = softmax_scale * m * m
        print(f"YaRN: factor={factor} low={low} high={high} mscale_all={mscale_all} -> softmax_scale={softmax_scale:.5f}")
    else:
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        print(f"no yarn; softmax_scale={softmax_scale:.5f}")

    inv_freq = inv_freq.astype("<f4")
    assert inv_freq.size == dim // 2
    # header: magic(4) + 14 i32 (56) -> floats at 60; inv_freq at 72
    with open(args.cbk, "r+b") as f:
        f.seek(4)
        assert True
        f.seek(60)
        f.write(struct.pack("<3f", eps, softmax_scale, rscale))
        f.write(inv_freq.tobytes())
    print(f"patched {args.cbk}: eps={eps} softmax_scale={softmax_scale:.5f} rscale={rscale} inv_freq[{inv_freq.size}]")


if __name__ == "__main__":
    main()
