#!/usr/bin/env python3
"""Analyze a TRAPETUM_LOG_EXPERTS dump (one comma-separated line of routed expert ids per
MoE call, token-major layer-minor). Answers the two placement questions:

1. HOT-EXPERT CACHE: per-layer routing skew -- what fraction of picks land in the N hottest
   experts of each layer (N = 8/16/32/64), and the VRAM it would take to pin them on the GPU
   in 4-bit. Decides whether the free ~27 GB of VRAM buys a real hit rate.
2. SPEC-DECODE AMORTIZATION: adjacent-token expert overlap per layer -- when verifying M
   drafted tokens in one batched pass, overlapping picks decode the expert's bytes once.

  python model/analyze_expert_log.py experts.log --moe-layers 58 --n-routed 256 \
      --expert-mb 22.4
"""
import argparse, collections, sys

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--moe-layers", type=int, default=58, help="MoE calls per token (R1: 61-3 dense)")
ap.add_argument("--n-routed", type=int, default=256)
ap.add_argument("--expert-mb", type=float, default=22.4, help="bytes of one 4-bit expert, MB")
a = ap.parse_args()

rows = [tuple(int(x) for x in ln.split(",")) for ln in open(a.log) if ln.strip()]
L = a.moe_layers
if len(rows) % L:
    print(f"warning: {len(rows)} lines not divisible by {L} MoE layers; truncating", file=sys.stderr)
    rows = rows[: len(rows) // L * L]
T = len(rows) // L
print(f"{T} tokens x {L} MoE layers, top_k={len(rows[0])}, n_routed={a.n_routed}")

# ---- 1. skew / hot-cache coverage ----
per_layer = [collections.Counter() for _ in range(L)]
for i, picks in enumerate(rows):
    per_layer[i % L].update(picks)
print("\nhot-expert cache coverage (mean over layers; hit rate if top-N pinned per layer):")
for N in (8, 16, 32, 64):
    covs = []
    for li in range(L):
        tot = sum(per_layer[li].values())
        top = sum(c for _, c in per_layer[li].most_common(N))
        covs.append(top / max(tot, 1))
    vram_gb = L * N * a.expert_mb / 1024
    print(f"  top-{N:>2}/layer: hit {sum(covs)/L:.3f}  (min {min(covs):.3f}, max {max(covs):.3f})"
          f"   VRAM to pin: {vram_gb:.1f} GB")

# ---- 2. adjacent-token overlap (spec-decode amortization) ----
ovl = [0.0] * L; cnt = [0] * L
for t in range(T - 1):
    for li in range(L):
        s0 = set(rows[t * L + li]); s1 = set(rows[(t + 1) * L + li])
        ovl[li] += len(s0 & s1) / max(len(s0), 1)
        cnt[li] += 1
means = [ovl[li] / max(cnt[li], 1) for li in range(L)]
m = sum(means) / L
print(f"\nadjacent-token expert overlap: mean {m:.3f} (min {min(means):.3f}, max {max(means):.3f})")
print(f"  -> batched verify of 2 tokens reads ~{(2 - m) / 2:.2f}x the unique-expert bytes of 2 sequential tokens")
gini = []
for li in range(L):
    c = sorted(per_layer[li].get(e, 0) for e in range(a.n_routed))
    n = len(c); s = sum(c)
    g = (2 * sum((i + 1) * v for i, v in enumerate(c)) / (n * s) - (n + 1) / n) if s else 0.0
    gini.append(g)
print(f"per-layer pick Gini: mean {sum(gini)/L:.3f} (0=uniform, 1=max skew)")
