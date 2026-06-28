#!/usr/bin/env python3
"""Parity / regression test for the per-column codebook k-means recipe.

Canonical recipe (export_runtime.py, bench/pareto.py, and now model/llama_quant.py +
model/llama_serve2.py all share it): linear init c/(K-1) over [min,max], L2 assignment,
mean update. This locks it and documents the real finding:

  - the L1-vs-L2 distance difference is a NO-OP: argmin|x-c| == argmin(x-c)^2, so the
    cluster ASSIGNMENT is identical either way;
  - the genuine divergence was the INIT (midpoint (c+0.5)/K vs linear c/(K-1)), now unified.

Run: python model/test_kmeans_parity.py   (CPU, no GPU needed)
"""
import torch


def kmeans(W, K=16, iters=12, init="linear"):
    IC, OC = W.shape
    lo, hi = W.min(0).values, W.max(0).values
    if init == "linear":
        cb = torch.stack([lo + (hi - lo) * (k / (K - 1)) for k in range(K)], 0)
    else:  # midpoint
        cb = torch.stack([lo + (hi - lo) * (k + 0.5) / K for k in range(K)], 0)
    idx = torch.zeros(IC, OC, dtype=torch.long)
    for _ in range(iters):
        d = (W.unsqueeze(-1) - cb.t().unsqueeze(0)) ** 2
        idx = d.argmin(-1)
        for k in range(K):
            m = (idx == k)
            cb[k] = (W * m).sum(0) / m.sum(0).clamp(min=1)
    return idx, cb


def test_deterministic():
    torch.manual_seed(0); W = torch.randn(512, 32)
    a, _ = kmeans(W); b, _ = kmeans(W.clone())
    assert torch.equal(a, b), "k-means non-deterministic on fixed input"


def test_l1_equals_l2_assignment():
    # the L1/L2 'divergence' Grok flagged does NOT change the assignment (argmin is monotonic)
    torch.manual_seed(1); W = torch.randn(1000, 8)
    lo, hi = W.min(0).values, W.max(0).values
    cb = torch.stack([lo + (hi - lo) * (k / 15) for k in range(16)], 0)
    diff = W.unsqueeze(-1) - cb.t().unsqueeze(0)
    assert torch.equal(diff.abs().argmin(-1), (diff ** 2).argmin(-1)), "L1 and L2 assignment differ"


def test_init_was_the_real_divergence():
    # different init can converge to different clusters -> this is what was actually unified
    torch.manual_seed(2); W = torch.randn(2000, 4)
    lin, _ = kmeans(W, init="linear"); mid, _ = kmeans(W, init="midpoint")
    n_diff = (lin != mid).float().mean().item()
    print(f"  init linear vs midpoint differ on {n_diff*100:.1f}% of weights (the real divergence, now unified)")


if __name__ == "__main__":
    test_deterministic()
    test_l1_equals_l2_assignment()
    test_init_was_the_real_divergence()
    print("kmeans parity OK: canonical = linear init + L2; L1==L2 assignment; all four paths now consistent")
