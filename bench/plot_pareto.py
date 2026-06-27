#!/usr/bin/env python3
"""
Pareto plot: decode throughput (tok/s, higher better) vs wikitext-2 perplexity
(lower better) from results.json. Each point is one (method, bits).

Usage: python plot_pareto.py results.json --out pareto.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"fp16": "#475569", "AQLM": "#4f46e5", "QTIP": "#15803d",
          "GPTQ": "#a16207", "AWQ": "#be185d", "Marlin": "#0891b2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--out", default="pareto.png")
    args = ap.parse_args()

    data = json.load(open(args.json))
    env = data.get("env", {})
    pts = [r for r in data["results"]
           if "wikitext2_ppl" in r and "decode_tok_per_s" in r]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    # bubble AREA proportional to peak VRAM -> shows the full speed/accuracy/memory trade-off
    for r in pts:
        c = COLORS.get(r["method"], "#334155")
        vram = r.get("peak_vram_gb", 4.0)
        ax.scatter(r["wikitext2_ppl"], r["decode_tok_per_s"], s=vram * 55, color=c,
                   alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate("%s (%sb)\n%.0f tok/s | %.2f PPL | %.1f GB" %
                    (r["method"], r.get("bits", "?"), r["decode_tok_per_s"],
                     r["wikitext2_ppl"], vram),
                    (r["wikitext2_ppl"], r["decode_tok_per_s"]),
                    textcoords="offset points", xytext=(10, 8), fontsize=9, color="#0f172a")

    gpu = env.get("gpu", "GPU")
    ax.set_xlabel("wikitext-2 perplexity  (lower is better)", fontsize=12)
    ax.set_ylabel("decode throughput, tok/s @ batch 1  (higher is better)", fontsize=12)
    ax.set_title("Quantization: speed vs accuracy vs memory on %s\n"
                 "bubble area = peak VRAM   (same HF forward harness, greedy, batch 1, seed 0)"
                 % gpu, fontsize=12)
    ax.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
