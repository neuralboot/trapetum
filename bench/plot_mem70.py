#!/usr/bin/env python3
"""Memory bar for Llama-2-70B: what fits on one GPU. python plot_mem70.py"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open("results_70b.json"))
rows = data["results"]
labels, vram, fits, est = [], [], [], []
for r in rows:
    labels.append("%s %s-bit" % (r["method"], int(r["bits"])))
    vram.append(r.get("peak_vram_gb", r.get("approx_vram_gb")))
    est.append("peak_vram_gb" not in r)
    fits.append(r.get("fits_on_80gb", True))

colors = []
for v, f in zip(vram, fits):
    colors.append("#dc2626" if not f else ("#16a34a" if v < 24 else "#d97706"))

fig, ax = plt.subplots(figsize=(9, 5.6))
bars = ax.bar(labels, vram, color=colors, edgecolor="white", linewidth=1.5, zorder=3, width=0.6)
for b, v, e in zip(bars, vram, est):
    txt = ("~%.0f GB (est., not run)" if e else "%.0f GB") % v
    if v < 30:   # short bar: value inside the bar, dark text, so it never meets the 24 GB line label
        ax.text(b.get_x() + b.get_width() / 2, v / 2, txt, ha="center", va="center", fontsize=11, fontweight="bold", color="#0f172a")
    else:
        ax.text(b.get_x() + b.get_width() / 2, v + 2, txt, ha="center", fontsize=11, fontweight="bold")

ax.axhline(80, ls="--", color="#1e293b", lw=1.5, zorder=2)
ax.text(0.985, 82, "1x H100  (80 GB)", color="#1e293b", fontsize=10, ha="right", transform=ax.get_yaxis_transform())   # inside the frame
ax.axhline(24, ls="--", color="#15803d", lw=1.5, zorder=2)
ax.text(0.985, 26, "RTX 4090  (24 GB)", color="#15803d", fontsize=10, ha="right", transform=ax.get_yaxis_transform())

ax.set_ylabel("VRAM (GB): peak measured, fp16 estimated", fontsize=12)
ax.set_ylim(0, max(vram) * 1.12)
ax.set_title("Llama-2-70B: fp16 needs 2+ GPUs. Quantized, it fits on ONE.\n"
             "AQLM 2-bit (21 GB) even fits a single RTX 4090.", fontsize=12)
ax.grid(True, axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig("mem70.png", dpi=160)
print("wrote mem70.png")
