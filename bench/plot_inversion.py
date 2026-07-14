# Inversion chain figure: 671B decode throughput, disk-offload baseline to CPU-experts
# steady state. Eight measured walls, each removed in turn (RESULTS_deepseek.md
# sessions B -> H). Clean white-background for the paper. English labels.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, tok/s) for the eight measured walls, in order, each one removed.
steps = [
    ("disk offload\nbaseline", 0.24),
    ("AVX2\nkernel", 0.44),
    ("MLA host\nabsorption", 0.96),
    ("register\ntranspose", 1.31),
    ("persistent\nscratch", 1.35),
    ("prewarm\ncache", 1.67),
    ("memset\nPREP fix", 1.84),
    ("attention\non GPU", 2.46),
]
labels = [s[0] for s in steps]
vals = [s[1] for s in steps]
x = list(range(len(vals)))

RUST = "#f74c00"
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9.6, 5.0))

bars = ax.bar(x, vals, width=0.66, color=RUST, edgecolor="#7a2a08", linewidth=0.8, zorder=3)
# fade the early bars into the final full-saturation win
for b, v in zip(bars, vals):
    b.set_alpha(0.45 + 0.55 * (v - 0.24) / (2.46 - 0.24))

for xi, v in zip(x, vals):
    ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom",
            fontsize=12.5, fontweight="bold", color="#2a1207")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("decode throughput (tok/s)", fontsize=13)
ax.set_ylim(0, 2.85)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e2e2e2", linewidth=0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# annotate the total gain
ax.annotate("", xy=(7, 2.66), xytext=(0, 0.42),
            arrowprops=dict(arrowstyle="->", color="#42863f", lw=1.6,
                            connectionstyle="arc3,rad=-0.22"))
ax.text(3.4, 2.62, "x10.2 measured, one machine", ha="center",
        fontsize=13.5, fontweight="bold", color="#2a6a28")

ax.set_title("DeepSeek-R1 671B, 4-bit: each wall measured, then removed",
             fontsize=14.5, fontweight="bold", pad=14, color="#1a1a1a")
plt.tight_layout()
plt.savefig("paper/figures/inversion_chain.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote paper/figures/inversion_chain.png")
