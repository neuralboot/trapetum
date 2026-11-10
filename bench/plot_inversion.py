# Inversion chain figure: 671B decode throughput on one g6e.16xlarge node (one L40S), from the
# first CPU-experts run to the steady state. Eight measured points, seven steps
# (RESULTS_deepseek.md sessions B -> H, each on a fresh g6e.16xlarge). Dashed line: GPU offload on the same node type, logged as 2.9 s/token
# (about 0.34 tok/s; 0.3 tok/s in the log's rounding). Clean white background for the paper. English labels.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, tok/s) for the eight measured points, in order.
steps = [
    ("CPU experts\n1st run", 0.24),
    ("work-steal\n+ AVX2", 0.44),
    ("MLA host\nabsorption", 0.96),
    ("register\ntranspose", 1.31),
    ("persistent\nscratch", 1.35),
    ("prewarm\ncache", 1.67),
    ("parallel\nmemset", 1.84),
    ("absorption\non GPU", 2.46),
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

OFFLOAD = 1 / 2.9   # GPU offload on the same node type: 2.9 s/token
for xi, v in zip(x, vals):
    y = v + 0.05 if v + 0.05 > OFFLOAD + 0.03 else OFFLOAD + 0.05   # keep the label clear of the dashed line
    ax.text(xi, y, f"{v:.2f}", ha="center", va="bottom", zorder=5,
            fontsize=12.5, fontweight="bold", color="#2a1207",
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("decode throughput (tok/s)", fontsize=13)
ax.set_ylim(0, 2.85)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e2e2e2", linewidth=0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# annotate the total gain
ax.annotate("", xy=(7, 2.66), xytext=(0, 0.58),
            arrowprops=dict(arrowstyle="->", color="#42863f", lw=1.6,
                            connectionstyle="arc3,rad=-0.22"))
ax.axhline(OFFLOAD, ls="--", color="#5b6572", lw=1.2, zorder=2)
ax.text(3.0, 2.62, "10.2x measured, same node type", ha="center",
        fontsize=13.5, fontweight="bold", color="#2a6a28")

ax.set_title("DeepSeek-R1 671B, 4-bit: each wall measured, then removed",
             fontsize=14.5, fontweight="bold", pad=14, color="#1a1a1a")
fig.text(0.5, -0.01, "Dashed line: GPU offload of the RAM-resident experts on the same node type, 2.9 s/token (about 0.34 tok/s).",
         ha="center", va="top", fontsize=10.5, color="#3b424c")
plt.tight_layout()
plt.savefig("paper/figures/inversion_chain.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote paper/figures/inversion_chain.png")
