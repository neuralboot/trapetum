import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
gpus=["RTX 4090\n~1.0 TB/s","A40\n~0.7 TB/s","H100 PCIe\n~3.3 TB/s"]
sp=[2.20,2.34,0.99]
col=["#16a34a" if s>1.05 else "#d97706" for s in sp]
fig,ax=plt.subplots(figsize=(8,5))
b=ax.bar(gpus,sp,color=col,edgecolor="white",linewidth=1.5,width=0.6,zorder=3)
for bar,s in zip(b,sp): ax.text(bar.get_x()+bar.get_width()/2,s+0.06,"x%.2f"%s,ha="center",fontsize=13,fontweight="bold")
ax.axhline(1.0,ls="--",color="#1e293b",lw=1.5,zorder=2)
ax.text(-0.45,1.06,"cuBLAS fp16 baseline (parity = x1.0)",ha="left",color="#1e293b",fontsize=10)
ax.set_ylabel("decode speedup vs cuBLAS fp16 GEMV",fontsize=12)
ax.set_ylim(0,2.7); ax.set_title("Fused 4-bit codebook GEMV: decode speedup (measured, batch 1, 4096x4096)\nThe more bandwidth-limited the GPU, the bigger the win. 4x less memory throughout.",fontsize=11)
ax.grid(True,axis="y",alpha=0.3,zorder=0); fig.tight_layout(); fig.savefig("gpu_speedup.png",dpi=160); print("wrote gpu_speedup.png")
