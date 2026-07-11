# DeepSeek-V2/V3 (MLA + MoE) support — end-to-end result

DeepSeek-V2-Lite (16B: MLA attention + 64-expert MoE, 2.4B active) runs **end-to-end in
the pure-Rust runtime** (RTX 4090), no Python at inference. Measured July 2026.

Prompt:  "The capital of France is"
Output:  " a city of 36 arrondissements, 20 of which are in the city of Paris. The"

Coherent English, mentions Paris. ~10 tok/s, loads in ~8s. The 4-bit codebook quantization
(experts) + dense-fp16 MLA projections; config L=27, kv_lora=512, nope=128, rope=64,
v_head_dim=128, n_routed=64, n_shared=2, top_k=6, vocab=102400.

## The three walls, solved
- **MLA** (Multi-head Latent Attention): `mla_attn` kernel (absorption form) + `MlaAttn`
  block (q/kv proj + W_UK/W_UV absorption + decoupled RoPE + shared low-rank latent cache).
- **MoE** (routing): `MoeBlock` (router dense fp16 + top-k + expert FFN + shared expert).
- **Memory** (671B doesn't fit): `MoeBlockOffload` streams top-k experts from host (LRU);
  validated lossless. (V2-Lite fits fully; offload is for V3-671B.)

## Pipeline
- `model/export_deepseek.py`: HF DeepSeek -> CBKD .cbk (MLA dense fp16, experts/MLP/lm_head 4-bit).
- `model/patch_deepseek_rope.py`: patches YaRN inv_freq + mscale softmax_scale + routed_scaling in place.
- `DeepSeekModel::load_deepseek` + forward; `deepseek_run` bin.

## Bugs found & fixed on the way (blind-written pipeline, debugged on GPU)
1. flash_attn required by remote modeling code -> prebuilt wheel.
2. CUDA OOM loading 16B on 24GB -> load on CPU, per-linear k-means on GPU.
3. router n_routed=64 not %256 -> router kept dense fp16.
4. moe_inter=1408 / inter_dense=10944 not %256 -> zero-pad intermediate (lossless).
5. shared expert inter = n_shared*moe_inter (bigger than routed) -> separate shared_inter + scratch.
6. YaRN RoPE scaling + mscale softmax scale + routed_scaling_factor -> patch_deepseek_rope.py.
7. **INTERLEAVED RoPE** (DeepSeek reshapes view(d/2,2).transpose before rotate_half), not
   Llama split-half -> the incoherent-output cause. Fixed = coherent text.

## Caveat (historical, now resolved)
V2-Lite has plain q_proj (q_lora_rank=null); the q_lora variant (V2 full / V3) needed
q_a_proj/q_b_proj export. Both walls fell in July 2026: see the 671B section below.

# DeepSeek-R1 671B: the full-size model, pure Rust (July 2026)

DeepSeek-R1 671B (the full V3 architecture: MLA with low-rank query projection,
256 routed experts, sigmoid noaux_tc router) loads and decodes coherently in the
pure-Rust runtime on a single node.

Prompt:  "The capital of France is"
Output:  " Paris. Paris is located in northern France and is known for its iconic
          landmarks such as the Eiffel Tower, Notre"

Measured numbers (raw logs in `runpod_logs/r1_671b_export.log` and
`runpod_logs/r1_671b_first_run.log`):

- **Export**: 1.34 TB bf16 checkpoint -> **326 GB** 4-bit CBKR artifact, produced by
  the torch-free streaming exporter (`model/export_deepseek_stream.py`): bounded RAM,
  resumable via a progress sidecar, ~7 h on one A100 80GB pod.
- **Load**: 43.3 s for all 61 layers (q_lora path: q_a 7168->1536, RMSNorm, q_b 1536->24576).
- **Memory**: **73 GB peak host RAM**. Routed experts are not loaded: their packed 4-bit
  indices stay mmap-backed on disk and are paged in on demand per token
  (`PackedBytes::Mmap`), while the f32 codebooks stay in RAM. Before the mmap path the
  same load needed 326 GB and died at layer 44 under the pod's 250 GB cgroup cap.
- **Speed**: 0.2 tok/s on the first pass (5.9 s/token), entirely first-touch disk paging
  on network-attached storage. Steady-state measured July 6 on local NVMe (64 tokens,
  per-token timing, immediate warm rerun): **~0.1 tok/s** (10.2 s/token), warm rerun
  identical. Verdict: with ~100 GB RAM against a 350 GB artifact the per-token expert
  working set (top-8 x 58 MoE layers) never fits in page cache, so decode stays
  disk-bandwidth-bound; the lever is RAM (or expert prefetch), not cache warm-up.
- **Quality**: the 64-token continuation is fully coherent: "Paris. Paris is located in
  northern France and is known for its iconic landmarks such as the Eiffel Tower,
  Notre-Dame Cathedral, and the Louvre Museum. It is also a global center for art,
  fashion, gastronomy, and culture..." (raw per-token logs in
  `runpod_logs/r1_671b_steady.log` and `r1_671b_warm.log`).

## The consumer money shot (July 7, 2026)
The same 350 GB artifact then ran on a **single RTX 4090** (consumer card, 24 GB), on a
host with 48 GB RAM and local NVMe:

- **20.9 GB peak VRAM** (dense MLA projections + codebooks on GPU): it fits, with margin.
- **0.24 tok/s steady state** (4.18 s/token over 64 tokens, warm rerun identical), faster
  than the A100 host thanks to a quicker NVMe: confirming decode is disk-bound, not
  GPU-bound, at this RAM size.
- Host RAM in use: 9 GB anonymous plus page cache; the experts stay mmap-backed on disk.
- Same fully coherent 64-token continuation (Paris, Eiffel Tower, Louvre...). Raw logs:
  `runpod_logs/r1_671b_4090.log` and `r1_671b_4090_warm.log`.

Headline: **DeepSeek-R1 671B, the largest open-source model, decodes coherently on one
consumer RTX 4090** in a from-scratch pure-Rust runtime. Throughput is disk-bound
(~0.24 tok/s): a proof of reach, not a serving speed; RAM or expert prefetch is the lever.

## What this establishes
The largest open-source model runs end to end in a from-scratch Rust runtime with no
Python, no PyTorch, and no GPU requirement for the expert weights: one box, 73 GB of
RAM, and 326 GB of disk. The V3-specific machinery validated on the way: q_lora MLA,
noaux_tc sigmoid routing with e_score_correction_bias and group top-k (n_group 8,
topk_group 4, routed_scaling 2.5, raw-sigmoid weight renormalization), first 3 layers
dense, 2048-dim expert FFNs.

## Speculative-decode ceiling: measured, and it is a no-go for the disk-bound case
Routed expert ids logged for 38 tokens x 58 MoE layers (raw:
`runpod_logs/r1_671b_expert_routing.csv`, greedy, "The capital of France is").
Adjacent tokens share only **2.22 of 8 experts (28%)**: the aux-loss-free router
actively decorrelates neighbors. Byte-amortization ceiling for a batched verify:
x1.16 (K=1), x1.28 (K=2), x1.37 (K=3), BEFORE acceptance and drafter cost. Net
expectation ~x1.1-1.25: not worth the machinery while decode is disk-bound.
Speculative decode remains a strong lever once weights are memory-resident
(compute-bound regime). The better disk-bound lever is asynchronous expert
prefetch: a layer's 8 experts are known before its FFN runs, and today's reads
are serial page faults.

## Expert prefetch (madvise WILLNEED): no effect on network-volume storage
A/B/A on one A100 reading the artifact from the RunPod network volume
(24 tokens each): baseline 22.6 s/token, prefetch 23.9, baseline-again 24.9.
The prefetch lands inside the storage drift: madvise readahead appears to be
a no-op on the network mount. The relevant verdict, on LOCAL NVMe where kernel
readahead actually queues reads, is still open. Code stays in the runtime
behind TRAPETUM_PREFETCH=1.

## 2-bit additive experts: quality measured (the x100 gate), V2-Lite proxy
Probe: model/probe_avq2bit_moe.py (in-place dequantized simulation, wikitext-2,
61k tokens, baseline bf16 PPL 6.8243). Routed experts to 2-bit additive (2x8,
group 8), everything else untouched:
- fast quantizer (beam 1):        PPL 8.6954  (+27.4%)
- paper quantizer (beam 4, LSQ3): PPL 8.2608  (+21.1%)
- dynamic mix (first 2 MoE layers spared, beam 4): PPL 8.1320 (+19.2%)
- paper quantizer 3-bit (M=3, beam 4, LSQ3): PPL 7.1755 (**+5.2%**, near-lossless), 228 GB artifact
Continuations stay coherent but flatten. Expert compression 7.75x vs fp16;
projected 671B artifact with 2-bit experts: 152 GB (fits 192 GB RAM, the
RAM-resident x100 path). Open question: the real 671B has 256 experts/layer
vs 64 here (more redundancy, quantizes better), but a direct PPL probe needs
1.34 TB of RAM to simulate, so the next honest step is the runtime AVQ port
plus a real 671B 2-bit export. A 3-bit variant (228 GB, ~x50 speedup path)
is the fallback if +19% is judged too costly.

## 4-bit 671B fully RAM-resident: the disk was NOT the only wall (July 10, 2026)
Measured on AWS g5.24xlarge (A10G, 373 GB RAM, us-east-1): the 350 GB 4-bit artifact
downloaded from S3 and read fully into page cache (339 GB cached, 38 s warm read),
so every expert byte is in RAM, zero disk paging.

Result: **~0.33 tok/s** steady state (3065 ms/token, warm rerun 0.31), coherent output.
Versus 0.24 tok/s disk-bound (RTX 4090): only **~1.4x**, not the 20-40x a pure
memory-bandwidth model predicts.

Verdict (important): RAM residency removes the disk read (~1 s/token saved) but reveals
the real per-token bottleneck: streaming ~10 GB of routed-expert weights from host RAM
to the GPU across PCIe, plus the single-GPU expert compute (weak on the A10G). The 671B
decode is not disk-bound-only; it is bound by getting the experts onto ONE GPU each
token. True interactive speed needs the experts resident in VRAM (multi-GPU), not just
in host RAM. This reframes the "x100 on a consumer box" goal: fitting the model in RAM
is necessary but not sufficient; the expert-to-GPU transfer is the next wall. Raw log:
runpod_logs/r1_671b_4bit_ram_resident.log. Caveat: A10G is a weak GPU; a 4090/A100 host
with RAM residency would be somewhat faster, but the host->device transfer wall stands.

## CPU probes S12/S13-M/S15: the inversion is measured (M4, 2026-07-11)

The K=16 codebook decode is one NEON `tbl` (x86: `pshufb`) per 32 weights, so a
CPU expert path is memory-bound, never decode-bound. Three probes, see
`bench/cpu_probes/`:

- S12 micro-kernel (fused tbl decode + SDOT GEMV): 53.1 GB/s packed at 8 threads,
  linear scaling (6.6 at 1T, 26.3 at 4T).
- S13-M full expert (gate+up+SiLU+int8 requant+down, per-row scales, 58 layers x
  8 random experts, work-stealing, barriers): 47.0 GB/s sustained, 217 ms/token
  expert-side = 4.6 tok/s. Full-structure overhead vs micro-kernel: 11.5%.
- S15 complete 671B decode token (MLA attention real dims + 4K KV + router +
  dense + shared + routed + lm_head = 18.46 GB/token): 361 ms/token = 2.77 tok/s
  FULL MODEL pure CPU on an entry-level laptop chip, 51.1 GB/s = 96% of the
  micro-kernel. Component ms: MoE 187.5, attention 138.5, dense 15.1, KV 11.5,
  lm_head 8.7.

Reference points: our 4090+NVMe system does 0.24 tok/s and RAM-resident 0.33.
Pure-CPU M4 full-pipeline is x11.5 the 4090 offload path, losslessly (same
artifact, same arithmetic). Residual-overlap accounting: MoE output is a
commutative sum (shared expert + routed), so on a discrete-GPU host the
non-routed terms (173.8 ms on CPU here) are VRAM-resident (~9 ms) and run
concurrently -> t_token ~ t_routed + eps. Projections (50% derate): desktop
DDR5 ~8-10 tok/s, EPYC 12ch ~22, M3 Ultra 512GB ~15-25 end-to-end lossless.
Next: S14 = hybrid path in the Rust runtime on V2-Lite (tok/s + greedy match).

## S14: hybrid CPU-experts path, measured end-to-end on real V2-Lite (pod 4090 + 128 vCPU, 2026-07-11)

Branch s14-cpu-experts, 8 increments. TRAPETUM_CPU_EXPERTS=1 keeps routed expert
weights host-side (never uploaded to VRAM) and executes them on CPU via a
row-major work-stealing engine; attention/router/shared/dense stay on GPU.

- Baseline pure GPU: 98 ms/token (10.2 tok/s). Hybrid: 150 ms/token (6.7 tok/s)
  at the 16-thread sweet spot (64 threads collapses to 5.5 ms/layer:
  oversubscription with spin barriers, same pathology as on the M4).
  Per-layer split: routed_cpu ~2 ms, shared_gpu 0.05 ms. Expected: V2-Lite fits
  in VRAM so pure GPU wins here; the hybrid is the only fast path for models
  that do not fit (671B-class).
- FIDELITY FINDINGS (the real yield of S14):
  1. The pure-GPU baseline itself is run-to-run NONDETERMINISTIC: the fused
     codebook GEMV accumulates IC-slices across grid.y=20 blocks with atomicAdd
     (both CUDA and Metal). Kernel probe: 30/30 runs bitwise-differ at GS=20,
     0/30 at grid.y=1. Near-tie logits flip; greedy diverges by token 3.
  2. TRAPETUM_DETERMINISTIC=1 (grid.y=1, both backends) makes decode
     byte-stable. Under it, pre-S14 main == branch flag-off BYTE-FOR-BYTE
     (2x2 runs): no regression from the S14 refactor, triple-confirmed
     (empirical A/B, author audit, independent adversarial source review).
  3. fp16-GPU vs f32-CPU expert arithmetic is a genuine ~1-logit systematic
     difference after 26 layers: deterministic GPU picks token 254 (margin
     0.66), deterministic hybrid picks 245 (margin 1.17). Two valid arithmetics
     of the same artifact: raw-id equality is not a meaningful lossless gate;
     per-path determinism + logit margins / PPL is (S17 reframed).
- Parked: Option B fast deterministic kernel (two-stage fixed-order reduction,
  est. +2-5%); persistent thread pool (per-layer spawn ~100us compounds);
  CUDA-side shared/routed overlap (Metal has it, bit-identical);
  export coherence check (nobody answers Paris: pre-existing).
- Pod gotchas burned: DeepSeek modeling file demands flash_attn at import
  (stub it), hf_xet crashes (HF_HUB_DISABLE_XET=1), HF cache must live on the
  volume not the 30GB container disk, pkill -f self-matches your own ssh
  command line.
