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
pure-Rust runtime on a single node. **Current result (session H, below): the full
model serves at 2.46 tok/s steady state via the CPU-experts inversion, a x10.2 gain
over the 0.24 tok/s disk-offload baseline. The sections below are the chronological
measured journey to it.**

Prompt:  "The capital of France is"
Output:  " Paris. Paris is located in northern France and is known for its iconic
          landmarks such as the Eiffel Tower, Notre"

Measured numbers (raw logs in `runpod_logs/r1_671b_export.log` and
`runpod_logs/r1_671b_first_run.log`):

- **Export**: 1.34 TB bf16 checkpoint -> **350 GB** 4-bit CBKR artifact, produced by
  the torch-free streaming exporter (`model/export_deepseek_stream.py`): bounded RAM,
  resumable via a progress sidecar, ~7 h on one A100 80GB pod.
- **Load**: 43.3 s for all 61 layers (q_lora path: q_a 7168->1536, RMSNorm, q_b 1536->24576).
- **Memory**: **73 GB peak host RAM**. Routed experts are not loaded: their packed 4-bit
  indices stay mmap-backed on disk and are paged in on demand per token
  (`PackedBytes::Mmap`), while the f32 codebooks stay in RAM. Before the mmap path the
  same load needed 350 GB and died at layer 44 under the pod's 250 GB cgroup cap.
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

Headline (July 7, historical): **DeepSeek-R1 671B decodes coherently on one consumer
RTX 4090** in a from-scratch pure-Rust runtime. Throughput here is disk-bound (~0.24
tok/s): a proof of reach, not a serving speed. **Superseded by the inversion (session H
below): the same 671B serves at 2.46 tok/s steady state on one 64-vCPU node, x10.2.**

## What this establishes
The largest open-source model runs end to end in a from-scratch Rust runtime with no
Python, no PyTorch, and no GPU requirement for the expert weights: one box, 73 GB of
RAM, and 350 GB of disk. The V3-specific machinery validated on the way: q_lora MLA,
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

## The Paris bug: found, fixed, validated (2026-07-11, branch det-kernel-pool)

Layer-bisect campaign (deliverables 4-6) on real V2-Lite, judged against HF with
IDENTICAL dequantized 4-bit weights:
- SYMPTOM: runtime output incoherent ("a big fan of the work...") while HF fp16
  answers " Paris." cleanly; divergence systematic from layer 0 (rel l2 0.26).
- ROOT CAUSE (commit 9c0f439): export_deepseek.py wrote rope_theta (10000) into
  the header slot the runtime reads as softmax_scale (correct: q_head_dim^-0.5
  * mscale^2 = 0.114721 for V2-Lite) -> near-one-hot attention since the FIRST
  DeepSeek run. Secondary: inv_freq was plain rope, not yarn-corrected (up to
  97% per-frequency error). The Rust MLA math itself was correct. The CBKR
  (V3/671B) path had the SAME two bugs: the S3 671B artifact is affected and
  should be re-exported.
- VALIDATION (pod, fixed artifact): attn_out Rust 2.311358 vs HF 2.311895
  (fp16 tolerance; was 28% off); GPU-det and HYBRID paths now produce IDENTICAL
  16-token greedy sequences; output is on-topic ("a city of 36 arrondissements,
  20 of which" = Paris semantics). Remaining gap to " Paris" as token 1 is the
  4-bit quantization near-tie (HF-quantized also has Paris only 3rd at ~0.7
  logits), not the runtime.
- S18 spectral probe (~5000 tensors): quantization damage is diffuse (median
  top-16 SV rel err 1.28%, max 2.65%, no catastrophic tensor), but 13 of the
  15 worst tensors are SHARED experts -> cheap mixed-precision policy: shared
  experts + lm_head at 6-8 bit (52 tensors, negligible size, every token path).

## 671B inversion, first real measure (AWS g6e.16xlarge, 2026-07-12): NEGATIVE, levers identified

Setup: 64 vCPU Xeon, 497 GB RAM, L40S 48 GB, artifact RAM-resident (336 GB page
cache verified), branch s19-mixed-precision (deliverable A CPU-experts path).
- Baseline GPU-offload: 0.3 tok/s (2.9 s/token): consistent with history.
- HYBRID TRAPETUM_CPU_EXPERTS=1: 0.24 tok/s (4.2 s/token) at 16/32/48 threads.
  THE INVERSION DID NOT PAY ON x86 YET. Effective expert throughput ~2.4 GB/s
  vs the 50-100 needed.
- Root cause (diagnosed live): (1) the CPU decode kernel is SCALAR on x86: the
  NEON tbl path is aarch64-only and the vpshufb x86 twin (probe S13's design)
  was never written; scalar i-outer measured ~1.2 GB/s single-thread on M4 and
  x86 is comparable; (2) per-GEMV pool dispatch grain: down-proj at ic=2048 has
  only 8 chunks for 32-48 threads, and 1392 dispatches/token serialize; both
  were explicitly flagged as deferred levers in deliverable A.
- The physics still stands: the model IS RAM-resident and the GPU sits idle at
  ~0-5% during hybrid decode: the bytes are in the right place, the compute
  kernel is not. Next: deliverable B = AVX2/AVX-512 vpshufb kernel + one phased
  work-steal over all (expert,chunk) tasks; validate on a cheap CPU-only spot
  box, then redo the 671B session (fully scripted, ~40 min to reproduce).
- Cost of the session: ~12 USD. Also noted: prompt.bin had a double BOS
  (tokenizer auto-adds id 0); fix the generator next run.

## 671B v3 rerun with deliverable B (AWS g6e.16xlarge, 2026-07-12): 0.44 tok/s, gather is the wall

- HYBRID AVX: 2271 ms/token (0.44 tok/s), thread-insensitive (32/48/60 identical).
- Scalar A/B: 2500 ms/token: the AVX gather kernel buys only 8%.
- MAP_POPULATE experiment: no change (minor-fault hypothesis refuted cleanly;
  populate verified working: load 26s -> 50.6s).
- Live utilization sampling: ~460-1000% CPU of 6400% = only ~9-16 of 32 workers
  effective; ~0.5-0.7 GB/s per core.
- DIAGNOSIS: two compounding ceilings. (1) per-core: AVX2 gather decode of the
  per-column f32 codebook is ~1 elem/cycle even L2-resident: ~0.7 GB/s/core;
  even perfect 32-core engagement would cap at ~22 GB/s = ~2 tok/s. (2)
  engagement: pool wake latency + phase barriers x58 calls/token + tail
  imbalance keep half the workers idle.
- Deliverable B improved 0.24 -> 0.44 tok/s (the work-steal grain paid x1.8);
  the remaining wall is the DECODE ITSELF, exactly as the deliverable-B report
  flagged: hardware gather cannot reach the tbl/pshufb 47 GB/s class.
- Deliverable C scoped, two candidate kernels to MEASURE (not assume): (C1)
  AVX-512 vpermps register-resident 16xf32 LUT with an (i-block x o-block)
  register-transpose tiling: EXACT, no contract change, hardest engineering;
  (C2) int8 codebook recode + vpshufb value-LUT (the C-probe design, 47 GB/s
  class): adds quantization of the codebook itself (error scale comparable to
  the already-accepted fp16-vs-f32 path difference; gates are margins/PPL since
  the determinism campaign). Build both behind flags, measure, pick.
- Session cost ~9 USD. All boxes terminated.

## 671B session 3 (2026-07-12): the THIRD wall is MLA attention + both prior levers exonerated

Deliverable C fully measured (engagement + lut8), and the floor did not move:
- Engagement A/B: futex 2316 ms/token vs spin 2201 (5%); ENGAGE_DEBUG proves
  workers=32 active=32, tasks balanced 13-26 -> ENGAGEMENT EXONERATED (the
  earlier 9-16/32 top-sampling read memory-stalled threads as idle).
- lut8 vs gather vs scalar: 2319 / 2210 / 2500 ms/token -> KERNEL EXONERATED.
- Box raw memory: 11.0 GB/s page-cache read on ONE thread -> BOX EXONERATED
  (and the sequential-TLB hypothesis with it).
- gdb thread-stack sampling during decode (PMU is blocked on virtualized EC2;
  stacks are not): main thread 2/5 samples inside MlaAttn::forward, 3/5 parked
  in Pool::run waiting on workers -> per token roughly ~0.9 s of GPU MLA and
  ~1.4 s of CPU-MoE phase.
- MLA at ~0.9 s/token with VRAM-resident weights is the new prime suspect:
  671B exercises the q_lora path (q_lora_rank=1536, absent on V2-Lite) and
  128 heads x 61 layers of per-op launches/syncs can alone cost ~0.8 s. The
  MoE-CPU phase at ~7 GB/s aggregate also has headroom (workers active but
  memory-stalled; per-GEMV x-quantization and reduce tails to profile).
- Next: deliverable E = (1) MLA batching/sync audit on the CUDA path for the
  671B config (batch per-head ops, kill per-op drains, inspect q_lora), with
  env-gated per-component timing (TRAPETUM_MLA_TIMING) so the box can split
  attention vs MoE vs rest per token; (2) MoE-phase micro-profiling hooks.
- Session ~2h ~15 USD. Boxes terminated. Cumulative finding chain:
  disk -> RAM -> faults -> grain -> gather -> engagement -> MLA serial path.

## 671B sessions 4+4b (2026-07-12): MLA fix pays x2.2 -> 0.96 tok/s; the per-core atom; F2 lesson

- MLA host-absorption fix (deliverable E): attention 900 -> 290 ms/token; TOTAL
  2271 -> 1046 ms/token = 0.96 tok/s. First time under 1.1 s.
- Dissection (TRAPETUM_MLA_TIMING/MOE_TIMING): attention=290 moe=831 other=1.6;
  moe split A(gate+up)=387 RA=21 C(down)=186 RC=101; decode dominates, ~140 ms
  unaccounted per-call overhead.
- THE ATOM: T=1 moe=15380 ms -> 0.66 GB/s per core on the gather kernel (warm
  mmap). Thread scaling GOOD (T16 12.5x, T32 18.5x): the wall is PER-CORE
  latency, not shared contention (the earlier thread-insensitivity was the
  MLA-host floor masking everything).
- F1 MADV_HUGEPAGE: -9% only. TLB/page-walk hypothesis largely refuted as the
  main wall; 0.66 GB/s/core is gather-latency territory, and lut8 (no gather)
  matching it suggests both kernels serialize ~equally for different reasons
  (vpgatherdd latency vs per-byte-position nibble extraction).
- F2 anon-THP arena: KILLED THE BOX: staging 350 GB anon while the page cache
  still held ~336 GB wedged the kernel in direct reclaim (SSH dead, instance
  terminated). LESSON: staging must fadvise(DONTNEED)/drop the model's page
  cache BEFORE allocating the arena, with a staging progress print + free-RAM
  guard. The scripted F2 run also failed silently earlier (empty output):
  same cause.
- lut8 correctness gate: cargo test output captured EMPTY twice; unresolved,
  suspicious (filter name? test compilation?). To nail next session.
- Next levers ranked: (1) per-core decode kernel: the in-register 16x16
  transpose on lut8 (marked hotspot) OR a gather-free vpermps tile: target
  2-4 GB/s/core = moe ~200-400 ms; (2) the ~140 ms per-call overhead gap;
  (3) GPU-side absorption for the remaining 290 ms attention; (4) F2 retried
  WITH cache-drop staging. Ceiling if (1)+(2) land: ~400-500 ms/token
  (~2-2.5 tok/s) on this box class.
- Session ~1.5 h ~11 USD. Program totals: ~55 USD AWS + 4 RunPod.

## 671B session 5 (2026-07-12): the transpose lands -- 1.31 tok/s (x5.5 from 0.24)

- lut8 16x16 in-register transpose (lever #1): per-core decode atom 0.66 ->
  3.3 GB/s (x5), gate bit-exact on AVX2. But total was masked until:
- recode+transpose HOISTED to a per-expert lazy cache (lever #2b): removed
  ~710 ms/token of per-call codebook prep.
- RESULT (T=32, steady, last 12): 762 ms/token = 1.31 tok/s. Chain:
  0.24 -> 0.44 -> 0.96 -> 1.31. Best config T=32 (T=48/60 regress: the reduce
  phase RC climbs 64->97->129 ms with threads = reduce contention/oversub).
- Dissection at 1.31: attention=287 (38%), moe=613. But moe phases only sum to
  A99+RA10+C54+RC64 = 227: ~386 ms/token STILL unaccounted inside the MoE
  forward (not the 4 timed phases): candidates = host activation copy +
  per-GEMV x-quantization + combine + pool handoff. This is the next lever, and
  it is now the BIGGEST single chunk (bigger than attention).
- Decode itself is essentially solved (A+C = 153 ms for 10 GB = ~67 GB/s
  aggregate, near the read floor): the wall moved from compute to per-call
  orchestration overhead + attention.
- Next levers ranked: (1) the ~386 ms unaccounted MoE per-call overhead
  (instrument the gap: activation quant, combine, handoff); (2) the 287 ms
  attention (GPU-side absorption); (3) reduce-phase RC contention at high T.
  If (1) halves and (2) drops to ~50: token ~ 480 ms = ~2 tok/s still in reach.
- Session ~1h ~8 USD. Program: ~63 USD AWS + 4 RunPod. Milestone: the inversion
  runs the full 671B at 1.31 tok/s lossless-4bit on ONE 64-vCPU box, no
  datacenter -- interactive-adjacent, x5.5 the disk-offload baseline.

## 671B session G (2026-07-12): the 386ms MoE gap SPLIT -- it is ws_entry (per-call overhead), NOT the shared expert

Whole-forward instrumentation (G-inc1) at steady state (pos 17-18):
- moe wall ~455 ms = phases(A98+RA10+C41+RC28 ~177) + ws_entry(~213) + xcopy(1.4)
  + shared(6.6).
- SHARED EXPERT = 6.6 ms: the residual-overlap prior is REFUTED, shared is already
  trivial (VRAM-resident, fast); no overlap lever there.
- XCOPY (device->host activation drain) = 1.4 ms: trivial.
- ws_entry = ws_total - timed phases = ~213 ms = THE GAP. It is the per-call
  work-steal overhead OUTSIDE decode/reduce: int8 activation x-quantization,
  codebook-cache HashMap lookup (464/token), pool submit/handoff (spin x58),
  and the routed combine/saxpy. ~3.7 ms/layer of pure orchestration.
- Token budget now: attention 291 + moe 455 (phases 177 + ws_entry 213 + misc) =
  ~776 ms = 1.29 tok/s steady.
- THE TWO REAL TARGETS, cleanly isolated: (1) ws_entry ~213 ms (per-call
  overhead, needs a sub-split: x-quant vs handoff vs lookup vs combine); (2)
  attention 291 ms (GPU-side absorption). Decode itself (phases 177) is DONE.
- If ws_entry halves (~100) and attention drops to ~80: token ~460 ms = ~2.2
  tok/s. Both are orchestration/overhead, not physics: reachable.
- Session ~1h ~8 USD. Program: ~71 USD AWS + 4 RunPod.

## 671B session G2 (2026-07-12): ws_entry sub-split -- barriers NOT the wall, "setup" is ~182ms

- Persistent lut8 scratch (G-inc2) barely moved ws_entry: 213 -> 204 ms. So the
  per-call malloc theory was mostly wrong for lut8 (or the alloc wasn't the
  bulk). Steady 742 ms = 1.35 tok/s (marginal vs 762).
- moe_ws_sub split: setup=~190 | run=~168 (the phases) | barrier+handoff=19.7 |
  combine=2.1. BOTH our bets REFUTED: barriers are 19.7 ms (not 70-115), combine
  is 2.1 ms. THE WALL IS "setup" = ~182-236 ms/token.
- setup = the pre-phase work in the lut8 work-steal: prime suspect = int8
  activation quantization done PER EXPERT (8x/layer x 58 = 464 quantizations of
  a 7168-vector/token) instead of ONCE per layer shared across the 8 picked
  experts. If so, hoisting the activation quant to once-per-layer cuts setup ~8x.
  Needs a sub-split of setup to confirm (quant vs layout-prep vs cache-resolve).
- Token budget: attention 296 + moe 426 (phases 168 + setup 190 + barrier 20 +
  misc). Decode (168) done; setup (190) and attention (296) are the targets.
- Next: G-inc3 = sub-split setup, then hoist the redundant per-expert work to
  per-layer. If setup 190->40 and attention later ->80: token ~450 = ~2.2 tok/s.
- Session ~1h ~8 USD. Program: ~79 USD AWS + 4 RunPod. Milestone: 1.35 tok/s.

## 671B session G3 (2026-07-12): prewarm confirmed -- 1.67 tok/s; attention is now the wall (49%)

- Prewarm-at-first-forward (G-inc3): recode 185 -> 0.3 ms CONFIRMED (the cold
  cache was exactly the 190ms "setup" wall; pos=15 was never steady-state).
- RESULT: 598 ms/token = 1.67 tok/s steady. Chain 0.24 -> 0.44 -> 0.96 -> 1.31
  -> 1.35 -> 1.67 = x7 from the disk-offload baseline.
- New breakdown at 598: attention=291 (49%!), moe=298 = phases 148 + ws_entry
  84 (setup 60 [recode 0.3, quant 1.2, PREP 58.8] + barrier 20 + combine 2) +
  xcopy 1.4 + shared 6.6.
- TWO TARGETS, cleanly isolated: (1) ATTENTION 291 ms is now the single biggest
  chunk (49% of the token) -- the host W_UK/W_UV absorption was parallelized (E)
  but still ~4 GB/token of host f32 matmul; the real fix is GPU-side absorption
  (do the two absorptions as device GEMMs, keep them off the 32-core CPU that is
  busy with MoE). (2) PREP 58.8 ms = the lut8 activation layout-prep in setup
  (deinterleave/transpose of the shared x per layer) -- likely hoistable to
  once-per-layer or fused into quant.
- If attention 291->~80 (GPU absorption) and prep 58->~10: token ~330 = ~3 tok/s.
  Attention is the big project; prep is a quick win.
- Session ~1h ~8 USD. Program: ~87 USD AWS + 4 RunPod. Milestone: 1.67 tok/s,
  full 671B lossless-4bit, ONE 64-vCPU box.

## 671B session H (2026-07-12): attention on GPU -- 2.46 tok/s, TARGET HIT, x10 from baseline

- GATE: mla_block_math_correct green (device MLA absorption numerically correct).
- H-HOST control (PREP-fix only, absorption still on host): 542 ms = 1.84 tok/s
  (the parallel-memset PREP fix alone lifted 1.67 -> 1.84; attention still 302).
- H-DEVICE (W_UK/W_UV absorptions as device GEMVs): attention 302 -> 170 ms,
  token 542 -> 416 = 2.40 tok/s. THE BIG WIN.
- H-DEVICE-OVERLAP (dev_flush drops the redundant per-layer attn drain): 407 ms
  = 2.46 tok/s. The residual-overlap saving is real but small (~9 ms): attention
  is mostly serial before o_proj, little to hide. Honest.
- FINAL: 2.46 tok/s. FULL CHAIN: 0.24 -> 0.44 -> 0.96 -> 1.31 -> 1.35 -> 1.67
  -> 1.84 -> 2.46 tok/s = x10.2 from the disk-offload baseline, on ONE g6e.16xlarge
  (64 vCPU + L40S), full DeepSeek-R1 671B, lossless 4-bit, pure Rust, no datacenter.
- Remaining budget at 407 ms: attention 164 + moe 242 (decode ~150 + ws ~90) +
  other. Both have more to give (attention GEMV batching, moe reduce tail) but
  the stated 2-2.5 tok/s target is HIT.
- Every wall in the chain, in order, each measured then killed: disk -> RAM ->
  minor-faults -> dispatch-grain -> gather-latency -> worker-engagement ->
  MLA-host-serial -> page-walks(refuted) -> per-core-decode -> recode-cache-cold
  -> scratch-memset -> attention-on-host. Eleven diagnoses, ~$95 total.
- Session ~1h ~8 USD. Program total: ~95 USD AWS + 4 RunPod.

## S19 validation gates (2026-07-13): 3 of 4 pass; Paris-rank-1 is NEGATIVE (honest)

- GATE 1 gemv8 K256 decode correctness: PASS (bit-exact, deterministic).
- GATE 1b k256_reconstruction_beats_k16: PASS.
- GATE 2 --mixed export + round-trip: PASS. prec_flags=0x3 (lm_head + shared
  expert K=256), magic CBKE, 10133 MB (vs 9510 plain = +6.5% for V2-Lite's small
  dense; the 671B projection was +0.7% since its experts dominate). Loads clean.
- GATE 3 Paris-rank-1: NEGATIVE, reported honestly. Both mixed and plain 4-bit
  greedy-emit token1 = 245 (" a", -> "a city of 36 arrondissements", on-topic
  but not the word). Paris (8913) is not even top-2 in either. Mixed precision
  DID shift logits (pos5 a-vs-the margin 0.557 -> 0.938) and lower reconstruction
  error, but did not lift " Paris" to rank 1. Conclusion: with correct MLA math,
  4-bit quantization of V2-Lite still degrades this factual prompt below the
  " Paris" threshold, and 8-bit on shared+lm_head alone is insufficient. Would
  need 8-bit on more tensors (routed experts) or it is a small-model+4bit floor.
  The claim was "plausible not promised"; the box says not-recovered. No
  regression -- the runtime is correct (fp16 says Paris, our 4-bit is on-topic).
- Program result stands: 671B at 2.46 tok/s (x10.2), determinism-by-default,
  mixed-precision format shipped and validated, Paris MLA export fix validated.
- Session ~40 min ~5 USD. Program: ~100 USD AWS + 4 RunPod.

## Quality: measured perplexity (wikitext-2, ctx 2048, DeepSeek-V2-Lite, our export's k-means)

Answers "how much does OUR 4-bit cost vs fp16", using the shipped export quantizer:
  fp16 baseline : PPL 5.6983  (reference)
  our 4-bit K16 : PPL 6.1024  (+7.09%)  -- the real cost of weight-only 4-bit
  mixed (S19)   : PPL 5.7832  (+1.49%)  -- shared experts + lm_head at 8-bit K256, routed 4-bit
  full 8-bit    : PPL 5.6998  (+0.03%)  -- all experts+lm_head K256, essentially fp16-lossless

Conclusion: full 8-bit is essentially lossless; the mixed-precision format captures most of
that recovery (7.1% -> 1.5%) by promoting only the shared experts + lm_head (0.7% of the 671B
footprint, the every-token path). The "Paris token-1" probe was a single-prompt red herring;
PPL is the honest metric and the mixed format is the right default. ~5 USD, cheap g6e.2xlarge.
