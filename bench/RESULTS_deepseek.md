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
  on network-attached storage. Steady-state throughput after page-cache warm-up is not
  yet measured; local NVMe should be substantially faster.

## What this establishes
The largest open-source model runs end to end in a from-scratch Rust runtime with no
Python, no PyTorch, and no GPU requirement for the expert weights: one box, 73 GB of
RAM, and 326 GB of disk. The V3-specific machinery validated on the way: q_lora MLA,
noaux_tc sigmoid routing with e_score_correction_bias and group top-k (n_group 8,
topk_group 4, routed_scaling 2.5, raw-sigmoid weight renormalization), first 3 layers
dense, 2048-dim expert FFNs.
