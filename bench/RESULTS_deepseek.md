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

## Caveat
V2-Lite has plain q_proj (q_lora_rank=null); the q_lora variant (V2 full / V3) needs
q_a_proj/q_b_proj export (not yet). DeepSeek-V3-671B additionally needs multi-GPU or the
NVMe/host offload path (mechanism validated) + big hardware.
