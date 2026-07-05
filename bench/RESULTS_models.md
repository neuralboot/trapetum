# Trapetum model coverage — measured on RTX 4090 (4-bit codebook)

Top local LLMs compressed to 4-bit and run in the pure-Rust runtime, July 2026.
Pareto = HF-loop decode (fp16 vs 4-bit): VRAM, tok/s, wikitext PPL, net J/token.
Runtime tok/s (fused, no Python) is separate and faster than the HF loop.

## Standard-attention models (Pareto: fp16 -> 4-bit)
| Model | VRAM (GB) | tok/s (HF loop) | Wikitext PPL | J/token net |
|---|---|---|---|---|
| **Qwen2.5-7B** fp16 | 15.23 | 39.0 | 7.24 | 4.66 |
| **Qwen2.5-7B** 4-bit | **5.49** | 24.7 | 8.22 | **2.45** |
| DeepSeek-R1-Distill-Qwen-7B fp16 | 15.23 | 38.5 | 25.78 | 4.63 |
| DeepSeek-R1-Distill-Qwen-7B 4-bit | **5.49** | 24.2 | 30.73 | **2.42** |
| Llama-2-7B fp16 (baseline) | 13.48 | 37.4 | — | 3.87 |
| Llama-2-7B 4-bit | **3.81** | 23.4 | — | **2.13** |
| Llama-3.1-8B fp16 | 16.06 | 33.2 | 7.35 | 4.96 |
| Llama-3.1-8B 4-bit | **5.64** | 21.1 | 8.45 | **2.38** |
| Mistral-7B fp16 | 14.50 | 36.7 | 5.77 | 4.68 |
| Mistral-7B 4-bit | **4.07** | 21.2 | 6.19 | **2.22** |

Consistent: ~2.8x less VRAM, ~47% less energy/token. (R1-Distill PPL is high because it is a
reasoning model measured on plain wikitext, not its domain.)

## New architectures (run end-to-end in pure Rust, coherent output)
| Model | Arch | Runtime | Output on "The capital of France is" |
|---|---|---|---|
| **DeepSeek-V2-Lite** (16B) | MLA + MoE (64 experts) | ~10 tok/s | "...Paris. The capital of France is Paris..." |
| **Gemma-2-9B** | GeGLU + softcaps + 4-norm | **74 tok/s** | "**Paris**. 🇫🇷 Let me know if you have any other questions!" |
| Llama-2-7B | standard | 136 tok/s | reproduces HF greedy 16/16 exactly |

Gemma-2 worked on the FIRST run (no debug iterations) — the port (attention+final logit
softcapping, GeGLU, RMSNorm(1+w), embedding*sqrt(hidden), 4-norm post-sublayer residual)
was correct by construction. DeepSeek took 7 fixes (see RESULTS_deepseek.md).

## Not yet measured
- Llama-3.1-8B, Mistral-7B: measured (table above) after fixing the sweep tooling (tokenizers upgrade + protobuf + uninstall the torch-2.2-pinned torchvision that broke transformers 4.48). Phi-4 (14B) still failed the harness (rerun TODO); NOT a Trapetum issue (the runtime is torch-free).
