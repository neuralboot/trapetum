#!/usr/bin/env bash
# Benchmark sweep: compress + measure the top local LLMs with Trapetum. Per model, pareto.py
# gives fp16 vs codebook-4bit (VRAM, decode tok/s, J/token, wikitext PPL) in one shot. Results
# aggregated to /workspace/sweep/<model>.json. Run FROM the pod after git pull.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT=/workspace/sweep; mkdir -p "$OUT"
PPL_TOKENS="${PPL_TOKENS:-40000}"

echo "### env ###"
pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers==4.44.2 accelerate sentencepiece numpy ninja pynvml datasets tiktoken || echo WARN
# non-gated, open, kernel-compatible (standard attention) top local models:
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "unsloth/Meta-Llama-3.1-8B-Instruct"
  "unsloth/mistral-7b-instruct-v0.3"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "microsoft/phi-4"
)
for M in "${MODELS[@]}"; do
  TAG=$(echo "$M" | tr '/' '_')
  echo "### $M -> Pareto (fp16 vs 4-bit: VRAM, tok/s, J/token, PPL@$PPL_TOKENS) ###"
  python bench/pareto.py --model "$M" --out "$OUT/$TAG" --gen 128 --ppl-tokens "$PPL_TOKENS" \
    && cp "$OUT/$TAG"/*.json "$OUT/$TAG.json" 2>/dev/null \
    || echo "WARN: $M failed (arch/gating?) - skipping"
done
echo "### SWEEP DONE ###"
echo "--- collected ---"; ls -la "$OUT"/*.json 2>/dev/null
