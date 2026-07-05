#!/usr/bin/env bash
# Sweep the 3 models that failed the tokenizers-version bug (Llama-3.1-8B, Mistral-7B, Phi-4),
# with an upgraded transformers/tokenizers that parses their newer tokenizer.json.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT=/workspace/sweep2; mkdir -p "$OUT"; PPL_TOKENS="${PPL_TOKENS:-40000}"
echo "### env (newer transformers/tokenizers) ###"
pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q -U "transformers>=4.47,<4.49" tokenizers accelerate sentencepiece numpy ninja pynvml datasets tiktoken || echo WARN
python -c "import transformers,tokenizers;print('tf',transformers.__version__,'tok',tokenizers.__version__)"
MODELS=(
  "unsloth/Meta-Llama-3.1-8B-Instruct"
  "unsloth/mistral-7b-instruct-v0.3"
  "microsoft/phi-4"
)
for M in "${MODELS[@]}"; do
  TAG=$(echo "$M" | tr '/' '_')
  echo "### $M -> Pareto (VRAM, tok/s, J/token, PPL@$PPL_TOKENS) ###"
  python bench/pareto.py --model "$M" --out "$OUT/$TAG" --gen 128 --ppl-tokens "$PPL_TOKENS" \
    || echo "WARN: $M failed - skipping"
done
echo "### SWEEP2 DONE ###"
