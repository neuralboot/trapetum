#!/usr/bin/env bash
# Phase 3: run a REAL DeepSeek (MLA+MoE) end-to-end in the pure-Rust runtime. Compresses
# DeepSeek-V2-Lite (16B, 4-bit ~8GB, fits a 24GB GPU fully resident), then decodes and
# checks coherence vs the HF fp16 continuation. Run FROM the pod after git pull.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
OUT=/workspace/ds; mkdir -p "$OUT"
PROMPT="${PROMPT:-The capital of France is}"

echo "### env ###"
pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers==4.44.2 accelerate sentencepiece numpy || echo "WARN pip"

echo "### compress DeepSeek ($MODEL) -> 4-bit CBKD ###"
python model/export_deepseek.py --model "$MODEL" --out "$OUT" --prompt "$PROMPT" --gen 16

echo "### build CUDA runtime + deepseek_run ###"
( cd runtime && cargo build --release --features cuda --bin deepseek_run )

echo "### run DeepSeek in the pure-Rust runtime ###"
runtime/target/release/deepseek_run "$OUT/model.cbk" "$OUT/prompt.bin" "$OUT/cont.bin"
echo "### DONE ###"
