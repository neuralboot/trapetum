#!/usr/bin/env bash
# Run Gemma-2-9B end-to-end in the pure-Rust runtime. Compresses (4-bit) + decodes.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
MODEL="${MODEL:-unsloth/gemma-2-9b-it}"; OUT=/workspace/gm; mkdir -p "$OUT"
echo "### env ###"
pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers==4.44.2 accelerate sentencepiece numpy || echo WARN
echo "### compress Gemma-2 ($MODEL) -> CBKG ###"
python model/export_gemma.py --model "$MODEL" --out "$OUT" --prompt "The capital of France is"
echo "### build + run ###"
( cd runtime && cargo build --release --features cuda --bin gemma_run )
runtime/target/release/gemma_run "$OUT/model.cbk" "$OUT/prompt.bin"
echo "### DONE ###"
