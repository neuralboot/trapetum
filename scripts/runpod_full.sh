#!/usr/bin/env bash
# Full CUDA validation + measurement for spec-dec items 1-5. Run FROM the pod after git pull.
#   bash scripts/runpod_full.sh
# Validates the CUDA batched-op port (#1) + K>1 (#3) + confidence (#4) are LOSSLESS on real
# GPU, measures alpha + K-sweep speedup for two drafters (#2), and the aqlm Pareto baseline (#5).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT=/workspace/cbk; mkdir -p "$OUT"
TARGET="NousResearch/Llama-2-7b-hf"
DRAFTER1="TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # 1.1B
DRAFTER2="JackFram/llama-160m"                  # 160M, Llama tokenizer (vocab 32000)
PROMPT="The capital of France is"

echo "### env ###"
pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers==4.44.2 accelerate sentencepiece numpy ninja aqlm[gpu,cpu] || echo "WARN: some pip failed"

echo "### build CUDA runtime + bins ###"
( cd runtime && cargo build --release --features cuda --bin generate --bin alpha_check --bin spec_check )

echo "### (#1,#3,#4) LOSSLESS validation of batched forward + K>1 + confidence on real CUDA ###"
runtime/target/release/spec_check || { echo "SPEC_CHECK FAILED"; exit 1; }

echo "### compress target + drafters ###"
python model/export_runtime.py --model "$TARGET"   --out "$OUT/target"  --prompt "$PROMPT" --gen 16
python model/export_runtime.py --model "$DRAFTER1" --out "$OUT/draft1" --prompt "$PROMPT" --gen 16
python model/export_runtime.py --model "$DRAFTER2" --out "$OUT/draft2" --prompt "$PROMPT" --gen 16

echo "### validate target reproduces HF ###"
runtime/target/release/generate "$OUT/target/model.cbk" "$OUT/target/prompt.bin" "$OUT/target/ref.bin" "$OUT/target/cont.bin"

PT=$(python -c "import numpy as np; print(' '.join(map(str, np.fromfile('$OUT/target/prompt.bin', dtype='<i4').tolist())))")
echo "### (#2) alpha + K-sweep: TinyLlama-1.1B drafter ###"
runtime/target/release/alpha_check "$OUT/target/model.cbk" "$OUT/draft1/model.cbk" 128 $PT
echo "### (#2) alpha + K-sweep: llama-160m drafter ###"
runtime/target/release/alpha_check "$OUT/target/model.cbk" "$OUT/draft2/model.cbk" 128 $PT

echo "### (#5) Pareto with aqlm-2bit baseline ###"
python bench/pareto.py --model "$TARGET" --out /workspace/bench --gen 128 --ppl-tokens 0 || echo "WARN: pareto partial"
echo "### ALL DONE (items 1-5). ###"
