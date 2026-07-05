#!/usr/bin/env bash
# ============================================================================
# Trapetum speculative decoding K=1 — REAL alpha + speedup measurement on a
# CUDA pod (RunPod / any 24GB+ NVIDIA box). Compresses a Llama-2-7B target and
# a TinyLlama-1.1B drafter to 4-bit .cbk, validates the target against HF, then
# measures the acceptance rate alpha and each model's latency to derive the
# projected spec-dec speedup. No CUDA batched-kernel port needed: alpha is a
# property of the two distributions, measured with the existing M=1 forward.
#
# RunPod gotchas (from prior sessions): secure cloud = TCP (not the HTTP proxy),
# `service ssh start` needs /run/sshd, scp uses -P (uppercase), nvcc needs
# `export PATH=/usr/local/cuda/bin:$PATH`. Run this FROM the pod after `git pull`.
#
# Usage (on the pod):
#   bash scripts/runpod_specdec.sh
# Env overrides: TARGET_MODEL, DRAFTER_MODEL, N (tokens), PROMPT.
# ============================================================================
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export TOKENIZERS_PARALLELISM=false

TARGET_MODEL="${TARGET_MODEL:-NousResearch/Llama-2-7b-hf}"     # ungated Llama-2-7B mirror
DRAFTER_MODEL="${DRAFTER_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"  # shares the Llama-2 tokenizer
N="${N:-128}"
PROMPT="${PROMPT:-The capital of France is}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-/workspace/cbk}"
mkdir -p "$OUT/target" "$OUT/drafter"

echo "### 1/6  python env (torch 2.4 cu121 + transformers 4.44.2, pinned) ###"
PIN="torch==2.4.0 transformers==4.44.2"
pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers==4.44.2 accelerate sentencepiece numpy $PIN

echo "### 2/6  compress TARGET  ($TARGET_MODEL) -> 4-bit .cbk ###"
python "$ROOT/model/export_runtime.py" --model "$TARGET_MODEL" --out "$OUT/target" --prompt "$PROMPT" --gen 16

echo "### 3/6  compress DRAFTER ($DRAFTER_MODEL) -> 4-bit .cbk ###"
python "$ROOT/model/export_runtime.py" --model "$DRAFTER_MODEL" --out "$OUT/drafter" --prompt "$PROMPT" --gen 16

echo "### 4/6  build the CUDA runtime + validate the target reproduces HF ###"
cd "$ROOT/runtime"
cargo build --release --features cuda --bin generate --bin alpha_check
./target/release/generate "$OUT/target/model.cbk" "$OUT/target/prompt.bin" "$OUT/target/ref.bin" "$OUT/target/cont.bin"

echo "### 5/6  measure REAL alpha + projected spec-dec K=1 speedup ###"
# prompt tokens from the exported prompt.bin (i32)
PROMPT_TOKS=$(python -c "import numpy as np,sys; print(' '.join(map(str, np.fromfile('$OUT/target/prompt.bin', dtype='<i4').tolist())))")
echo "prompt tokens: $PROMPT_TOKS"
./target/release/alpha_check "$OUT/target/model.cbk" "$OUT/drafter/model.cbk" "$N" $PROMPT_TOKS

echo "### 6/6  decode/Pareto benchmark (quantization vs cuBLAS/Marlin/AQLM + energy) ###"
# best-effort: pareto.py skips any baseline whose backend is absent. PPL skipped by default
# (--ppl-tokens 0) to keep the run short; add it back for the full paper table.
pip install -q pynvml || true
python "$ROOT/bench/pareto.py" --model "$TARGET_MODEL" --out /workspace/bench --gen 128 --ppl-tokens 0 || echo "WARN: pareto bench partial/failed (non-fatal)"
echo "### ALL BENCHMARKS DONE. Spec-dec alpha/speedup + decode Pareto printed above. ###"
echo "### DONE. alpha and projected speedup printed above. ###"
echo "Note: the projected speedup = (1+alpha)/(1 + t_draft/t_target); the M=2 verify is"
echo "bandwidth-bound (M0 bench: ms/token flat to M=4), so it costs ~one target forward."
echo "The end-to-end WALL-CLOCK spec run additionally needs the 5 batched ops ported to"
echo "CUDA (they exist + are validated in the Metal backend); alpha+latency give the number."
