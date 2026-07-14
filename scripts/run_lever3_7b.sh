#!/usr/bin/env bash
# Lever 3 (OA-EM init) A/B on the real Llama-2-7B at 2-BIT (M=2, D=8, K=256), on a 4090.
#   ARM A  OAEM=0  greedy init (current pipeline)
#   ARM B  OAEM=1  output-aware EM init (diag-Hessian-weighted init k-means; beam+LSQ unweighted)
# 2-bit is the regime where init matters most ("Initialisation Determines the Basin").
# References: real AQLM 2x8 trained = PPL 7.63; our greedy residual-VQ M=2 (no beam) diverged.
# Net win iff ARM B PPL < ARM A PPL. M=2 quantization is faster than M=4 (~20-25 min/arm).
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd /work
pip -q install "transformers==4.44.2" "datasets" "sentencepiece" "accelerate" 2>&1 | tail -1 || true

echo "===== ARM A: OAEM=0 (greedy init baseline, 2-bit M=2) ====="
MC=2 ROUNDS=1 CALIB=0 ROT=0 OAEM=0 python model/llama_aqlm_train.py 2>&1 | tee /work/lever3_oaem0.log

echo "===== ARM B: OAEM=1 (output-aware EM init, 2-bit M=2) ====="
MC=2 ROUNDS=1 CALIB=0 ROT=0 OAEM=1 python model/llama_aqlm_train.py 2>&1 | tee /work/lever3_oaem1.log

echo "===== VERDICT ====="
grep "AQLM-trained" /work/lever3_oaem0.log /work/lever3_oaem1.log
echo "Net win iff OAEM=1 PPL < OAEM=0 PPL (both M=2 2-bit, same beam+LSQ)."
echo "ALLDONE_LEVER3"
