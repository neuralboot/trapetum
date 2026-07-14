#!/usr/bin/env bash
# Lever 3b: OA-EM init at 4-BIT (M=4) -- does the confirmed 2-bit win (-28.5%) carry to 4-bit?
#   ARM A  OAEM=0  greedy init  (reference: PPL 6.1391 measured on 4090, ROUNDS=1)
#   ARM B  OAEM=1  output-aware EM init (same beam+LSQ)
# At 4 bits the init matters less (more codebooks to correct); win = any PPL below ARM A.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd /work
pip -q install "transformers==4.44.2" "datasets" "sentencepiece" "accelerate" 2>&1 | tail -1 || true

echo "===== ARM A: OAEM=0 (greedy init, 4-bit M=4, expect ~6.139) ====="
MC=4 ROUNDS=1 CALIB=0 ROT=0 OAEM=0 python model/llama_aqlm_train.py 2>&1 | tee /work/lever3b_oaem0.log

echo "===== ARM B: OAEM=1 (OA-EM init, 4-bit M=4) ====="
MC=4 ROUNDS=1 CALIB=0 ROT=0 OAEM=1 python model/llama_aqlm_train.py 2>&1 | tee /work/lever3b_oaem1.log

echo "===== VERDICT ====="
grep "AQLM-trained" /work/lever3b_oaem0.log /work/lever3b_oaem1.log
echo "Win iff OAEM=1 PPL < OAEM=0 PPL (both M=4 4-bit, same beam+LSQ)."
echo "ALLDONE_LEVER3B"
