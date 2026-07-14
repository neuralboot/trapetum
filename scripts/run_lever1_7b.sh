#!/usr/bin/env bash
# Lever 1 SELECTIVE rotation A/B on the real Llama-2-7B (run on a 4090, not A40 -- A40 is ~2-3x
# slower for this beam-search). Fair test at identical MC/ROUNDS:
#   ARM A  ROT=0  baseline (should reproduce ~6.13)
#   ARM B  ROT=2  selective Hadamard on o_proj + down_proj only (where the 7B proxy showed the gain)
# Net free win iff ARM B PPL < ARM A PPL.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd /work
pip -q install "transformers==4.44.2" "datasets" "sentencepiece" "accelerate" 2>&1 | tail -1 || true

echo "===== ARM A: ROT=0 (baseline, must reproduce ~6.13) ====="
MC=4 ROUNDS=1 CALIB=0 ROT=0 python model/llama_aqlm_train.py 2>&1 | tee /work/lever1sel_rot0.log

echo "===== ARM B: ROT=2 (selective Hadamard: o_proj + down_proj) ====="
MC=4 ROUNDS=1 CALIB=0 ROT=2 python model/llama_aqlm_train.py 2>&1 | tee /work/lever1sel_rot2.log

echo "===== VERDICT ====="
grep "AQLM-trained" /work/lever1sel_rot0.log /work/lever1sel_rot2.log
echo "Net free win iff ROT=2 PPL < ROT=0 PPL (both M=4 4-bit, same beam+LSQ)."
echo "ALLDONE_LEVER1SEL"
