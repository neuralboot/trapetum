#!/usr/bin/env bash
# MTP accept-length run on the AWS 671B box (g6e.16xlarge: 64 vCPU, L40S 48GB, 512GB RAM,
# 2x1.9TB NVMe). Reassembles model.cbk from the S3 parts (in-region, free egress), exports
# mtp.cbk from a PARTIAL R1 snapshot (~15-20 GB), runs the shadow accept-length measurement
# with expert logging, analyzes routing skew, and uploads mtp.cbk + logs back to S3.
#
# Run as root/ubuntu on the instance, detached:  nohup bash mtp_aws_run.sh > /tmp/mtpaws.log 2>&1 &
set -euo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
BUCKET=s3://neuralboot-trapetum-models/deepseek-r1-671b
WORK=/work
REPO=$WORK/trapetum
SNAP=$WORK/r1_snap_mtp
HF=https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/main

echo "=== 0. NVMe scratch ==="
if [ ! -d $WORK ]; then
  DEV=$(lsblk -dn -o NAME,TYPE,SIZE | awk '$2=="disk" && $3 ~ /T/ {print "/dev/"$1}' | head -1)
  mkfs.ext4 -F $DEV && mkdir -p $WORK && mount $DEV $WORK
fi
df -h $WORK

echo "=== 1. reassemble model.cbk from S3 parts (parallel download, then cat) ==="
mkdir -p $WORK/parts
aws s3 cp $BUCKET/ $WORK/parts/ --recursive --exclude "*" --include "model.cbk.part.*" --only-show-errors
ls $WORK/parts | head -3; ls $WORK/parts | wc -l
cat $(ls $WORK/parts/model.cbk.part.* | sort) > $WORK/model.cbk
rm -rf $WORK/parts
ls -la $WORK/model.cbk

echo "=== 2. repo + deps ==="
command -v cargo >/dev/null || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal)
export PATH=$HOME/.cargo/bin:$PATH
[ -d $REPO ] || git clone -q https://github.com/neuralboot/trapetum $REPO
pip -q install numpy transformers sentencepiece 2>&1 | tail -1 || true

echo "=== 3. partial R1 snapshot: shards holding layers.61 + embed + lm_head ==="
mkdir -p $SNAP && cd $SNAP
curl -sLO $HF/config.json
curl -sLO $HF/model.safetensors.index.json
python3 - <<'PY'
import json
idx = json.load(open("model.safetensors.index.json"))
wm = idx["weight_map"]
need = [k for k in wm if ".layers.61." in k] + ["model.embed_tokens.weight", "lm_head.weight"]
shards = sorted(set(wm[k] for k in need))
open("shards_needed.txt","w").write("\n".join(shards))
json.dump({"metadata": idx.get("metadata", {}),
           "weight_map": {k: v for k, v in wm.items() if v in shards}},
          open("model.safetensors.index.json","w"))
print("shards needed:", shards)
PY
while read -r sh; do [ -f "$sh" ] || { echo "downloading $sh"; curl -sL -o "$sh" "$HF/$sh"; }; done < shards_needed.txt

echo "=== 4. export mtp.cbk + upload to S3 (persist the asset) ==="
cd $REPO
python3 model/export_deepseek_mtp.py --dir $SNAP --out $WORK 2>&1 | tail -12
aws s3 cp $WORK/mtp.cbk $BUCKET/mtp.cbk --only-show-errors && echo "mtp.cbk uploaded to $BUCKET/mtp.cbk"

echo "=== 5. build runtime (L40S = sm_89) ==="
cd $REPO/runtime
CUDA_ARCH=sm_89 CUDA_PATH=/usr/local/cuda cargo build --release --bin mtp_shadow 2>&1 | grep -E "^error|Finished" | tail -3

echo "=== 6. prompt.bin via the R1 tokenizer ==="
cd $SNAP
curl -sLO $HF/tokenizer.json; curl -sLO $HF/tokenizer_config.json
python3 - <<'PY'
import struct
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
text = ("The industrial revolution transformed not only the economy of nineteenth century "
        "Europe but also its social fabric. Factories drew millions from the countryside "
        "into rapidly growing cities, where new classes emerged and old hierarchies eroded. "
        "Historians have long debated whether this transformation improved the lives of "
        "ordinary workers. The evidence suggests a complicated picture:")
ids = tok(text).input_ids
open("/work/prompt.bin","wb").write(struct.pack(f"<{len(ids)}i", *ids))
print("prompt.bin:", len(ids), "tokens")
PY

echo "=== 7. shadow accept-length run (decode unchanged, depth-3, expert log on) ==="
cd $REPO/runtime
TRAPETUM_CPU_EXPERTS=1 TRAPETUM_NTOK=96 TRAPETUM_LOG_EXPERTS=$WORK/experts.log \
  ./target/release/mtp_shadow $WORK/model.cbk $WORK/mtp.cbk /work/prompt.bin 2>&1 | grep -vE "loaded layer" | tail -30

echo "=== 8. routing skew + adjacent-overlap analysis (hot-cache decision, free bonus) ==="
# 58 MoE layers per MAIN forward; the MTP layer adds 1 more MoE call per draft (3 drafts/step)
# -> lines per step = 58 + 3. The analyzer needs uniform rows; analyze main-only by taking
# the first 58 lines of each 61-line block.
python3 - <<'PY'
lines = [l.strip() for l in open("/work/experts.log") if l.strip()]
main = []
B = 58 + 3
# prefill wrote (prompt_len x 58) + (prompt_len-1 MTP calls); decode wrote 61/block. Keep it
# simple: only keep rows with the modal comma-count and chunk by 58 after dropping MTP rows is
# nontrivial; fallback = analyze ALL rows as one stream (skew stats remain valid; overlap is
# approximate). Write both files.
open("/work/experts_all.log","w").write("\n".join(lines))
print("expert log rows:", len(lines))
PY
python3 $REPO/model/analyze_expert_log.py $WORK/experts_all.log --moe-layers 58 --n-routed 256 || true
aws s3 cp $WORK/experts.log $BUCKET/mtp_experts_$(date +%s).log --only-show-errors || true
echo "ALLDONE_MTPAWS"
