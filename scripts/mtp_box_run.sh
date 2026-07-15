#!/usr/bin/env bash
# MTP step 3 on the 671B box: export mtp.cbk from a PARTIAL R1 snapshot (only the shards
# holding layers.61 + embed + lm_head, ~15-20 GB instead of ~700 GB), then run the shadow
# accept-length measurement against the full model.cbk on the attached network volume.
#
# Assumes: network volume mounted at /workspace (model.cbk present), repo at /work/cuda-codebook,
# CUDA toolkit + rust available (install below), HF token not needed (R1 is public).
set -euo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
VOL=/workspace
SNAP=$VOL/r1_snap_mtp
REPO=/work/cuda-codebook
HF=https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/main

echo "=== 0. sanity: model.cbk on the volume ==="
ls -la $VOL/*.cbk

echo "=== 1. partial snapshot: only shards containing layers.61 + embed + lm_head ==="
mkdir -p $SNAP && cd $SNAP
curl -sLO $HF/config.json
curl -sLO $HF/model.safetensors.index.json
python3 - <<'PY'
import json, os
idx = json.load(open("model.safetensors.index.json"))
wm = idx["weight_map"]
need_keys = [k for k in wm if ".layers.61." in k] + ["model.embed_tokens.weight", "lm_head.weight"]
shards = sorted(set(wm[k] for k in need_keys))
print("shards needed:", shards)
open("shards_needed.txt", "w").write("\n".join(shards))
# filtered index: keep ONLY entries living in the shards we download (LazySafetensors
# indexes every listed shard, so absent shards must not be referenced).
wm2 = {k: v for k, v in wm.items() if v in shards}
json.dump({"metadata": idx.get("metadata", {}), "weight_map": wm2},
          open("model.safetensors.index.json", "w"))
print("filtered index:", len(wm2), "tensors")
PY
while read -r sh; do
  if [ ! -f "$sh" ]; then echo "downloading $sh"; curl -sL -o "$sh" "$HF/$sh"; fi
done < shards_needed.txt
ls -la $SNAP/*.safetensors | head

echo "=== 2. export mtp.cbk (streaming, CPU k-means) ==="
cd $REPO
python3 model/export_deepseek_mtp.py --dir $SNAP --out $VOL 2>&1 | tail -20
ls -la $VOL/mtp.cbk

echo "=== 3. build runtime (CUDA sm_89 for L40S; sm_86 for A40) ==="
cd $REPO/runtime
ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "4090\|L40" && echo sm_89 || echo sm_86)
CUDA_ARCH=$ARCH CUDA_PATH=/usr/local/cuda cargo build --release --bin mtp_shadow 2>&1 | grep -E "^error|Finished" | tail -3

echo "=== 4. real prompt.bin via the R1 tokenizer ==="
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
open("/workspace/prompt.bin", "wb").write(struct.pack(f"<{len(ids)}i", *ids))
print("prompt.bin:", len(ids), "tokens")
PY

echo "=== 5. shadow accept-length run (decode unchanged, depth-3 drafting) ==="
cd $REPO/runtime
TRAPETUM_CPU_EXPERTS=1 TRAPETUM_NTOK=96 \
  ./target/release/mtp_shadow $VOL/model.cbk $VOL/mtp.cbk /workspace/prompt.bin 2>&1 | grep -vE "loaded layer" | tail -30
echo "ALLDONE_MTPBOX"
