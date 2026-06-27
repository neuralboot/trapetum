#!/usr/bin/env bash
# Environment setup for the quantization benchmark, on an H100/H200 pod.
#
# THE TRAP (learned the hard way): RunPod pytorch images may ship an OLD torch
# (e.g. 2.1+cu118 / CUDA 11.8), while modern transformers + aqlm[gpu] require
# torch >= 2.4 / CUDA 12. A plain `pip install -U ...` upgrades torch to a build
# that does not match the driver and breaks CUDA. Fix: install a KNOWN torch
# 2.4.0+cu121 first, then PIN torch (and transformers) in every later install so
# no backend silently bumps them.
#
# This exact recipe was verified working on an H100 PCIe (driver 12.4):
#   torch 2.4.0+cu121, transformers 4.44.2, aqlm[gpu]  -> fp16 + AQLM ran clean.
#
# Multi-backend note: autoawq / gptqmodel / auto-gptq each pin their own
# transformers range and can conflict. They are installed in SEPARATE pip calls
# (each pinning torch+transformers) so a failure of one does not break the others;
# bench_quant.py then simply skips any method whose backend did not install.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH

PIN="torch==2.4.0 transformers==4.44.2"

# 1) correct CUDA torch FIRST (cu121 works on Hopper, driver 12.x)
pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 2) base, torch pinned so nothing moves it
pip install -q transformers==4.44.2 accelerate datasets $PIN

# 3) quantization backends, each isolated and pinned (failures are non-fatal)
pip install -q "aqlm[gpu]"     $PIN || echo "WARN: aqlm install failed"
pip install -q autoawq         $PIN || echo "WARN: autoawq install failed"
# gptqmodel: build against the ALREADY-installed torch (the earlier failure was pip
# building it in an isolated env with no torch). --no-build-isolation fixes that.
pip install -q gptqmodel --no-build-isolation $PIN || echo "WARN: gptqmodel failed (GPTQ will skip)"

# 4) sanity
python - <<'PY'
import torch, transformers
from transformers import AutoModelForCausalLM  # fails loudly if torch not seen
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__,
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

echo "setup done. Run:"
echo "  python bench_quant.py --only fp16,aqlm-2bit,gptq-4bit,awq-4bit --ppl-samples 40 --out results.json"
echo "  python plot_pareto.py results.json --out pareto.png"
