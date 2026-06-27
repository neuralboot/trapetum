#!/usr/bin/env python3
"""Speedup of the fused 4-bit codebook GEMV vs a torch dense fp16 mat-vec, across
layer shapes, on the current GPU. The speedup GROWS with layer size: it loses on
tiny matrices (overhead bound) and wins big on the large LLM layers that dominate
parameter count. Timing is value-independent, so random data is fine.

Run on a GPU box:  pip install ninja ; python shapes_test.py
"""
import torch, time
import quant_demo as q   # importing compiles/loads the codebook_gemv op (cached)

def t(f, n=400):
    f(); torch.cuda.synchronize(); s = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - s) / n * 1e3

print("shape (IC x OC)      kernel      torch-dense   speedup")
for IC, OC in [(3072, 768), (4096, 4096), (11008, 4096), (4096, 11008)]:
    cb = torch.randn(16, OC, device="cuda", dtype=torch.float16)
    packed = torch.randint(0, 256, (IC, OC // 2), device="cuda", dtype=torch.uint8)
    x = torch.randn(IC, device="cuda", dtype=torch.float16)
    W = torch.randn(OC, IC, device="cuda", dtype=torch.float16)
    tk = t(lambda: q.ext.codebook_gemv(x, packed, cb))
    td = t(lambda: torch.mv(W, x))
    print("%6dx%-6d  %.4f ms   %.4f ms     x%.2f" % (IC, OC, tk, td, td / tk))
