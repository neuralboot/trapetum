#!/usr/bin/env python3
"""
Op-level decode latency: our fused additive-VQ GEMV (avq_gemv3, 2-bit AQLM 2x8 config
M=2/K=256/D=8) vs AQLM's official kernel, on the REAL Llama-2-7B attention shape
(4096->4096), batch-1, same machine.

We time the real AQLM `QuantizedLinear.forward` (which dispatches `aqlm::code2x8_matmat`)
so there is no signature guessing. For our kernel we use matching-dimension random codes
(decode latency is data-independent; correctness that avq decodes real AQLM weights is
established separately by the PPL run). fp16 cuBLAS is the dense reference.

Note: avq_gemv3 requires OC % 1024 == 0, so this benchmarks the 4096-output projections
(q/k/v/o); the 11008-output MLP shapes need a tail-handling variant (TODO).

Usage: python avq_oplevel.py
"""
import time
import torch
from torch.utils.cpp_extension import load_inline

KERNEL = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#define M 2
#define K 256
#define D 8
#define CPB 256
#ifndef GT
#define GT 16
#endif
__global__ void avq_gemv3(const __half* __restrict__ X, const unsigned char* __restrict__ codes,
                          const __half* __restrict__ CB, float* __restrict__ Y, int IC, int OC) {
    int ng = IC / D;
    int o = (blockIdx.x * CPB + threadIdx.x) * 4;
    int g0 = blockIdx.y * GT;
    __shared__ __half s_CB[M*K*D];
    __shared__ float s_LUT[M*GT*K];
    __shared__ __half s_x[GT*D];
    for (int t = threadIdx.x; t < M*K*D; t += CPB) s_CB[t] = CB[t];
    for (int t = threadIdx.x; t < GT*D; t += CPB) { int gg = g0 + t/D; s_x[t] = (gg<ng) ? X[gg*D + t%D] : __float2half(0.f); }
    __syncthreads();
    for (int t = threadIdx.x; t < M*GT*K; t += CPB) {
        int m = t/(GT*K), r = t%(GT*K), gt = r/K, k = r%K; float dd = 0;
        #pragma unroll
        for (int e = 0; e < D; e++) dd += __half2float(s_x[gt*D+e]) * __half2float(s_CB[(m*K+k)*D+e]);
        s_LUT[t] = dd;
    }
    __syncthreads();
    if (o < OC) {
        float a0=0,a1=0,a2=0,a3=0;
        #pragma unroll
        for (int gt = 0; gt < GT; gt++) {
            int g = g0 + gt; if (g >= ng) break;
            #pragma unroll
            for (int m = 0; m < M; m++) {
                unsigned cc = *reinterpret_cast<const unsigned*>(&codes[((size_t)m*ng + g)*OC + o]);
                const float* L = &s_LUT[(m*GT + gt)*K];
                a0 += L[cc & 0xFF]; a1 += L[(cc>>8) & 0xFF]; a2 += L[(cc>>16) & 0xFF]; a3 += L[(cc>>24) & 0xFF];
            }
        }
        atomicAdd(&Y[o], a0); atomicAdd(&Y[o+1], a1); atomicAdd(&Y[o+2], a2); atomicAdd(&Y[o+3], a3);
    }
}
torch::Tensor avq_gemv(torch::Tensor X, torch::Tensor codes, torch::Tensor CB) {
    int IC = X.size(0), OC = codes.size(2), ng = IC / D;
    auto Y = torch::zeros({OC}, X.options().dtype(torch::kFloat));
    dim3 grid(OC/(CPB*4), (ng+GT-1)/GT), block(CPB);
    avq_gemv3<<<grid, block>>>((const __half*)X.data_ptr<at::Half>(),
        (const unsigned char*)codes.data_ptr<uint8_t>(), (const __half*)CB.data_ptr<at::Half>(),
        Y.data_ptr<float>(), IC, OC);
    return Y;
}
"""

ext = load_inline(name="avq_op", cpp_sources="torch::Tensor avq_gemv(torch::Tensor, torch::Tensor, torch::Tensor);",
                  cuda_sources=KERNEL, functions=["avq_gemv"], extra_cuda_cflags=["-O3", "-DGT=16"], verbose=False)


def bench(fn, iters=500):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms


def main():
    IC, OC = 4096, 4096
    D, Mc, Kc = 8, 2, 256
    ng = IC // D
    dev = "cuda"

    # ours: matching-dim random codes (latency is data-independent)
    X = torch.randn(IC, dtype=torch.float16, device=dev)
    codes = torch.randint(0, 256, (Mc, ng, OC), dtype=torch.uint8, device=dev)
    CB = torch.randn(Mc, Kc, D, dtype=torch.float16, device=dev) * 0.05
    y = ext.avq_gemv(X, codes, CB)
    assert y.shape[0] == OC
    t_ours = bench(lambda: ext.avq_gemv(X, codes, CB))

    # fp16 cuBLAS reference
    W = torch.randn(OC, IC, dtype=torch.float16, device=dev) * 0.02
    x2 = X.view(1, IC)
    t_fp16 = bench(lambda: torch.nn.functional.linear(x2, W))

    print(f"shape {IC}->{OC}, batch-1 decode, 2-bit (M=2,K=256,D=8):")
    print(f"  ours avq_gemv3 : {t_ours:.4f} ms")
    print(f"  fp16 cuBLAS    : {t_fp16:.4f} ms   (ours x{t_fp16/t_ours:.2f} vs fp16)")

    # AQLM official kernel: time the real QuantizedLinear.forward (dispatches code2x8_matmat)
    try:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf", torch_dtype=torch.float16,
            trust_remote_code=True).cuda().eval()
        lin = m.model.layers[0].self_attn.q_proj  # QuantizedLinear, 4096->4096
        print(f"  aqlm layer: {type(lin).__name__}, in={IC}")
        xa = torch.randn(1, 1, IC, dtype=torch.float16, device=dev)
        with torch.no_grad():
            lin(xa)
            t_aqlm = bench(lambda: lin(xa))
        print(f"  aqlm code2x8   : {t_aqlm:.4f} ms   (ours x{t_aqlm/t_ours:.2f} vs aqlm)")
    except Exception as e:
        print("  aqlm skipped:", str(e)[:160])


if __name__ == "__main__":
    main()
