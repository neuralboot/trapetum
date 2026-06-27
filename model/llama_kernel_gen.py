#!/usr/bin/env python3
"""End-to-end decode throughput of Llama-2-7B with the fused 4-bit codebook kernel.

Replaces every projection Linear with a CodebookLinear that stores 4-bit packed
indices + fp16 codebook (4x less memory), uses the kernel for batch-1 decode and a
reconstruct-then-matmul path for prefill. Measures decode tokens/s and peak VRAM
against the fp16 baseline. Honest: a naive per-layer custom-op swap pays python /
launch overhead per layer per token, so the per-GEMV speedup may not fully survive.

Run:  pip install ninja transformers==4.44.2 ; python llama_kernel_gen.py
"""
import torch, torch.nn as nn, time
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-hf"
K = 16
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

CUDA = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>
#define K 16
#define CPB 256
#define TY 8
__global__ void __launch_bounds__(32*TY)
gemv4(const __half* __restrict__ X, const unsigned char* __restrict__ packed,
      const __half* __restrict__ cb, float* __restrict__ Yacc, int IC, int OC) {
    extern __shared__ char sm[];
    __half* s_cb = (__half*)sm; float* red = (float*)(s_cb + K*CPB);
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty*32+tx, nth = 32*TY;
    int j0 = blockIdx.x*CPB;
    for (int t = tid; t < K*CPB/2; t += nth) {
        int idx = t*2, k = idx/CPB, jj = j0 + (idx%CPB);
        *reinterpret_cast<__half2*>(&s_cb[idx]) = *reinterpret_cast<const __half2*>(&cb[(size_t)k*OC+jj]);
    }
    __syncthreads();
    int per = (IC+gridDim.y-1)/gridDim.y, ic0 = blockIdx.y*per, ic1 = min(IC, ic0+per);
    int jbase = j0 + tx*8; size_t OCp = OC/2;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int ic = ic0+ty; ic < ic1; ic += TY) {
        unsigned f = __ldg((const unsigned*)&packed[(size_t)ic*OCp + jbase/2]);
        float xx = __half2float(__ldg(&X[ic]));
        #pragma unroll
        for (int c = 0; c < 8; c++) { unsigned char id = (f>>(4*c))&0xF; acc[c] += xx*__half2float(s_cb[id*CPB+tx*8+c]); }
    }
    #pragma unroll
    for (int c = 0; c < 8; c++) red[ty*CPB+tx*8+c] = acc[c];
    __syncthreads();
    if (ty == 0) {
        #pragma unroll
        for (int c = 0; c < 8; c++) { float s = 0; for (int y = 0; y < TY; y++) s += red[y*CPB+tx*8+c]; atomicAdd(&Yacc[j0+tx*8+c], s); }
    }
}
torch::Tensor codebook_gemv(torch::Tensor x, torch::Tensor packed, torch::Tensor cb) {
    int IC = x.size(0), OC = cb.size(1);
    auto y = torch::zeros({OC}, x.options().dtype(torch::kFloat32));
    size_t smem = (size_t)K*CPB*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
    cudaFuncSetAttribute(gemv4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(OC/CPB, 20), block(32, TY);
    gemv4<<<grid, block, smem>>>((const __half*)x.data_ptr<at::Half>(),
        packed.data_ptr<unsigned char>(), (const __half*)cb.data_ptr<at::Half>(),
        y.data_ptr<float>(), IC, OC);
    return y;
}
'''
CPP = "torch::Tensor codebook_gemv(torch::Tensor x, torch::Tensor packed, torch::Tensor cb);"
print("compiling kernel op...")
ext = load_inline(name="cb_ext", cpp_sources=[CPP], cuda_sources=[CUDA], functions=["codebook_gemv"],
                  with_cuda=True, verbose=False, extra_cuda_cflags=["-O3"])

def quantize_per_column(W, k=16, iters=10, chunk=2048):
    # chunk over OC columns so the (IC, chunk, K) temp stays small (24 GB-safe).
    IC, OC = W.shape
    idx = torch.empty(IC, OC, dtype=torch.uint8, device=W.device)
    cbf = torch.empty(k, OC, dtype=torch.float16, device=W.device)
    for c0 in range(0, OC, chunk):
        c1 = min(OC, c0 + chunk); Wc = W[:, c0:c1]; cw = c1 - c0
        cb = torch.zeros(k, cw, device=W.device); lo = Wc.min(0).values; hi = Wc.max(0).values
        for c in range(k): cb[c] = lo + (hi - lo) * (c + 0.5) / k
        ii = None
        for _ in range(iters):
            d = (Wc.unsqueeze(-1) - cb.t().unsqueeze(0)).abs(); ii = d.argmin(-1); del d
            for c in range(k):
                m = (ii == c); cb[c] = (Wc * m).sum(0) / m.sum(0).clamp(min=1)
        idx[:, c0:c1] = ii.to(torch.uint8); cbf[:, c0:c1] = cb.to(torch.float16)
    return idx, cbf

def pack4(idx):
    a = idx[:, 0::2].to(torch.int32); b = idx[:, 1::2].to(torch.int32)
    return (a | (b<<4)).to(torch.uint8).contiguous()

class CodebookLinear(nn.Module):
    def __init__(self, packed, cb):
        super().__init__()
        self.register_buffer("packed", packed)   # (IC, OC/2)
        self.register_buffer("cb", cb)            # (K, OC)
        self.IC, self.OC = packed.shape[0], cb.shape[1]
    def forward(self, x):
        flat = x.reshape(-1, self.IC)
        if flat.shape[0] == 1:                    # decode: kernel
            y = ext.codebook_gemv(flat[0].half().contiguous(), self.packed, self.cb)
            return y.half().reshape(*x.shape[:-1], self.OC)
        # prefill: reconstruct W from packed and matmul
        p = self.packed
        idx = torch.empty(self.IC, self.OC, dtype=torch.long, device=p.device)
        idx[:, 0::2] = (p & 0xF).long(); idx[:, 1::2] = (p >> 4).long()
        Wdeq = self.cb[idx, torch.arange(self.OC, device=p.device)]   # (IC, OC) fp16
        return (flat.half() @ Wdeq).reshape(*x.shape[:-1], self.OC)

@torch.no_grad()
def quantize_model(model):
    import re
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
            Wt = mod.weight.data.t().float().contiguous()    # (IC, OC)
            idx, cb = quantize_per_column(Wt, K)
            packed = pack4(idx)
            parent = model.get_submodule(name.rsplit(".", 1)[0]); child = name.rsplit(".", 1)[1]
            setattr(parent, child, CodebookLinear(packed, cb).to(mod.weight.device))
            del mod, Wt, idx, cb
    torch.cuda.empty_cache()

@torch.no_grad()
def decode_toks(model, tok, new=128, warmup=1, reps=3):
    ids = tok("The history of computing began", return_tensors="pt").input_ids.to(model.device)
    for _ in range(warmup):
        model.generate(ids, max_new_tokens=new, do_sample=False)
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s = time.time(); model.generate(ids, max_new_tokens=new, do_sample=False)
        torch.cuda.synchronize(); ts.append(new/(time.time()-s))
    ts.sort(); return ts[len(ts)//2]

def vram(): return torch.cuda.max_memory_allocated()/1e9

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()
    torch.cuda.reset_peak_memory_stats()
    tps_fp16 = decode_toks(model, tok)
    print("fp16            : decode %.1f tok/s | peak VRAM %.2f GB" % (tps_fp16, vram()))
    torch.cuda.empty_cache()
    print("quantizing to 4-bit codebook + kernel...")
    quantize_model(model)
    torch.cuda.reset_peak_memory_stats()
    tps_cb = decode_toks(model, tok)
    print("codebook 4-bit  : decode %.1f tok/s | peak VRAM %.2f GB" % (tps_cb, vram()))
    print("\nDECODE: fp16 %.1f -> codebook-kernel %.1f tok/s  (x%.2f)" % (tps_fp16, tps_cb, tps_cb/tps_fp16))

if __name__ == "__main__":
    main()
