#!/usr/bin/env python3
"""Serving without per-op casts, take 2: MANUAL CUDA-graph capture of the decode
step (torch.compile cannot trace the pybind op). Both fp16 and codebook models run
through the SAME manual-graph decode loop with a StaticCache, so the only difference
is the linear kernel. This isolates whether the op-level x2.4 survives once the
per-token python overhead is removed by graph capture.

Run:  pip install ninja transformers==4.44.2 ; python llama_serve2.py
"""
import os, torch, torch.nn as nn, time
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = os.environ.get("SERVE_MODEL", "NousResearch/Llama-2-7b-hf")
K = 16; NEW = 128
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
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
void codebook_gemv_out(torch::Tensor x, torch::Tensor packed, torch::Tensor cb, torch::Tensor y) {
    int IC = x.size(0), OC = cb.size(1);
    y.zero_();
    size_t smem = (size_t)K*CPB*sizeof(__half) + (size_t)TY*CPB*sizeof(float);
    dim3 grid(OC/CPB, 20), block(32, TY);
    auto stream = at::cuda::getCurrentCUDAStream();
    gemv4<<<grid, block, smem, stream>>>((const __half*)x.data_ptr<at::Half>(),
        packed.data_ptr<unsigned char>(), (const __half*)cb.data_ptr<at::Half>(),
        y.data_ptr<float>(), IC, OC);
}
'''
CPP = "void codebook_gemv_out(torch::Tensor x, torch::Tensor packed, torch::Tensor cb, torch::Tensor y);"
print("compiling kernel op...", flush=True)
ext = load_inline(name="serve2_ext", cpp_sources=[CPP], cuda_sources=[CUDA],
                  functions=["codebook_gemv_out"], with_cuda=True, verbose=False, extra_cuda_cflags=["-O3"])

def quantize_per_column(W, k=16, iters=10, chunk=2048):
    IC, OC = W.shape
    idx = torch.empty(IC, OC, dtype=torch.uint8, device=W.device)
    cbf = torch.empty(k, OC, dtype=torch.float16, device=W.device)
    for c0 in range(0, OC, chunk):
        c1 = min(OC, c0+chunk); Wc = W[:, c0:c1]; cw = c1-c0
        cb = torch.zeros(k, cw, device=W.device); lo = Wc.min(0).values; hi = Wc.max(0).values
        for c in range(k): cb[c] = lo + (hi-lo)*(c/(k-1))   # linear init, canonical
        ii = None
        for _ in range(iters):
            d = (Wc.unsqueeze(-1)-cb.t().unsqueeze(0)) ** 2; ii = d.argmin(-1); del d   # L2/squared, canonical
            for c in range(k):
                m = (ii==c); cb[c] = (Wc*m).sum(0)/m.sum(0).clamp(min=1)
        idx[:, c0:c1] = ii.to(torch.uint8); cbf[:, c0:c1] = cb.to(torch.float16)
    return idx, cbf

def pack4(idx):
    a = idx[:, 0::2].to(torch.int32); b = idx[:, 1::2].to(torch.int32)
    return (a | (b<<4)).to(torch.uint8).contiguous()

class CodebookLinear(nn.Module):
    def __init__(self, packed, cb, dev):
        super().__init__()
        self.register_buffer("packed", packed); self.register_buffer("cb", cb)
        self.IC, self.OC = packed.shape[0], cb.shape[1]
        self.y32 = torch.zeros(self.OC, device=dev, dtype=torch.float32)
        self.y16 = torch.zeros(self.OC, device=dev, dtype=torch.float16)
    def forward(self, x):
        if x.shape[:-1].numel() == 1:                 # decode
            ext.codebook_gemv_out(x.reshape(-1), self.packed, self.cb, self.y32)
            self.y16.copy_(self.y32)
            return self.y16.view(*x.shape[:-1], self.OC)
        p = self.packed                                # prefill: reconstruct (no persistent cache)
        idx = torch.empty(self.IC, self.OC, dtype=torch.long, device=p.device)
        idx[:, 0::2] = (p & 0xF).long(); idx[:, 1::2] = (p >> 4).long()
        Wdeq = self.cb[idx, torch.arange(self.OC, device=p.device)]
        return (x.reshape(-1, self.IC).half() @ Wdeq).view(*x.shape[:-1], self.OC)

@torch.no_grad()
def quantize_model(model, dev):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
            Wt = mod.weight.data.t().float().contiguous()
            idx, cb = quantize_per_column(Wt, K); packed = pack4(idx)
            parent = model.get_submodule(name.rsplit(".",1)[0]); child = name.rsplit(".",1)[1]
            setattr(parent, child, CodebookLinear(packed, cb, dev).to(dev))
            del mod, Wt, idx, cb
    torch.cuda.empty_cache()

@torch.no_grad()
def graph_decode_tps(model, tok, dev, new=NEW):
    ids = tok("The history of computing began in", return_tensors="pt").input_ids.to(dev)
    P = ids.shape[1]; L = P + new + 8
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=L, device=dev, dtype=torch.float16)
    model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device=dev))
    static_in = ids[:, -1:].clone()
    static_pos = torch.tensor([P], device=dev)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model(static_in, past_key_values=cache, use_cache=True, cache_position=static_pos)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_logits = model(static_in, past_key_values=cache, use_cache=True, cache_position=static_pos).logits
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(new):
        g.replay()
        static_in.copy_(static_logits[:, -1:].argmax(-1)); static_pos.add_(1)
    torch.cuda.synchronize()
    return new / (time.time() - t0)

def vram(): return torch.cuda.max_memory_allocated()/1e9

def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()
    torch.cuda.reset_peak_memory_stats()
    tps_fp16 = graph_decode_tps(model, tok, dev)
    print("fp16     (CUDA graph) : decode %.1f tok/s | VRAM %.2f GB" % (tps_fp16, vram()), flush=True)
    print("quantizing...", flush=True); quantize_model(model, dev)
    torch.cuda.reset_peak_memory_stats()
    tps_cb = graph_decode_tps(model, tok, dev)
    print("codebook (CUDA graph) : decode %.1f tok/s | VRAM %.2f GB" % (tps_cb, vram()), flush=True)
    print("\nGRAPH end-to-end: fp16 %.1f -> codebook %.1f tok/s  (x%.2f)  [eager was x0.85]"
          % (tps_fp16, tps_cb, tps_cb/tps_fp16), flush=True)

if __name__ == "__main__":
    main()
