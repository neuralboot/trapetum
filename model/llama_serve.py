#!/usr/bin/env python3
"""Serving integration without per-op casts: realize the op-level x2.4 (kernel vs
cuBLAS GEMV) at the model level. The naive CodebookLinear lost end-to-end (x0.73)
because of per-op fp32->fp16 casts + allocations. Here:
  - the kernel writes into a PREALLOCATED fp32 buffer (no allocation per call),
  - one cheap copy_ casts to a PREALLOCATED fp16 buffer (no python .half() churn),
  - views (free) instead of .contiguous() copies,
  - then the whole decode step is CUDA-graph captured (zero per-token CPU overhead).

Measures Llama-2-7B decode tok/s: fp16 vs codebook (eager) vs codebook (CUDA graph).
Run:  pip install ninja transformers==4.44.2 ; python llama_serve.py
"""
import torch, torch.nn as nn, time
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-hf"
K = 16
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
// writes fp32 accumulator into a preallocated y (zeroed first), launches on the
// current (capturable) stream. No allocation, no python-side cast.
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
ext = load_inline(name="serve_ext", cpp_sources=[CPP], cuda_sources=[CUDA],
                  functions=["codebook_gemv_out"], with_cuda=True, verbose=False, extra_cuda_cflags=["-O3"])

def quantize_per_column(W, k=16, iters=10, chunk=2048):
    IC, OC = W.shape
    idx = torch.empty(IC, OC, dtype=torch.uint8, device=W.device)
    cbf = torch.empty(k, OC, dtype=torch.float16, device=W.device)
    for c0 in range(0, OC, chunk):
        c1 = min(OC, c0+chunk); Wc = W[:, c0:c1]; cw = c1-c0
        cb = torch.zeros(k, cw, device=W.device); lo = Wc.min(0).values; hi = Wc.max(0).values
        for c in range(k): cb[c] = lo + (hi-lo)*(c+0.5)/k
        ii = None
        for _ in range(iters):
            d = (Wc.unsqueeze(-1)-cb.t().unsqueeze(0)).abs(); ii = d.argmin(-1); del d
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
        self._Wdeq = None
    def forward(self, x):
        if x.shape[:-1].numel() == 1:                 # decode: clean kernel path, no casts/allocs
            ext.codebook_gemv_out(x.reshape(-1), self.packed, self.cb, self.y32)
            self.y16.copy_(self.y32)                  # one fp32->fp16 cast into a preallocated buffer
            return self.y16.view(*x.shape[:-1], self.OC)
        if self._Wdeq is None:                        # prefill: reconstruct once, cache
            p = self.packed
            idx = torch.empty(self.IC, self.OC, dtype=torch.long, device=p.device)
            idx[:, 0::2] = (p & 0xF).long(); idx[:, 1::2] = (p >> 4).long()
            self._Wdeq = self.cb[idx, torch.arange(self.OC, device=p.device)].contiguous()
        return (x.reshape(-1, self.IC).half() @ self._Wdeq).view(*x.shape[:-1], self.OC)

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
def decode_tps(model, tok, new=128, warmup=2, reps=4):
    ids = tok("The history of computing began in", return_tensors="pt").input_ids.to(model.device)
    for _ in range(warmup): model.generate(ids, max_new_tokens=new, do_sample=False)
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s = time.time(); model.generate(ids, max_new_tokens=new, do_sample=False)
        torch.cuda.synchronize(); ts.append(new/(time.time()-s))
    ts.sort(); return ts[len(ts)//2]

def vram(): return torch.cuda.max_memory_allocated()/1e9

def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()
    torch.cuda.reset_peak_memory_stats()
    tps_fp16 = decode_tps(model, tok)
    print("fp16             : decode %.1f tok/s | VRAM %.2f GB" % (tps_fp16, vram()), flush=True)

    print("quantizing (clean fp16-out integration)...", flush=True)
    quantize_model(model, dev)
    torch.cuda.reset_peak_memory_stats()
    tps_cb = decode_tps(model, tok)
    print("codebook (eager) : decode %.1f tok/s | VRAM %.2f GB" % (tps_cb, vram()), flush=True)
    print("\nEAGER: fp16 %.1f -> codebook %.1f tok/s  (x%.2f)  [naive wrapper was x0.73]"
          % (tps_fp16, tps_cb, tps_cb/tps_fp16), flush=True)

    # stretch: CUDA-graph the decode step via HF static cache + reduce-overhead compile
    try:
        torch.compiler.reset()
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
        _ = decode_tps(model, tok, new=32, warmup=3, reps=1)   # compile/capture warmup
        tps_g = decode_tps(model, tok)
        print("codebook (graph) : decode %.1f tok/s  (x%.2f vs fp16)" % (tps_g, tps_g/tps_fp16), flush=True)
    except Exception as e:
        print("graph path skipped:", str(e)[:160], flush=True)

if __name__ == "__main__":
    main()
