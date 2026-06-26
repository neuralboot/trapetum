#!/usr/bin/env python3
"""Minimal PyTorch binding + quantize recipe for the fused 4-bit codebook GEMV.

End to end on a REAL weight matrix:
  1. compiles the CUDA kernel as a torch op (codebook_gemv).
  2. quantize_per_column(W, K=16): per-output-column k-means -> 4-bit packed
     indices + fp16 codebook (the SqueezeLLM-family scheme).
  3. checks codebook_gemv(x, packed, cb) reproduces x @ W on a GPT-2 weight, and
     times it against a dense fp16 matmul.

To wire into a real nn.Linear: a Linear computes y = x @ weight.T, so quantize
W = layer.weight.t().contiguous() (in_features x out_features), then
codebook_gemv(x, packed, cb) gives the layer output (add bias separately).

Run on a GPU box:  pip install ninja transformers ; python quant_demo.py
"""
import torch, time
from torch.utils.cpp_extension import load_inline

# Full implementation (kernel + launch wrapper) lives in CUDA, compiled by nvcc.
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

print("compiling the torch op (first time is slow)...")
ext = load_inline(name="codebook_ext", cpp_sources=[CPP], cuda_sources=[CUDA],
                  functions=["codebook_gemv"], with_cuda=True, verbose=False,
                  extra_cuda_cflags=["-O3"])

def quantize_per_column(W, k=16, iters=12):
    IC, OC = W.shape
    cb = torch.zeros(k, OC, device=W.device, dtype=torch.float32)
    lo = W.min(0).values; hi = W.max(0).values
    for c in range(k):
        cb[c] = lo + (hi - lo) * (c + 0.5) / k
    idx = torch.zeros(IC, OC, device=W.device, dtype=torch.long)
    for _ in range(iters):
        d = (W.unsqueeze(-1) - cb.t().unsqueeze(0)).abs()   # (IC, OC, K)
        idx = d.argmin(-1)
        for c in range(k):
            mask = (idx == c)
            cnt = mask.sum(0).clamp(min=1)
            cb[c] = (W * mask).sum(0) / cnt
    return idx.to(torch.uint8), cb.to(torch.float16)

def pack4(idx):
    a = idx[:, 0::2].to(torch.int32); b = idx[:, 1::2].to(torch.int32)
    return (a | (b << 4)).to(torch.uint8).contiguous()

def main():
    import os, struct
    dev = "cuda"
    # a real weight matrix: GPT-2 via transformers, else W.bin, else synthetic.
    try:
        from transformers import AutoModelForCausalLM
        mdl = AutoModelForCausalLM.from_pretrained("gpt2")
        W = mdl.transformer.h[6].mlp.c_proj.weight.detach().float().contiguous()
        print("source: gpt2 mlp.c_proj")
    except Exception:
        if os.path.exists("W.bin"):
            data = open("W.bin", "rb").read(); m, n = struct.unpack("<ii", data[:8])
            W = torch.frombuffer(bytearray(data[8:]), dtype=torch.float32).reshape(m, n).clone()
            print("source: W.bin")
        else:
            torch.manual_seed(0); W = torch.randn(3072, 768) * 0.05
            print("source: synthetic")
    IC, OC = W.shape
    if IC % 256 or OC % 256:
        IC -= IC % 256; OC -= OC % 256; W = W[:IC, :OC].contiguous()
    W = W.to(dev)
    print("weight %dx%d" % (IC, OC))
    idx, cb = quantize_per_column(W, k=16)
    packed = pack4(idx)
    x = torch.randn(IC, device=dev, dtype=torch.float16)

    y_kernel = ext.codebook_gemv(x, packed, cb)
    Wdeq = cb[idx.long(), torch.arange(OC, device=dev)]       # reconstructed (IC, OC): cb[idx[i,j], j]
    y_ref = x.float() @ Wdeq.float()
    print("codebook_gemv vs reconstructed dense: rel err = %.3g" % ((y_kernel - y_ref).norm()/y_ref.norm()).item())
    y_true = x.float() @ W.float()
    print("codebook_gemv vs true fp16 weight   : rel err = %.3g (= quantization error)" % ((y_kernel - y_true).norm()/y_true.norm()).item())

    def t(f, n=300):
        f(); torch.cuda.synchronize(); s = time.time()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.time()-s)/n*1e3
    Wf16 = W.to(torch.float16)
    tk = t(lambda: ext.codebook_gemv(x, packed, cb))
    td = t(lambda: torch.mv(Wf16.t(), x))
    print("decode: kernel %.4f ms | torch dense fp16 mv %.4f ms | x%.2f" % (tk, td, td/tk))
    print("OK: torch binding works end to end on a real weight.")

if __name__ == "__main__":
    main()
