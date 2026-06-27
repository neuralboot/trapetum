#!/usr/bin/env python3
"""Decisive decode-op test: the codebook kernel vs the REAL fp16 path (cuBLAS GEMV
via F.linear, not the weak torch.mv), both with CUDA graphs (no launch overhead).
This is the fair "does the kernel beat cuBLAS at M=1 once integration overhead is
removed" question, on the 224 GEMVs of a Llama-2-7B decode token.

Run on a GPU box:  pip install ninja ; python graph_decode.py
"""
import torch, torch.nn.functional as Fn, time
from torch.utils.cpp_extension import load_inline

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
print("compiling kernel op...")
ext = load_inline(name="gd_ext", cpp_sources=[CPP], cuda_sources=[CUDA], functions=["codebook_gemv_out"],
                  with_cuda=True, verbose=False, extra_cuda_cflags=["-O3"])

dev = "cuda"
DIMS = ([(4096, 4096)] * 4 + [(4096, 11008), (4096, 11008), (11008, 4096)]) * 32

cb_x  = [torch.randn(IC, device=dev, dtype=torch.float16) for IC, OC in DIMS]
cb_y  = [torch.zeros(OC, device=dev, dtype=torch.float32) for IC, OC in DIMS]
cb_pk = [torch.randint(0, 256, (IC, OC // 2), device=dev, dtype=torch.uint8) for IC, OC in DIMS]
cb_cb = [torch.randn(16, OC, device=dev, dtype=torch.float16) for IC, OC in DIMS]
fp_W  = [torch.randn(OC, IC, device=dev, dtype=torch.float16) for IC, OC in DIMS]
fx    = [torch.randn(1, IC, device=dev, dtype=torch.float16) for IC, OC in DIMS]

def step_codebook():
    for i in range(len(DIMS)): ext.codebook_gemv_out(cb_x[i], cb_pk[i], cb_cb[i], cb_y[i])

def step_fp16():
    # the REAL fp16 path: F.linear -> cuBLAS GEMV
    for i in range(len(DIMS)): Fn.linear(fx[i], fp_W[i])

def capture(step):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): step()
    return g

def tps(run, n=200):
    run(); torch.cuda.synchronize(); s = time.time()
    for _ in range(n): run()
    torch.cuda.synchronize(); return n / (time.time() - s)

print("layers/token = %d (fp16 baseline = F.linear / cuBLAS)" % len(DIMS))
print("fp16(cuBLAS) eager : %.1f tok/s" % tps(step_fp16))
print("codebook     eager : %.1f tok/s" % tps(step_codebook))
g_fp = capture(step_fp16); g_cb = capture(step_codebook)
print("fp16(cuBLAS) graph : %.1f tok/s" % tps(g_fp.replay))
print("codebook     graph : %.1f tok/s" % tps(g_cb.replay))
print("DONE")
