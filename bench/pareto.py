#!/usr/bin/env python3
"""
Honest Pareto on a real Llama-2-7B, one machine, equal-effective-bits where possible.

For each method it reports the four axes the literature usually splits across papers:
  - Wikitext-2 perplexity (accuracy)
  - model weight memory (GB) and effective bits/weight
  - single-stream decode throughput (tokens/s, batch 1)
  - energy per token (J/token), sampled from nvidia-smi power.draw

Methods:
  - fp16            : dense cuBLAS baseline (HF)
  - codebook-4bit   : ours, the fused 4-bit per-column codebook (this repo's kernel)
  - aqlm-2bit       : ISTA-DASLab AQLM 2x8 checkpoint, best-effort (needs the `aqlm` pkg)

Decode tok/s is batch 1 (this kernel's regime); batched serving throughput and a Marlin
(uniform 4-bit) column are honest TODOs, printed as such rather than faked. PPL for the
codebook method is computed on the dequantized weights (the kernel is an exact decoder of
those weights), so it is the true accuracy of the quantization.

Usage: python pareto.py --model NousResearch/Llama-2-7b-hf --ctx 2048 --ppl-tokens 40000 \
                        --gen 128 --out /root/bench
"""
import argparse, json, os, subprocess, threading, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

K = 16
GS = 20

KERNEL = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#define K 16
#define CPB 256
#define TY 8
__global__ void gemv4(const __half* X, const unsigned char* packed, const __half* cb,
                      float* Yacc, int IC, int OC, int GSg) {
    extern __shared__ char sm[];
    __half* s_cb=(__half*)sm; float* red=(float*)(s_cb+K*CPB);
    int tx=threadIdx.x, ty=threadIdx.y, tid=ty*32+tx, nth=32*TY, j0=blockIdx.x*CPB;
    for(int t=tid;t<K*CPB/2;t+=nth){int idx=t*2,k=idx/CPB,jj=j0+(idx%CPB);
        *reinterpret_cast<__half2*>(&s_cb[idx])=*reinterpret_cast<const __half2*>(&cb[(size_t)k*OC+jj]);}
    __syncthreads();
    int per=(IC+GSg-1)/GSg, ic0=blockIdx.y*per, ic1=min(IC,ic0+per), jbase=j0+tx*8; size_t OCp=OC/2;
    float acc[8]={0,0,0,0,0,0,0,0};
    for(int ic=ic0+ty;ic<ic1;ic+=TY){unsigned f=__ldg((const unsigned*)&packed[(size_t)ic*OCp+jbase/2]);
        float xx=__half2float(__ldg(&X[ic]));
        #pragma unroll
        for(int c=0;c<8;c++){unsigned char id=(f>>(4*c))&0xF; acc[c]+=xx*__half2float(s_cb[id*CPB+tx*8+c]);}}
    #pragma unroll
    for(int c=0;c<8;c++) red[ty*CPB+tx*8+c]=acc[c];
    __syncthreads();
    if(ty==0){
        #pragma unroll
        for(int c=0;c<8;c++){float s=0; for(int y=0;y<TY;y++) s+=red[y*CPB+tx*8+c]; atomicAdd(&Yacc[j0+tx*8+c],s);}}
}
torch::Tensor codebook_gemv(torch::Tensor x, torch::Tensor packed, torch::Tensor cb, int GSg){
    int IC=x.size(0), OC=cb.size(1);
    auto y=torch::zeros({OC}, x.options().dtype(torch::kFloat));
    size_t smem=(size_t)K*CPB*sizeof(__half)+(size_t)TY*CPB*sizeof(float);
    dim3 grid(OC/CPB,GSg), block(32,TY);
    gemv4<<<grid,block,smem>>>((const __half*)x.data_ptr<at::Half>(),
        (const unsigned char*)packed.data_ptr<uint8_t>(),(const __half*)cb.data_ptr<at::Half>(),
        y.data_ptr<float>(), IC, OC, GSg);
    return y;
}
"""

ext = load_inline(
    name="pareto_cb",
    cpp_sources="torch::Tensor codebook_gemv(torch::Tensor x, torch::Tensor packed, torch::Tensor cb, int g);",
    cuda_sources=KERNEL,
    functions=["codebook_gemv"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


@torch.no_grad()
def quantize(weight):
    # weight [out,in] -> packed [in,out/2] u8, cb [K,out] half, W_dq [out,in] half (all cuda)
    Wt = weight.t().contiguous().float().cuda()
    inn, out = Wt.shape
    lo, hi = Wt.min(0).values, Wt.max(0).values
    cen = torch.stack([lo + (hi - lo) * (k / (K - 1)) for k in range(K)], 0)
    bk = torch.zeros(inn, out, dtype=torch.long, device="cuda")
    for _ in range(12):
        bd = torch.full((inn, out), float("inf"), device="cuda")
        for k in range(K):
            d = (Wt - cen[k:k + 1]) ** 2
            b = d < bd
            bk = torch.where(b, torch.full_like(bk, k), bk)
            bd = torch.where(b, d, bd)
        for k in range(K):
            m = (bk == k).float()
            c = m.sum(0)
            cen[k] = torch.where(c > 0, (Wt * m).sum(0) / c.clamp_min(1), cen[k])
    Wt_dq = torch.gather(cen, 0, bk)
    idx = bk.to(torch.uint8)
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous()
    return packed, cen.half().contiguous(), Wt_dq.t().contiguous().half()


class CodebookLinear(nn.Module):
    """Decode-only (batch-1) linear using the fused codebook kernel."""
    def __init__(self, lin):
        super().__init__()
        packed, cb, _ = quantize(lin.weight.data)
        self.packed, self.cb = packed, cb
        self.out = cb.size(1)
        self.bias = lin.bias

    def forward(self, x):
        # x: [.., in]. Decode is one token (one kernel call); a short prefill loops the
        # rows through the batch-1 GEMV (one-time, not on the timed decode path).
        flat = x.reshape(-1, x.size(-1)).half().contiguous()
        outs = [ext.codebook_gemv(flat[i].contiguous(), self.packed, self.cb, GS)
                for i in range(flat.size(0))]
        y = torch.stack(outs, 0).to(x.dtype).reshape(*x.shape[:-1], self.out)
        if self.bias is not None:
            y = y + self.bias
        return y


class PowerSampler(threading.Thread):
    """Samples GPU power via pynvml (low-overhead, in-process) or nvidia-smi as fallback.
    Stores (t, watts) pairs so energy is INTEGRATED (trapezoid) over the real window,
    not approximated by mean_watts/tps."""
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []      # watts
        self.times = []        # monotonic timestamps (s)
        self.run_flag = True
        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            self.nvml = None

    def run(self):
        while self.run_flag:
            try:
                if self.nvml is not None:
                    w = self._pynvml.nvmlDeviceGetPowerUsage(self.nvml) / 1000.0
                else:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2).stdout.strip().splitlines()[0]
                    w = float(out)
                self.samples.append(w); self.times.append(time.monotonic())
            except Exception:
                pass
            time.sleep(0.02)

    def mean_watts(self):
        return float(np.mean(self.samples)) if self.samples else float("nan")
    def std_watts(self):
        return float(np.std(self.samples)) if self.samples else float("nan")
    def energy_joules(self):
        """Trapezoidal integral of power over the sampled window (J) -- the rigorous number."""
        if len(self.samples) < 2:
            return float("nan")
        return float(np.trapz(np.array(self.samples), np.array(self.times)))


def measure_idle_watts(seconds=3.0):
    """GPU idle-power baseline (no compute): lets us report NET (active) energy too,
    so the headline is not inflated by the card's static draw."""
    torch.cuda.synchronize()
    ps = PowerSampler(); ps.start()
    time.sleep(seconds)
    ps.run_flag = False; ps.join()
    return ps.mean_watts()


def linears(model):
    for L in model.model.layers:
        a = L.self_attn
        yield from [a.q_proj, a.k_proj, a.v_proj, a.o_proj]
        m = L.mlp
        yield from [m.gate_proj, m.up_proj, m.down_proj]


@torch.no_grad()
def wikitext_ppl(model, tok, ctx, max_tokens):
    if max_tokens <= 0:
        return None   # skip PPL (already known) -> energy-only run, avoids datasets-lib version issues
    from datasets import load_dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids[0][:max_tokens].cuda()
    nll, cnt = 0.0, 0
    for i in range(0, ids.size(0) - 1, ctx):
        chunk = ids[i:i + ctx + 1]
        if chunk.size(0) < 2:
            break
        out = model(chunk[:-1].unsqueeze(0))
        lp = torch.log_softmax(out.logits[0].float(), -1)
        tgt = chunk[1:]
        nll += -lp[torch.arange(tgt.size(0)), tgt].sum().item()
        cnt += tgt.size(0)
    return float(np.exp(nll / cnt))


@torch.no_grad()
def decode_energy(model, tok, n_gen, repeats=3, idle_w=None):
    """Returns (tps, jpt, extra). jpt is energy/token from the TRAPEZOIDAL integral of
    sampled power (not mean_watts/tps), averaged over `repeats` runs with std reported.
    extra['jpt_net'] subtracts the idle baseline so the headline is the ACTIVE energy."""
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    # warmup once
    past = model(ids, use_cache=True).past_key_values
    cur = ids[:, -1:]
    for _ in range(8):
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        cur = o.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    tps_l, jpt_l, w_l = [], [], []
    for _ in range(repeats):
        past = model(ids, use_cache=True).past_key_values   # fresh context per run (comparable)
        cur = ids[:, -1:]
        torch.cuda.synchronize()
        ps = PowerSampler(); ps.start()
        t0 = time.time()
        for _ in range(n_gen):
            o = model(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        dt = time.time() - t0
        ps.run_flag = False; ps.join()
        e = ps.energy_joules()
        jpt = (e / n_gen) if e == e else (ps.mean_watts() / (n_gen / dt))  # integral, fallback to mean
        tps_l.append(n_gen / dt); jpt_l.append(jpt); w_l.append(ps.mean_watts())
    tps = float(np.mean(tps_l)); jpt = float(np.mean(jpt_l))
    extra = {"jpt_std": float(np.std(jpt_l)), "watts": float(np.mean(w_l)),
             "idle_w": idle_w, "repeats": repeats,
             "jpt_net": (jpt - idle_w / tps) if isinstance(idle_w, (int, float)) else None}
    return tps, jpt, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ppl-tokens", type=int, default=40000)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--out", default="/root/bench")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = []

    def load_fp16():
        return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()

    # fp16 baseline
    print("== fp16 ==", flush=True)
    m = load_fp16()
    idle_w = measure_idle_watts()       # GPU static draw, for NET (active) energy
    print("idle baseline: %.1f W" % idle_w, flush=True)
    nparam = sum(p.numel() for p in m.parameters())
    ppl = wikitext_ppl(m, tok, args.ctx, args.ppl_tokens)
    tps, jpt, ex = decode_energy(m, tok, args.gen, idle_w=idle_w)
    rows.append(dict(method="fp16", bits=16.0, gb=nparam * 2 / 1e9, ppl=ppl, tps=tps, jpt=jpt,
                     jpt_std=ex["jpt_std"], jpt_net=ex["jpt_net"], watts=ex["watts"], idle_w=idle_w))
    print(rows[-1], flush=True)
    del m; torch.cuda.empty_cache()

    # ours: 4-bit codebook. PPL on the dequantized weights; tok/s + energy with the kernel.
    print("== codebook-4bit (ours) ==", flush=True)
    md = load_fp16()
    qbytes, qnumel = 0, 0
    for lin in linears(md):
        packed, cb, w_dq = quantize(lin.weight.data)
        lin.weight.data = w_dq.to(lin.weight.device)            # dequantized for PPL
        qbytes += packed.numel() + cb.numel() * 2               # 4-bit codes + fp16 codebook
        qnumel += lin.weight.numel()
    ppl_q = wikitext_ppl(md, tok, args.ctx, args.ppl_tokens)
    del md; torch.cuda.empty_cache()
    eff_bits = qbytes * 8 / qnumel                              # effective bits on the quantized weights
    ours_gb = (qbytes + (nparam - qnumel) * 2) / 1e9           # quantized linears + the rest in fp16
    mk = load_fp16()
    for L in mk.model.layers:
        a = L.self_attn
        a.q_proj, a.k_proj, a.v_proj, a.o_proj = (CodebookLinear(a.q_proj), CodebookLinear(a.k_proj),
                                                  CodebookLinear(a.v_proj), CodebookLinear(a.o_proj))
        mm = L.mlp
        mm.gate_proj, mm.up_proj, mm.down_proj = (CodebookLinear(mm.gate_proj), CodebookLinear(mm.up_proj),
                                                  CodebookLinear(mm.down_proj))
    tps_q, jpt_q, exq = decode_energy(mk, tok, args.gen, idle_w=idle_w)
    rows.append(dict(method="codebook-4bit", bits=round(eff_bits, 2), gb=round(ours_gb, 2),
                     ppl=ppl_q, tps=tps_q, jpt=jpt_q,
                     jpt_std=exq["jpt_std"], jpt_net=exq["jpt_net"], watts=exq["watts"], idle_w=idle_w))
    print(rows[-1], flush=True)
    del mk; torch.cuda.empty_cache()

    # AQLM 2-bit, best effort
    print("== aqlm-2bit (best-effort) ==", flush=True)
    try:
        ma = AutoModelForCausalLM.from_pretrained(
            "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf", torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
        ppl_a = wikitext_ppl(ma, tok, args.ctx, args.ppl_tokens)
        tps_a, jpt_a, exa = decode_energy(ma, tok, args.gen, idle_w=idle_w)
        gb_a = sum(p.numel() * p.element_size() for p in ma.parameters()) / 1e9
        rows.append(dict(method="aqlm-2bit", bits=2.0, gb=gb_a, ppl=ppl_a, tps=tps_a, jpt=jpt_a,
                         jpt_std=exa["jpt_std"], jpt_net=exa["jpt_net"], watts=exa["watts"], idle_w=idle_w))
        print(rows[-1], flush=True)
    except Exception as e:
        rows.append(dict(method="aqlm-2bit", bits=2.0, gb=None, ppl=None, tps=None, jpt=None,
                         note="unavailable: " + str(e)[:120]))
        print("aqlm skipped:", str(e)[:160], flush=True)

    # gCO2 is a DERIVED, secondary axis: J/token x grid intensity. We do NOT know RunPod's
    # actual grid mix, so this is a projection at a stated intensity, not a measurement.
    # Reported per 1000 tokens for readability, at France-like (~50) and US-like (~400) grids.
    def gco2_per_ktok(jpt, intensity):
        return None if not isinstance(jpt, (int, float)) else jpt / 3.6e6 * intensity * 1000
    for r in rows:
        r["gco2_per_ktok_fr50"] = gco2_per_ktok(r.get("jpt"), 50)
        r["gco2_per_ktok_us400"] = gco2_per_ktok(r.get("jpt"), 400)

    json.dump(rows, open(os.path.join(args.out, "pareto.json"), "w"), indent=2)
    print("\n| method | bits | mem GB | PPL | tok/s | J/token (gross, +/-std) | J/token (net of idle) | gCO2/1k tok (FR/US) |", flush=True)
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for r in rows:
        def f(x, p="{:.2f}"):
            return p.format(x) if isinstance(x, (int, float)) else "n/a"
        co2 = f"{f(r['gco2_per_ktok_fr50'],'{:.3f}')} / {f(r['gco2_per_ktok_us400'],'{:.2f}')}"
        jcell = f"{f(r['jpt'])} +/- {f(r.get('jpt_std'),'{:.2f}')}"
        print(f"| {r['method']} | {f(r['bits'])} | {f(r['gb'])} | {f(r['ppl'])} | {f(r['tps'],'{:.1f}')} | {jcell} | {f(r.get('jpt_net'))} | {co2} |", flush=True)
    print("\nNOTE J/token gross = trapezoidal integral of sampled GPU power / tokens, mean over %d runs."
          % rows[0].get("repeats", 3) if rows else "", flush=True)
    print("J/token NET subtracts the idle baseline (%.1f W) so the figure is the ACTIVE decode energy."
          % (rows[0].get("idle_w") or float("nan")) if rows and rows[0].get("idle_w") else "", flush=True)
    print("NOTE J/token is the measured primary axis. gCO2/1k-tok = J/token x grid intensity", flush=True)
    print("(FR ~50, US ~400 gCO2/kWh) -- a projection, not a measurement (RunPod grid mix unknown).", flush=True)
    print("Batch-1 decode; Marlin (uniform 4-bit) and batched throughput are TODO.", flush=True)


if __name__ == "__main__":
    main()
