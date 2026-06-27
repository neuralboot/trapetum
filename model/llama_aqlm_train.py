#!/usr/bin/env python3
"""AQLM-core training at a kernel-friendly config, now with diagonal-Hessian
calibration + multiple rounds. Per-output scale + Hessian-weighted BEAM-SEARCH code
assignment + Hessian-weighted least-squares codebook updates, alternating ROUNDS
times. The diagonal Hessian h_i = mean(x_i^2) (per input channel, from calibration
data) approximates AQLM's output-error objective ||(W-What)X||^2 while staying
per-group separable. Goal: push M=4 below the no-calibration 6.13, toward fp16 5.83,
still decodable by the avq kernel.

Run:  MC=4 ROUNDS=3 python llama_aqlm_train.py
"""
import torch, torch.nn as nn, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "NousResearch/Llama-2-7b-hf"
K = 256; D = 8; SEQLEN = 2048; PPL_SAMPLES = 30; CALIB_SEQS = 8
MC = int(os.environ.get("MC", "4")); ROUNDS = int(os.environ.get("ROUNDS", "1")); BEAM = 4
# CALIB=0 (default) = the win: no-calibration beam+LSQ, M=4 -> PPL 6.13 (beats scalar 6.34).
# CALIB=1 = naive diagonal-Hessian weighting: a NEGATIVE result, it regresses (PPL 8.6,
# worse than no-calib). Real AQLM calibration needs the full Hessian + sequential GPTQ-style
# error correction, not diagonal MSE weighting; that is the multi-week reproduction.
CALIB = int(os.environ.get("CALIB", "0"))
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def kmeans(P, k=K, iters=5, chunk=1_000_000):
    N = P.shape[0]; C = P[torch.randperm(N, device=P.device)[:k]].clone()
    asg = torch.zeros(N, dtype=torch.long, device=P.device)
    for _ in range(iters):
        Cn2 = (C*C).sum(1)
        for s in range(0, N, chunk):
            p = P[s:s+chunk]; asg[s:s+chunk] = ((p*p).sum(1,keepdim=True) - 2*(p@C.t()) + Cn2).argmin(1)
        Cnew = torch.zeros_like(C); cnt = torch.zeros(k, device=P.device)
        Cnew.index_add_(0, asg, P); cnt.index_add_(0, asg, torch.ones(N, device=P.device)); C = Cnew/cnt.clamp(min=1).unsqueeze(1)
    return C

@torch.no_grad()
def beam_search(W, C, Hw, B=BEAM, chunk=40000):
    # W (N,d), C (M,K,d), Hw (N,d) per-dim weights. minimize sum_e Hw[e]*(w-recon)^2.
    N, d = W.shape; Mc = C.shape[0]
    codes = torch.empty(N, Mc, dtype=torch.long, device=W.device)
    for s in range(0, N, chunk):
        w = W[s:s+chunk]; hw = Hw[s:s+chunk]; n = w.shape[0]
        d0 = (hw[:,None,:] * (w[:,None,:] - C[0][None,:,:])**2).sum(-1)
        _, idx = d0.topk(B, dim=1, largest=False)
        bc = idx.unsqueeze(-1); br = C[0][idx]
        for m in range(1, Mc):
            cand = br[:,:,None,:] + C[m][None,None,:,:]
            sc = (hw[:,None,None,:] * (w[:,None,None,:] - cand)**2).sum(-1).reshape(n, B*K)
            _, flat = sc.topk(B, dim=1, largest=False)
            bsel = flat // K; ksel = flat % K
            bc = torch.cat([torch.gather(bc, 1, bsel.unsqueeze(-1).expand(-1,-1,m)), ksel.unsqueeze(-1)], -1)
            br = torch.gather(br, 1, bsel.unsqueeze(-1).expand(-1,-1,d)) + C[m][ksel]
        codes[s:s+chunk] = bc[:,0,:]
    return codes

@torch.no_grad()
def lsq_update(W, codes, hr, Mc, reg=1e-2):
    # per-row scalar weights hr (N,). Weighted normal equations, one solve for all dims.
    N, d = W.shape; P = Mc*K
    feat = codes + (torch.arange(Mc, device=W.device)*K)[None,:]
    AtW = torch.zeros(P, d, device=W.device)
    for m in range(Mc): AtW.index_add_(0, feat[:,m], hr[:,None]*W)
    AtA = torch.zeros(P, P, device=W.device)
    for m in range(Mc):
        for mp in range(Mc):
            AtA.view(-1).index_add_(0, feat[:,m]*P + feat[:,mp], hr)
    AtA += reg*torch.eye(P, device=W.device)
    return torch.linalg.solve(AtA, AtW).reshape(Mc, K, d)

@torch.no_grad()
def aqlm_quantize(W, h, Mc):
    OC, IC = W.shape
    s = W.abs().amax(1, keepdim=True).clamp(min=1e-8)
    P = (W/s).reshape(OC, IC//D, D).reshape(-1, D).contiguous()
    if CALIB and h is not None:   # diagonal-Hessian weighting (clamped); NEGATIVE result, regresses
        Hw = (h / h.mean().clamp(min=1e-8)).clamp(0.25, 4.0).reshape(IC//D, D).repeat(OC, 1)
    else:                          # uniform = the winning no-calibration path
        Hw = torch.ones(IC, device=W.device).reshape(IC//D, D).repeat(OC, 1)
    hr = Hw.mean(1).contiguous()                                        # (N,) per-row weight
    R = P.clone(); cb = []
    for m in range(Mc):
        cm = kmeans(R, K); cb.append(cm)
        codes_m = beam_search(R, cm.unsqueeze(0), Hw)[:,0]; R = R - cm[codes_m]
    C = torch.stack(cb)
    for _ in range(ROUNDS):
        codes = beam_search(P, C, Hw); C = lsq_update(P, codes, hr, Mc)
    codes = beam_search(P, C, Hw)
    recon = sum(C[m][codes[:,m]] for m in range(Mc))
    return (recon.reshape(OC, IC//D, D).reshape(OC, IC)) * s

@torch.no_grad()
def collect_hessian(model, enc):
    hs = {}
    def mk(n):
        def hook(m, inp, out): hs[n] = hs.get(n, 0) + (inp[0].detach().float()**2).reshape(-1, inp[0].shape[-1]).sum(0)
        return hook
    hooks = [m.register_forward_hook(mk(n)) for n,m in model.named_modules()
             if isinstance(m, nn.Linear) and any(t in n for t in TARGETS)]
    for i in range(CALIB_SEQS): model(enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device))
    for h in hooks: h.remove()
    return hs

@torch.no_grad()
def quantize_model(model, h_all, Mc):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
            h = h_all.get(name); h = h.to(mod.weight.device) if h is not None else None
            mod.weight.data = aqlm_quantize(mod.weight.data.float(), h, Mc).to(mod.weight.dtype); torch.cuda.empty_cache()

@torch.no_grad()
def ppl(model, enc):
    nseq = min(PPL_SAMPLES, enc.size(1)//SEQLEN); lf = nn.CrossEntropyLoss(reduction="sum"); tot=0.0; nt=0
    for i in range(nseq):
        ids = enc[:, i*SEQLEN:(i+1)*SEQLEN].to(model.device); o = model(ids).logits
        tot += lf(o[:, :-1, :].reshape(-1, o.size(-1)).float(), ids[:, 1:].reshape(-1)).item(); nt += ids[:,1:].numel()
    return float(torch.tensor(tot/nt).exp())

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0}).eval()
    print("fp16 PPL = %.4f" % ppl(model, enc), flush=True)
    h_all = collect_hessian(model, enc) if CALIB else {}
    t0 = time.time(); quantize_model(model, h_all, MC)
    print("AQLM-trained M=%d (%.0f-bit, ROUNDS=%d, CALIB=%d): PPL = %.4f  (%.0fs)"
          % (MC, MC*8/D, ROUNDS, CALIB, ppl(model, enc), time.time()-t0), flush=True)
    print("(refs: fp16 5.83 | scalar 4-bit 6.34 | no-calib beam M=4 = 6.13 WIN | naive diag-Hessian regresses 8.6)", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
