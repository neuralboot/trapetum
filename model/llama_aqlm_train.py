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
# ROT (Lever 1): randomized-Hadamard rotation of the INPUT dim before the AVQ, folded back.
# Deployment-faithful: quantizing W_r = W Q^T and folding W_eff = deq_r Q reproduces exactly
# the QuaRot serving path (online Hadamard on activations x' = x Q^T + AVQ codes of the rotated
# weight), because y = x W^T = (x Q^T)(W Q^T)^T. Zero change to the avq decode kernel.
#   ROT=0  no rotation (baseline).
#   ROT=1  rotate ALL target layers (uniform; dilutes the gain -> only marginal on 7B).
#   ROT=2  SELECTIVE: rotate only o_proj + down_proj (the wide-outlier projections QuaRot rotates).
#          The 7B AVQ proxy showed the win is concentrated there (o_proj ratio ~0.63), rest neutral.
ROT = int(os.environ.get("ROT", "0"))
# OAEM=1 (Lever 3, "Initialisation Determines the Basin" 2604.08118): output-aware EM INIT.
# The greedy residual k-means init is weighted by the diagonal Hessian (Mahalanobis-diagonal
# distance + weighted centroid updates). Beam search and LSQ stay UNWEIGHTED -- weighting those
# is the known negative result (PPL 8.6). Only the codebook init basin changes.
OAEM = int(os.environ.get("OAEM", "0"))
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SEL_TARGETS = ("o_proj", "down_proj")           # layers rotated under ROT=2 (selective)

def _should_rotate(name):
    if ROT == 1: return True
    if ROT == 2: return any(s in name for s in SEL_TARGETS)
    return False

def _hadamard_pow2(n, device):
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H

def rand_hadamard(n, device, seed=0):
    # orthonormal (block-)Hadamard, n = p2 * odd -> block-diag of p2-Hadamards, with random signs.
    import math
    p2 = 1
    while (p2 * 2) <= n and (n % (p2 * 2) == 0): p2 *= 2
    Hb = _hadamard_pow2(p2, device) / math.sqrt(p2); odd = n // p2
    Q = torch.zeros(n, n, device=device)
    for b in range(odd): Q[b*p2:(b+1)*p2, b*p2:(b+1)*p2] = Hb
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1).to(device)
    return Q * d.unsqueeze(0)

def kmeans(P, k=K, iters=5, chunk=1_000_000, Hw=None):
    # Plain k-means, or (Hw given, (N,d)) output-aware EM: diagonal-Mahalanobis assignment
    # d(p,c) = sum_e Hw[i,e]*(p_e-c_e)^2 and Hw-weighted centroid updates (per-dim).
    N = P.shape[0]; C = P[torch.randperm(N, device=P.device)[:k]].clone()
    asg = torch.zeros(N, dtype=torch.long, device=P.device)
    for _ in range(iters):
        for s in range(0, N, chunk):
            p = P[s:s+chunk]
            if Hw is None:
                Cn2 = (C*C).sum(1)
                asg[s:s+chunk] = ((p*p).sum(1,keepdim=True) - 2*(p@C.t()) + Cn2).argmin(1)
            else:
                hw = Hw[s:s+chunk]                                   # (n,d)
                # sum_e hw*(p-c)^2 = (hw*p*p).sum - 2*(hw*p)@C.T-ish + hw@ (C*C).T
                asg[s:s+chunk] = ((hw*p*p).sum(1,keepdim=True) - 2*((hw*p)@C.t()) + hw@(C*C).t()).argmin(1)
        if Hw is None:
            Cnew = torch.zeros_like(C); cnt = torch.zeros(k, device=P.device)
            Cnew.index_add_(0, asg, P); cnt.index_add_(0, asg, torch.ones(N, device=P.device))
            C = Cnew/cnt.clamp(min=1).unsqueeze(1)
        else:
            num = torch.zeros_like(C); den = torch.zeros_like(C)
            num.index_add_(0, asg, Hw*P); den.index_add_(0, asg, Hw)
            C = num/den.clamp(min=1e-9)
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
def aqlm_quantize(W, h, Mc, rotate=False):
    OC, IC = W.shape
    Q = None
    if rotate:                                # rotate input dim: W_r = W Q^T ; quantize W_r
        Q = rand_hadamard(IC, W.device)
        W = W @ Q.t()
        if CALIB and h is not None: h = (Q * Q) @ h   # diag Hessian into the rotated basis
    s = W.abs().amax(1, keepdim=True).clamp(min=1e-8)
    P = (W/s).reshape(OC, IC//D, D).reshape(-1, D).contiguous()
    if CALIB and h is not None:   # diagonal-Hessian weighting (clamped); NEGATIVE result, regresses
        Hw = (h / h.mean().clamp(min=1e-8)).clamp(0.25, 4.0).reshape(IC//D, D).repeat(OC, 1)
    else:                          # uniform = the winning no-calibration path
        Hw = torch.ones(IC, device=W.device).reshape(IC//D, D).repeat(OC, 1)
    hr = Hw.mean(1).contiguous()                                        # (N,) per-row weight
    # OA-EM init weights (Lever 3): diag-Hessian per-dim, used ONLY in the init k-means.
    Hw_init = None
    if OAEM and h is not None:
        Hw_init = (h / h.mean().clamp(min=1e-8)).clamp(0.25, 4.0).reshape(IC//D, D).repeat(OC, 1)
    R = P.clone(); cb = []
    for m in range(Mc):
        cm = kmeans(R, K, Hw=Hw_init); cb.append(cm)
        codes_m = beam_search(R, cm.unsqueeze(0), Hw)[:,0]; R = R - cm[codes_m]
    C = torch.stack(cb)
    for _ in range(ROUNDS):
        codes = beam_search(P, C, Hw); C = lsq_update(P, codes, hr, Mc)
    codes = beam_search(P, C, Hw)
    recon = sum(C[m][codes[:,m]] for m in range(Mc))
    deq = (recon.reshape(OC, IC//D, D).reshape(OC, IC)) * s
    if rotate: deq = deq @ Q                  # fold back: W_eff = deq_r Q (== online Hadamard at inference)
    return deq

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
            mod.weight.data = aqlm_quantize(mod.weight.data.float(), h, Mc, rotate=_should_rotate(name)).to(mod.weight.dtype); torch.cuda.empty_cache()

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
    h_all = collect_hessian(model, enc) if (CALIB or OAEM) else {}
    t0 = time.time(); quantize_model(model, h_all, MC)
    print("AQLM-trained M=%d (%.0f-bit, ROUNDS=%d, CALIB=%d, ROT=%d, OAEM=%d): PPL = %.4f  (%.0fs)"
          % (MC, MC*8/D, ROUNDS, CALIB, ROT, OAEM, ppl(model, enc), time.time()-t0), flush=True)
    print("(refs: fp16 5.83 | scalar 4-bit 6.34 | no-calib beam M=4 = 6.13 WIN | naive diag-Hessian regresses 8.6)", flush=True)
    print("(Lever 1: ROT 0=none 1=all 2=selective[o_proj+down_proj]. Net win iff ROT=2 PPL < ROT=0 PPL)", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
