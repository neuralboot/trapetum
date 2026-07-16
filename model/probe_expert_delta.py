# probe_expert_delta.py -- Phase-1 go/no-go for the "shared base + low-rank delta" architecture
# rethink (D2-MoE / MoBE family) on the REAL DeepSeek-R1 expert weights.
#
# Question: do the 256 routed experts of an R1 MoE layer share enough structure that
#   W_e = base + delta_e   with delta_e of LOW effective rank?
# If yes: keep the base resident on the GPU (same for all experts of the layer) and stream
# only rank-r delta factors from RAM -> per-token expert bytes drop by ~ d_in*d_out / r(d_in+d_out)
# (~5-7x at r~10-15%), which attacks the ONE wall that killed all other speed levers.
#
# Decision metric (per projection type, median over sampled experts):
#   r95(delta) vs r95(W)   -- the rank capturing 95% of Frobenius energy.
#   If r95(delta) ~= r95(W): the base buys nothing (experts don't share structure) -> NO-GO.
#   If r95(delta) << r95(W): green light for base+delta streaming.
#
# Efficient: HTTP Range reads straight out of the HF fp8 shards (safetensors headers first,
# then only the wanted tensor byte ranges). ~2.8 GB per probed layer instead of 700 GB.
#
# Run:  /tmp/lev_venv/bin/python model/probe_expert_delta.py     (writes probe_expert_delta.json)

import json, math, os, struct, time, urllib.request
import numpy as np

REPO = "https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/main"
LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", "10,45").split(",")]
N_BASE = int(os.environ.get("PROBE_NBASE", "64"))     # experts sampled to estimate the base
N_ANALYZE = int(os.environ.get("PROBE_NSVD", "32"))   # deltas spectrally analyzed
N_CTRL = 8                                             # control: spectra of raw W
PROJS = ["gate_proj", "up_proj", "down_proj"]
CACHE = os.path.expanduser("~/.cache/r1_probe")
os.makedirs(CACHE, exist_ok=True)

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def http_get(url, start=None, end=None):
    req = urllib.request.Request(url, headers={"User-Agent": "trapetum-probe"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end-1}")
    with urllib.request.urlopen(req) as r:
        return r.read()

# ---- fp8 e4m3 LUT + block dequant (same math as export_deepseek_stream) ----
def _f8_lut():
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) & 1 else 1.0
        e = (b >> 3) & 0xF; m = b & 0x7
        if e == 0: v = s * (m/8.0) * 2.0**(1-7)
        elif e == 15 and m == 7: v = float("nan")
        else: v = s * (1.0 + m/8.0) * 2.0**(e-7)
        lut[b] = v
    return lut
LUT = _f8_lut()

def dequant_fp8(raw, shape, sraw, sshape, block=128):
    w = LUT[np.frombuffer(raw, np.uint8)].reshape(shape)
    sc = np.frombuffer(sraw, np.float32).reshape(sshape)
    r, c = shape
    out = np.empty(shape, np.float32)
    for i in range(sshape[0]):
        r0, r1 = i*block, min((i+1)*block, r)
        for j in range(sshape[1]):
            c0, c1 = j*block, min((j+1)*block, c)
            out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * sc[i, j]
    return out

# ---- lazy shard headers: name -> (shard, data_start_abs, end_abs, dtype, shape) ----
log("fetching index")
idx = json.loads(http_get(f"{REPO}/model.safetensors.index.json"))["weight_map"]
_hdr_cache = {}
def shard_header(shard):
    if shard in _hdr_cache: return _hdr_cache[shard]
    n = struct.unpack("<Q", http_get(f"{REPO}/{shard}", 0, 8))[0]
    hdr = json.loads(http_get(f"{REPO}/{shard}", 8, 8+n))
    _hdr_cache[shard] = (hdr, 8+n)
    return _hdr_cache[shard]

def fetch(name):
    p = os.path.join(CACHE, name.replace("/", "_") + ".npy")
    if os.path.exists(p): return np.load(p)
    shard = idx[name]
    hdr, base = shard_header(shard)
    m = hdr[name]; s, e = m["data_offsets"]
    raw = http_get(f"{REPO}/{shard}", base+s, base+e)
    if m["dtype"] == "F8_E4M3":
        sname = name.replace(".weight", ".weight_scale_inv")
        sshard = idx[sname]                        # scale may live in a DIFFERENT shard
        shdr, sbase = shard_header(sshard)
        ms = shdr[sname]; ss, se = ms["data_offsets"]
        sraw = http_get(f"{REPO}/{sshard}", sbase+ss, sbase+se)
        w = dequant_fp8(raw, m["shape"], sraw, ms["shape"])
    elif m["dtype"] == "BF16":
        u16 = np.frombuffer(raw, np.uint16).astype(np.uint32)
        w = (u16 << 16).view(np.float32).reshape(m["shape"])
    else:
        w = np.frombuffer(raw, {"F32": np.float32, "F16": np.float16}[m["dtype"]]).astype(np.float32).reshape(m["shape"])
    np.save(p, w)
    return w

def spectrum(W):
    # full singular-value spectrum via the small-side gram matrix (min dim 2048 on R1 experts)
    W = W.astype(np.float32)
    if W.shape[0] <= W.shape[1]: G = W @ W.T
    else: G = W.T @ W
    ev = np.linalg.eigvalsh(G)
    return np.sqrt(np.clip(ev[::-1], 0, None))       # descending singular values

def rank_at(sv, frac):
    e = sv**2; c = np.cumsum(e) / max(e.sum(), 1e-30)
    return int(np.searchsorted(c, frac) + 1)

rng = np.random.RandomState(0)
results = {"layers": {}, "config": {"n_base": N_BASE, "n_analyze": N_ANALYZE}}
for L in LAYERS:
    picks = rng.choice(256, size=N_BASE, replace=False)
    lay = {}
    for proj in PROJS:
        log(f"layer {L} {proj}: fetching {N_BASE} experts (range reads)")
        Ws = [fetch(f"model.layers.{L}.mlp.experts.{e}.{proj}.weight") for e in picks]
        base = np.mean(Ws, axis=0)
        r95_d, r99_d, relF = [], [], []
        for W in Ws[:N_ANALYZE]:
            sv = spectrum(W - base)
            r95_d.append(rank_at(sv, 0.95)); r99_d.append(rank_at(sv, 0.99))
            relF.append(float(np.linalg.norm(W - base) / np.linalg.norm(W)))
        r95_w = [rank_at(spectrum(W), 0.95) for W in Ws[:N_CTRL]]
        d_out, d_in = Ws[0].shape
        med95d = int(np.median(r95_d)); med95w = int(np.median(r95_w))
        bytes_ratio = med95d * (d_in + d_out) / (d_in * d_out)
        lay[proj] = {
            "shape": [int(d_out), int(d_in)],
            "r95_delta_median": med95d, "r99_delta_median": int(np.median(r99_d)),
            "r95_W_median": med95w, "rank_ratio": med95d / max(med95w, 1),
            "deltaF_over_WF_median": float(np.median(relF)),
            "delta_bytes_ratio_at_r95": bytes_ratio,
        }
        log(f"  L{L} {proj}: r95(delta)={med95d} vs r95(W)={med95w} "
            f"(ratio {med95d/max(med95w,1):.2f}), |delta|F/|W|F={np.median(relF):.3f}, "
            f"delta bytes@r95={bytes_ratio:.2f}x of full")
    results["layers"][str(L)] = lay

out = os.path.join(os.path.dirname(__file__), "probe_expert_delta.json")
open(out, "w").write(json.dumps(results, indent=2))
log("RESULT written", out)
print(json.dumps(results, indent=2))
