#!/usr/bin/env python3
"""
STREAMING CBKR exporter for DeepSeek-V3/R1 (671B, q_lora MLA + V3 sigmoid/grouped router).
No transformers, no torch required for CPU k-means (torch only if EXPORT_DEV=cuda for the
GPU k-means path). export_deepseek.py loads the whole model via `transformers` on CPU first
-- impossible at 671B (1.34TB bf16). This script never holds more than a few tensors at a
time: a lazy safetensors index is built from shard HEADERS only (name -> byte range), and
`load_deepseek_qlora`'s exact CBKR byte layout is written layer-by-layer, one tensor (one
expert, for MoE) fetched, quantized and freed at a time.

  python export_deepseek_stream.py --dir /path/to/DeepSeek-V3-bf16 --out /workspace/ds3
  python export_deepseek_stream.py --dir ... --out /workspace/ds3 --resume   # after a crash
  python export_deepseek_stream.py --selftest --out /tmp/cbkr_selftest       # no download

The --dir snapshot must contain config.json + model.safetensors.index.json (+ shards), or a
single model.safetensors. Reuses quantize()/wf16/wf32/qwrite from export_safetensors.py (same
k-means + packing: low nibble first, per-output-channel codebook [K][oc]) so the two exporters
never drift on how a 4-bit tensor is packed.
"""
import argparse, glob, json, math, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_safetensors import qwrite, wf16, wf32  # noqa: E402  (shared 4-bit packing + k-means)


def pad256(n):
    return ((n + 255) // 256) * 256


# ---------------------------------------------------------------------------
# Lazy safetensors index: parse every shard's JSON header ONCE (8-byte length +
# JSON), keep only {name: (shard_path, base_offset, start, end, dtype, shape)}.
# get(name) opens the shard, seeks, reads ONLY that tensor's bytes -- peak RAM
# is one tensor (+ quantize()'s workspace) at a time, never the whole model.
# ---------------------------------------------------------------------------
_ST_DT = {"F32": np.float32, "F16": np.float16, "F64": np.float64,
          "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
          "U8": np.uint8, "BOOL": np.bool_}


def _build_f8e4m3_lut():
    # F8_E4M3FN: 1 sign + 4 exp (bias 7) + 3 mantissa; exp=15,mant=7 is NaN (no inf).
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b >> 7) & 1 else 1.0
        exp = (b >> 3) & 0xF
        mant = b & 0x7
        if exp == 0:
            val = sign * (mant / 8.0) * (2.0 ** (1 - 7))
        elif exp == 15 and mant == 7:
            val = float("nan")
        else:
            val = sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        lut[b] = val
    return lut


_F8E4M3_LUT = _build_f8e4m3_lut()


def _dequant_fp8_blocks(w_f8, scale_inv, block=128):
    # w_f8: [rows,cols] raw (unscaled) fp8-decoded f32; scale_inv: [ceil(rows/blk),ceil(cols/blk)].
    rows, cols = w_f8.shape
    br, bc = scale_inv.shape
    out = np.empty_like(w_f8, dtype=np.float32)
    for i in range(br):
        r0, r1 = i * block, min((i + 1) * block, rows)
        for j in range(bc):
            c0, c1 = j * block, min((j + 1) * block, cols)
            out[r0:r1, c0:c1] = w_f8[r0:r1, c0:c1] * scale_inv[i, j]
    return out


class LazySafetensors:
    def __init__(self, d):
        self.dir = d
        self.index = {}  # name -> (shard_path, base_offset, start, end, dtype, shape)
        idxfile = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idxfile):
            shards = sorted(set(json.load(open(idxfile))["weight_map"].values()))
        else:
            shards = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(d, "*.safetensors")))]
        assert shards, f"no .safetensors shards found in {d}"
        for shard in shards:
            path = os.path.join(d, shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            base = 8 + n
            for name, meta in hdr.items():
                if name == "__metadata__":
                    continue
                s, e = meta["data_offsets"]
                self.index[name] = (path, base, s, e, meta["dtype"], tuple(meta["shape"]))
        self._fh_path = None
        self._fh = None
        print(f"  indexed {len(shards)} shard(s), {len(self.index)} tensors (headers only)", flush=True)

    def has(self, name):
        return name in self.index

    def _raw(self, name):
        path, base, s, e, dt, shape = self.index[name]
        if self._fh_path != path:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(path, "rb")
            self._fh_path = path
        self._fh.seek(base + s)
        return self._fh.read(e - s), dt, shape

    def get(self, name):
        raw, dt, shape = self._raw(name)
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        elif dt == "F8_E4M3":
            arr = _F8E4M3_LUT[np.frombuffer(raw, dtype=np.uint8)].reshape(shape)
            sname = name[:-len(".weight")] + ".weight_scale_inv" if name.endswith(".weight") else None
            if not sname or sname not in self.index:
                raise SystemExit(f"{name} is FP8 (F8_E4M3) with no matching weight_scale_inv "
                                  f"-- re-export from a BF16 mirror instead")
            sraw, sdt, sshape = self._raw(sname)
            assert sdt == "F32", f"unexpected weight_scale_inv dtype {sdt}"
            scale = np.frombuffer(sraw, dtype=np.float32).reshape(sshape)
            return _dequant_fp8_blocks(arr, scale)
        else:
            arr = np.frombuffer(raw, dtype=_ST_DT[dt]).astype(np.float32)
        return arr.reshape(shape) if shape else arr.reshape(())


# ---------------------------------------------------------------------------
# YaRN inv_freq + mscale-adjusted softmax_scale -- exact port of patch_deepseek_rope.py.
# ---------------------------------------------------------------------------
def yarn_find_correction_dim(num_rot, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base, max_pos):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(mn, mx, dim):
    if mn == mx:
        mx += 0.001
    f = (np.arange(dim, dtype=np.float32) - mn) / (mx - mn)
    return np.clip(f, 0, 1)


def yarn_get_mscale(scale, mscale):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_rope(cfg):
    dim, base = cfg["rope"], cfg["rope_theta"]
    q_head_dim = cfg["nope"] + cfg["rope"]
    softmax_scale = q_head_dim ** -0.5
    rs = cfg.get("rope_scaling")
    if rs and rs.get("type") == "yarn":
        factor = float(rs["factor"])
        beta_fast = float(rs.get("beta_fast", 32))
        beta_slow = float(rs.get("beta_slow", 1))
        orig_max = float(rs.get("original_max_position_embeddings", 4096))
        mscale_all = float(rs.get("mscale_all_dim", 0))
        idx = np.arange(0, dim, 2, dtype=np.float32)
        freq_extra = 1.0 / (base ** (idx / dim))
        freq_inter = 1.0 / (factor * base ** (idx / dim))
        low, high = yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max)
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        if mscale_all:
            m = yarn_get_mscale(factor, mscale_all)
            softmax_scale = softmax_scale * m * m
        print(f"  YaRN: factor={factor} low={low} high={high} mscale_all={mscale_all} "
              f"-> softmax_scale={softmax_scale:.5f}", flush=True)
    else:
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        print(f"  no yarn; softmax_scale={softmax_scale:.5f}", flush=True)
    return inv_freq.astype(np.float32), np.float32(softmax_scale)


def build_cfg_from_json(c):
    return dict(
        hidden=c["hidden_size"], n_heads=c["num_attention_heads"],
        kv_lora=c["kv_lora_rank"], nope=c["qk_nope_head_dim"], rope=c["qk_rope_head_dim"],
        vhd=c["v_head_dim"], q_lora_rank=c.get("q_lora_rank") or 0,
        inter_dense=c["intermediate_size"], moe_inter=c["moe_intermediate_size"],
        n_routed=c["n_routed_experts"], n_shared=c["n_shared_experts"], top_k=c["num_experts_per_tok"],
        vocab=c["vocab_size"], first_k_dense=c.get("first_k_dense_replace", 0),
        n_layers=c["num_hidden_layers"],
        n_group=c.get("n_group", 1) or 1, topk_group=c.get("topk_group", 1) or 1,
        sigmoid=1 if c.get("scoring_func", "softmax") == "sigmoid" else 0,
        eps=c["rms_norm_eps"], rope_theta=float(c.get("rope_theta", 10000.0)),
        rscale=float(c.get("routed_scaling_factor", 1.0)), rope_scaling=c.get("rope_scaling"))


# ---------------------------------------------------------------------------
# The writer: byte-for-byte what DeepSeekModel::load_deepseek_qlora (CBKR) expects.
# Parameterized over `get(name) -> np.ndarray[out,in]` so the SAME code path is used
# for the real streaming export and the --selftest in-memory synthetic model (the
# whole point: no layout drift between the two).
# ---------------------------------------------------------------------------
def write_cbkr(out_path, cfg, get, resume=False, progress_path=None):
    progress_path = progress_path or (out_path + ".progress.json")
    hidden, n_heads = cfg["hidden"], cfg["n_heads"]
    kv_lora, nope, rope, vhd = cfg["kv_lora"], cfg["nope"], cfg["rope"], cfg["vhd"]
    q_lora_rank = cfg["q_lora_rank"]
    assert q_lora_rank > 0, "CBKR requires q_lora_rank set (use export_deepseek.py's CBKD branch otherwise)"
    inter_dense, moe_inter = cfg["inter_dense"], cfg["moe_inter"]
    n_routed, n_shared, top_k, vocab = cfg["n_routed"], cfg["n_shared"], cfg["top_k"], cfg["vocab"]
    first_k_dense, n_layers = cfg["first_k_dense"], cfg["n_layers"]
    n_group, topk_group, sigmoid = cfg["n_group"], cfg["topk_group"], cfg["sigmoid"]
    eps, rscale = cfg["eps"], cfg["rscale"]

    qdim = n_heads * (nope + rope)
    inter_dense_pad, moe_inter_pad = pad256(inter_dense), pad256(moe_inter)
    shared_inter_pad = pad256(n_shared * moe_inter_pad)
    assert qdim % 256 == 0, f"n_heads*(nope+rope)={qdim} must be %256 for the quantized q_b"
    assert hidden % 256 == 0, f"hidden={hidden} must be %256 for the quantized o_proj"
    assert vocab % 256 == 0, f"vocab={vocab} must be %256 for the quantized LM head"
    inv_freq, softmax_scale = compute_rope(cfg)

    layers_done = 0
    if resume and os.path.exists(progress_path) and os.path.exists(out_path):
        prog = json.load(open(progress_path))
        layers_done = prog["layers_done"]
        f = open(out_path, "r+b")
        f.truncate(prog["file_offset"])
        f.seek(prog["file_offset"])
        print(f"  RESUME: {layers_done}/{n_layers} layers already on disk, "
              f"continuing from byte offset {prog['file_offset']}", flush=True)
    else:
        if os.path.exists(out_path) and not resume:
            raise SystemExit(f"{out_path} already exists -- pass --resume to continue an "
                              f"interrupted export, or remove it to start fresh")
        f = open(out_path, "wb")
        f.write(b"CBKR")
        f.write(struct.pack("<18i", n_layers, hidden, n_heads, kv_lora, nope, rope, vhd,
                             inter_dense_pad, moe_inter_pad, n_routed, n_shared, top_k, vocab,
                             first_k_dense, q_lora_rank, n_group, topk_group, sigmoid))
        f.write(struct.pack("<3f", eps, softmax_scale, rscale))
        wf32(f, inv_freq)
        wf16(f, get("model.embed_tokens.weight"))
        f.flush()

    def _pad_out(w, oc):
        if w.shape[0] >= oc:
            return w
        return np.concatenate([w, np.zeros((oc - w.shape[0], w.shape[1]), w.dtype)], 0)

    def _pad_in(w, ic):
        if w.shape[1] >= ic:
            return w
        return np.concatenate([w, np.zeros((w.shape[0], ic - w.shape[1]), w.dtype)], 1)

    def qwrite_ffn(prefix, inter_pad):
        # gate/up padded on OUTPUT to inter_pad, down padded on INPUT -- zeros are lossless
        # (silu(0)=0), matches export_deepseek.py's qffn().
        qwrite(f, _pad_out(get(f"{prefix}.gate_proj.weight"), inter_pad))
        qwrite(f, _pad_out(get(f"{prefix}.up_proj.weight"), inter_pad))
        qwrite(f, _pad_in(get(f"{prefix}.down_proj.weight"), inter_pad))

    t0 = time.time()
    for li in range(layers_done, n_layers):
        tl0 = time.time()
        P = f"model.layers.{li}"
        wf32(f, get(f"{P}.input_layernorm.weight"))
        wf16(f, get(f"{P}.self_attn.q_a_proj.weight"))                # [q_lora_rank][hidden]
        wf32(f, get(f"{P}.self_attn.q_a_layernorm.weight"))           # [q_lora_rank]
        qwrite(f, get(f"{P}.self_attn.q_b_proj.weight"))              # [qdim][q_lora_rank] 4-bit
        wf16(f, get(f"{P}.self_attn.kv_a_proj_with_mqa.weight"))      # [kv_lora+rope][hidden]
        wf32(f, get(f"{P}.self_attn.kv_a_layernorm.weight"))          # [kv_lora]
        wf16(f, get(f"{P}.self_attn.kv_b_proj.weight"))               # [n_heads*(nope+vhd)][kv_lora]
        qwrite(f, get(f"{P}.self_attn.o_proj.weight"))                # [hidden][n_heads*vhd] 4-bit
        wf32(f, get(f"{P}.post_attention_layernorm.weight"))
        if li < first_k_dense:
            qwrite_ffn(f"{P}.mlp", inter_dense_pad)
            kind = "dense"
        else:
            wf32(f, get(f"{P}.mlp.gate.e_score_correction_bias"))     # [n_routed] BEFORE the router
            wf16(f, get(f"{P}.mlp.gate.weight"))                      # router [n_routed][hidden]
            for ei in range(n_routed):
                qwrite_ffn(f"{P}.mlp.experts.{ei}", moe_inter_pad)
            qwrite_ffn(f"{P}.mlp.shared_experts", shared_inter_pad)
            kind = "moe"
        f.flush()
        file_offset = f.tell()
        json.dump({"layers_done": li + 1, "file_offset": file_offset}, open(progress_path, "w"))
        dt = time.time() - tl0
        elapsed = time.time() - t0
        done = li + 1 - layers_done
        avg = elapsed / max(done, 1)
        eta_min = avg * (n_layers - (li + 1)) / 60
        print(f"  [{time.strftime('%H:%M:%S')}] layer {li+1}/{n_layers} ({kind}) {dt:.1f}s  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta_min:.1f}min", flush=True)

    wf32(f, get("model.norm.weight"))
    qwrite(f, get("lm_head.weight"))
    f.close()
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print(f"wrote {out_path}  {os.path.getsize(out_path)//(1024*1024)} MB", flush=True)


# ---------------------------------------------------------------------------
# --selftest: tiny synthetic q_lora+MoE model, in memory, no download. Runs the
# EXACT SAME write_cbkr() as production, just fed from a dict instead of a
# LazySafetensors index -- catches byte-layout drift between writer and reader.
# ---------------------------------------------------------------------------
SELFTEST_CFG = dict(hidden=512, n_heads=8, kv_lora=128, nope=64, rope=32, vhd=64,
                     q_lora_rank=128, inter_dense=512, moe_inter=256, n_routed=8, n_shared=1,
                     top_k=2, vocab=512, first_k_dense=1, n_layers=3, n_group=2, topk_group=1,
                     sigmoid=1, eps=1e-6, rope_theta=10000.0, rscale=2.5, rope_scaling=None)


def make_selftest_weights(cfg, seed=0):
    rng = np.random.RandomState(seed)
    hidden, n_heads = cfg["hidden"], cfg["n_heads"]
    kv_lora, nope, rope, vhd = cfg["kv_lora"], cfg["nope"], cfg["rope"], cfg["vhd"]
    q_lora_rank, qdim = cfg["q_lora_rank"], n_heads * (nope + rope)
    inter_dense, moe_inter = cfg["inter_dense"], cfg["moe_inter"]
    n_routed, n_shared, vocab, first_k_dense = cfg["n_routed"], cfg["n_shared"], cfg["vocab"], cfg["first_k_dense"]

    def rnd(*shape):
        return (rng.randn(*shape) * 0.02).astype(np.float32)

    def onesish(n):
        return (1.0 + rng.randn(n) * 0.02).astype(np.float32)

    W = {"model.embed_tokens.weight": rnd(vocab, hidden)}
    for li in range(cfg["n_layers"]):
        P = f"model.layers.{li}"
        W[f"{P}.input_layernorm.weight"] = onesish(hidden)
        W[f"{P}.self_attn.q_a_proj.weight"] = rnd(q_lora_rank, hidden)
        W[f"{P}.self_attn.q_a_layernorm.weight"] = onesish(q_lora_rank)
        W[f"{P}.self_attn.q_b_proj.weight"] = rnd(qdim, q_lora_rank)
        W[f"{P}.self_attn.kv_a_proj_with_mqa.weight"] = rnd(kv_lora + rope, hidden)
        W[f"{P}.self_attn.kv_a_layernorm.weight"] = onesish(kv_lora)
        W[f"{P}.self_attn.kv_b_proj.weight"] = rnd(n_heads * (nope + vhd), kv_lora)
        W[f"{P}.self_attn.o_proj.weight"] = rnd(hidden, n_heads * vhd)
        W[f"{P}.post_attention_layernorm.weight"] = onesish(hidden)
        if li < first_k_dense:
            W[f"{P}.mlp.gate_proj.weight"] = rnd(inter_dense, hidden)
            W[f"{P}.mlp.up_proj.weight"] = rnd(inter_dense, hidden)
            W[f"{P}.mlp.down_proj.weight"] = rnd(hidden, inter_dense)
        else:
            W[f"{P}.mlp.gate.e_score_correction_bias"] = rnd(n_routed)
            W[f"{P}.mlp.gate.weight"] = rnd(n_routed, hidden)
            for ei in range(n_routed):
                EP = f"{P}.mlp.experts.{ei}"
                W[f"{EP}.gate_proj.weight"] = rnd(moe_inter, hidden)
                W[f"{EP}.up_proj.weight"] = rnd(moe_inter, hidden)
                W[f"{EP}.down_proj.weight"] = rnd(hidden, moe_inter)
            SP = f"{P}.mlp.shared_experts"
            si = n_shared * moe_inter
            W[f"{SP}.gate_proj.weight"] = rnd(si, hidden)
            W[f"{SP}.up_proj.weight"] = rnd(si, hidden)
            W[f"{SP}.down_proj.weight"] = rnd(hidden, si)
    W["model.norm.weight"] = onesish(hidden)
    W["lm_head.weight"] = rnd(vocab, hidden)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="HF snapshot dir: config.json + safetensors shard(s)")
    ap.add_argument("--out", default="/workspace/ds3")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted export")
    ap.add_argument("--selftest", action="store_true", help="tiny synthetic model, no download")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.selftest:
        cfg = SELFTEST_CFG
        W = make_selftest_weights(cfg)
        out_path = os.path.join(args.out, "model_selftest.cbk")
        if os.path.exists(out_path):
            os.remove(out_path)  # selftest always starts fresh
        prog_path = out_path + ".progress.json"
        if os.path.exists(prog_path):
            os.remove(prog_path)
        write_cbkr(out_path, cfg, W.__getitem__, resume=False, progress_path=prog_path)
        print(f"\nselftest export OK: {out_path}")
        return

    assert args.dir, "--dir is required (or pass --selftest)"
    c = json.load(open(os.path.join(args.dir, "config.json")))
    cfg = build_cfg_from_json(c)
    print(f"cfg: L={cfg['n_layers']} hidden={cfg['hidden']} heads={cfg['n_heads']} "
          f"kv_lora={cfg['kv_lora']} nope={cfg['nope']} rope={cfg['rope']} vhd={cfg['vhd']} "
          f"q_lora_rank={cfg['q_lora_rank']} n_routed={cfg['n_routed']} top_k={cfg['top_k']} "
          f"n_group={cfg['n_group']} topk_group={cfg['topk_group']} sigmoid={cfg['sigmoid']} "
          f"first_k_dense={cfg['first_k_dense']} vocab={cfg['vocab']}", flush=True)
    idx = LazySafetensors(args.dir)
    out_path = os.path.join(args.out, "model.cbk")
    write_cbkr(out_path, cfg, idx.get, resume=args.resume, progress_path=out_path + ".progress.json")


if __name__ == "__main__":
    main()
