#!/usr/bin/env python3
"""
cbk_audit: read a compressed .cbk model (CBK1, CBK2, CBK3 scalar codebooks; CBK4 additive
codebooks, or scalar ones with an 8-bit LM head when avq_m = 0) and report how well its
codebooks are used, without loading the file in memory.

For every scalar projection (one codebook of K entries per output column) it measures:
  dead centroids    entries no weight of the column uses (codebook range spent on nothing)
  bulk99            how many entries hold 99% of the column's weights (low = resolution wasted
                    on a few outliers, the pattern of massive-activation / super-weight columns)
  index entropy     bits per index actually needed (4 for K=16 if every entry is used equally);
                    1 - H/log2(K) is the most lossless entropy coding could save on disk
  outlier ratio     largest |used centroid - column mean| / column std (super-weight style)
For CBK4 additive records (AQLM-style, M codebooks of 256 x 8) it reports, per codebook, the
number of entries never used and the entropy of the codes.

Usage:
  python model/cbk_audit.py <model.cbk> [--json out.json] [--html out.html] [--top 15]
Pure numpy; reads one matrix at a time. No em dashes or en dashes anywhere.
"""
import argparse, json, math, os, struct, sys, time
import numpy as np

PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ROW_CHUNK = 1024


def col_counts_k16(f, ic, oc):
    """Per-column usage counts of a nibble-packed K=16 matrix ([ic][oc/2] bytes, low nibble =
    even column). Returns counts (oc, 16) int64 and the packed bytes are not kept."""
    half = oc // 2
    ce = np.zeros(half * 16, np.int64)
    co = np.zeros(half * 16, np.int64)
    base = (np.arange(half, dtype=np.int64) * 16)[None, :]
    left = ic
    while left > 0:
        n = min(ROW_CHUNK, left)
        a = np.fromfile(f, dtype=np.uint8, count=n * half).reshape(n, half)
        ce += np.bincount(((a & 0x0F).astype(np.int64) + base).ravel(), minlength=half * 16)
        co += np.bincount(((a >> 4).astype(np.int64) + base).ravel(), minlength=half * 16)
        left -= n
    counts = np.empty((oc, 16), np.int64)
    counts[0::2] = ce.reshape(half, 16)
    counts[1::2] = co.reshape(half, 16)
    return counts


def col_counts_k256(f, ic, oc):
    """Per-column usage counts of a K=256 matrix ([ic][oc] u8 indices). Returns (oc, 256)."""
    acc = np.zeros(oc * 256, np.int64)
    base = (np.arange(oc, dtype=np.int64) * 256)[None, :]
    left = ic
    while left > 0:
        n = min(max(1, ROW_CHUNK // 4), left)
        a = np.fromfile(f, dtype=np.uint8, count=n * oc).reshape(n, oc)
        acc += np.bincount((a.astype(np.int64) + base).ravel(), minlength=oc * 256)
        left -= n
    return acc.reshape(oc, 256)


def find_rows(f, start, ic, oc, k, col, code):
    """Rows of column `col` whose index equals `code` (re-reads only what is needed)."""
    rows = []
    if k == 16:
        half = oc // 2
        f.seek(start)
        left, r0 = ic, 0
        while left > 0:
            n = min(ROW_CHUNK, left)
            a = np.fromfile(f, dtype=np.uint8, count=n * half).reshape(n, half)
            b = a[:, col // 2]
            v = (b & 0x0F) if col % 2 == 0 else (b >> 4)
            rows += [int(r0 + i) for i in np.nonzero(v == code)[0]]
            left -= n; r0 += n
    return rows


def scalar_stats(name, counts, cb, ic, top):
    """Column statistics for one scalar projection. cb is (K, oc) float32."""
    K = counts.shape[1]
    p = counts / float(ic)
    used = counts > 0
    dead = (~used).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(used, p * np.log2(np.where(used, p, 1.0)), 0.0)).sum(1)
    srt = np.sort(counts, axis=1)[:, ::-1]
    cum = np.cumsum(srt, axis=1)
    bulk99 = (cum < 0.99 * ic).sum(1) + 1
    c = cb.T.astype(np.float64)                       # (oc, K)
    mean = (p * c).sum(1)
    std = np.sqrt(np.maximum((p * (c - mean[:, None]) ** 2).sum(1), 1e-30))
    dev = np.where(used, np.abs(c - mean[:, None]), 0.0)
    worst_k = dev.argmax(1)
    ratio = dev.max(1) / std
    order = np.argsort(-ratio)[:top]
    tops = []
    for j in order:
        kk = int(worst_k[j])
        tops.append(dict(proj=name, col=int(j), ratio=float(ratio[j]), value=float(c[j, kk]),
                         code=kk, uses=int(counts[j, kk]), dead=int(dead[j]), bulk99=int(bulk99[j])))
    return dict(
        name=name, ic=int(ic), oc=int(counts.shape[0]), K=int(K),
        dead_slots=int(dead.sum()), slots=int(dead.size * K),
        cols_any_dead=int((dead > 0).sum()), cols_dead4=int((dead >= 4).sum()),
        cols_bulk_le_half=int((bulk99 <= K // 2).sum()),
        mean_entropy=float(ent.mean()), max_ratio=float(ratio.max()),
    ), tops


def cbka_stats(name, f, m_expected=None):
    magic = f.read(4)
    assert magic == b"CBKA", "expected a CBKA record at %s" % name
    m, d, k, rows, cols = struct.unpack("<5i", f.read(20))
    f.seek(2 * m * k * d + 2 * rows, 1)               # codebooks f16 + scales f16
    ng = cols // d
    per_cb = []
    for mi in range(m):
        acc = np.zeros(k, np.int64)
        left = ng * rows
        while left > 0:
            n = min(1 << 24, left)
            acc += np.bincount(np.fromfile(f, dtype=np.uint8, count=n), minlength=k)
            left -= n
        pr = acc / acc.sum()
        h = float(-(pr[pr > 0] * np.log2(pr[pr > 0])).sum())
        per_cb.append(dict(dead_entries=int((acc == 0).sum()), entropy=h))
    return dict(name=name, kind="cbka", M=m, K=k, rows=rows, cols=cols, codebooks=per_cb)


def audit(path, top=15, log=print):
    t0 = time.time()
    f = open(path, "rb")
    magic = f.read(4)
    if magic not in (b"CBK1", b"CBK2", b"CBK3", b"CBK4"):
        raise SystemExit("%s: not a .cbk file (magic %r)" % (path, magic))
    v2, v4 = magic == b"CBK2", magic == b"CBK4"
    v3 = magic == b"CBK3" or v4
    n_layers, hidden, n_heads, n_kv, head_dim, inter, vocab = struct.unpack("<7i", f.read(28))
    f.read(8)                                          # eps, rope base
    if v2 or v3:
        f.read(4)                                      # rope scale
    has_bias = struct.unpack("<i", f.read(4))[0] != 0 if v3 else False
    if v3:
        f.seek(4 * (head_dim // 2), 1)                 # inv_freq
    avq_m, lm_k = (struct.unpack("<2i", f.read(8)) if v4 else (0, 16))
    f.seek(2 * vocab * hidden, 1)                      # f16 embedding
    qd, kvd = n_heads * head_dim, n_kv * head_dim
    shapes = {"q_proj": (hidden, qd), "k_proj": (hidden, kvd), "v_proj": (hidden, kvd), "o_proj": (qd, hidden),
              "gate_proj": (hidden, inter), "up_proj": (hidden, inter), "down_proj": (inter, hidden)}
    meta = dict(file=os.path.abspath(path), bytes=os.path.getsize(path), format=magic.decode(),
                n_layers=n_layers, hidden=hidden, n_heads=n_heads, n_kv=n_kv, head_dim=head_dim,
                inter=inter, vocab=vocab, has_bias=has_bias, avq_m=avq_m, lm_k=lm_k)
    log("%s %s: %d layers, hidden %d, inter %d, vocab %d" % (os.path.basename(path), meta["format"], n_layers, hidden, inter, vocab))
    projs, tops, supers = [], [], []

    def scalar(name, ic, oc, k=16):
        start = f.tell()
        counts = col_counts_k16(f, ic, oc) if k == 16 else col_counts_k256(f, ic, oc)
        cb = np.fromfile(f, dtype="<f4", count=k * oc).reshape(k, oc)
        st, tp = scalar_stats(name, counts, cb, ic, top)
        return st, tp, start, counts, cb

    for li in range(n_layers):
        f.seek(4 * hidden, 1)                          # input norm
        for pn in PROJ_NAMES:
            if pn == "gate_proj":
                f.seek(4 * hidden, 1)                  # post-attention norm
            ic, oc = shapes[pn]
            name = "layers.%d.%s" % (li, pn)
            if v4 and avq_m:                          # CBK4 with avq_m = 0 keeps CBK3's scalar projections
                projs.append(cbka_stats(name, f))
            else:
                st, tp, start, counts, cb = scalar(name, ic, oc)
                st.update(layer=li, proj=pn)
                projs.append(st)
                tops += tp
                if pn == "down_proj" and li < 4 and tp and tp[0]["ratio"] > 20 and tp[0]["uses"] <= 2:
                    here = f.tell()
                    rows = find_rows(f, start, ic, oc, 16, tp[0]["col"], tp[0]["code"])
                    f.seek(here)
                    supers.append(dict(proj=name, col=tp[0]["col"], rows=rows, value=tp[0]["value"],
                                       ratio=tp[0]["ratio"], dead_in_column=tp[0]["dead"], bulk99=tp[0]["bulk99"]))
            if pn in ("q_proj", "k_proj", "v_proj") and has_bias:
                f.seek(4 * oc, 1)
        log("  layer %d/%d  (%.0fs)" % (li + 1, n_layers, time.time() - t0))
    f.seek(4 * hidden, 1)                              # final norm
    st, tp, _, _, _ = scalar("lm_head", hidden, vocab, lm_k)
    st.update(layer=-1, proj="lm_head")
    projs.append(st)
    tail = f.read(1)
    meta["trailing_bytes_ok"] = (tail == b"")
    f.close()

    sc = [p for p in projs if p.get("kind") != "cbka"]
    summary = {}
    if sc:
        slots = sum(p["slots"] for p in sc)
        cols = sum(p["oc"] for p in sc)
        w = [(p["mean_entropy"], p["oc"] * p["ic"], p["K"]) for p in sc]
        nidx = sum(n for _, n, _ in w)
        h_avg = sum(h * n for h, n, _ in w) / nidx
        bits = sum(math.log2(k) * n for _, n, k in w) / nidx
        by_proj = {}
        for pn in PROJ_NAMES + ["lm_head"]:
            ps = [p for p in sc if p["proj"] == pn]
            if ps:
                by_proj[pn] = dict(dead_pct=100.0 * sum(p["dead_slots"] for p in ps) / sum(p["slots"] for p in ps),
                                   cols_any_dead_pct=100.0 * sum(p["cols_any_dead"] for p in ps) / sum(p["oc"] for p in ps),
                                   cols_bulk_le_half=sum(p["cols_bulk_le_half"] for p in ps),
                                   mean_entropy=sum(p["mean_entropy"] * p["oc"] for p in ps) / sum(p["oc"] for p in ps))
        summary = dict(
            scalar_columns=cols, dead_slots=sum(p["dead_slots"] for p in sc), slots=slots,
            dead_pct=100.0 * sum(p["dead_slots"] for p in sc) / slots,
            cols_any_dead=sum(p["cols_any_dead"] for p in sc), cols_dead4=sum(p["cols_dead4"] for p in sc),
            cols_bulk_le_half=sum(p["cols_bulk_le_half"] for p in sc),
            index_bits=bits, index_entropy=h_avg, lossless_index_saving_pct=100.0 * (1 - h_avg / bits),
            by_proj=by_proj)
    tops.sort(key=lambda t: -t["ratio"])
    meta["seconds"] = round(time.time() - t0, 1)
    return dict(meta=meta, summary=summary, projections=projs, top_outlier_columns=tops[:top], super_weight_candidates=supers)


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trapetum codebook audit</title><style>
:root{--bg:#0d1117;--card:#161b22;--line:#21262d;--ink:#e6edf3;--muted:#8b949e;--rust:#f74c00;--green:#3fb950}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:26px 16px 56px}h1{font-size:26px;margin:4px 0 6px;color:var(--ink)}h2{font-size:19px;margin:30px 0 10px;color:var(--ink)}
.sub{color:var(--muted)}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:12px;padding:12px 14px}.kpi .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em;font-weight:700}
.kpi .v{font-size:24px;font-weight:800;margin-top:4px;color:var(--ink)}.kpi .n{color:var(--muted);font-size:12.5px}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}table{border-collapse:collapse;width:100%;min-width:680px;font-size:13px;color:var(--ink)}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}th{background:#1f2630;color:var(--ink)}td.n{text-align:right;font-variant-numeric:tabular-nums}
.bar{display:inline-block;height:9px;border-radius:0 3px 3px 0;background:var(--rust);vertical-align:middle;margin-right:6px}code{background:#0b0f15;color:var(--ink);padding:1px 5px;border-radius:4px}
@media(max-width:760px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><div class="wrap">
<div class="sub">Trapetum codebook audit</div><h1>__TITLE__</h1><div class="sub">__META__</div>
<div class="kpis">__KPIS__</div>
<h2>By projection type</h2><div class="tw"><table><thead><tr><th>Projection</th><th>Dead centroid slots</th><th>Columns with a dead centroid</th><th>Columns with 99% of weights on half the codebook or less</th><th>Index entropy (bits)</th></tr></thead><tbody>__PROJ__</tbody></table></div>
<h2>Most outlier-dominated columns</h2><div class="tw"><table><thead><tr><th>Projection</th><th>Column</th><th>Outlier ratio</th><th>Extreme centroid</th><th>Weights on it</th><th>Dead centroids</th><th>Entries holding 99%</th></tr></thead><tbody>__TOPS__</tbody></table></div>
<h2>Super-weight candidates</h2><div class="tw"><table><thead><tr><th>Projection</th><th>Column</th><th>Rows</th><th>Value</th><th>Outlier ratio</th><th>Dead centroids in column</th></tr></thead><tbody>__SUPERS__</tbody></table></div>
<p class="sub">Dead centroid: a codebook entry that no weight of its column uses. Outlier ratio: largest distance of a used centroid from the column mean, in column standard deviations. Index entropy below the index width is the most a lossless entropy coder could save on disk (it saves no GPU memory: the fused kernel reads fixed-width indices).</p>
</div></body></html>"""


def to_html(rep):
    m, s = rep["meta"], rep["summary"]
    title = "%s (%s)" % (os.path.basename(os.path.dirname(m["file"])) or os.path.basename(m["file"]), m["format"])
    meta = "%s &middot; %.2f GB &middot; %d layers &middot; hidden %d &middot; audited in %.0f s" % (
        m["file"], m["bytes"] / 1e9, m["n_layers"], m["hidden"], m["seconds"])
    if not s:
        kpis = '<div class="kpi"><div class="l">Format</div><div class="v">%s</div><div class="n">additive codebooks: see JSON</div></div>' % m["format"]
        return HTML.replace("__TITLE__", title).replace("__META__", meta).replace("__KPIS__", kpis).replace("__PROJ__", "").replace("__TOPS__", "").replace("__SUPERS__", "")
    k = [("Dead centroid slots", "%.2f%%" % s["dead_pct"], "%s of %s slots" % (format(s["dead_slots"], ","), format(s["slots"], ","))),
         ("Columns with a dead centroid", "%.2f%%" % (100.0 * s["cols_any_dead"] / s["scalar_columns"]), "%s columns, %s with 4 or more" % (format(s["cols_any_dead"], ","), format(s["cols_dead4"], ","))),
         ("Resolution wasted", format(s["cols_bulk_le_half"], ","), "columns where 99% of weights use half the codebook or less"),
         ("Index entropy", "%.2f bits" % s["index_entropy"], "of %.0f: lossless coding could save %.1f%% of index bytes" % (s["index_bits"], s["lossless_index_saving_pct"]))]
    kpis = "".join('<div class="kpi"><div class="l">%s</div><div class="v">%s</div><div class="n">%s</div></div>' % x for x in k)
    mx = max([v["dead_pct"] for v in s["by_proj"].values()] + [1e-9])
    proj = "".join('<tr><td>%s</td><td class="n"><span class="bar" style="width:%dpx"></span>%.3f%%</td><td class="n">%.2f%%</td><td class="n">%s</td><td class="n">%.3f</td></tr>' % (
        pn, int(120 * v["dead_pct"] / mx), v["dead_pct"], v["cols_any_dead_pct"], format(v["cols_bulk_le_half"], ","), v["mean_entropy"]) for pn, v in s["by_proj"].items())
    tops = "".join('<tr><td><code>%s</code></td><td class="n">%d</td><td class="n">%.1f</td><td class="n">%.4f</td><td class="n">%d</td><td class="n">%d</td><td class="n">%d</td></tr>' % (
        t["proj"], t["col"], t["ratio"], t["value"], t["uses"], t["dead"], t["bulk99"]) for t in rep["top_outlier_columns"])
    sup = "".join('<tr><td><code>%s</code></td><td class="n">%d</td><td class="n">%s</td><td class="n">%.4f</td><td class="n">%.1f</td><td class="n">%d</td></tr>' % (
        x["proj"], x["col"], ", ".join(str(r) for r in x["rows"][:4]), x["value"], x["ratio"], x["dead_in_column"]) for x in rep["super_weight_candidates"]) or '<tr><td colspan="6">none found</td></tr>'
    return (HTML.replace("__TITLE__", title).replace("__META__", meta).replace("__KPIS__", kpis)
            .replace("__PROJ__", proj).replace("__TOPS__", tops).replace("__SUPERS__", sup))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cbk")
    ap.add_argument("--json")
    ap.add_argument("--html")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    rep = audit(a.cbk, a.top)
    s = rep["summary"]
    if s:
        print("dead centroid slots: %.3f%% (%d of %d) | columns with a dead centroid: %d of %d | "
              "resolution wasted (99%% on half the codebook or less): %d columns | index entropy %.3f of %.0f bits"
              % (s["dead_pct"], s["dead_slots"], s["slots"], s["cols_any_dead"], s["scalar_columns"],
                 s["cols_bulk_le_half"], s["index_entropy"], s["index_bits"]))
    for x in rep["super_weight_candidates"]:
        print("super-weight candidate: %s col %d rows %s value %.4f ratio %.1f" % (x["proj"], x["col"], x["rows"][:4], x["value"], x["ratio"]))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if a.html:
        open(a.html, "w", encoding="utf-8").write(to_html(rep))
    if not rep["meta"]["trailing_bytes_ok"]:
        print("WARNING: bytes left after the LM head: the file does not match the expected layout", file=sys.stderr)


if __name__ == "__main__":
    main()
