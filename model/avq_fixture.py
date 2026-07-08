#!/usr/bin/env python3
"""Generate a tiny CBKA fixture + its expected dequantized weight, for the Rust CPU
decoder test (runtime/src/bin/avq_check.rs). Uses cbka_format.write_cbka_record (the SAME
byte layout the exporter emits) with a small standalone numpy encoder, so the test proves
the Rust decoder reads the exporter's format and reconstructs W identically.

  python avq_fixture.py --out /tmp/scratch --M 2
Writes <out>/avq_m<M>.cbka (the record) and <out>/avq_m<M>.ref (rows*cols little-endian f32,
row-major [rows][cols]). No em dashes or en dashes anywhere.
"""
import argparse
import os
import numpy as np

from cbka_format import AVQ_D, AVQ_K, avq_reconstruct, write_cbka_record


def encode(W, M, seed=0):
    # Small residual additive-VQ encode: for each codebook, sample K group-vectors as the
    # codebook, assign each group its nearest entry, subtract, repeat. Codebooks rounded to
    # f16 before assignment so codes match the on-disk (f16) codebook.
    rng = np.random.RandomState(seed)
    OC, IC = W.shape
    assert IC % AVQ_D == 0 and OC % 4 == 0
    ng = IC // AVQ_D
    s = np.abs(W).max(1, keepdims=True).astype(np.float32)
    s[s < 1e-8] = 1e-8
    Pg = (W / s).reshape(-1, AVQ_D).astype(np.float32)   # (OC*ng, D), row = o*ng + g
    N = Pg.shape[0]
    C = np.zeros((M, AVQ_K, AVQ_D), np.float32)
    codes = np.zeros((N, M), np.int64)
    R = Pg.copy()
    for m in range(M):
        idx = rng.randint(0, N, size=AVQ_K)
        Cm = R[idx].astype(np.float16).astype(np.float32)
        dist = ((R[:, None, :] - Cm[None, :, :]) ** 2).sum(-1)
        a = dist.argmin(1)
        codes[:, m] = a
        R = R - Cm[a]
        C[m] = Cm
    codes_ogm = codes.reshape(OC, ng, M).astype(np.uint16)
    return codes_ogm, C, s.reshape(OC).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=2, choices=[2, 3])
    ap.add_argument("--rows", type=int, default=12)   # OC, must be %4
    ap.add_argument("--cols", type=int, default=40)   # IC, must be %8
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    W = (rng.randn(args.rows, args.cols) * 0.05).astype(np.float32)
    codes_ogm, C, scale = encode(W, args.M, seed=args.seed)

    cbka_path = os.path.join(args.out, f"avq_m{args.M}.cbka")
    ref_path = os.path.join(args.out, f"avq_m{args.M}.ref")
    with open(cbka_path, "wb") as f:
        write_cbka_record(f, codes_ogm, C, scale, args.M)
    W_ref = avq_reconstruct(codes_ogm, C, scale).astype("<f4")   # [rows][cols] row-major f32
    W_ref.tofile(ref_path)
    print(f"wrote {cbka_path} ({os.path.getsize(cbka_path)} B) and {ref_path} "
          f"(rows={args.rows} cols={args.cols} M={args.M})")


if __name__ == "__main__":
    main()
