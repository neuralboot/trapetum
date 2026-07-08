#!/usr/bin/env python3
"""Shared on-disk layout for the CBKA additive-codebook expert record (2/3-bit AQLM-style).

One place defines the byte format so the streaming exporter (export_deepseek_stream.py) and
the validation fixture (avq_fixture.py) can never drift, exactly like export_safetensors.qwrite
is the single source of truth for the 4-bit scalar packing. The Rust decoder lives in
runtime/src/lib.rs (read_cbka / AvqLinear) and kernels/avq_gemv3.cu; keep the three in sync.

Record (one weight matrix; a routed expert = 3 records: gate, up, down):
    magic  "CBKA"                 (4 bytes)
    M      i32                    (2 or 3 additive codebooks)
    D      i32                    (group size, 8)
    K      i32                    (entries per codebook, 256)
    rows   i32                    (OC, output channels; rows % 4 == 0)
    cols   i32                    (IC, input channels; cols % D == 0)
    codebooks   M*K*D    f16      layout [M][K][D], flat (m*K + k)*D + e
    scales      rows     f16      per output channel
    indices     M*ng*rows u8      ng = cols/D, layout [M][ng][rows], flat (m*ng + g)*rows + o

Reconstruction:  W[o, g*D + e] = scale[o] * sum_m codebooks[m][code_m[o,g]][e].
No em dashes or en dashes anywhere.
"""
import struct
import numpy as np

AVQ_K = 256
AVQ_D = 8


def write_cbka_record(f, codes_ogm, C, scale, M):
    """Append one CBKA record to an open binary file.

    codes_ogm : integer array [OC, ng, M], the M code indices per (output row, group)
    C         : float array [M, K, D] codebooks
    scale     : float array [OC] per-output-channel scale
    """
    codes_ogm = np.asarray(codes_ogm)
    OC, ng, Mc = codes_ogm.shape
    assert Mc == M, f"codes last dim {Mc} != M {M}"
    K, D = C.shape[1], C.shape[2]
    assert C.shape[0] == M and K == AVQ_K and D == AVQ_D, f"bad codebook shape {C.shape}"
    assert scale.shape[0] == OC, f"scale len {scale.shape[0]} != OC {OC}"
    assert OC % 4 == 0, f"rows (OC={OC}) must be a multiple of 4"
    f.write(b"CBKA")
    f.write(struct.pack("<5i", M, D, K, OC, ng * D))       # M, D, K, rows, cols
    f.write(np.ascontiguousarray(C, np.float32).astype(np.float16).tobytes())
    f.write(np.ascontiguousarray(scale, np.float32).astype(np.float16).tobytes())
    # indices to [M][ng][OC] with o fastest (C order), matching flat (m*ng + g)*OC + o
    idx = np.ascontiguousarray(np.transpose(codes_ogm, (2, 1, 0)).astype(np.uint8))
    f.write(idx.tobytes())


def avq_reconstruct(codes_ogm, C, scale):
    """Dequantize a CBKA record to W [OC, cols] float32, using f16-rounded codebooks and
    scale so the result matches what the Rust/CUDA decoder reads back off disk bit-for-bit
    (up to the f32 accumulation order, which both sides do identically)."""
    codes_ogm = np.asarray(codes_ogm)
    OC, ng, M = codes_ogm.shape
    D = C.shape[2]
    C16 = C.astype(np.float16).astype(np.float32)
    s16 = np.asarray(scale, np.float32).astype(np.float16).astype(np.float32)
    recon = np.zeros((OC, ng, D), np.float32)
    for m in range(M):
        recon += C16[m][codes_ogm[:, :, m]]
    recon *= s16[:, None, None]
    return recon.reshape(OC, ng * D)
