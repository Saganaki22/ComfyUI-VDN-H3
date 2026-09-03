"""CPU-only test for the FlexAttention mask logic (window.py flex path).

The flex kernel itself needs CUDA + triton; this test validates the PARTITION it
computes: a dense masked SDPA built from _build_window_tables must equal
window_softmax_grouped (already verified against the official reference).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vdn_h3.window import (_build_window_tables, window_bounds,
                           window_softmax_grouped)


def _dense_mask(seq, vs, ve, nf, tpf, lo, hi, anchor_frames, device):
    q = torch.arange(seq, device=device).unsqueeze(1)
    kv = torch.arange(seq, device=device).unsqueeze(0)
    gq = (q < vs) | (q >= ve)
    gk = (kv < vs) | (kv >= ve)
    qf = torch.div(q - vs, tpf, rounding_mode="trunc").clamp(0, nf - 1)
    kf = torch.div(kv - vs, tpf, rounding_mode="trunc").clamp(0, nf - 1)
    allowed = gq | gk | ((kf >= lo[q]) & (kf <= hi[q]))
    if anchor_frames in ("columns", "both"):
        allowed = allowed | (kf == 0) | (kf == nf - 1)
    if anchor_frames in ("rows", "both"):
        allowed = allowed | (qf == 0) | (qf == nf - 1)
    return allowed


def test_partition(anchor_frames):
    torch.manual_seed(0)
    vs, tpf, nf, heads, dim = 5, 8, 12, 3, 16
    seq = vs + nf * tpf + 3
    q = torch.randn(seq, heads, dim)
    k = torch.randn(seq, heads, dim)
    v = torch.randn(seq, heads, dim)
    bounds = window_bounds(nf, 1, 5)
    lo, hi = _build_window_tables(seq, vs, seq - 3, nf, tpf, bounds, q.device)
    mask = _dense_mask(seq, vs, seq - 3, nf, tpf, lo, hi, anchor_frames, q.device)

    want = F.scaled_dot_product_attention(
        q.permute(1, 0, 2).unsqueeze(0), k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        attn_mask=mask.unsqueeze(0).unsqueeze(0), scale=dim ** -0.5
    ).squeeze(0).permute(1, 0, 2)
    got = window_softmax_grouped(q, k, v, vs, seq - 3, nf, tpf, bounds,
                                 dim ** -0.5, anchor_frames=anchor_frames)
    assert torch.allclose(got, want, atol=1e-5), anchor_frames
    print(f"flex-mask partition [{anchor_frames}]: PASS")


if __name__ == "__main__":
    for mode in ("none", "columns", "rows", "both"):
        test_partition(mode)
    print("ALL PASS")
