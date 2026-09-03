"""CPU-only test for window.py's optimized_attention dispatch routing.

Stubs comfy.ldm.modules.attention so no real ComfyUI import (and no CUDA
context) is needed. Verifies that with transformer_options the window groups
route through the dispatched attention function with the [B, S, H*D] contract,
and that the dispatched result (scale = head_dim ** -0.5, the contract every
comfy attention function implements) equals raw SDPA with the same scale.
Run from anywhere with the ComfyUI venv python.
"""
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

calls = []


def _stub_optimized_attention(q, k, v, heads, mask=None, **kwargs):
    calls.append({"heads": heads, "transformer_options": kwargs.get("transformer_options"),
                  "q_shape": tuple(q.shape), "k_shape": tuple(k.shape)})
    assert q.dim() == 3 and q.shape[0] == 1, "expected [1, S, H*D] layout"
    assert mask is None, "window groups are dense per group; no mask expected"
    rows, inner = q.shape[1], q.shape[2]
    keys = k.shape[1]
    head_dim = inner // heads
    scale = head_dim ** -0.5
    q4 = q.view(1, rows, heads, head_dim).transpose(1, 2).float()
    k4 = k.view(1, keys, heads, head_dim).transpose(1, 2).float()
    v4 = v.view(1, keys, heads, head_dim).transpose(1, 2).float()
    out = F.scaled_dot_product_attention(q4, k4, v4, scale=scale)
    return out.transpose(1, 2).reshape(1, rows, inner).to(q.dtype)


def _install_stub():
    for name in ("comfy", "comfy.ldm", "comfy.ldm.modules"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    stub = types.ModuleType("comfy.ldm.modules.attention")
    stub.optimized_attention = _stub_optimized_attention
    sys.modules["comfy.ldm.modules.attention"] = stub


def test_dispatch():
    from vdn_h3.window import window_bounds, window_softmax_grouped

    torch.manual_seed(0)
    video_start, tokens, frames, heads, dim = 5, 8, 12, 2, 16
    seq = video_start + frames * tokens + 3
    q = torch.randn(seq, heads, dim)
    k = torch.randn(seq, heads, dim)
    v = torch.randn(seq, heads, dim)
    to = {"anything": True}

    got = window_softmax_grouped(q, k, v, video_start, seq - 3, frames, tokens,
                                 window_bounds(frames, 1, 5), dim ** -0.5,
                                 anchor_frames="both", transformer_options=to)
    assert len(calls) >= 3, "expected several grouped dispatch calls"
    for c in calls:
        assert c["heads"] == heads
        assert c["transformer_options"] is to, "transformer_options not passed through"
        assert c["q_shape"][2] == heads * dim and c["q_shape"][0] == 1

    # exactness: routed result matches a direct SDPA reference per call scale
    want = window_softmax_grouped(q, k, v, video_start, seq - 3, frames, tokens,
                                  window_bounds(frames, 1, 5), dim ** -0.5,
                                  anchor_frames="both")
    assert torch.allclose(got, want, atol=1e-5), "dispatch changed the math"
    print(f"dispatch routing: PASS ({len(calls)} grouped calls, math identical)")


if __name__ == "__main__":
    _install_stub()
    test_dispatch()
    print("ALL PASS")
