"""Numerical verification of the VDN-H3 port against naive reference implementations.

Run with ComfyUI's venv python:
    <ComfyUI>/venv/Scripts/python.exe custom_nodes/ComfyUI-VDN/tests/test_vdn_math.py
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import torch

from vdn_h3.window import (window_bounds, full_coverage,
                        window_softmax_grouped)
from vdn_h3 import branch as B


def window_softmax_reference(query, key, value, video_start, video_end, num_frames,
                             tokens_per_frame, bounds, scale, anchor_frames):
    """One SDPA per frame + dense globals -- the official correctness oracle."""
    H = query.shape[1]
    out = torch.empty_like(query)
    idx = torch.cat([torch.arange(video_start), torch.arange(video_end, query.shape[0])])

    def sdpa(q, k, v):
        a = torch.nn.functional.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0), k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0), scale=scale)
        return a.squeeze(0).permute(1, 0, 2)

    out[idx] = sdpa(query[idx], key, value)
    for f in range(num_frames):
        lo = max(bounds[f][0], 0)
        hi = min(bounds[f][1], num_frames - 1)
        if anchor_frames in ("rows", "both") and f in (0, num_frames - 1):
            lo, hi = 0, num_frames - 1
        extra = [x for x in ((0, num_frames - 1)
                             if anchor_frames in ("columns", "both") else ())
                 if not lo <= x <= hi]
        a = video_start + f * tokens_per_frame
        b = a + tokens_per_frame
        rows = [idx] + [torch.arange(video_start + fr * tokens_per_frame,
                                     video_start + (fr + 1) * tokens_per_frame)
                        for fr in range(lo, hi + 1)] \
            + [torch.arange(video_start + fr * tokens_per_frame,
                            video_start + (fr + 1) * tokens_per_frame) for fr in extra]
        k_rows = torch.cat([key[r] for r in rows])
        v_rows = torch.cat([value[r] for r in rows])
        out[a:b] = sdpa(query[a:b], k_rows, v_rows)
    return out


def test_window_softmax():
    torch.manual_seed(0)
    F_, S, H, D = 23, 7, 3, 16
    video_rows = F_ * S
    text_rows = 11
    audio_rows = 5
    total = text_rows + video_rows + audio_rows
    video_start, video_end = text_rows, text_rows + video_rows
    q = torch.randn(total, H, D)
    k = torch.randn(total, H, D)
    v = torch.randn(total, H, D)
    scale = D ** -0.5

    for chunk, radius, anchors in ((0, 3, "none"), (5, 1, "both"), (5, 1, "none"),
                                   (5, 1, "rows"), (5, 1, "columns"), (4, 2, "both")):
        bounds = window_bounds(F_, radius, chunk)
        got = window_softmax_grouped(q, k, v, video_start, video_end, F_, S, bounds,
                                     scale, anchor_frames=anchors)
        want = window_softmax_reference(q, k, v, video_start, video_end, F_, S,
                                        bounds, scale, anchors)
        err = (got - want).abs().max().item()
        assert err < 2e-3, f"chunk={chunk} r={radius} anchors={anchors}: {err}"
        print(f"  window chunk={chunk} r={radius} anchors={anchors}: max err {err:.2e}")

    bounds = window_bounds(6, 1, 5)
    assert full_coverage(bounds, 6), "F<=2*chunk+... must be full cover"
    bounds = window_bounds(23, 1, 5)
    assert not full_coverage(bounds, 23)
    print("  full_coverage: ok")


def test_delta_scan():
    torch.manual_seed(1)
    F_, H, D = 17, 4, 32
    alpha = torch.rand(F_, H, D) * 0.5 + 0.5
    a_raw = torch.randn(F_, H, D, D) * 0.05
    a_raw = 0.5 * (a_raw + a_raw.transpose(-1, -2))
    b_raw = torch.randn(F_, H, D, D) * 0.3
    text = torch.randn(H, D, D) * 0.2

    for name, backend in (("vdn_solve", B.VdnDelta(None)),
                          ("sana_scaled", B.SanaDelta(S := 48))):
        with torch.no_grad():
            trans, inj = backend.factor_apply(alpha, a_raw, b_raw)
            # step-by-step recurrence (the official step_ref spelling)
            fwd_ref, rev_ref = [], [None] * F_
            st = text.clone()
            for f in range(F_):
                st = st @ trans[f] + inj[f]
                fwd_ref.append(st.clone())
            st = text.clone()
            for f in range(F_ - 1, -1, -1):
                st = st @ trans[f] + inj[f]
                rev_ref[f] = st.clone()
            fwd_ref = torch.stack(fwd_ref)
            rev_ref = torch.stack(rev_ref)
            fwd, rev = B.run_scans(backend, alpha, a_raw, b_raw, text_state=text)
            assert (fwd - fwd_ref).abs().max() < 1e-4, name
            assert (rev - rev_ref).abs().max() < 1e-4, name
        print(f"  scan {name}: ok")

    # vdn_solve transition/injection against the closed form (I+A)^-1
    a32 = a_raw.float()
    eye = torch.eye(D).expand_as(a32)
    inv = torch.linalg.inv(a32 + eye)
    trans_want = alpha.unsqueeze(-1) * inv
    inj_want = b_raw.float() @ inv
    backend = B.VdnDelta(None)
    trans, inj = backend.factor_apply(alpha, a_raw, b_raw)
    assert (trans.float() - trans_want).abs().max() < 1e-4
    assert (inj - inj_want).abs().max() < 1e-4
    print("  vdn_solve closed form: ok")


def test_gather():
    torch.manual_seed(2)
    F_, H, D = 13, 2, 8
    prefix = torch.randn(F_, H, D, D)
    suffix = torch.randn(F_, H, D, D)
    alpha = torch.rand(F_, H, D) * 0.9 + 0.1
    bounds = window_bounds(F_, 1, 5)
    text = torch.randn(H, D, D)

    got = B.gather_linear_state(prefix, suffix, alpha, bounds, bridge="alpha",
                                text_state=text)
    # naive per-frame reconstruction; alpha is per KEY channel, so the decay terms
    # broadcast along the last axis ([H, 1, D] against [H, dv, dk])
    log_prefix = torch.cat([torch.zeros(1, H, D), alpha.clamp_min(1e-12).log().cumsum(0)])
    for t in range(F_):
        lo, hi = max(bounds[t][0], 0), min(bounds[t][1], F_ - 1)
        if lo - 1 >= 0:
            # advance the state from after frame lo-1 through frame t's own
            # transition: prod alpha[lo..t] (official includes the query frame)
            want_l = prefix[lo - 1] * torch.exp(log_prefix[t + 1] - log_prefix[lo]).unsqueeze(1)
        else:
            # scan start (text) sits at virtual index -1; reaching frame t decays
            # through frames 0..t, i.e. the inclusive prefix product
            want_l = text * torch.exp(log_prefix[t + 1] - log_prefix[0]).unsqueeze(1)
        if hi + 1 <= F_ - 1:
            want_r = suffix[hi + 1] * torch.exp(log_prefix[hi + 1] - log_prefix[t]).unsqueeze(1)
        else:
            want_r = text * torch.exp(log_prefix[F_] - log_prefix[t]).unsqueeze(1)
        want = (want_l + want_r).to(got.dtype)
        err = (got[t] - want).abs().max().item()
        assert err < 1e-4, f"frame {t}: {err}"
    print("  gather vs naive bridge: ok")


def test_adapter_fold():
    torch.manual_seed(3)
    from vdn_h3.adapters import convert_adapter

    r, hidden, inner = 8, 32, 48
    sd = {}
    for proj in ("to_q", "to_k", "to_v"):
        sd[f"transformer_blocks.4.attn.orig.{proj}.lora_A.default.weight"] = \
            torch.randn(r, hidden)
        sd[f"transformer_blocks.4.attn.orig.{proj}.lora_B.default.weight"] = \
            torch.randn(inner, r)
    sd["transformer_blocks.4.ff.net.0.proj.lora_A.default.weight"] = torch.randn(r, hidden)
    b = torch.randn(2 * inner, r)
    sd["transformer_blocks.4.ff.net.0.proj.lora_B.default.weight"] = b
    cfg = {"config": {"rank": r, "alpha": r}}

    conv = convert_adapter(sd, cfg)
    a_f, b_f, scale = conv["blocks.4.attn.qkv_proj"]
    delta = b_f @ a_f
    want = torch.cat([
        sd[f"transformer_blocks.4.attn.orig.{p}.lora_B.default.weight"]
        @ sd[f"transformer_blocks.4.attn.orig.{p}.lora_A.default.weight"]
        for p in ("to_q", "to_k", "to_v")], dim=0)
    assert torch.allclose(delta, want, atol=1e-5), "fused qkv delta mismatch"
    assert scale == 1.0

    a_f, b_f, _ = conv["blocks.4.mlp.fc1"]
    assert torch.allclose(b_f, torch.cat([b[inner:], b[:inner]], dim=0)), \
        "swiglu halves must swap"
    print("  adapter fold (qkv block-diag, swiglu swap): ok")


if __name__ == "__main__":
    print("VDN-H3 port numerical tests")
    test_window_softmax()
    test_delta_scan()
    test_gather()
    test_adapter_fold()
    print("ALL PASS")
