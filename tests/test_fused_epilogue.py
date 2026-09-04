"""The compile-fused branch epilogue must equal the eager one (same math, one
inductor kernel when compilation succeeds; falls back to eager on failure)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vdn_h3.branch import linear_epilogue
from vdn_h3 import branch as B


def test_epilogue_parity():
    torch.manual_seed(0)
    frames, heads, per_frame, dim = 4, 3, 5, 8
    readout = torch.randn(frames, heads, per_frame, dim).to(torch.bfloat16)
    weight = torch.randn(dim).to(torch.bfloat16)
    gate = torch.rand(frames * per_frame, heads * dim).to(torch.bfloat16)

    eager = linear_epilogue(readout, weight, gate, 1e-6, fuse=False)
    fused = linear_epilogue(readout, weight, gate, 1e-6, fuse=True)
    assert torch.equal(eager.float(), fused.float()), "fused epilogue differs"
    assert eager.shape == (frames * per_frame, heads * dim)
    print("epilogue fused/eager: PASS (identical)")


def test_q_fhsd_store():
    """fast_kernels stores q frame-major straight out of the activation; layout and
    values must match the eager view/permute it replaces."""
    torch.manual_seed(1)
    frames, per_frame, heads, dim = 4, 5, 3, 8
    x = torch.randn(frames * per_frame, heads, dim).to(torch.bfloat16)
    want = B._activate(x, True).view(frames, per_frame, heads, dim) \
        .permute(0, 2, 1, 3).contiguous()
    got = B._run_compiled(("test_act_fhsd", True), B._activate_fhsd_body,
                          x, True, frames, per_frame)
    assert got.shape == (frames, heads, per_frame, dim) and got.is_contiguous()
    assert (got.float() - want.float()).abs().max() < 1e-3
    print("q fhsd store fused/eager: PASS")


def test_readout_parity():
    """End to end: LinearBranch._readout under fast_kernels (fused epilogue + fused
    gather + frame-major q store) must match the default eager path."""
    torch.manual_seed(2)
    frames, per_frame, heads, dim, hidden = 6, 4, 2, 4, 16
    channels = heads * dim
    w = {
        "beta_proj.weight": torch.randn(heads, hidden),
        "alpha.down.weight": torch.randn(dim, hidden),
        "alpha.up.weight": torch.randn(heads * dim, dim),
        "alpha.dt_bias": torch.randn(heads * dim),
        "alpha.A_log": torch.randn(heads),
        "output_gate.down.weight": torch.randn(dim, hidden),
        "output_gate.up.weight": torch.randn(heads * dim, dim),
        "output_gate.up.bias": torch.randn(heads * dim),
        "norm.weight": torch.randn(dim),
        "short_conv.k_sp.weight": torch.randn(channels, 1, 5, 5) * 0.2,
        "short_conv.k_tm.weight": torch.randn(channels, 1, 5) * 0.2,
        "short_conv.v_sp.weight": torch.randn(channels, 1, 5, 5) * 0.2,
        "short_conv.v_tm.weight": torch.randn(channels, 1, 5) * 0.2,
    }
    rows = frames * per_frame
    xv = torch.randn(rows, hidden)
    q_raw = torch.randn(rows, heads, dim)
    k_raw = torch.randn(rows, heads, dim)
    v_raw = torch.randn(rows, heads, dim)
    bounds = B.window_bounds(frames, 1, 2) if hasattr(B, "window_bounds") else None
    from vdn_h3.window import window_bounds
    bounds = window_bounds(frames, 1, 2)

    branch = B.LinearBranch(w, heads, dim, delta_rule="vdn_solve", bridge="alpha",
                            a_fp32=True, short_conv=("k", "v"),
                            enable_text_state=False)
    eager = branch.readout(w, xv, q_raw, k_raw, v_raw, frames, per_frame, bounds,
                           frame_size=(2, 2))
    branch.fuse_epilogue = True
    fused = branch.readout(w, xv, q_raw, k_raw, v_raw, frames, per_frame, bounds,
                           frame_size=(2, 2))
    err = (fused - eager).abs().max().item()
    assert err < 1e-4, f"fast_kernels readout differs: {err}"
    print(f"readout fast_kernels/eager: PASS (max err {err:.2e})")


if __name__ == "__main__":
    test_epilogue_parity()
    test_q_fhsd_store()
    test_readout_parity()
    print("ALL PASS")
