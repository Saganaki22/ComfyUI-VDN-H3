"""The compile-fused branch epilogue must equal the eager one (same math, one
inductor kernel when compilation succeeds; falls back to eager on failure)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vdn_h3.branch import linear_epilogue


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


if __name__ == "__main__":
    test_epilogue_parity()
    print("ALL PASS")
