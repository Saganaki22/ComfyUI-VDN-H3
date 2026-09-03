"""Regression test for the RecursionError after model reload (observed on the
second generation when two adapter sets stack bypass hooks on the same modules).

Legacy behavior: injection sets are ejected in list order, which restores a
stale hook as module.forward; the next load then captures that hook as its own
"original" and the forward self-recurses. _install_injection's LIFO eject must
keep load/unload cycles stable. Run from anywhere with the ComfyUI venv python.
"""
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "custom_nodes" / "ComfyUI-VDN"))

import comfy.model_management
import comfy.weight_adapter
from comfy.weight_adapter.bypass import BypassForwardHook
from vdn_h3.apply import _FrugalLoRA, _install_injection

comfy.model_management.get_torch_device = lambda: torch.device("cpu")


def _adapter():
    up = torch.randn(8, 4)
    down = torch.randn(4, 8) * 0.1
    return _FrugalLoRA(set(), (up, down, torch.tensor(4.0)))  # alpha/rank = 1


class _Patcher:
    def __init__(self):
        self.injections = {}

    def set_injections(self, key, value):
        self.injections[key] = value


def _fresh_module():
    torch.manual_seed(0)
    mod = nn.Linear(8, 8)
    return mod, mod.forward


def _legacy_cycle_breaks():
    """Documents the bug: forward-order eject leaves a stale hook in place, so
    re-injecting makes the hook its own original forward."""
    mod, true_fwd = _fresh_module()
    hook_d = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    hook_t = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    hook_d.inject()
    hook_t.inject()                     # hook_t.original = hook_d._bypass_forward
    hook_d.eject()                      # mod.forward = true forward
    hook_t.eject()                      # mod.forward = hook_d._bypass_forward (!)
    hook_d.inject()                     # hook_d.original = its own bypass forward
    broken = hook_d.original_forward == hook_d._bypass_forward
    mod.forward = true_fwd              # restore for cleanliness
    return broken


def test_lifo_cycles():
    mod, true_fwd = _fresh_module()
    hook_d = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    hook_t = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    patcher = _Patcher()
    _install_injection(patcher, [hook_d, hook_t])
    injection = patcher.injections["vdn_lora"][0]

    x = torch.randn(3, 8)
    base = true_fwd(x)
    want = base + _adapter_lora(hook_d, x) + _adapter_lora(hook_t, x)

    for cycle in range(3):
        injection.inject(patcher)
        assert mod.forward == hook_t._bypass_forward, f"cycle {cycle}: wrong outermost hook"
        assert hook_t.original_forward == hook_d._bypass_forward, f"cycle {cycle}: T should wrap D"
        assert hook_d.original_forward == true_fwd, f"cycle {cycle}: D should wrap the true forward"
        got = mod(x)
        assert torch.allclose(got, want, atol=1e-5), f"cycle {cycle}: wrong value"
        injection.eject(patcher)
        assert mod.forward == true_fwd, f"cycle {cycle}: true forward not restored"
        assert hook_d.original_forward is None and hook_t.original_forward is None
    print("lifo cycles: PASS (3x inject/eject, values correct, true forward restored)")


def _adapter_lora(hook, x):
    up, down, _ = hook.adapter.weights
    return torch.nn.functional.linear(
        torch.nn.functional.linear(x, down), up)


if __name__ == "__main__":
    assert _legacy_cycle_breaks(), "legacy cycle did not reproduce the bug"
    print("legacy cycle: reproduces the self-hook bug as expected")
    test_lifo_cycles()
    print("ALL PASS")





