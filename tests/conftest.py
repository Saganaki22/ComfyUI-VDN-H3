"""Pytest path setup for the VDN-H3 suite.

Pytest imports the custom node's package __init__ (which pulls real ComfyUI
modules through vdn_h3.hybrid) during test setup, so the ComfyUI checkout root
must be importable for the whole session -- not only when a particular test
file happens to be collected first. The per-file sys.path insertions remain
for standalone `python tests\\test_x.py` runs.
"""
import sys
from pathlib import Path

_COMFYUI_ROOT = Path(__file__).resolve().parents[3]  # the ComfyUI checkout
_PACKAGE = Path(__file__).resolve().parents[1]       # this package, any folder name
for _p in (str(_COMFYUI_ROOT), str(_PACKAGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
