"""ComfyUI-VDN: Video Delta Net (VDN-H3) hybrid attention for MiniMax-H3.

Reference implementation: github.com/OpenVDN/vdn-minimax-h3 (Apache-2.0).
This package ports the released Video Delta Attention onto ComfyUI's native
MiniMax-H3 model as model patches; no ComfyUI core files are modified.
"""

import os, sys
_PKG = os.path.dirname(__file__)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from vdn_h3.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
