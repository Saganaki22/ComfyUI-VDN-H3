"""VDN-H3 checkpoint discovery, loading and validation.

A VDN checkpoint is the official exploded release directory (model_spec.json +
linear_branch/ + adapters/) from huggingface.co/OpenVDN/vdn-minimax-h3, dropped under
ComfyUI's models/vdn/ folder. Nothing is converted on disk; tensors are re-keyed in
memory onto ComfyUI's MiniMax-H3 module paths.
"""
import functools
import json
import logging
import os

import folder_paths
import torch
from safetensors.torch import load_file

_log = logging.getLogger("comfy.vdn")

SUPPORTED_DELTA_RULES = ("vdn_solve", "sana_scaled", "vdn_scaled")
SUPPORTED_ANCHORS = ("none", "columns", "rows", "both")
SHORT_CONV_TARGETS = ("q", "k", "v")

BRANCH_FILE = "model.safetensors"
BRANCH_FILE_INT8 = "model_int8_convrot_comfyui.safetensors"

try:
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
    _KITCHEN_OK = True
except ImportError:  # pragma: no cover - kitchen ships with comfy
    QuantizedTensor = TensorWiseINT8Layout = None
    _KITCHEN_OK = False


def _branch_file(path, prefer_int8=False):
    """The stage's branch file: plain, or the pre-quantized int8_convrot_comfyui
    one. Plain wins when both exist unless prefer_int8 (auto mode under VRAM
    pressure; see auto_branch_policy)."""
    plain = os.path.join(path, "linear_branch", BRANCH_FILE)
    quant = os.path.join(path, "linear_branch", BRANCH_FILE_INT8)
    if prefer_int8:
        if os.path.isfile(quant):
            return quant
        return plain
    if os.path.isfile(plain):
        return plain
    if os.path.isfile(quant):
        return quant
    return plain


def auto_branch_policy(path, free_bytes):
    """branch_weights="auto": cache_gpu when the free VRAM (queried after the base
    load) exceeds 1.5x the stage size plus 4 GiB of headroom, else stream. Under
    memory pressure the int8_convrot branch file is preferred over the plain bf16
    one when both exist (2.2 GB vs 4.3 GB, identical output). Returns
    (mode, prefer_int8); the choice is logged."""
    plain = os.path.join(path, "linear_branch", BRANCH_FILE)
    quant = os.path.join(path, "linear_branch", BRANCH_FILE_INT8)
    have_plain, have_quant = os.path.isfile(plain), os.path.isfile(quant)
    size_plain = os.path.getsize(plain) if have_plain else 0
    size_quant = os.path.getsize(quant) if have_quant else 0
    gib = 1 << 30

    def fits(size):
        return free_bytes > 1.5 * size + 4 * gib

    prefer_int8 = False
    if have_plain and fits(size_plain):
        mode = "cache_gpu"                        # no pressure: plain wins, resident
    elif have_quant:
        prefer_int8 = have_plain                  # pressure (or int8-only stage)
        mode = "cache_gpu" if fits(size_quant) else "stream"
    else:
        mode = "stream"
    chosen = quant if prefer_int8 else plain
    _log.info("[vdn] branch_weights=auto: %s, %s (%.1f GiB VRAM free, stage "
              "%.2f GiB%s)", os.path.basename(chosen), mode, free_bytes / gib,
              (size_quant if prefer_int8 else size_plain) / gib,
              ", int8 preferred under memory pressure"
              if prefer_int8 and have_plain else "")
    return mode, prefer_int8


def auto_retain_policy(path, prefer_int8, free_bytes):
    """retain_buffers="auto": keep the retained scratch/bank buffers when VRAM
    has headroom (free >= stage + 10 GiB) -- they buy allocator-churn-free
    steps and the one-block prefetch; go transient (the v1.3.1 per-call
    allocation pattern, prefetch skipped) under pressure, where the ~0.5-1 GiB
    of retained buffers can cost more than the churn they save. Logged."""
    branch_path = _branch_file(path, prefer_int8=prefer_int8)
    stage = os.path.getsize(branch_path) if os.path.isfile(branch_path) else 0
    gib = 1 << 30
    retain = free_bytes >= stage + 10 * gib
    _log.info("[vdn] retain_buffers=auto: %s (%.1f GiB VRAM free; stage "
              "%.2f GiB + 10 GiB headroom)",
              "retained" if retain else "transient", free_bytes / gib,
              stage / gib)
    return retain


def _read_header(path):
    """The safetensors JSON header: {key: {"dtype": ..., "shape": [...]}}."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16,
                      "F32": torch.float32, "I8": torch.int8,
                      "U8": torch.uint8}

# One mmap handle per (file, device). The OS page cache manages residency, so
# branch weights are read from disk on demand and never accumulated as owned
# CPU tensors (the same disk-backed philosophy as comfy's --fast-disk).
_HANDLES = {}


def _handle(path, device):
    h = _HANDLES.get((path, device))
    if h is None:
        from safetensors import safe_open
        h = safe_open(path, framework="pt", device=device)
        _HANDLES[(path, device)] = h
    return h


class LazyBranchTensor:
    """A branch weight kept on disk: reads straight from the safetensors mmap to
    the target device on resolve(). Per-block stream reads replace the old
    full-branch load_file() into committed CPU RAM."""

    __slots__ = ("_path", "_key", "_scale_key", "_conf", "shape", "dtype")

    def __init__(self, path, key, shape, dtype, scale_key=None, conf=None):
        self._path = path
        self._key = key
        self._scale_key = scale_key
        self._conf = conf
        self.shape = shape
        self.dtype = dtype

    def resolve(self, device, dtype=None):
        h = _handle(self._path, str(device))
        if self._conf is None:
            t = h.get_tensor(self._key)
            return t if dtype is None or t.dtype == dtype else t.to(dtype)
        qdata = h.get_tensor(self._key)
        scale = h.get_tensor(self._scale_key)
        return QuantizedTensor(
            qdata, "TensorWiseINT8Layout",
            TensorWiseINT8Layout.Params(
                scale=scale, orig_dtype=dtype or torch.bfloat16,
                orig_shape=tuple(self.shape), is_weight=True,
                convrot=bool(self._conf.get("convrot", False)),
                convrot_groupsize=int(self._conf.get("convrot_groupsize", 256))))


def _lazy_branch_sd(path):
    """Descriptors for every tensor in a branch file, quantization included."""
    header = _read_header(path)
    conf_keys = [k for k in header if k.endswith(".comfy_quant")]
    confs = {}
    if conf_keys:
        h = _handle(path, "cpu")
        for k in conf_keys:
            layer = k[: -len(".comfy_quant")]
            confs[layer] = json.loads(
                bytes(h.get_tensor(k).tolist()).decode("utf-8"))
    out = {}
    for key, meta in header.items():
        if key == "__metadata__" or key.endswith(".comfy_quant"):
            continue
        if key.endswith(".weight_scale") \
                and key[: -len(".weight_scale")] in confs:
            continue
        layer = key[: -len(".weight")] if key.endswith(".weight") else None
        conf = confs.get(layer) if layer else None
        scale_key = key + "_scale" if conf else None
        if conf and scale_key not in header:
            conf = None
            scale_key = None
        out[key] = LazyBranchTensor(
            path, key, torch.Size(meta["shape"]),
            SAFETENSORS_DTYPES.get(meta["dtype"]), scale_key, conf)
    return out


def register_folder():
    for base in {os.path.dirname(p) for p in folder_paths.get_folder_paths("loras")}:
        folder_paths.add_model_folder_path("vdn", os.path.join(base, "vdn"))


def vdn_folders():
    if "vdn" not in folder_paths.folder_names_and_paths:
        register_folder()
    return folder_paths.get_folder_paths("vdn")


def list_vdn_checkpoints():
    """Relative names of directories holding a linear_branch branch file (plain
    bf16 or pre-quantized int8_convrot_comfyui)."""
    found = []
    for root in vdn_folders():
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _files in os.walk(root):
            if os.path.isfile(_branch_file(dirpath)):
                rel = os.path.relpath(dirpath, root)
                found.append(rel.replace("\\", "/"))
                dirnames[:] = []
    return sorted(found)


def resolve_vdn_checkpoint(name):
    for root in vdn_folders():
        path = os.path.join(root, *name.split("/"))
        if os.path.isfile(_branch_file(path)):
            return path
    raise FileNotFoundError(
        f"VDN checkpoint {name!r} not found under {vdn_folders()}. Download the "
        "release from huggingface.co/OpenVDN/vdn-minimax-h3 (hf download "
        "OpenVDN/vdn-minimax-h3 --local-dir <ComfyUI>/models/vdn) and keep the "
        "stage-... directory layout intact.")


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def transform_config(spec):
    """Validate the ModelSpec's hybrid_attention transform; return its config."""
    transforms = [t for t in spec.get("transforms", []) if t.get("type") == "hybrid_attention"]
    if len(transforms) != 1:
        raise ValueError("model_spec.json must carry exactly one hybrid_attention "
                         f"transform, got {len(transforms)}")
    t = transforms[0]
    if t.get("version") != 2:
        raise ValueError(f"hybrid_attention transform version {t.get('version')}; "
                         "this node reads version 2 (the released VDN-H3 format)")
    cfg = t["config"]
    lin, soft = cfg["linear_attention"], cfg["softmax_attention"]
    if lin["delta_rule"] not in SUPPORTED_DELTA_RULES:
        raise ValueError(f"delta_rule {lin['delta_rule']!r} not supported")
    if lin["bridge"] not in ("alpha", "none"):
        raise ValueError(f"bridge {lin['bridge']!r} not supported")
    targets = lin.get("short_conv", {}).get("targets", [])
    if any(x not in SHORT_CONV_TARGETS for x in targets):
        raise ValueError(f"short_conv targets {targets} not supported")
    if cfg.get("anchor_frames") not in SUPPORTED_ANCHORS:
        raise ValueError(f"anchor_frames {cfg.get('anchor_frames')!r} not supported")
    if not isinstance(soft["radius"], int) or not isinstance(lin["linear_head_dim"], int):
        raise ValueError("radius and linear_head_dim must be resolved ints")
    return dict(
        enable_softmax_gate=bool(cfg.get("enable_softmax_gate", True)),
        anchor_frames=cfg["anchor_frames"],
        radius=int(soft["radius"]),
        chunk=int(soft.get("chunk", 0)),
        delta_rule=lin["delta_rule"],
        bridge=lin["bridge"],
        a_fp32=bool(lin.get("a_fp32", True)),
        linear_head_dim=int(lin["linear_head_dim"]),
        short_conv=tuple(targets),
        enable_text_state=bool(lin.get("enable_text_state", False)),
    )


_CACHE = {}


def load_vdn_checkpoint(path, prefer_int8=False):
    """Read model_spec.json + linear_branch + adapters. Returns
    (cfg, branch_weights_by_block, {adapter_name: (loader, adapter_spec)}). Cached
    by (path, branch file, mtime) so re-running the node doesn't re-read the
    metadata. Branch tensors stay disk-backed (LazyBranchTensor); adapters are
    loaded from their safetensors at APPLY time (loader()), not cached as loaded
    dicts for the process lifetime."""
    branch_path = _branch_file(path, prefer_int8=prefer_int8)
    stamp = (path, branch_path, os.path.getmtime(branch_path))
    hit = _CACHE.get(stamp)
    if hit is not None:
        return hit

    spec_path = os.path.join(path, "model_spec.json")
    if not os.path.isfile(spec_path):
        raise FileNotFoundError(
            f"{path} has linear_branch/ but no model_spec.json; keep the official "
            "release directory layout intact")
    spec = _read_json(spec_path)
    cfg = transform_config(spec)

    branch_sd = _lazy_branch_sd(branch_path)
    num_blocks = 0
    for key in branch_sd:
        if ".attn.to_out_linear.weight" in key:
            num_blocks = max(num_blocks, int(key.split(".")[1]) + 1)
    branches = []
    for i in range(num_blocks):
        prefix = f"transformer_blocks.{i}.attn."
        w = {}
        for key, tensor in branch_sd.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            # branch internals sit one level deeper in the checkpoint
            # (...attn.linear_attention.alpha.A_log) and the branch reads bare names
            if name.startswith("linear_attention."):
                name = name[len("linear_attention."):]
            w[name] = tensor
        missing = {"to_out_linear.weight", "beta_proj.weight", "norm.weight",
                   "alpha.A_log", "alpha.dt_bias", "alpha.down.weight",
                   "alpha.up.weight", "output_gate.down.weight",
                   "output_gate.up.weight", "output_gate.up.bias"} - set(w)
        if missing:
            raise ValueError(f"block {i} of {path} is missing branch tensors: "
                             f"{sorted(missing)}")
        branches.append(w)

    adapters = {}
    adapters_root = os.path.join(path, "adapters")
    if os.path.isdir(adapters_root):
        for name in sorted(os.listdir(adapters_root)):
            adir = os.path.join(adapters_root, name)
            cfg_file = os.path.join(adir, "adapter_config.json")
            weights_file = os.path.join(adir, "adapter_model.safetensors")
            if os.path.isfile(cfg_file) and os.path.isfile(weights_file):
                # Loader, not the loaded dict: the (fp32-on-disk) adapter tensors
                # are read from the safetensors at apply time and upcast there,
                # not held in RAM for the process lifetime.
                adapters[name] = (functools.partial(load_file, weights_file),
                                  _read_json(cfg_file))

    result = (cfg, branches, adapters)
    _CACHE.clear()
    _CACHE[stamp] = result
    return result
