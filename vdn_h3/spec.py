"""VDN-H3 checkpoint discovery, loading and validation.

A VDN checkpoint is the official exploded release directory (model_spec.json +
linear_branch/ + adapters/) from huggingface.co/OpenVDN/vdn-minimax-h3, dropped under
ComfyUI's models/vdn/ folder. Nothing is converted on disk; tensors are re-keyed in
memory onto ComfyUI's MiniMax-H3 module paths.
"""
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


def _branch_file(path):
    """The stage's branch file: plain, or the pre-quantized int8_convrot_comfyui one."""
    plain = os.path.join(path, "linear_branch", BRANCH_FILE)
    if os.path.isfile(plain):
        return plain
    quant = os.path.join(path, "linear_branch", BRANCH_FILE_INT8)
    if os.path.isfile(quant):
        return quant
    return plain


def _restore_quantized(sd):
    """Rebuild Comfy Kitchen QuantizedTensors from comfy_quant metadata keys.

    Mirrors ComfyUI's own int8-convrot model format (see comfy/ops.py: the
    <layer>.comfy_quant JSON + <layer>.weight_scale companion tensors): the
    weights stay QuantizedTensor / TensorWiseINT8Layout -- nothing is
    dequantized at load time. orig_dtype is bf16: the branch always computes
    in bf16 and the format does not carry it.
    """
    if not _KITCHEN_OK:
        return sd
    conf_keys = [k for k in sd if k.endswith(".comfy_quant")]
    if not conf_keys:
        return sd
    for conf_key in conf_keys:
        conf = json.loads(bytes(sd[conf_key].tolist()).decode("utf-8"))
        if conf.get("format") != "int8_tensorwise":
            raise ValueError(f"{conf_key}: unsupported quantization format "
                             f"{conf.get('format')!r}")
        layer = conf_key[: -len(".comfy_quant")]
        wkey = layer + ".weight"
        skey = wkey + "_scale"
        if wkey not in sd or skey not in sd:
            raise ValueError(f"{conf_key}: missing {wkey} or {skey}")
        qdata = sd.pop(wkey)
        scale = sd.pop(skey)
        params = TensorWiseINT8Layout.Params(
            scale=scale, orig_dtype=torch.bfloat16,
            orig_shape=tuple(qdata.shape), is_weight=True,
            convrot=bool(conf.get("convrot", False)),
            convrot_groupsize=int(conf.get("convrot_groupsize", 256)))
        sd[wkey] = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
        del sd[conf_key]
    return sd


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


def load_vdn_checkpoint(path):
    """Read model_spec.json + linear_branch + adapters. Returns
    (cfg, branch_weights_by_block, {adapter_name: (sd, adapter_spec)}). Cached by
    (path, mtime) so re-running the node doesn't re-read 5 GB."""
    branch_path = _branch_file(path)
    stamp = (path, os.path.getmtime(branch_path))
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

    branch_sd = _restore_quantized(load_file(branch_path))
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
            w[name] = tensor.contiguous()
        missing = {"to_out_linear.weight", "beta_proj.weight", "norm.weight",
                   "alpha.A_log", "alpha.dt_bias", "alpha.down.weight",
                   "alpha.up.weight", "output_gate.down.weight",
                   "output_gate.up.weight", "output_gate.up.bias"} - set(w)
        if missing:
            raise ValueError(f"block {i} of {path} is missing branch tensors: "
                             f"{sorted(missing)}")
        branches.append(w)
    del branch_sd

    adapters = {}
    adapters_root = os.path.join(path, "adapters")
    if os.path.isdir(adapters_root):
        for name in sorted(os.listdir(adapters_root)):
            adir = os.path.join(adapters_root, name)
            cfg_file = os.path.join(adir, "adapter_config.json")
            weights_file = os.path.join(adir, "adapter_model.safetensors")
            if os.path.isfile(cfg_file) and os.path.isfile(weights_file):
                adapters[name] = (load_file(weights_file), _read_json(cfg_file))

    result = (cfg, branches, adapters)
    _CACHE.clear()
    _CACHE[stamp] = result
    return result
