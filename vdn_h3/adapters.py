"""VDN-H3 adapter (LoRA) loading for ComfyUI.

The released adapters are stored against the diffusers MiniMax-H3 layout with peft
key names. ComfyUI's MiniMax-H3 instead fuses the attention projections into one
qkv_proj and packs the MLP swiglu as [gate; value] where diffusers packs [value;
gate]. This module rewrites adapter tensors onto ComfyUI module paths in memory at
load time; nothing on disk is converted.

Fusion of three per-projection LoRA pairs (to_q/to_k/to_v) into one fused-qkv pair is
exact: delta_W = concat(B_i @ A_i) = B_fused @ A_fused with A_fused = vstack(A_i) and
B_fused block-diagonal.
"""
import torch

# diffusers target suffix (after transformer_blocks.N. / token_refiner.refiner_blocks.N.)
# -> (kind, comfy suffix)
_ATTN_QKV = ("attn.orig.to_q", "attn.orig.to_k", "attn.orig.to_v")
_ATTN_OUT = "attn.orig.to_out.0"


def _comfy_path(key, is_refiner):
    """diffusers module path -> (comfy module path, conversion kind)."""
    if is_refiner:
        stem = key.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    else:
        stem = key.replace("transformer_blocks.", "blocks.")
    for proj in _ATTN_QKV:
        if stem.endswith("." + proj):
            return stem[: -len(proj)] + "attn.qkv_proj", "qkv"
    if stem.endswith("." + _ATTN_OUT):
        return stem[: -len(_ATTN_OUT)] + "attn.out_proj", "out"
    if stem.endswith(".ff.net.0.proj"):
        return stem[: -len("ff.net.0.proj")] + "mlp.fc1", "swiglu"
    if stem.endswith(".ff.net.2"):
        return stem[: -len("ff.net.2")] + "mlp.fc2", "plain"
    if stem.endswith(".adaln_proj.linear") or stem == "norm_out.linear":
        path = stem if stem.endswith(".adaln_proj.linear") \
            else stem.replace("norm_out.linear", "final_layer.adaln_proj.linear")
        return path, "adaln"
    return stem, "plain"


def parse_adapter_state(sd):
    """peft-named safetensors dict -> {(diffusers module, infix-free): {lora_A/lora_B}}."""
    out = {}
    for key, tensor in sd.items():
        if ".lora_" not in key:
            continue
        module, rest = key.split(".lora_", 1)
        side = rest.split(".")[0]                    # A or B
        out.setdefault(module, {})[side] = tensor.float()
    return out


def per_module_scale(adapter_cfg, module):
    """peft scaling alpha/rank for one module, honoring rank_pattern/alpha_pattern."""
    cfg = adapter_cfg.get("config", adapter_cfg)
    rank = cfg.get("rank", 64)
    alpha = cfg.get("alpha", rank)
    for pattern, value in (cfg.get("rank_pattern") or {}).items():
        if pattern in module:
            rank = value
    for pattern, value in (cfg.get("alpha_pattern") or {}).items():
        if pattern in module:
            alpha = value
    return alpha / rank


def convert_adapter(sd, adapter_cfg):
    """{diffusers module: {A, B}} -> {comfy module: (lora_A, lora_B, scale)}.

    qkv targets fold into one block-diagonal pair; swiglu halves swap; refiner keys
    reroot onto ComfyUI's token_refiner.blocks naming."""
    parsed = parse_adapter_state(sd)
    qkv_groups = {}
    out = {}
    for module, sides in parsed.items():
        is_refiner = module.startswith("token_refiner.")
        path, kind = _comfy_path(module, is_refiner)
        a, b = sides["A"], sides["B"]
        scale = per_module_scale(adapter_cfg, module)
        if kind == "qkv":
            qkv_groups.setdefault(path, []).append((module, a, b, scale))
            continue
        if kind == "swiglu":
            half = b.shape[0] // 2
            value_half, gate_half = b[:half], b[half:]
            b = torch.cat([gate_half, value_half], dim=0)
        out[path] = (a, b, scale)
    for path, group in qkv_groups.items():
        order = ("to_q", "to_k", "to_v")

        def qkv_index(item):
            return next(i for i, p in enumerate(order) if item[0].endswith("." + p))

        group.sort(key=qkv_index)
        rank = group[0][1].shape[0]
        in_dim = group[0][1].shape[1]
        out_dim = group[0][2].shape[0]
        a_fused = torch.cat([g[1] for g in group], dim=0)          # [3r, in]
        b_fused = torch.zeros(out_dim * 3, rank * 3, dtype=a_fused.dtype)
        scales = []
        for i, (_mod, a, b, scale) in enumerate(group):
            b_fused[i * out_dim:(i + 1) * out_dim, i * rank:(i + 1) * rank] = b
            scales.append(scale)
        if len(set(scales)) != 1:
            raise ValueError(f"mixed alpha/rank across qkv projections of {path}: "
                             f"{scales}; cannot fold into one fused pair")
        out[path] = (a_fused, b_fused, scales[0])
    return out


def load_comfy_lora_format(converted, strength=1.0):
    """{comfy module: (A, B, scale)} -> the dict comfy.lora.load_lora consumes, keyed
    by diffusion-model weight key with peft lora_A/lora_B names. ComfyUI divides the
    stored `.alpha` by the rank itself, so write alpha = scale * rank; the node's
    strength is applied separately at injection time."""
    lora = {}
    for path, (a, b, scale) in converted.items():
        lora[path + ".lora_A.weight"] = a.contiguous()
        lora[path + ".lora_B.weight"] = b.contiguous()
        lora[path + ".alpha"] = torch.tensor(scale * a.shape[0] * strength)
    return lora
