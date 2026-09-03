"""Apply VDN-H3 (Video Delta Net hybrid attention) to a loaded MiniMax-H3 model."""

import logging

import folder_paths

from vdn_h3.apply import apply_adapters
from vdn_h3.hybrid import VDNState, apply_vdn
from vdn_h3.branch import LinearBranch
import vdn_h3.spec as spec

_log = logging.getLogger("comfy.vdn")


class ApplyVDNH3:
    @classmethod
    def INPUT_TYPES(cls):
        names = spec.list_vdn_checkpoints()
        return {"required": {
            "model": ("MODEL",),
            "vdn_checkpoint": (names or ["<place a VDN stage-... directory under models/vdn>"],),
            "apply_turbo_adapter": ("BOOLEAN", {
                "default": True,
                "tooltip": "Apply the 'turbo' adapter when the checkpoint carries one "
                           "(stage-dmd = the 8-step VDN-H3 model). OFF gives the "
                           "50-step model the checkpoint was distilled from. Use 8 "
                           "sampler steps with it ON, 50 with it OFF."}),
            "strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Adapter strength. 1.0 is the released model."}),
            "lora_mode": (["bypass", "merge"], {
                "default": "bypass",
                "tooltip": "bypass: adapters applied at run time in activation space "
                           "(sharp; slightly more VRAM). merge: folded into the "
                           "weights (lowest VRAM; partly rounded away on int8/fp8 "
                           "bases)."}),
            "branch_weights": (["stream", "cache_gpu"], {
                "default": "stream",
                "tooltip": "stream: the ~4.3 GB of linear-branch weights are moved to "
                           "the GPU per block per step (safe on small cards, a little "
                           "slower). cache_gpu: resident on the GPU after the first "
                           "step (faster; keep ~4.3 GB VRAM free)."}),
            "verbose": ("BOOLEAN", {"default": False}),
            "attention_backend": (["grouped", "flex"], {
                "default": "grouped",
                "tooltip": "How the windowed softmax runs. grouped: one dense SDPA "
                           "per window group (portable, exact). flex: the whole "
                           "pattern as one compiled FlexAttention kernel over the "
                           "full sequence (faster on long clips; first run compiles, "
                           "falls back to grouped if compile fails)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patch/video"
    DESCRIPTION = (
        "VDN-H3: hybrid attention for MiniMax-H3. Nearby frames keep exact softmax "
        "attention; distant context goes through the checkpoint's linear Video Delta "
        "Attention branch. Requires a VDN checkpoint (models/vdn, from "
        "OpenVDN/vdn-minimax-h3) and a MiniMax-H3 base model.")

    def apply(self, model, vdn_checkpoint, apply_turbo_adapter, strength, lora_mode,
              branch_weights, attention_backend, verbose):
        path = spec.resolve_vdn_checkpoint(vdn_checkpoint)
        cfg, branch_weights_by_block, adapters = spec.load_vdn_checkpoint(path)

        dm = model.get_model_object("diffusion_model")
        blocks = getattr(dm, "blocks", None)
        if blocks is None or not hasattr(getattr(blocks[0], "attn", None), "qkv_proj"):
            raise RuntimeError(
                "ApplyVDNH3 needs a MiniMax-H3 MODEL (blocks[].attn.qkv_proj). "
                "Connect a MiniMax-H3 diffusion model loader first.")
        if len(blocks) != len(branch_weights_by_block):
            raise RuntimeError(
                f"VDN checkpoint has {len(branch_weights_by_block)} blocks but the "
                f"loaded model has {len(blocks)}; this VDN checkpoint belongs to a "
                "different base.")

        for key in model.object_patches:
            if key.endswith(".attn.forward"):
                if getattr(model.object_patches[key], "_vdn_forward", False):
                    raise RuntimeError("This MODEL already has VDN-H3 applied; chain "
                                       "it only once.")
                logging.warning("[vdn] replacing an existing attention forward patch "
                                "(%s); attention patches applied before VDN will no "
                                "longer run on the softmax path", key)

        attn0 = blocks[0].attn
        num_heads, head_dim = attn0.heads, attn0.head_dim
        hidden = dm.hidden_size
        lin_dim = cfg["linear_head_dim"]
        expected = {"to_out_linear.weight": (hidden, num_heads * lin_dim),
                    "beta_proj.weight": (num_heads, hidden),
                    "alpha.A_log": (num_heads,),
                    "alpha.dt_bias": (num_heads * lin_dim,),
                    "alpha.down.weight": (lin_dim, hidden),
                    "alpha.up.weight": (num_heads * lin_dim, lin_dim),
                    "output_gate.down.weight": (lin_dim, hidden),
                    "output_gate.up.weight": (num_heads * lin_dim, lin_dim),
                    "output_gate.up.bias": (num_heads * lin_dim,),
                    "softmax_gate.up.weight": (num_heads, hidden),
                    "softmax_gate.up.bias": (num_heads,),
                    "norm.weight": (lin_dim,)}
        sample = branch_weights_by_block[0]
        for key, shape in expected.items():
            if key in sample and tuple(sample[key].shape) != shape:
                raise RuntimeError(
                    f"VDN checkpoint/{path}: {key} has shape "
                    f"{tuple(sample[key].shape)}, expected {shape} for this base "
                    f"(heads={num_heads}, head_dim={head_dim}, hidden={hidden}).")

        branches = [LinearBranch(w, num_heads, head_dim,
                                 delta_rule=cfg["delta_rule"], bridge=cfg["bridge"],
                                 a_fp32=cfg["a_fp32"], short_conv=cfg["short_conv"],
                                 enable_text_state=cfg["enable_text_state"])
                    for w in branch_weights_by_block]
        state = VDNState(vdn_checkpoint, cfg, branches, num_heads, head_dim)
        state.cache_gpu = branch_weights == "cache_gpu"
        state.softmax_backend = attention_backend

        new_model = model.clone()
        apply_vdn(new_model, state)

        wanted = {"default"}
        if apply_turbo_adapter:
            wanted.add("turbo")
        missing = wanted - set(adapters)
        if "default" in missing:
            raise RuntimeError(
                f"{vdn_checkpoint}: the 'default' (Stage-B) adapter is missing; this "
                "checkpoint cannot reproduce the released model. Re-download the "
                "stage directory.")
        converted = {}
        for name in sorted(wanted & set(adapters)):
            sd, adapter_cfg = adapters[name]
            from vdn_h3.adapters import convert_adapter
            converted[name] = convert_adapter(sd, adapter_cfg)
            if verbose:
                _log.info("[vdn] adapter %s: %d modules (%s)", name,
                          len(converted[name]),
                          ", ".join(sorted(converted[name])[:3]) + ", ...)")
        report = apply_adapters(new_model, converted, strength, lora_mode)

        _log.info("[vdn] %s applied on %d blocks (%s): %s", vdn_checkpoint,
                  len(branches),
                  f"r={cfg['radius']} c={cfg['chunk']} anchors={cfg['anchor_frames']} "
                  f"rule={cfg['delta_rule']}", report)
        return (new_model,)


NODE_CLASS_MAPPINGS = {"ApplyVDNH3": ApplyVDNH3}
NODE_DISPLAY_NAME_MAPPINGS = {"ApplyVDNH3": "Apply VDN-H3 (MiniMax-H3 Hybrid Attention)"}
