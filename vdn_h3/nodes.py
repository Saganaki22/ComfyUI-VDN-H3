"""Apply VDN-H3 (Video Delta Net hybrid attention) to a loaded MiniMax-H3 model."""

import logging
import os

import folder_paths

import comfy.model_management

from vdn_h3.apply import apply_adapters
from vdn_h3.hybrid import VDNState, apply_vdn
from vdn_h3.branch import LinearBranch
import vdn_h3.spec as spec

_log = logging.getLogger("comfy.vdn")


def _apply_vdn(model, vdn_checkpoint, strength, lora_mode, branch_weights,
               attention_backend, verbose, apply_turbo_adapter=True,
               cfg_overrides=None, fast_kernels=False, retain_buffers="auto"):
    """Shared core of ApplyVDNH3 and ApplyVDNH3Advanced. `strength` is a float or a
    {adapter_name: float} map; `cfg_overrides` deviates from the checkpoint's trained
    spec (ablation knobs); `fast_kernels` torch.compiles the branch's hot spots
    (epilogue, state gather, frame-major q store, bidirectional scan)."""
    path = spec.resolve_vdn_checkpoint(vdn_checkpoint)
    prefer_int8 = False
    retain = True
    if branch_weights == "auto" or retain_buffers == "auto":
        # free VRAM right now = after the base model's load in the same run
        free = comfy.model_management.get_free_memory(
            comfy.model_management.get_torch_device())
    else:
        free = None
    if branch_weights == "auto":
        branch_weights, prefer_int8 = spec.auto_branch_policy(path, free)
    cfg, branch_weights_by_block, adapters = spec.load_vdn_checkpoint(
        path, prefer_int8=prefer_int8)
    if retain_buffers == "auto":
        retain = spec.auto_retain_policy(path, prefer_int8, free)
    else:
        retain = retain_buffers == "on"

    if cfg_overrides:
        changed = {k: (cfg.get(k), v) for k, v in cfg_overrides.items()
                   if k in cfg and cfg.get(k) != v}
        if changed:
            logging.warning("[vdn] deviating from the trained spec: %s", changed)
        cfg = dict(cfg, **cfg_overrides)

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
                             enable_text_state=cfg["enable_text_state"],
                             retain_buffers=retain)
                for w in branch_weights_by_block]
    for b in branches:
        b.fuse_epilogue = fast_kernels
    if fast_kernels and "dmd" in os.path.basename(path).lower():
        _log.warning(
            "[vdn] fast_kernels on an 8-step DMD stage (%s): the compiled branch "
            "kernels are known to drift on 8-step DMD checkpoints (measurably "
            "visible output on torch 2.10) -- ablation use only, do not use for "
            "final renders", os.path.basename(path))
    state = VDNState(vdn_checkpoint, cfg, branches, num_heads, head_dim)
    state.retain_buffers = retain
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
        loader, adapter_cfg = adapters[name]
        from vdn_h3.adapters import convert_adapter
        converted[name] = convert_adapter(loader(), adapter_cfg)
        if verbose:
            _log.info("[vdn] adapter %s: %d modules (%s)", name,
                      len(converted[name]),
                      ", ".join(sorted(converted[name])[:3]) + ", ...)")
    report = apply_adapters(new_model, converted, strength, lora_mode)

    _log.info("[vdn] %s applied on %d blocks (%s): %s", vdn_checkpoint,
              len(branches),
              f"r={cfg['radius']} c={cfg['chunk']} anchors={cfg['anchor_frames']} "
              f"rule={cfg['delta_rule']}"
              + (" fused" if fast_kernels else ""), report)
    return (new_model,)


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
                "default": "merge",
                "tooltip": "merge: adapters folded into the weights -- reproduces "
                           "the validated model exactly. REQUIRED for 8-step DMD "
                           "checkpoints (stage-dmd-*): bypass's activation-space "
                           "rounding noise is amplified by the deep blocks and "
                           "visibly degrades output."}),
            "branch_weights": (["auto", "stream", "cache_gpu"], {
                "default": "auto",
                "tooltip": "auto (default): cache_gpu when the free VRAM after the "
                           "base load exceeds 1.5x the stage size + 4 GiB headroom, "
                           "else stream (prefers the int8_convrot stage file under "
                           "memory pressure). stream: the ~4.3 GB of linear-branch "
                           "weights are moved to the GPU per block per step, with a "
                           "one-block lookahead prefetch (safe on small cards). "
                           "cache_gpu: resident on the GPU after the first step "
                           "(faster; keep ~4.3 GB VRAM free)."}),
            "retain_buffers": (["auto", "on", "off"], {
                "default": "auto",
                "tooltip": "Retained branch scratch/banks (scan banks, delta "
                           "solve, window gather, q/k/v copies + prefetch) trade "
                           "~0.5-1 GiB VRAM for churn-free steps. auto: retain "
                           "when free VRAM >= stage + 10 GiB headroom, else "
                           "transient (v1.3.1 allocation pattern, peak VRAM "
                           "priority on small cards). on/off override."}),
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
              branch_weights, attention_backend, verbose, retain_buffers="auto"):
        return _apply_vdn(model, vdn_checkpoint, strength, lora_mode, branch_weights,
                          attention_backend, verbose,
                          apply_turbo_adapter=apply_turbo_adapter,
                          retain_buffers=retain_buffers)


class ApplyVDNH3Advanced:
    """Everything the base node does, plus per-adapter strengths, ablation knobs that
    deviate from the released spec (window radius/chunk, anchor frames, text state,
    linear branch), and compile-fused branch kernels (fast_kernels)."""

    @classmethod
    def INPUT_TYPES(cls):
        names = spec.list_vdn_checkpoints()
        return {"required": {
            "model": ("MODEL",),
            "vdn_checkpoint": (names or ["<place a VDN stage-... directory under models/vdn>"],),
            "apply_turbo_adapter": ("BOOLEAN", {
                "default": True,
                "tooltip": "Apply the 'turbo' adapter when the checkpoint carries "
                           "one (8-step model). See the base node's tooltip."}),
            "stage_b_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Strength of the 'default' (Stage-B) adapter."}),
            "turbo_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Strength of the 'turbo' (8-step DMD) adapter."}),
            "lora_mode": (["bypass", "merge"], {
                "default": "merge",
                "tooltip": "merge required for 8-step DMD checkpoints; see the "
                           "base node's tooltip."}),
            "branch_weights": (["auto", "stream", "cache_gpu"], {"default": "auto"}),
            "retain_buffers": (["auto", "on", "off"], {"default": "auto"}),
            "verbose": ("BOOLEAN", {"default": False}),
            "attention_backend": (["grouped", "flex"], {"default": "grouped"}),
        }, "optional": {
            "window_radius": ("INT", {
                "default": 1, "min": 0, "max": 8,
                "tooltip": "Softmax window radius in chunks. Trained: 1."}),
            "window_chunk": ("INT", {
                "default": 5, "min": 0, "max": 64,
                "tooltip": "Chunk size for the aligned window; 0 = per-frame "
                           "centered window. Trained: 5."}),
            "anchor_frames": (["both", "columns", "rows", "none"], {
                "default": "both",
                "tooltip": "Boundary-frame anchors. Trained: both."}),
            "text_state": ("BOOLEAN", {
                "default": True,
                "tooltip": "Write the prompt into the linear branch's states at "
                           "init. Trained: on."}),
            "linear_branch": ("BOOLEAN", {
                "default": True,
                "tooltip": "Off = window-only ablation (debug; output then lacks "
                           "all long-range context on clips longer than the "
                           "window)."}),
            "fast_kernels": ("BOOLEAN", {
                "default": False,
                "tooltip": "torch.compile the branch's hot spots (RMSNorm+gate "
                           "epilogue, state gather, frame-major q store, and the "
                           "bidirectional scan as one CUDA-graph replay). Same "
                           "math; falls back to eager if compile fails. First run "
                           "compiles."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patch/video"
    DESCRIPTION = (
        "VDN-H3 advanced: per-adapter strengths, window/anchor/text/branch ablations, "
        "and compile-fused branch kernels. Defaults reproduce the released "
        "model exactly.")

    def apply(self, model, vdn_checkpoint, apply_turbo_adapter, stage_b_strength,
              turbo_strength, lora_mode, branch_weights, attention_backend, verbose,
              retain_buffers="auto",
              window_radius=1, window_chunk=5, anchor_frames="both", text_state=True,
              linear_branch=True, fast_kernels=False):
        strength = {"default": stage_b_strength, "turbo": turbo_strength}
        cfg_overrides = {"radius": window_radius, "chunk": window_chunk,
                         "anchor_frames": anchor_frames,
                         "enable_text_state": text_state,
                         "linear_enabled": linear_branch}
        return _apply_vdn(model, vdn_checkpoint, strength, lora_mode, branch_weights,
                          attention_backend, verbose,
                          apply_turbo_adapter=apply_turbo_adapter,
                          cfg_overrides=cfg_overrides, fast_kernels=fast_kernels,
                          retain_buffers=retain_buffers)


NODE_CLASS_MAPPINGS = {"ApplyVDNH3": ApplyVDNH3,
                       "ApplyVDNH3Advanced": ApplyVDNH3Advanced}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyVDNH3": "Apply VDN-H3 (MiniMax-H3 Hybrid Attention)",
    "ApplyVDNH3Advanced": "Apply VDN-H3 Advanced (Ablations & Fast Kernels)"}
