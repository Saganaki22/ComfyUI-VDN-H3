"""Windowed softmax branch of VDN-H3 for ComfyUI's MiniMax-H3.

Ports the window geometry and softmax semantics of the official VDN release
(github.com/OpenVDN/vdn-minimax-h3, src/models/softmax_attention/) onto ComfyUI's
packed sequence. The released 8-step checkpoint uses radius=1, chunk=5 (chunk-aligned
windows), anchor_frames="both".

The released inference path runs block-sparse FlexAttention over a BlockMask. This
port groups queries that share a window -- under chunk-aligned bounds every frame of
a chunk has the same window -- and runs one dense SDPA call per distinct window, so
it needs no Triton and no torch.compile while keeping the exact same softmax
partition (the official window_softmax_reference is the same arithmetic spelled as
one SDPA per frame instead of per chunk).
"""
import collections
import logging
import os
import warnings

import torch
import torch.nn.functional as F

_log = logging.getLogger("comfy.vdn")

ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")


def window_bounds(num_frames, radius, chunk=0):
    """Per-frame inclusive softmax-window bounds [lo, hi], unclamped. Verbatim port.

    chunk == 0: frame mode, centered window |t_q - t_k| <= radius.
    chunk == K: chunk-aligned mode, frame t sees whole chunks [t//K - r, t//K + r].
    """
    if chunk <= 0:
        return [(t - radius, t + radius) for t in range(num_frames)]
    return [(((t // chunk) - radius) * chunk, ((t // chunk) + radius + 1) * chunk - 1)
            for t in range(num_frames)]


def full_coverage(bounds, num_frames):
    """True when every window already covers all frames (softmax IS dense and the
    linear branch must go inactive so nothing is counted twice)."""
    return all(lo <= 0 and hi >= num_frames - 1 for lo, hi in bounds)


# --------------------------------------------------- grouped-path plan & scratch --

MAX_CACHED_PLANS = 8
_PLAN_CACHE = collections.OrderedDict()
_KV_SCRATCH = {}


def _window_plan(video_start, video_end, num_frames, tokens_per_frame, bounds,
                 anchor_frames, seq, device):
    """Everything about the window partition that is identical for every block and
    every step of a run: the global-row index, per-group query/window row indices,
    and the anchor-row slices. Cached per layout instead of rebuilding ~50
    arange/cat index tensors per block per step."""
    key = (video_start, video_end, num_frames, tokens_per_frame,
           tuple(map(tuple, bounds)), anchor_frames, seq, str(device))
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        _PLAN_CACHE.move_to_end(key)
        return hit

    def frame_rows(f):
        a = video_start + f * tokens_per_frame
        return torch.arange(a, a + tokens_per_frame, device=device)

    global_idx = torch.cat([torch.arange(video_start, device=device),
                            torch.arange(video_end, seq, device=device)])
    anchors = (0, num_frames - 1)
    anchor_rows = sorted(f for f in anchors if anchor_frames in ("rows", "both"))
    anchor_set = set(anchor_rows)

    grouped = collections.OrderedDict()
    for f in range(num_frames):
        if f in anchor_set:
            continue
        lo = max(bounds[f][0], 0)
        hi = min(bounds[f][1], num_frames - 1)
        grouped.setdefault((lo, hi), []).append(f)

    groups = []
    max_rows = global_idx.numel()
    for (lo, hi), frames in grouped.items():
        extra = [f for f in anchors
                 if anchor_frames in ("columns", "both") and not lo <= f <= hi]
        key_frames = sorted(set(range(lo, hi + 1)) | set(extra))
        win_idx = torch.cat([frame_rows(f) for f in key_frames])
        q_idx = torch.cat([frame_rows(f) for f in frames])
        groups.append((q_idx, win_idx))
        max_rows = max(max_rows, global_idx.numel() + win_idx.numel())
    plan = dict(global_idx=global_idx, groups=groups,
                anchor_slices=[(video_start + f * tokens_per_frame,
                                video_start + (f + 1) * tokens_per_frame)
                               for f in anchor_rows],
                max_kv_rows=max_rows)
    _PLAN_CACHE[key] = plan
    while len(_PLAN_CACHE) > MAX_CACHED_PLANS:
        _PLAN_CACHE.popitem(last=False)
    return plan


def _kv_scratch(rows, heads, head_dim, device, dtype, retain=True):
    """Retained mode: ONE grow-only k/v buffer pair per (device, dtype), sized
    to the largest window group; every group's gather lands in a slice of it
    instead of a fresh per-group torch.cat allocation. Transient mode (VRAM
    pressure): fresh per call -- the v1.3.1 pattern. Same-stream execution makes
    the reuse safe (each group's SDPA is enqueued before the next group's
    gather overwrites)."""
    if not retain:
        shape = (rows, heads, head_dim)
        return (torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype))
    key = (str(device), dtype)
    pair = _KV_SCRATCH.get(key)
    need = rows * heads * head_dim
    if pair is None or pair[0].numel() < need:
        pair = (torch.empty(need, device=device, dtype=dtype),
                torch.empty(need, device=device, dtype=dtype))
        _KV_SCRATCH[key] = pair
    return (pair[0][:need].view(rows, heads, head_dim),
            pair[1][:need].view(rows, heads, head_dim))


def clear_window_state():
    """Drop cached window plans and the k/v scratch (run interrupt / cleanup)."""
    _PLAN_CACHE.clear()
    _KV_SCRATCH.clear()


def window_softmax_grouped(query, key, value, video_start, video_end,
                           num_frames, tokens_per_frame, bounds, scale,
                           anchor_frames="none", transformer_options=None,
                           retain_buffers=True):
    """Windowed softmax over the packed sequence [globals | video], one dense SDPA
    call per distinct query group.

    query/key/value: [seq, H, d], already QK-normed and RoPE'd, full sequence.
    Returns [seq, H, d]: every pair involving a global row (text/cond/audio) stays
    dense in both directions; (video, video) pairs are restricted to the window,
    widened by the anchor frames per `anchor_frames` (official semantics: "columns"
    makes frames 0 and F-1 visible to every query, "rows" makes those two frames'
    queries see everything, "both" is exact on both sides).
    """
    heads, head_dim = query.shape[1], query.shape[2]
    seq = query.shape[0]
    out = torch.empty_like(query)
    plan = _window_plan(video_start, video_end, num_frames, tokens_per_frame,
                        bounds, anchor_frames, seq, query.device)
    global_idx = plan["global_idx"]
    global_q = query[global_idx]
    g = global_idx.numel()
    if g:
        # globals (text/cond/audio) are dense in both directions: every key
        out[global_idx] = _sdpa(global_q, key, value, scale, transformer_options)

    groups = plan["groups"]
    if groups:
        global_k = key[global_idx]
        global_v = value[global_idx]
        k_scratch, v_scratch = _kv_scratch(plan["max_kv_rows"], heads, head_dim,
                                           key.device, key.dtype,
                                           retain=retain_buffers)
        if g:
            k_scratch[:g].copy_(global_k)
            v_scratch[:g].copy_(global_v)
        for q_idx, win_idx in groups:
            w = win_idx.numel()
            torch.index_select(key, 0, win_idx, out=k_scratch[g:g + w])
            torch.index_select(value, 0, win_idx, out=v_scratch[g:g + w])
            q_rows = query.index_select(0, q_idx)
            out[q_idx] = _sdpa(q_rows, k_scratch[:g + w], v_scratch[:g + w],
                               scale, transformer_options)

    for a, b in plan["anchor_slices"]:
        out[a:b] = _sdpa(query[a:b], key, value, scale, transformer_options)

    return out


# ------------------------------------------------------------- backend visibility --

_BACKEND_LOGGED = False
_FORCED_BACKEND = ...  # lazily parsed sentinel
_FORCED_BROKEN = set()

# Exact SDPA backends only, in ComfyUI's priority order. The window softmax must
# never route through sage/kitchen int8 overrides (that measurably softens
# output); choosing AMONG the exact kernels is pure perf, no quality surface.
_BACKEND_PRIORITY = ("flash", "cudnn", "mem_efficient", "math")


def _sdpa_backend_enum(name):
    from torch.nn.attention import SDPBackend
    return {"flash": SDPBackend.FLASH_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH}[name]


def _forced_backend():
    """VDN_H3_WINDOW_SDPA=flash|cudnn|mem_efficient|math forces one exact backend
    for the window groups (default: ComfyUI's priority chain)."""
    global _FORCED_BACKEND
    if _FORCED_BACKEND is ...:
        raw = os.environ.get("VDN_H3_WINDOW_SDPA", "auto").strip().lower()
        if raw in ("", "auto"):
            _FORCED_BACKEND = None
        elif raw in _BACKEND_PRIORITY:
            _FORCED_BACKEND = raw
        else:
            _log.warning("[vdn] VDN_H3_WINDOW_SDPA=%r not one of %s; ignoring",
                         raw, _BACKEND_PRIORITY)
            _FORCED_BACKEND = None
    return _FORCED_BACKEND


def _sdpa_call(q4, k4, v4, scale, backend=None):
    """One SDPA under a single-backend (or default-chain) dispatch context."""
    if backend is None:
        return F.scaled_dot_product_attention(q4, k4, v4, scale=scale)
    from torch.nn.attention import sdpa_kernel
    with sdpa_kernel([_sdpa_backend_enum(backend)], set_priority=True):
        return F.scaled_dot_product_attention(q4, k4, v4, scale=scale)


def _log_backend_once(q4, k4, v4, scale):
    """Name the exact kernel the priority chain actually picks for these window
    shapes: probe flash -> cuDNN -> mem-efficient -> math once, in order."""
    global _BACKEND_LOGGED
    if _BACKEND_LOGGED:
        return
    _BACKEND_LOGGED = True
    forced = _forced_backend()
    if forced is not None:
        _log.info("[vdn] window SDPA backend: %s (forced via "
                  "VDN_H3_WINDOW_SDPA)", forced)
        return
    chosen = "math"
    # torch emits a UserWarning per rejected backend while probing ("not used
    # because ...", "runtime disabled"); users don't need the spam -- the info
    # line below reports the winner.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in _BACKEND_PRIORITY:
            try:
                _sdpa_call(q4, k4, v4, scale, backend=name)
                chosen = name
                break
            except Exception:
                continue
    _log.info("[vdn] window SDPA backend: %s (priority flash -> cuDNN -> "
              "mem-efficient; override with VDN_H3_WINDOW_SDPA=flash|cudnn|"
              "mem_efficient)", chosen)


def _sdpa(q_rows, k_rows, v_rows, scale, transformer_options=None):
    """[rows, H, d] x [keys, H, d] -> [rows, H, d] via one dense attention.

    With transformer options the call goes through ComfyUI's dispatched
    attention, so any optimized_attention_override on the model (e.g. a sage
    patch) applies to the window groups exactly as it does to the base model's
    dense attention. The dispatched functions scale by head_dim ** -0.5
    internally, which is the `scale` the caller passes. Without them (unit
    tests, CPU) raw SDPA keeps the path dependency-free."""
    if transformer_options is not None:
        from comfy.ldm.modules import attention as comfy_attention
        rows, heads, dim = q_rows.shape
        out = comfy_attention.optimized_attention(
            q_rows.reshape(1, rows, heads * dim),
            k_rows.reshape(1, -1, heads * dim),
            v_rows.reshape(1, -1, heads * dim),
            heads, transformer_options=transformer_options)
        return out.reshape(rows, heads, dim)
    # No override: still dispatch through comfy's backend-priority chain
    # (flash -> cuDNN -> mem-efficient), since Windows torch builds ship without
    # the flash kernel and raw F.sdpa lands on the slow mem-efficient backend.
    q4 = q_rows.permute(1, 0, 2).unsqueeze(0)
    k4 = k_rows.permute(1, 0, 2).unsqueeze(0)
    v4 = v_rows.permute(1, 0, 2).unsqueeze(0)
    _log_backend_once(q4, k4, v4, scale)
    forced = _forced_backend()
    if forced is not None and forced not in _FORCED_BROKEN:
        try:
            return _sdpa_call(q4, k4, v4, scale, backend=forced) \
                .squeeze(0).permute(1, 0, 2)
        except RuntimeError as e:
            _FORCED_BROKEN.add(forced)
            _log.warning("[vdn] forced window SDPA backend %s unavailable (%s); "
                         "using the default chain from here on", forced, e)
    try:
        from comfy.ops import scaled_dot_product_attention as comfy_sdpa
    except ImportError:                      # unit tests run without comfy on path
        comfy_sdpa = F.scaled_dot_product_attention
    attended = comfy_sdpa(q4, k4, v4, scale=scale)
    return attended.squeeze(0).permute(1, 0, 2)


# ---------------------------------------------------------------- flex path --

_FLEX = None
_BM_CACHE = {}


def _build_window_tables(seq, video_start, video_end, num_frames,
                         tokens_per_frame, bounds, device):
    """Per-token [lo, hi] allowed video-frame ranges; the table both the flex
    mask_mod and the dense test oracle index into."""
    lo = torch.zeros(seq, dtype=torch.long, device=device)
    hi = torch.full((seq,), num_frames - 1, dtype=torch.long, device=device)
    for f in range(num_frames):
        a = video_start + f * tokens_per_frame
        lo[a:a + tokens_per_frame] = max(bounds[f][0], 0)
        hi[a:a + tokens_per_frame] = min(bounds[f][1], num_frames - 1)
    return lo, hi


def _window_mask_mod(video_start, video_end, num_frames, tokens_per_frame,
                     lo, hi, anchor_frames):
    """mask_mod over token indices: globals dense both ways, video restricted to
    its chunk window, plus anchor columns and/or anchor rows."""
    allow_k = anchor_frames in ("columns", "both")
    allow_q = anchor_frames in ("rows", "both")

    def mask_mod(b, h, q, kv):
        gq = (q < video_start) | (q >= video_end)
        gk = (kv < video_start) | (kv >= video_end)
        qf = (q - video_start) // tokens_per_frame
        kf = (kv - video_start) // tokens_per_frame
        allowed = gq | gk | ((kf >= lo[q]) & (kf <= hi[q]))
        if allow_k:
            allowed = allowed | (kf == 0) | (kf == num_frames - 1)
        if allow_q:
            allowed = allowed | (qf == 0) | (qf == num_frames - 1)
        return allowed

    return mask_mod


def window_softmax_flex(query, key, value, video_start, video_end, num_frames,
                        tokens_per_frame, bounds, scale, anchor_frames="none"):
    """The same window partition as window_softmax_grouped, executed as one fused
    FlexAttention kernel over the full sequence with a BlockMask -- the official
    release's softmax architecture, minus its FA4 backend. Needs torch.compile +
    triton; the first call per sequence shape compiles (and the BlockMask is
    cached per shape)."""
    global _FLEX
    from torch.nn.attention.flex_attention import (create_block_mask,
                                                   flex_attention)
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention)
    seq = query.shape[0]
    ck = (seq, video_start, video_end, num_frames, tokens_per_frame,
          anchor_frames, tuple(tuple(b) for b in bounds), query.device.type)
    bm = _BM_CACHE.get(ck)
    if bm is None:
        lo, hi = _build_window_tables(seq, video_start, video_end, num_frames,
                                      tokens_per_frame, bounds, query.device)
        bm = create_block_mask(
            _window_mask_mod(video_start, video_end, num_frames,
                             tokens_per_frame, lo, hi, anchor_frames),
            None, None, seq, seq, query.device, _compile=True)
        _BM_CACHE[ck] = bm
    out = _FLEX(query.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                block_mask=bm, scale=scale)
    return out.squeeze(0).transpose(0, 1)
