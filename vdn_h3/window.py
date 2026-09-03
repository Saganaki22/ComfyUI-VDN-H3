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
import torch
import torch.nn.functional as F

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


def window_softmax_grouped(query, key, value, video_start, video_end,
                           num_frames, tokens_per_frame, bounds, scale,
                           anchor_frames="none", transformer_options=None):
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
    out = torch.empty_like(query)

    global_idx = torch.cat([torch.arange(video_start, device=query.device),
                            torch.arange(video_end, query.shape[0], device=query.device)])
    global_q = query[global_idx]
    global_k, global_v = key[global_idx], value[global_idx]
    if global_idx.numel():
        # globals (text/cond/audio) are dense in both directions: every key
        out[global_idx] = _sdpa(global_q, key, value, scale, transformer_options)

    def frame_slice(f):
        a = video_start + f * tokens_per_frame
        b = a + tokens_per_frame
        return a, b

    def video_keys(frames):
        rows = [slice(*frame_slice(f)) for f in frames]
        return (torch.cat([key[r] for r in rows]),
                torch.cat([value[r] for r in rows]))

    anchors = (0, num_frames - 1)
    anchor_rows = {f for f in anchors if anchor_frames in ("rows", "both")}

    # group frames by their clamped window; anchor-row frames split out as dense
    groups = {}
    for f in range(num_frames):
        if f in anchor_rows:
            continue
        lo = max(bounds[f][0], 0)
        hi = min(bounds[f][1], num_frames - 1)
        groups.setdefault((lo, hi), []).append(f)

    for (lo, hi), frames in groups.items():
        q_rows = torch.cat([query[slice(*frame_slice(f))] for f in frames])
        key_frames = list(range(lo, hi + 1))
        extra = [f for f in anchors
                 if anchor_frames in ("columns", "both") and not lo <= f <= hi]
        k_w, v_w = video_keys(sorted(set(key_frames) | set(extra)))
        k = torch.cat([global_k, k_w])
        v = torch.cat([global_v, v_w])
        out[torch.cat([torch.arange(*frame_slice(f)) for f in frames])] = \
            _sdpa(q_rows, k, v, scale, transformer_options)

    for f in anchor_rows:
        a, b = frame_slice(f)
        out[a:b] = _sdpa(query[a:b], key, value, scale, transformer_options)

    return out


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
    attended = F.scaled_dot_product_attention(
        q_rows.permute(1, 0, 2).unsqueeze(0),
        k_rows.permute(1, 0, 2).unsqueeze(0),
        v_rows.permute(1, 0, 2).unsqueeze(0), scale=scale)
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
