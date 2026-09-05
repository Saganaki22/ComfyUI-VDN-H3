"""VDN-H3 hybrid attention for ComfyUI's MiniMax-H3: the integration layer.

Replaces each DiT block's Attention.forward on a model CLONE (object patch, the same
mechanism the MiniMax-H3-Turbo node uses) with the official hybrid:

    softmax_out = window_softmax(roped q, k, v)          # local frames + globals
    out         = out_proj(softmax_gate(x) * softmax_out)
    out[video] += to_out_linear(branch(video rows))       # everything the window can't see

The base QKV projection, QK-norm and RoPE are reused verbatim from
comfy/ldm/minimax/model.py (the checkpoint's own weights), and the linear branch
consumes the raw pre-norm pre-RoPE q/k/v exactly like the official HybridAttention.

Per-forward packed-sequence geometry (video span, frame grid, text span) is published
by a DIFFUSION_MODEL wrapper that reads the payload's PackedLayout -- the same object
the model itself consumes.
"""
import logging
import queue
import threading

import torch
import torch.nn.functional as F

import comfy.ldm.minimax.model as minimax_model
import comfy.cli_args
import comfy.model_management
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.patcher_extension import WrappersMP

from vdn_h3.branch import LinearBranch
from vdn_h3.window import full_coverage, window_bounds

_log = logging.getLogger("comfy.vdn")
_seen = set()


def _once(key, message):
    if key not in _seen:
        _seen.add(key)
        _log.info(f"[vdn] {message}")


_RECORD_STREAM_NEEDED = None


def _record_stream_needed():
    """record_stream exists for torch's caching allocator; under cudaMallocAsync
    it is a no-op (the allocator is stream-ordered natively) and torch 2.10
    warns on every call."""
    global _RECORD_STREAM_NEEDED
    if _RECORD_STREAM_NEEDED is None:
        try:
            _RECORD_STREAM_NEEDED = (
                torch.cuda.get_allocator_backend() != "cudaMallocAsync")
        except Exception:
            _RECORD_STREAM_NEEDED = True
    return _RECORD_STREAM_NEEDED


class _StreamPrefetcher:
    """One-block lookahead for branch_weights="stream": while block i computes, a
    daemon thread reads block i+1's weights from the page cache to the GPU on its
    own (non-default) CUDA stream, so the H2D copy overlaps the compute instead of
    stalling it. The consumer stream waits on the copy's event before use;
    record_stream on the consumer keeps the allocator's reuse safe. At most one
    block in flight -- the extra residency is one block (~86 MB bf16 / 43 MB
    int8)."""

    def __init__(self):
        self._queue = queue.Queue(maxsize=1)
        self._done = {}
        self._inflight = set()
        self._lock = threading.Lock()
        self._gen = 0
        self._stream = None
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="vdn-branch-prefetch")
        self._thread.start()

    def request(self, index, fetch):
        with self._lock:
            if index in self._done or index in self._inflight:
                return
            gen = self._gen
            self._inflight.add(index)
        try:
            self._queue.put_nowait((gen, index, fetch))
        except queue.Full:
            with self._lock:
                self._inflight.discard(index)

    def _record(self, t, stream):
        """Mark every storage of t (plain or kitchen QuantizedTensor) as used on
        the consumer stream, so freeing it on the main thread can't be reused by
        the prefetch stream while the consumer is still reading."""
        if not _record_stream_needed():
            return
        seen = [t]
        inner = getattr(t, "_qdata", None)
        if inner is not None:
            seen.append(inner)
        params = getattr(t, "_params", None)
        for name in ("scale", "orig_weight", "bias"):
            sub = getattr(params, name, None)
            if isinstance(sub, torch.Tensor):
                seen.append(sub)
        for x in seen:
            try:
                x.record_stream(stream)
            except Exception:
                pass

    def _worker(self):
        while True:
            gen, index, fetch = self._queue.get()
            try:
                if gen != self._gen:
                    continue
                if self._stream is None:
                    self._stream = torch.cuda.Stream()
                with torch.cuda.stream(self._stream):
                    w = fetch()
                    ev = torch.cuda.Event()
                    ev.record(self._stream)
                with self._lock:
                    if gen == self._gen:
                        self._done[index] = (w, ev)
            except Exception as e:
                _log.warning("[vdn] branch prefetch failed (%s); the consumer "
                             "will read synchronously", e)
            finally:
                with self._lock:
                    self._inflight.discard(index)

    def take(self, index):
        with self._lock:
            hit = self._done.pop(index, None)
        if hit is None:
            return None
        w, ev = hit
        cur = torch.cuda.current_stream()
        cur.wait_event(ev)
        for t in w.values():
            self._record(t, cur)
        return w

    def reset(self):
        """Cancel pending/in-flight work (run interrupt): stale entries refer to
        weights of a cancelled forward and are dropped, not consumed."""
        with self._lock:
            self._gen += 1
            self._done.clear()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass


class VDNLayout:
    """Published once per forward: the packed-sequence geometry the branches need."""

    __slots__ = ("video_start", "video_end", "num_frames", "tokens_per_frame",
                 "frame_size", "text_start", "text_len", "bounds", "full_cover",
                 "seq_len", "anchor_frames")

    def __init__(self, video_start, video_end, num_frames, tokens_per_frame,
                 frame_size, text_start, text_len, seq_len, radius, chunk,
                 anchor_frames):
        self.video_start = video_start
        self.video_end = video_end
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.frame_size = frame_size
        self.text_start = text_start
        self.text_len = text_len
        self.seq_len = seq_len
        self.bounds = window_bounds(num_frames, radius, chunk)
        self.full_cover = full_coverage(self.bounds, num_frames)
        self.anchor_frames = anchor_frames


class VDNState:
    """Everything one Apply-VDN application owns: config, per-block branch weights,
    the per-forward layout, and the runtime weight-placement policy."""

    def __init__(self, name, cfg, branches, num_heads, head_dim):
        self.name = name
        self.cfg = cfg
        self.branches = branches              # [num_blocks] LinearBranch or None
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layout = None                    # published by the wrapper each forward
        self.cache_gpu = False
        self.retain_buffers = True            # auto-resolved at apply time
        self._gpu_cache = {}
        self._act = None                      # per-geometry activation scratch
        self._act_key = None
        self._prefetcher = None
        self.forwards = 0

    def act_scratch(self, video_rows, text_rows, device, dtype):
        """The raw pre-RoPE q/k/v copies the linear branch reads. Retained mode:
        one buffer set lives across blocks and is dropped after each block's
        readout (the allocator re-serves it, so only one set is ever live).
        Transient mode (VRAM pressure): fresh per block -- the v1.3.1 pattern.
        The branch never writes to these."""
        key = (video_rows, text_rows, self.num_heads, self.head_dim,
               str(device), dtype)
        if not self.retain_buffers:
            shape = (video_rows, self.num_heads, self.head_dim)
            tshape = (text_rows, self.num_heads, self.head_dim)
            return {"q": torch.empty(shape, device=device, dtype=dtype),
                    "k": torch.empty(shape, device=device, dtype=dtype),
                    "v": torch.empty(shape, device=device, dtype=dtype),
                    "tk": torch.empty(tshape, device=device, dtype=dtype),
                    "tv": torch.empty(tshape, device=device, dtype=dtype)}
        if self._act is None or self._act_key != key:
            shape = (video_rows, self.num_heads, self.head_dim)
            tshape = (text_rows, self.num_heads, self.head_dim)
            self._act = {"q": torch.empty(shape, device=device, dtype=dtype),
                         "k": torch.empty(shape, device=device, dtype=dtype),
                         "v": torch.empty(shape, device=device, dtype=dtype),
                         "tk": torch.empty(tshape, device=device, dtype=dtype),
                         "tv": torch.empty(tshape, device=device, dtype=dtype)}
            self._act_key = key
        return self._act

    def _prefetch(self):
        if self._prefetcher is None:
            self._prefetcher = _StreamPrefetcher()
        return self._prefetcher

    def weights_on(self, index, device, dtype):
        w = self.branches[index].w

        def fetch(t, copy=False):
            resolve = getattr(t, "resolve", None)
            if resolve is not None:
                return resolve(device, dtype)  # disk-backed: read straight to device
            return comfy.model_management.cast_to(t, dtype=dtype, device=device,
                                                  copy=copy)

        if self.cache_gpu:
            key = (index, str(device), str(dtype))
            hit = self._gpu_cache.get(key)
            if hit is None:
                hit = {k: fetch(t, copy=True) for k, t in w.items()}
                self._gpu_cache[key] = hit
            return hit
        if torch.device(device).type == "cuda":
            if not self.retain_buffers:
                # pressure mode: skip the prefetch side-stream too -- its pool
                # costs residency + fragmentation and only pays off when there
                # is headroom to spend
                return {k: fetch(t) for k, t in w.items()}
            # stream with a one-block lookahead: block i+1's page-cache->GPU copy
            # runs on the prefetch thread while block i computes. The chain wraps
            # around (last block prefetches block 0), so steps after the first
            # start with block 0 already in flight; the fetched weights are the
            # same disk tensors every forward, so a wrapped entry stays valid.
            pf = self._prefetch()
            hit = pf.take(index)
            if hit is None:
                hit = {k: fetch(t) for k, t in w.items()}
            nxt = (index + 1) % len(self.branches)
            if self.branches[nxt] is not None:
                wn = self.branches[nxt].w
                pf.request(nxt, lambda: {k: fetch(t) for k, t in wn.items()})
            return hit
        return {k: fetch(t) for k, t in w.items()}


def layout_from_payload(payload, x, context, cfg):
    """Rebuild/adopt the PackedLayout the model itself uses, and derive the VDN
    geometry from it. Mirrors MiniMaxH3Model._forward's shape handling (including the
    patch-size padding of the video latent)."""
    payload = payload or {}
    layout = payload.get("layout")
    video_x = x[0]
    padded = comfy.ldm.common_dit.pad_to_patch_size(video_x, (1, 2, 2))
    latent_t, lat_h, lat_w = padded.shape[2], padded.shape[3], padded.shape[4]
    audio_t = x[1].shape[-1]
    text_len = context.shape[1]
    signature = (text_len, latent_t, lat_h, lat_w, audio_t)
    if layout is None or layout.signature != signature:
        layout = minimax_model.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                            keyframes=payload.get("keyframes"),
                                            refs=payload.get("refs"))
    seg = next(s for s in layout.segments if s[2] == "video")
    text_seg = next(s for s in layout.segments if s[2] == "text")
    tokens_per_frame = (lat_h // 2) * (lat_w // 2)
    return VDNLayout(seg[0], seg[1], (seg[1] - seg[0]) // tokens_per_frame,
                     tokens_per_frame, (lat_h // 2, lat_w // 2),
                     text_seg[0], text_seg[1] - text_seg[0], layout.seq_len,
                     cfg["radius"], cfg["chunk"], cfg["anchor_frames"])


def make_layout_wrapper(state):
    """DIFFUSION_MODEL wrapper: publish the layout, run the model, clear it."""

    def wrap(executor, *args, **kwargs):
        # comfy builds with the model compiler crash on VDN forwards (the
        # malloc-graph planner cannot trace them); nodes.py flips the switch
        # off when the compiler stack exists, and we scope it to exactly this
        # forward so non-VDN workflows keep it.
        owns_switch = getattr(state, "owns_compiler_switch", False)
        if owns_switch:
            comfy.cli_args.args.disable_comfy_compiler = True
        state.layout = layout_from_payload(kwargs.get("minimax_payload"),
                                           args[0], args[2], state.cfg)
        state.forwards += 1
        lay = state.layout
        _once(("layout", lay.seq_len, lay.num_frames, lay.tokens_per_frame),
              f"layout: seq {lay.seq_len} rows, video [{lay.video_start}, "
              f"{lay.video_end}), F={lay.num_frames}, S={lay.tokens_per_frame}, "
              f"frame {lay.frame_size}, text {lay.text_len} rows, "
              f"window {'dense (full cover)' if lay.full_cover else lay.bounds[0]}")
        try:
            return executor(*args, **kwargs)
        except comfy.model_management.InterruptProcessingException:
            # A cancelled mid-run leaves this node's GPU cache behind and the
            # CUDA allocator pool fragmented; drop everything the node owns so
            # the next run starts clean instead of OOM-ing on its first big
            # activation. (The base model's own residency is comfy's to manage.)
            state._gpu_cache.clear()
            state._act = None
            state._act_key = None
            if state._prefetcher is not None:
                state._prefetcher.reset()
            from vdn_h3 import branch as _b, window as _w
            _b.clear_scan_banks()
            _w.clear_window_state()
            torch.cuda.empty_cache()
            raise
        finally:
            if owns_switch:
                comfy.cli_args.args.disable_comfy_compiler = False
            state.layout = None

    return wrap


def _base_attention(attn, x, rope_freqs, transformer_options):
    """comfy/ldm/minimax/model.py Attention.forward, verbatim (the dense teacher)."""
    s = x.shape[0]
    q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
    v = v.view(s, attn.heads, attn.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, attn.heads, attn.head_dim)
        k = k.view(1, s, attn.heads, attn.head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
        k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
    v = v.clone()
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(q, k, v, attn.heads, mask=None, skip_reshape=True,
                              transformer_options=transformer_options)
    return attn.out_proj(out.squeeze(0))


def make_vdn_forward(attn, state, block_index):
    """The object-patched Attention.forward for one DiT block."""
    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim
    qkv_proj, out_proj = attn.qkv_proj, attn.out_proj
    q_norm, k_norm = attn.q_norm, attn.k_norm
    branch = state.branches[block_index]
    cfg = state.cfg

    def vdn_forward(x, rope_freqs=None, transformer_options={}):
        lay = state.layout
        if lay is None or branch is None:
            return _base_attention(attn, x, rope_freqs, transformer_options)

        s = x.shape[0]
        device, dtype = x.device, x.dtype
        q, k, v = qkv_proj(x).split(inner, dim=-1)
        v = v.view(s, heads, head_dim)
        q_raw = q.view(s, heads, head_dim)
        k_raw = k.view(s, heads, head_dim)

        window_active = not lay.full_cover
        linear_active = window_active and cfg.get("linear_enabled", True)
        q_raw_video = k_raw_video = v_video = None
        text_x = text_k_raw = text_v_raw = None
        if linear_active:
            v_s, e_s = lay.video_start, lay.video_end
            buf = state.act_scratch(
                e_s - v_s,
                lay.text_len if branch.enable_text_state else 0, device, dtype)
            q_raw_video = buf["q"].copy_(q_raw[v_s:e_s])
            k_raw_video = buf["k"].copy_(k_raw[v_s:e_s])
            v_video = buf["v"].copy_(v[v_s:e_s])
            if branch.enable_text_state and lay.text_len:
                t_a, t_b = lay.text_start, lay.text_start + lay.text_len
                text_x = x[t_a:t_b]
                text_k_raw = buf["tk"].copy_(k_raw[t_a:t_b])
                text_v_raw = buf["tv"].copy_(v[t_a:t_b])

        if rope_freqs is not None:
            q4 = q.view(1, s, heads, head_dim)
            k4 = k.view(1, s, heads, head_dim)
            qw = comfy.model_management.cast_to(q_norm.weight, device=device)
            kw = comfy.model_management.cast_to(k_norm.weight, device=device)
            rot = rope_freqs.shape[-3] * 2
            comfy.quant_ops.ck.rms_rope_split_half_(
                q4, k4, rope_freqs, qw, kw, epsilon=q_norm.eps, rot_dim=rot)
            q = q4[0]
            k = k4[0]
        else:
            q = q_norm(q_raw)
            k = k_norm(k_raw)
        # v is NOT cloned: nothing downstream mutates it. The grouped window path
        # gathers k/v rows through index_select (which copies into contiguous
        # scratch), so the strided split view never reaches an SDPA kernel. The
        # two paths that feed v to an attention call directly get a contiguous
        # copy at the call site instead of one clone per block per step.

        if window_active:
            if getattr(state, "softmax_backend", "grouped") == "flex":
                from vdn_h3.window import window_softmax_flex
                try:
                    softmax_out = window_softmax_flex(
                        q, k, v.contiguous(), lay.video_start, lay.video_end,
                        lay.num_frames, lay.tokens_per_frame, lay.bounds,
                        head_dim ** -0.5, anchor_frames=cfg["anchor_frames"])
                except Exception as e:
                    state.softmax_backend = "grouped"
                    _log.warning("[vdn] flex attention failed (%s); falling back "
                                 "to grouped SDPA", e)
            if getattr(state, "softmax_backend", "grouped") != "flex":
                from vdn_h3.window import window_softmax_grouped
                # Windows always run exact SDPA: routing them through the model's
                # optimized_attention_override (sage/kitchen int8) measurably
                # softens output -- the released model validated exact local
                # attention. Overrides still apply to the base model's own
                # attention (text refiner, full-cover fallback).
                softmax_out = window_softmax_grouped(
                    q, k, v, lay.video_start, lay.video_end, lay.num_frames,
                    lay.tokens_per_frame, lay.bounds, head_dim ** -0.5,
                    anchor_frames=cfg["anchor_frames"],
                    retain_buffers=state.retain_buffers)
        else:
            q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
            k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
            v = AttentionTensorContainer(
                v.contiguous().transpose(0, 1).unsqueeze(0))
            softmax_out = optimized_attention(
                q, k, v, heads, mask=None, skip_reshape=True,
                transformer_options=transformer_options).squeeze(0)

        # The roped/raw projections are dead from here (~3 GiB at H3 scale, held
        # alive by views of the qkv_proj buffer); free them before the gate, the
        # out projection and the branch readout (the official inference body dels
        # them for the same reason). The branch works from the clones taken above.
        del q, k, v, q_raw, k_raw

        w = state.weights_on(block_index, device, dtype)
        if cfg["enable_softmax_gate"]:
            gate = torch.sigmoid(F.linear(x, w["softmax_gate.up.weight"],
                                          w["softmax_gate.up.bias"]))
            flat = (softmax_out * gate.view(s, heads, 1).to(softmax_out.dtype)) \
                .reshape(s, -1)
        else:
            flat = softmax_out.reshape(s, -1)
        out = out_proj(flat.type_as(x))
        del softmax_out

        if linear_active:
            readout = branch.readout(
                w, x[lay.video_start:lay.video_end], q_raw_video, k_raw_video,
                v_video, lay.num_frames, lay.tokens_per_frame, lay.bounds,
                frame_size=lay.frame_size, text_x=text_x, text_k_raw=text_k_raw,
                text_v_raw=text_v_raw, skip_ends=(cfg["anchor_frames"] == "both"))
            # release the block-local activation scratch: the readout was its
            # last consumer, and the allocator re-serves the same block next
            # time (no churn, no residency past one block)
            state._act = None
            out[lay.video_start:lay.video_end] += F.linear(
                readout.type_as(x), w["to_out_linear.weight"])
        return out

    vdn_forward._vdn_forward = True
    return vdn_forward


def apply_vdn(new_model, state):
    """Install the layout wrapper and one object patch per DiT block on a cloned
    ModelPatcher."""
    dm = new_model.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if blocks is None or not hasattr(getattr(blocks[0], "attn", None), "qkv_proj"):
        raise RuntimeError(
            "ApplyVDNH3: the MODEL's diffusion model is not a ComfyUI MiniMax-H3 "
            "(expected blocks[].attn.qkv_proj). Load a MiniMax-H3 checkpoint first.")
    if len(blocks) != len(state.branches):
        raise RuntimeError(
            f"ApplyVDNH3: checkpoint has {len(state.branches)} blocks but the loaded "
            f"model has {len(blocks)}; the VDN checkpoint and the base model do not "
            "belong together.")
    for i, block in enumerate(blocks):
        new_model.add_object_patch(
            f"diffusion_model.blocks.{i}.attn.forward",
            make_vdn_forward(block.attn, state, i))
    new_model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "vdn_h3",
                                   make_layout_wrapper(state))
