"""VDN-H3 linear-attention branch (Video Delta Attention) for ComfyUI's MiniMax-H3.

Port of the official release's BidirectionalLinearBranch
(github.com/OpenVDN/vdn-minimax-h3, src/models/linear_attention/) with the checkpoint
held as plain tensors instead of a module tree, so ComfyUI's model patcher stays the
sole owner of the diffusion model's parameter tree.

The released 8-step checkpoint configuration: delta_rule="vdn_solve", bridge="alpha",
a_fp32=True, enable_text_state=True, short_conv on (k, v), linear_head_dim=128.
Everything here is eager PyTorch -- no Triton, no torch.compile, no CUDA kernels.
Numerics follow the reference inference bodies: A statistics in fp32 (TF32 GEMM), the
recurrence in fp32 via preallocated banks, bf16 features and readout.
"""
import collections
import logging
import math

import torch
import torch.nn.functional as F

_log = logging.getLogger("comfy.vdn")


# ---------------------------------------------------------------- delta rules --

class VdnDelta:
    """S_out = (S_in Diag(alpha) + B)(I + A)^{-1}, the rule the released checkpoints
    use. I + A is SPD: one batched Cholesky on cuBLAS/cuSOLVER, then L^{-T}L^{-1} as a
    triangular solve and a matmul (cheaper than two triangular solves at 128x128)."""

    def __init__(self, tokens_per_frame=None):
        pass

    def factor_apply(self, alpha, a_raw, b_raw):
        a32 = a_raw.float()
        eye = torch.eye(a32.shape[-1], device=a32.device,
                        dtype=torch.float32).expand_as(a32)
        chol = torch.linalg.cholesky(a32 + eye)
        linv = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
        inv = linv.transpose(-1, -2) @ linv
        transition = alpha.unsqueeze(-1) * inv
        injection = b_raw.float() @ inv
        return transition.to(a_raw.dtype), injection.to(b_raw.dtype)


class SanaDelta:
    """Scaled subtractive delta: S_out = (S_in Diag(D))(I - c^2 A) + c B."""

    def __init__(self, tokens_per_frame):
        self.inv_tokens = 1.0 / tokens_per_frame
        self.inv_sqrt_tokens = self.inv_tokens ** 0.5

    def factor_apply(self, alpha, a_raw, b_raw):
        eye = torch.eye(a_raw.shape[-1], device=a_raw.device, dtype=a_raw.dtype)
        transition = alpha.unsqueeze(-1) * (eye - self.inv_tokens * a_raw)
        injection = self.inv_sqrt_tokens * b_raw
        return transition, injection


class VdnScaledDelta(VdnDelta):
    """Exact joint solve WITH SANA's key scaling:
    S_out = (S_in Diag(D) + cB)(I + c^2 A)^-1, c = 1/sqrt(S).

    A control arm, kept for interpretability, not to train with: once c^2 = 1/S
    forces trace(c^2 A) <= 1, the exact inverse and the first-order truncation
    (I - c^2 A) are very nearly the same operator. spec.py accepts the rule, so
    a checkpoint that names it must find it here."""

    def __init__(self, tokens_per_frame):
        super().__init__(tokens_per_frame)
        self.inv_tokens = 1.0 / tokens_per_frame              # c^2
        self.inv_sqrt_tokens = self.inv_tokens ** 0.5         # c

    def factor_apply(self, alpha, a_raw, b_raw):
        a32 = a_raw.float() * self.inv_tokens
        eye = torch.eye(a32.shape[-1], device=a32.device,
                        dtype=torch.float32).expand_as(a32)
        chol = torch.linalg.cholesky(a32 + eye)
        inv = torch.cholesky_solve(eye.contiguous(), chol)
        transition = alpha.unsqueeze(-1) * inv
        injection = (b_raw.float() * self.inv_sqrt_tokens) @ inv
        return transition.to(a_raw.dtype), injection.to(b_raw.dtype)


DELTA_BACKENDS = {"vdn_solve": VdnDelta, "sana_scaled": SanaDelta,
                  "vdn_scaled": VdnScaledDelta}

TEXT_STATE_SCALE = 0.5


# ------------------------------------------------------------- frame statistics --

def _tf32_matmul():
    class _Ctx:
        def __enter__(self):
            self.prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

        def __exit__(self, *a):
            torch.backends.cuda.matmul.allow_tf32 = self.prev
    return _Ctx()


def frame_statistics(kf, vf, beta, a_fp32=True):
    """A[f,h,k,l] = sum_s k beta k,  B[f,h,v,k] = sum_s v beta k, over one chunk's
    rows. A in fp32 (bf16's 8 mantissa bits break the conditioning I+A needs), B left
    in bf16 for the tensor-core GEMM and promoted on the store. Operates with autocast
    off implicitly -- callers run under inference no_grad, no ambient autocast."""
    with torch.autocast(device_type=kf.device.type, enabled=False):
        kf16 = kf.contiguous()
        kf32 = kf16.float()
        scaled32 = (kf32 * beta.unsqueeze(-1).float()).contiguous()
        vb = (vf * beta.unsqueeze(-1).to(vf.dtype)).contiguous()
        if a_fp32:
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                a = torch.matmul(scaled32.transpose(-1, -2), kf32)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev
        else:
            a = torch.matmul((kf * beta.unsqueeze(-1).to(kf.dtype)).contiguous()
                             .transpose(-1, -2), kf).float()
        a = 0.5 * (a + a.transpose(-1, -2))
        b = torch.matmul(vb.transpose(-1, -2), kf).float()
        return a, b


# ------------------------------------------------------- compile small helpers --

_COMPILED_CACHE = {}
_COMPILED_BROKEN = set()


def _run_compiled(key, body, *args, **kwargs):
    """torch.compile(body, dynamic=False), built once per key, with a permanent
    eager fallback on failure -- the same policy linear_epilogue already uses:
    same math, one rounding at the store instead of one per op, just slower."""
    if key in _COMPILED_BROKEN:
        return body(*args, **kwargs)
    try:
        if key not in _COMPILED_CACHE:
            _COMPILED_CACHE[key] = torch.compile(body, dynamic=False)
        return _COMPILED_CACHE[key](*args, **kwargs)
    except Exception as e:
        _COMPILED_BROKEN.add(key)
        _log.warning("[vdn] compile of %s failed (%s); using eager", key, e)
        return body(*args, **kwargs)


# ---------------------------------------------------------------------- scans --

def run_scans(backend, alpha, a_raw, b_raw, text_state=None):
    """Forward/reverse state banks; the plain linear recurrence
    state_t = state_{t-1} @ transition_t + injection_t, one baddbmm per frame."""
    with torch.autocast(device_type=a_raw.device.type, enabled=False):
        transitions, injections = backend.factor_apply(alpha, a_raw, b_raw)
        num_frames = transitions.shape[0]
        start = (torch.zeros_like(injections[0]) if text_state is None
                 else text_state.to(injections.dtype))
        prefix = torch.empty((num_frames, *start.shape), dtype=injections.dtype,
                             device=injections.device)
        suffix = torch.empty_like(prefix)
        state = start
        for frame in range(num_frames):
            torch.baddbmm(injections[frame], state, transitions[frame],
                          out=prefix[frame])
            state = prefix[frame]
        state = start
        for frame in range(num_frames - 1, -1, -1):
            torch.baddbmm(injections[frame], state, transitions[frame],
                          out=suffix[frame])
            state = suffix[frame]
        return prefix, suffix


MAX_CACHED_GATHERS = 64
_GATHER_INDEX_CACHE = collections.OrderedDict()


def gather_indices(bounds, num_frames, device):
    key = (tuple(bounds), num_frames, str(device))
    hit = _GATHER_INDEX_CACHE.get(key)
    if hit is not None:
        _GATHER_INDEX_CACHE.move_to_end(key)
        return hit
    last_before = torch.tensor([lo for lo, _ in bounds], device=device) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=device) + 1
    hit = dict(
        before_idx=last_before.clamp(min=0),
        after_idx=first_after.clamp(max=num_frames - 1),
        has_before=(last_before >= 0),
        has_after=(first_after < num_frames),
        bridge_before=(last_before + 1).clamp(min=0),
        bridge_after=first_after.clamp(max=num_frames),
        frames=torch.arange(num_frames, device=device),
    )
    _GATHER_INDEX_CACHE[key] = hit
    while len(_GATHER_INDEX_CACHE) > MAX_CACHED_GATHERS:
        _GATHER_INDEX_CACHE.popitem(last=False)
    return hit


def _gather_body(prefix_states, suffix_states, alpha, text_state, bridge_alpha,
                 out_dtype, before_idx, after_idx, has_before, has_after,
                 bridge_before, bridge_after, frames):
    """The arithmetic of gather_linear_state, with the index tensors already built.

    Split out so fast_kernels can hand the whole thing to one compiled kernel:
    eager it is two gathers, two wheres, two multiplies and a combine over the
    fp32 state bank -- seven passes for what is one read of each side and one
    store."""
    state_before = prefix_states[before_idx]
    state_after = suffix_states[after_idx]
    if text_state is not None:
        text_state = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before,
                                   text_state)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after,
                                  text_state)
    if bridge_alpha:
        log_alpha = torch.log(alpha.clamp_min(1e-12))
        log_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
        alpha_from_before = torch.exp(
            log_prefix[frames + 1] - log_prefix[bridge_before])
        alpha_from_after = torch.exp(
            log_prefix[bridge_after] - log_prefix[frames])
        # alpha is per KEY channel: broadcast over d_v, not d_k
        state_before = state_before * alpha_from_before.unsqueeze(2)
        state_after = state_after * alpha_from_after.unsqueeze(2)
    if text_state is not None:
        out = state_before + state_after
    else:
        out = (state_before * has_before.view(-1, 1, 1, 1)
               + state_after * has_after.view(-1, 1, 1, 1))
    return out if out_dtype is None else out.to(out_dtype)


def gather_linear_state(prefix_states, suffix_states, alpha, bounds, bridge="alpha",
                        text_state=None, out_dtype=None, fuse=False):
    """The state of everything OUTSIDE the softmax window, in the query frame's frame
    of reference: prefix_states[lo-1] + suffix_states[hi+1], decayed in by the product
    of alpha over the window span (bridge="alpha"), with the scan start (the text
    state, when given) read by out-of-range sides.

    fuse=True (fast_kernels) runs the arithmetic as one compiled kernel, keyed on
    (bridge, text_state?, out_dtype); same math, rounded once at the store."""
    assert bridge in ("alpha", "none")
    num_frames = prefix_states.shape[0]
    idx = gather_indices(bounds, num_frames, prefix_states.device)
    if not fuse:
        return _gather_body(prefix_states, suffix_states, alpha, text_state,
                            bridge == "alpha", out_dtype, **idx)
    key = ("gather", bridge, text_state is not None, str(out_dtype))
    return _run_compiled(key, _gather_body, prefix_states, suffix_states, alpha,
                         text_state, bridge == "alpha", out_dtype, **idx)


# -------------------------------------------------------------------- features --

def _activate(tokens, l2norm):
    x = F.silu(tokens)
    if l2norm:
        return F.normalize(x, dim=-1, eps=1e-6).to(x.dtype)
    return x


def _activate_fhsd_body(tokens, l2norm, num_frames, per_frame):
    """_activate storing q frame-major, [F, H, S, d] instead of [F*S, H, d].

    The readout below is a frame-major batched matmul; storing q this way (the
    official inference body) means the matmul consumes it without a permute-in
    copy. Only pays off compiled, where the strided store rides the activation
    kernel for free -- so callers gate it behind fast_kernels."""
    x = _activate(tokens, l2norm)
    heads, dim = x.shape[-2], x.shape[-1]
    return x.view(num_frames, per_frame, heads, dim).permute(0, 2, 1, 3).contiguous()


def _temporal_shift(x, w, kernel):
    """Depthwise k-tap conv over frames as shift-multiply-add. x [F, S, C]; w [C, k];
    zero-padded, symmetric."""
    pad = kernel // 2
    xp = F.pad(x, (0, 0, 0, 0, pad, pad))
    out = None
    for dt in range(kernel):
        part = xp[dt:dt + x.shape[0]] * w[:, dt].view(1, 1, -1)
        out = part if out is None else out + part
    return out


def conv_features(tokens, sp_weight, tm_weight, num_frames, frame_size, l2norm):
    """Separable short conv: depthwise 5x5 spatial per frame (cudnn NHWC via a
    channels-last view), then the 5-tap temporal shift, then SiLU [+ L2Norm]."""
    heads, head_dim = tokens.shape[-2], tokens.shape[-1]
    grid_h, grid_w = frame_size
    channels = heads * head_dim
    volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
    volume = F.conv2d(volume, sp_weight, padding=2, groups=channels)
    x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
    tm = tm_weight.squeeze(1)                     # Conv1d [C, 1, K] -> [C, K]
    out = _temporal_shift(x, tm.to(x.dtype), tm.shape[-1])
    return _activate(out.reshape(-1, heads, head_dim), l2norm)


def alpha_gate(frame_mean, w_down, w_up, dt_bias, a_log, num_heads, head_dim):
    """alpha_t = exp(-exp(A_log) * softplus(delta + dt_bias)) per frame/head/channel,
    KDA's double-exponential gate in fla layout. fp32 throughout."""
    with torch.autocast(device_type=frame_mean.device.type, enabled=False):
        delta = F.linear(frame_mean.float(), w_down.float())
        delta = F.linear(delta, w_up.float())
        delta = delta + dt_bias.float()
        scale = torch.exp(a_log.float())[:, None]
        delta = delta.view(-1, num_heads, head_dim)
        return torch.exp(-scale * F.softplus(delta.float()))


def rms_norm(x, weight, eps):
    """Weighted RMSNorm with fp32 second-moment accumulation (vector_norm spelling)."""
    ms = torch.linalg.vector_norm(
        x, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / x.shape[-1]
    return x * torch.rsqrt(ms + eps).to(x.dtype) * weight.to(x.dtype)


def _linear_epilogue_body(readout_fhsd, norm_weight, gate, eps):
    """RMSNorm + output gate over a readout still in [F, H, S, d], with the transpose
    back to token order folded into the store."""
    ms = torch.linalg.vector_norm(
        readout_fhsd, dim=-1, keepdim=True, dtype=torch.float32).pow(2) \
        / readout_fhsd.shape[-1]
    normed = readout_fhsd * torch.rsqrt(ms + eps).to(readout_fhsd.dtype) \
        * norm_weight.to(readout_fhsd.dtype)
    frames, heads, per_frame, dim = normed.shape
    rows = frames * per_frame
    return (normed.permute(0, 2, 1, 3).reshape(rows, heads * dim)
            * gate.reshape(rows, heads * dim))


_EPILOGUE_FUSED = None
_EPILOGUE_FUSED_BROKEN = False


def linear_epilogue(readout_fhsd, norm_weight, gate, eps, fuse=False):
    """RMSNorm + output gate, optionally under torch.compile. Eager this walks the
    full readout several times (norm, rsqrt, two multiplies, the gated store); the
    fused variant is one inductor kernel. Compilation failure falls back to eager
    permanently (same math, just slower)."""
    global _EPILOGUE_FUSED, _EPILOGUE_FUSED_BROKEN
    if fuse and not _EPILOGUE_FUSED_BROKEN:
        try:
            if _EPILOGUE_FUSED is None:
                _EPILOGUE_FUSED = torch.compile(_linear_epilogue_body)
            return _EPILOGUE_FUSED(readout_fhsd, norm_weight, gate, eps)
        except Exception as e:
            _EPILOGUE_FUSED_BROKEN = True
            _log.warning("[vdn] fused epilogue compile failed (%s); using eager", e)
    return _linear_epilogue_body(readout_fhsd, norm_weight, gate, eps)


# ---------------------------------------------------------------- the branch --

class LinearBranch:
    """The checkpoint-backed linear-attention branch for ONE transformer block.

    Weights are plain CPU tensors under `w` (checkpoint keys minus the per-block
    prefix). Call `readout(...)` inside the block's attention forward; it consumes the
    raw (pre-QK-norm, pre-RoPE) q/k/v of the video rows and the hidden states, and
    returns the gated readout [video_rows, H*d_linear] pre-to_out_linear.
    """

    def __init__(self, w, num_heads, head_dim, delta_rule="vdn_solve", bridge="alpha",
                 a_fp32=True, short_conv=("k", "v"), enable_text_state=True):
        self.w = w
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bridge = bridge
        self.a_fp32 = a_fp32
        self.short_conv = tuple(short_conv) or None
        self.enable_text_state = enable_text_state
        self.delta_rule = delta_rule
        self.fuse_epilogue = False
        self._backend = None
        self._backend_key = None

    def _features(self, w, q_raw, k_raw, v_raw, num_frames, frame_size, q_fhsd=False):
        """[ShortConv ->] SiLU [-> L2Norm for q/k]. NoPE: the branch consumes raw
        pre-RoPE features. q_fhsd (fast_kernels) stores q frame-major [F, H, S, d]
        straight out of the fused activation; n/a when q itself is convolved."""
        conv = self.short_conv
        if q_fhsd and not (conv and "q" in conv):
            query = _run_compiled(("act_fhsd", True), _activate_fhsd_body, q_raw,
                                  True, num_frames, q_raw.shape[0] // num_frames)
        else:
            query = _activate(q_raw, l2norm=True)
        if conv and "k" in conv:
            key = conv_features(k_raw, w["short_conv.k_sp.weight"],
                                w["short_conv.k_tm.weight"], num_frames, frame_size,
                                l2norm=True)
        else:
            key = _activate(k_raw, l2norm=True)
        if conv and "v" in conv:
            value = conv_features(v_raw, w["short_conv.v_sp.weight"],
                                  w["short_conv.v_tm.weight"], num_frames, frame_size,
                                  l2norm=False)
        else:
            value = _activate(v_raw, l2norm=False)
        return query, key, value

    def _delta_backend(self, tokens_per_frame):
        key = (self.delta_rule, tokens_per_frame)
        if self._backend is None or self._backend_key != key:
            self._backend = DELTA_BACKENDS[self.delta_rule](tokens_per_frame)
            self._backend_key = key
        return self._backend

    def _text_state(self, w, text_x, text_k_raw, text_v_raw):
        """TEXT_STATE_SCALE * S_text: the whole prompt written into a zero state as ONE
        delta-rule chunk; both directional scans start from it."""
        if not self.enable_text_state or text_x is None:
            return None
        length = text_x.shape[0]
        n_heads, head_dim = self.num_heads, self.head_dim
        key = _activate(text_k_raw, l2norm=True)
        value = _activate(text_v_raw, l2norm=False)
        key = key.view(1, length, n_heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, n_heads, head_dim).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(text_x, w["beta_proj.weight"]))
        beta = beta.view(1, length, n_heads).permute(0, 2, 1)
        a, b = frame_statistics(key, value, beta, a_fp32=self.a_fp32)
        backend = self._delta_backend(length)
        with torch.autocast(device_type=a.device.type, enabled=False):
            ones = torch.ones(1, n_heads, head_dim, device=a.device, dtype=a.dtype)
            _, injection = backend.factor_apply(ones, a, b)
        return TEXT_STATE_SCALE * injection[0]

    def readout(self, w, xv, q_raw, k_raw, v_raw, num_frames, tokens_per_frame,
                bounds, frame_size=None, text_x=None, text_k_raw=None,
                text_v_raw=None, skip_ends=False):
        """Everything the softmax window cannot see, summarised for every video row.

        w: the branch weights, already moved to the activations' device/dtype (see
        VDNState.weights_on). xv: [video_rows, hidden]; q/k/v_raw: [video_rows, H, d]
        raw features. bounds: per-frame inclusive window [lo, hi]. Returns
        [video_rows, H*d_linear] (gated + normalised; the caller adds
        to_out_linear(...) into the attention output's video rows).
        """
        n_heads, head_dim = self.num_heads, self.head_dim
        ref = xv

        if skip_ends:
            if num_frames <= 2:
                return ref.new_zeros(num_frames * tokens_per_frame, n_heads * head_dim)
            inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
            readout = self._readout(
                w, xv[inner] if xv is not None else None,
                tuple(t[inner] for t in (q_raw, k_raw, v_raw)),
                num_frames - 2, tokens_per_frame,
                [(lo - 1, hi - 1) for lo, hi in bounds[1:num_frames - 1]],
                frame_size, text_x, text_k_raw, text_v_raw)
            out = readout.new_empty(num_frames * tokens_per_frame, readout.shape[-1])
            out[:tokens_per_frame].zero_()
            out[(num_frames - 1) * tokens_per_frame:].zero_()
            out[inner] = readout
            return out
        return self._readout(w, xv, (q_raw, k_raw, v_raw), num_frames,
                             tokens_per_frame, bounds, frame_size, text_x,
                             text_k_raw, text_v_raw)

    def _readout(self, w, xv, qkv_raw, num_frames, tokens_per_frame, bounds,
                 frame_size, text_x, text_k_raw, text_v_raw):
        n_heads, head_dim = self.num_heads, self.head_dim
        num_tokens = num_frames * tokens_per_frame
        backend = self._delta_backend(tokens_per_frame)
        shape = (num_frames, tokens_per_frame, n_heads, head_dim)

        query, key, value = self._features(w, *qkv_raw, num_frames, frame_size,
                                           q_fhsd=self.fuse_epilogue)
        key_by_frame = key.view(shape).permute(0, 2, 1, 3)       # [F, H, S, d]
        value_by_frame = value.view(shape).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(xv, w["beta_proj.weight"]))
        beta = beta.view(num_frames, tokens_per_frame, n_heads).permute(0, 2, 1)

        a, b = frame_statistics(key_by_frame, value_by_frame, beta, a_fp32=self.a_fp32)

        # fp32 on the mean, not just inside alpha: bf16 rounding before the fp32 island
        # would throw away what alpha's fp32 math cannot recover
        frame_mean = xv.view(num_frames, tokens_per_frame, -1).mean(
            dim=1, dtype=torch.float32)
        alpha = alpha_gate(frame_mean, w["alpha.down.weight"], w["alpha.up.weight"],
                           w["alpha.dt_bias"], w["alpha.A_log"], n_heads, head_dim)

        text_state = self._text_state(w, text_x, text_k_raw, text_v_raw)
        prefix_states, suffix_states = run_scans(backend, alpha, a, b,
                                                text_state=text_state)
        gate = torch.sigmoid(F.linear(xv, w["output_gate.down.weight"])
                             @ w["output_gate.up.weight"].T
                             + w["output_gate.up.bias"])
        linear_state = gather_linear_state(
            prefix_states, suffix_states, alpha, bounds, bridge=self.bridge,
            text_state=text_state, out_dtype=gate.dtype, fuse=self.fuse_epilogue)
        del prefix_states, suffix_states

        # q is [F, S, H, d]; the readout is frame-major, so align to [F, H, S, dk]
        # before the batched matmul. fast_kernels stored q frame-major already
        # (the official inference body), skipping the permute-in copy.
        if query.dim() == 4:
            query_fhsd = query                                   # [F, H, S, d]
        else:
            query_fhsd = query.view(shape).permute(0, 2, 1, 3)
        readout = torch.matmul(query_fhsd, linear_state.transpose(-1, -2))
        return linear_epilogue(readout, w["norm.weight"], gate,
                               w["norm.weight"].new_tensor(1e-6).item(),
                               fuse=self.fuse_epilogue)
