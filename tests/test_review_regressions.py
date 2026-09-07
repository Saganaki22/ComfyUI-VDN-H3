"""Checkpoint mapping, short-clip gating, and inference buffer lifetime."""
import weakref
import queue
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from comfy.ldm.modules.attention import attention_pytorch
from vdn_h3 import adapters, branch, hybrid, nodes, spec


@pytest.mark.parametrize("name", ["default", "turbo"])
def test_refiner_attention_adapters(name):
    torch.manual_seed(10)
    sd = {}
    deltas = []
    for projection in ("to_q", "to_k", "to_v", "to_out.0"):
        stem = f"token_refiner.refiner_blocks.0.attn.{projection}"
        a, b = torch.randn(3, 8), torch.randn(8, 3)
        sd[f"{stem}.lora_A.{name}.weight"] = a
        sd[f"{stem}.lora_B.{name}.weight"] = b
        deltas.append(b @ a)
    converted = adapters.convert_adapter(sd, {"config": {"rank": 3, "alpha": 6}})
    assert set(converted) == {"token_refiner.blocks.0.attn.qkv_proj",
                              "token_refiner.blocks.0.attn.out_proj"}
    for path, expected in (("qkv_proj", torch.cat(deltas[:3])),
                           ("out_proj", deltas[3])):
        a, b, scale = converted[f"token_refiner.blocks.0.attn.{path}"]
        torch.testing.assert_close(scale * (b @ a), 2 * expected)


@pytest.mark.parametrize("frame_major", [False, True])
def test_query_short_conv_is_applied(frame_major):
    torch.manual_seed(13)
    q = torch.randn(12, 2, 4)
    # A zero checkpoint convolution must produce zero Q features, not raw SiLU(Q).
    w = {"short_conv.q_sp.weight": torch.zeros(8, 1, 5, 5),
         "short_conv.q_tm.weight": torch.ones(8, 1, 5)}
    lin = branch.LinearBranch(w, 2, 4, short_conv=("q",))
    got, _, _ = lin._features(w, q, q, q, 3, (2, 2), q_fhsd=frame_major)
    assert torch.count_nonzero(got) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_temporal_shift_preserves_rounding(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(14)
    x = torch.randn(7, 15, 64, device=device, dtype=dtype)
    w = torch.randn(64, 5, device=device, dtype=dtype)
    padded = F.pad(x, (0, 0, 0, 0, 2, 2))
    expected = padded[:7] * w[:, 0]
    for tap in range(1, 5):
        expected = expected + padded[tap:tap + 7] * w[:, tap]
    got = branch._temporal_shift(x, w, 5)
    assert torch.equal(got, expected)


def test_quantized_branch_requires_scale(tmp_path, monkeypatch):
    path = tmp_path / "branch.safetensors"
    conf = b'{"format":"int8_tensorwise","convrot":true}'
    save_file({"proj.weight": torch.ones(2, 256, dtype=torch.int8),
               "proj.comfy_quant": torch.tensor(list(conf), dtype=torch.uint8)}, str(path))
    monkeypatch.setattr(spec, "_HANDLES", {})
    with pytest.raises(ValueError, match="missing proj.weight_scale"):
        spec._lazy_branch_sd(str(path))
    spec._HANDLES.clear()


def _attention():
    return SimpleNamespace(heads=2, head_dim=4,
                           qkv_proj=torch.nn.Linear(8, 24, bias=False),
                           out_proj=torch.nn.Linear(8, 8, bias=False),
                           q_norm=torch.nn.RMSNorm(4, eps=1e-6),
                           k_norm=torch.nn.RMSNorm(4, eps=1e-6))


@pytest.mark.parametrize("gated", [False, True])
def test_full_coverage_attention(gated, monkeypatch):
    torch.manual_seed(11)
    attn = _attention()
    cfg = {"enable_softmax_gate": gated, "anchor_frames": "both"}
    state = hybrid.VDNState("test", cfg, [SimpleNamespace()], 2, 4)
    state.layout = hybrid.VDNLayout(2, 8, 3, 2, (1, 2), 0, 2, 9, 1, 5, "both")
    assert state.layout.full_cover
    weights = {"softmax_gate.up.weight": torch.randn(2, 8),
               "softmax_gate.up.bias": torch.randn(2)}
    state.weights_on = lambda *args: weights
    monkeypatch.setattr(hybrid, "optimized_attention", attention_pytorch)
    x = torch.randn(9, 8)
    with torch.inference_mode():
        q, k, v = attn.qkv_proj(x).reshape(9, 3, 2, 4).unbind(1)
        q, k = attn.q_norm(q), attn.k_norm(k)
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        expected = expected.transpose(0, 1)
        if gated:
            gate = F.linear(x, weights["softmax_gate.up.weight"],
                            weights["softmax_gate.up.bias"]).sigmoid()
            expected = expected * gate.unsqueeze(-1)
        expected = attn.out_proj(expected.reshape(9, 8))
        got = hybrid.make_vdn_forward(attn, state, 0)(x)
    torch.testing.assert_close(got, expected)


def test_hybrid_releases_projection_buffers(monkeypatch):
    attn = _attention()
    refs = {}
    attn.qkv_proj.register_forward_hook(
        lambda module, args, out: refs.update(qkv=weakref.ref(out)))
    attn.out_proj.register_forward_pre_hook(
        lambda module, args: refs.update(flat=weakref.ref(args[0])))

    def readout(w, xv, *args, **kwargs):
        assert refs["qkv"]() is None, "RoPE views retained the QKV allocation"
        assert refs["flat"]() is None, "softmax projection input survived into branch"
        refs["raw"] = [weakref.ref(t) for t in args[:3]]
        return torch.zeros(xv.shape[0], 8)

    lin = SimpleNamespace(enable_text_state=False, readout=readout)
    cfg = {"enable_softmax_gate": True, "anchor_frames": "both"}
    state = hybrid.VDNState("test", cfg, [lin], 2, 4)
    state.layout = hybrid.VDNLayout(2, 24, 11, 2, (1, 2), 0, 2, 25, 1, 5, "both")
    weights = {
        "softmax_gate.up.weight": torch.randn(2, 8),
        "softmax_gate.up.bias": torch.randn(2),
        "to_out_linear.weight": torch.randn(8, 8)}
    state.weights_on = lambda *args: weights
    original_linear = F.linear

    def linear(x, weight, *args, **kwargs):
        if weight is weights["to_out_linear.weight"]:
            assert all(ref() is None for ref in refs["raw"]), "raw QKV survived into final projection"
        return original_linear(x, weight, *args, **kwargs)

    monkeypatch.setattr(F, "linear", linear)
    # Exercise the lifetime of the RoPE views; the CUDA kernel itself is external.
    monkeypatch.setattr(hybrid.comfy.quant_ops.ck, "rms_rope_split_half_",
                        lambda *args, **kwargs: None)
    with torch.inference_mode():
        got = hybrid.make_vdn_forward(attn, state, 0)(
            torch.randn(25, 8), rope_freqs=torch.zeros(1, 1, 2))
    assert got.shape == (25, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity check")
@pytest.mark.parametrize("tokens", [64, 920])
def test_frame_statistics_cuda(tokens):
    torch.manual_seed(12)
    shape = (4, tokens, 56, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    v = torch.randn_like(k)
    beta = torch.rand(4, 56, tokens, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        _, got = branch.frame_statistics(k, v, beta)
        # Original strided B GEMM: repacking must preserve every output bit.
        vb = (v * beta.unsqueeze(-1)).contiguous()
        expected = (vb.transpose(-1, -2) @ k).float()
    assert torch.equal(got, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("a_fp32", [False, True])
def test_frame_statistics_batches_preserve_results(device, dtype, a_fp32, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(15)
    k = torch.randn(7, 23, 3, 16, device=device, dtype=dtype).permute(0, 2, 1, 3)
    v = torch.randn_like(k)
    beta = torch.rand(7, 3, 23, device=device, dtype=dtype)
    reference = branch._frame_statistics_chunk(k, v, beta, a_fp32)
    per_frame = 3 * 23 * 16 * (3 * k.element_size() + (8 if a_fp32 else k.element_size()))
    monkeypatch.setattr(branch, "_STATISTICS_WORKSPACE_BYTES", 3 * per_frame)
    sizes = []
    original = branch._frame_statistics_chunk

    def chunk(k, v, beta, a_fp32):
        sizes.append(k.shape[0])
        return original(k, v, beta, a_fp32)

    monkeypatch.setattr(branch, "_frame_statistics_chunk", chunk)
    got = branch.frame_statistics(k, v, beta, a_fp32)
    assert sizes == [3, 3, 1]
    assert all(torch.equal(a, b) for a, b in zip(got, reference))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_readout_epsilon_preserves_rounding(dtype):
    expected = torch.tensor(1e-6, dtype=dtype).item()
    assert branch._readout_eps(dtype) == expected
    if torch.cuda.is_available():
        # Unlike Tensor.item() on a device tensor, this is legal during capture.
        graph = torch.cuda.CUDAGraph()
        x = torch.zeros(8, device="cuda")
        with torch.cuda.graph(graph):
            result = x + branch._readout_eps(dtype)
        graph.replay()
        torch.testing.assert_close(result, torch.full_like(x, expected))


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("failure", [None, "layout", "forward"])
def test_compiler_flag_restored(previous, failure, monkeypatch):
    state = hybrid.VDNState("test", {}, [], 2, 4)
    state.owns_compiler_switch = True
    monkeypatch.setattr(hybrid.comfy.cli_args.args, "disable_comfy_compiler", previous)

    def layout(*args):
        assert hybrid.comfy.cli_args.args.disable_comfy_compiler
        if failure == "layout":
            raise ValueError("layout failure")
        return hybrid.VDNLayout(2, 8, 3, 2, (1, 2), 0, 2, 9, 1, 5, "both")

    def execute(*args):
        assert hybrid.comfy.cli_args.args.disable_comfy_compiler
        if failure == "forward":
            raise ValueError("forward failure")
        return "done"

    monkeypatch.setattr(hybrid, "layout_from_payload", layout)
    wrapped = hybrid.make_layout_wrapper(state)

    def model_forward(*args):
        # This check is before the inner DIFFUSION_MODEL wrapper, like the
        # allocation-graph decision in MiniMaxH3Model.forward.
        assert hybrid.comfy.cli_args.args.disable_comfy_compiler
        result = wrapped(execute, *args)
        assert hybrid.comfy.cli_args.args.disable_comfy_compiler
        return result

    if failure:
        with pytest.raises(ValueError, match=f"{failure} failure"):
            hybrid._without_comfy_compiler(model_forward, None, None, None)
    else:
        for _ in range(2):
            assert hybrid._without_comfy_compiler(model_forward, None, None, None) == "done"
    assert hybrid.comfy.cli_args.args.disable_comfy_compiler is previous
    assert state.layout is None


def test_compiler_detection_does_not_change_global_flag(monkeypatch):
    monkeypatch.setattr(hybrid.comfy.cli_args.args, "disable_comfy_compiler", False)
    monkeypatch.setattr(nodes.comfy.model_prefetch, "comfy_aimdo",
                        SimpleNamespace(malloc_graph=object()))
    assert nodes._needs_comfy_compiler_workaround()
    assert hybrid.comfy.cli_args.args.disable_comfy_compiler is False


def test_compiler_wrapper_encloses_model_forward():
    attn = _attention()
    dm = SimpleNamespace(blocks=[SimpleNamespace(attn=attn)])
    wrappers = {}
    patcher = SimpleNamespace(
        get_model_object=lambda key: dm,
        add_object_patch=lambda *args: None,
        add_wrapper_with_key=lambda kind, key, fn: wrappers.update({kind: fn}))
    state = hybrid.VDNState("test", {}, [SimpleNamespace()], 2, 4)
    state.owns_compiler_switch = True
    hybrid.apply_vdn(patcher, state)
    assert wrappers[hybrid.WrappersMP.APPLY_MODEL] is hybrid._without_comfy_compiler
    assert hybrid.WrappersMP.DIFFUSION_MODEL in wrappers


@pytest.mark.parametrize("allocator", ["native", "cudaMallocAsync"])
@pytest.mark.parametrize("quantized", [False, True])
def test_prefetch_take_protects_storage_on_consumer_stream(allocator, quantized, monkeypatch):
    # CPU tensors and stream spies exercise the handoff without launching CUDA.
    data = torch.ones(2, 256, dtype=torch.int8 if quantized else torch.bfloat16)
    scale = torch.ones((), dtype=torch.float32)
    if quantized:
        weight = spec.QuantizedTensor(
            data, "TensorWiseINT8Layout",
            spec.TensorWiseINT8Layout.Params(
                scale=scale, orig_dtype=torch.bfloat16,
                orig_shape=(2, 256), is_weight=True, convrot=True,
                convrot_groupsize=256))
        expected = [data, scale]
    else:
        weight = data
        expected = [data]
    events = []
    ready = object()
    consumer = SimpleNamespace(wait_event=lambda event: events.append(("wait", event)))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: consumer)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: allocator)
    monkeypatch.setattr(torch.Tensor, "record_stream",
                        lambda tensor, stream: events.append((id(tensor), stream)))
    pf = hybrid._StreamPrefetcher.__new__(hybrid._StreamPrefetcher)
    pf._lock = threading.Lock()
    weights = {"weight": weight}
    pf._done = {3: (weights, ready)}

    assert pf.take(3) is weights
    assert events == [("wait", ready)] + [(id(t), consumer) for t in expected]
    assert pf.take(3) is None


def test_prefetch_record_failure_is_not_hidden(monkeypatch):
    def fail(tensor, stream):
        raise RuntimeError("stream registration failed")

    monkeypatch.setattr(torch.Tensor, "record_stream", fail)
    pf = hybrid._StreamPrefetcher.__new__(hybrid._StreamPrefetcher)
    with pytest.raises(RuntimeError, match="stream registration failed"):
        pf._record(torch.ones(1), object())


def test_cancelled_prefetch_can_request_same_block_again():
    # Pause before the worker takes the queued request: cancellation used to
    # drain that queue but leave its block permanently marked as in flight.
    pf = hybrid._StreamPrefetcher.__new__(hybrid._StreamPrefetcher)
    pf._queue = queue.Queue(maxsize=1)
    pf._lock = threading.Lock()
    pf._gen = 0
    pf._done = {}
    pf._inflight = set()
    fetch = lambda: {}
    pf.request(1, fetch)
    pf.reset()
    pf.request(1, fetch)
    assert pf._queue.get_nowait() == (1, 1, fetch)
    # Completion of work from the cancelled generation cannot clear new work.
    pf._inflight.discard((0, 1))
    assert (1, 1) in pf._inflight
