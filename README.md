# ComfyUI-VDN-H3 — VDN-H3 (Video Delta Net) for MiniMax-H3


<img width="1039" height="505" alt="image" src="https://github.com/user-attachments/assets/ab4c1691-bff5-46fe-8b3e-635429b0700f" />



**English** | [中文](README_ZH.md)

Video Delta Net hybrid attention for MiniMax-H3 as a native ComfyUI node. Nearby
frames keep exact softmax attention; distant temporal context goes through the
checkpoint's **Video Delta Attention** linear branch, replacing the quadratic
long-range attention with a constant-cost recurrent state.

Reference implementation: [OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3)
(Apache-2.0). Weights: [OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3)
(MiniMax H3 Community License — **read it before use**; the license excludes some
territories).

This package is a **port, not a fork**: it reproduces the official hybrid-attention
math on ComfyUI's native MiniMax-H3 model as runtime model patches. No ComfyUI core
files are modified.

**Why this repo exists (and what it isn't).** The official VDN-H3 release targets a datacenter stack: 8× B200 GPUs with Ulysses sequence parallelism, and FlashAttention-4 kernels that only support Hopper and datacenter Blackwell — consumer Blackwell (sm_120) isn't supported, and there are no Windows builds. Upstream also uses FP8 linears and custom fused Triton kernels; this port substitutes those with portable PyTorch equivalents that run everywhere ComfyUI runs.

What you get: the same released checkpoints and the same architecture — windowed softmax + Video Delta Attention branch, unit-tested against the official implementation — with zero new dependencies. The 8-step distilled model, near-lossless quality versus dense H3, and an attention cost that grows linearly with clip length instead of quadratically — the longer the video, the more this matters.

What you don't get: the headline numbers. The official 74.5× figure combines 8-GPU parallelism, FA4, FP8, and 8-step distillation; upstream's own single-GPU measurement is ~2.6× at 50 steps, and this port's portable kernels land somewhat under that (measured ~17 s/it at 1280×736 / 145 frames on an RTX 5090 — see Benchmarks.md). If you want to experiment with the architecture on your own hardware, this is for you; if you want the streaming-real-time numbers, that takes their B200 cluster.

| CK, Sol-attn, res_multi / simple — 20 Steps, 1280x736, 3:05 | LightXv2 4-Step Turbo v1.1, CK, Sol-attn, er_sde / beta — 8 steps, 1280x736, 1:24 |
|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/7120657d-af61-4414-b621-53b39208ffe0" controls></video> | <video src="https://github.com/user-attachments/assets/b0373566-fc78-4616-b591-13462c4b50e6" controls></video> |

| VDN-H3 Turbo, er_sde / beta — 8 steps, 1280x736, 2:04 | VDN-H3 Advanced fast_kernels Turbo, er_sde / beta — 8 steps, 1280x736, 1:13 |
|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/89cc7155-ca89-459e-9996-5b5f6bfcd284" controls></video> | <video src="https://github.com/user-attachments/assets/5cc9906e-acec-4c61-a3b9-17c79153945b" controls></video> |


### Same seed 
`981445682258077`


## Install

1. Clone into `ComfyUI/custom_nodes/` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-VDN-H3
```

2. Download the VDN checkpoint stage you want into `ComfyUI/models/vdn/`:

```bash
hf download OpenVDN/vdn-minimax-h3 --include "stage-dmd-step-250/*" --local-dir <ComfyUI>/models/vdn
```

Keep the release directory layout intact (`model_spec.json`, `linear_branch/`,
`adapters/`). Nothing is converted on disk — the node re-keys the diffusers-format
tensors onto ComfyUI module paths in memory.

**No new Python dependencies.** The node runs the official math in eager PyTorch
that ships with ComfyUI (torch + safetensors). No Triton, no flash-attn-4, no CUDA
builds, no `pip install`.

## Nodes

**Apply VDN-H3 (MiniMax-H3 Hybrid Attention)** — `MODEL -> MODEL`

| Input | Meaning |
|---|---|
| `vdn_checkpoint` | a stage directory under `models/vdn` |
| `apply_turbo_adapter` | ON = the released **8-step** model (use 8 sampler steps); OFF = the **50-step** model (use ~50 steps) |
| `strength` | adapter strength, 1.0 = released model |
| `lora_mode` | `bypass` (runtime, sharp) / `merge` (folded into weights; lowest VRAM, softer on int8/fp8 bases) |
| `branch_weights` | `stream` (~4.3 GB of branch weights move to GPU per block per step — safe on small cards) / `cache_gpu` (resident, faster, keep ~4.3 GB VRAM free) |
| `attention_backend` | `grouped` (default; one dense SDPA per window group) / `flex` (one compiled FlexAttention kernel; opt-in, see Benchmarks.md) |
| `verbose` | log the applied adapters and per-forward layout |

Drop it between your MiniMax-H3 loader and the sampler; conditioning, LoRAs,
samplers, VAE decode and video/audio output nodes are unchanged. Example workflow:
`example_workflows/vdn_h3_t2v_8step.json`.

**Apply VDN-H3 Advanced** — everything above plus, for experimenters:

| Input | Meaning |
|---|---|
| `stage_b_strength` / `turbo_strength` | per-adapter strengths (default node applies one global strength) |
| `window_radius`, `window_chunk` | deviate from the trained c=5 r=1 window (ablation) |
| `anchor_frames` | `both` / `columns` / `rows` / `none` (trained: `both`) |
| `text_state` | write the prompt into the branch's states at init (trained: on) |
| `linear_branch` | off = window-only ablation (debug — output loses all long-range context) |
| `fast_kernels` | torch.compile the branch's RMSNorm+gate epilogue into one kernel (same math; falls back to eager if compile fails) |

Ablation inputs warn in the console when they deviate from the checkpoint's
trained spec; defaults reproduce the released model exactly.

## Attention backends and stacking

VDN's windowed softmax always runs exact SDPA — dispatched through ComfyUI's
backend-priority chain (flash / cuDNN / mem-efficient), but never through
quantized backends: routing the windows through sage/kitchen int8 measurably
softens output, and the released model validated exact local attention. Backend
override patches (SageAttention, kitchen-int8, KJNodes) still apply to the base
model's own attention (text refiner, and the dense fallback on very short
clips). The delta-rule branch never calls softmax kernels and is unaffected by
backend patches.

**Do not stack the "MiniMax H3 Scheduled Sol Attention" patch with this node.**
It replaces `blocks.*.attn.forward` — the same path VDN owns — so wherever SOL
handles a call, VDN's linear branch is skipped and you are no longer running
VDN-H3 (with VDN's LoRAs applied to an attention they were not trained for).
Use SOL-H3 for plain H3 runs; use VDN alone for VDN runs. SOL's FFN-chunking
node and general attention overrides do compose.

## Required models

| Component | File | Source | Place in |
|---|---|---|---|
| Base diffusion model | `minimax_h3_fl2va_int8_convrot.safetensors` (recommended with torch cu130; use the `fp8_scaled` variant only if you can't) | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/diffusion_models` |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/text_encoders` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/vae` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/vae` |
| VDN branch + adapters | `stage-dmd-step-250/` (8-step) and/or `stage-b-step-2000/` (50-step) | [OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3) | `models/vdn` |

The VDN release **does not contain base weights** — it is branch + LoRA adapters
only, applied at runtime on whatever MiniMax-H3 base you load. The 72 GB diffusers
base (`h3-base/`) in the HF repo is *not* needed.

The 8-step model's `turbo` adapter replaces (does not stack with) community
MiniMax-H3 turbo LoRAs — do not run both.

## What is official vs adapted

**Faithful to the official implementation** (verified against the reference math by
unit tests in `tests/`): chunk-aligned softmax window with anchor frames
(`radius=1, chunk=5, anchor_frames=both` in the released spec), the `vdn_solve`
delta rule, bidirectional frame scans with the alpha bridge and prompt text state,
the K/V short conv, output gates, and both LoRA adapters.

**ComfyUI-specific adaptations:**

- The default windowed softmax runs as one dense SDPA per chunk-group instead of
  block-sparse FlexAttention. Same partition, same math; needs no Triton and no
  torch.compile. A FlexAttention + BlockMask path IS included (opt-in via
  `attention_backend: flex`) and compiled fine on triton-windows — measured
  parity with grouped on RTX 5090 at 34.5k tokens (see Benchmarks.md), so grouped
  stays the default. The official FA4 backend is faster still but needs
  Linux + datacenter Blackwell.
- Eager pointwise ops instead of the official Triton/compiled fusions (temporal
  conv, RMSNorm epilogue, gather). Correct, somewhat slower; the scan's kernel
  launches are the remaining optimization target (CUDA-graph via torch.compile).
- LoRA applied through ComfyUI's bypass/merge machinery (int8-fused `fc2` weights
  route through merge automatically; pruned/curve bases get the e-grid adaln
  re-injection).
- The packed-sequence geometry is read from ComfyUI's own `PackedLayout`, so
  conditioning variants (t2va / fl2va / ref2va) keep working; only t2va-style
  layouts were exercised by VDN's training.

## GPUs / platform

- **Windows + NVIDIA**: primary target, tested (RTX 5090, torch 2.10+cu130).
- **Linux + NVIDIA**: should work identically (pure PyTorch).
- Single GPU only in this port. The official Ulysses 8-GPU path is not implemented
  (it is distribution, not algorithm).
- AMD/Intel/CPU: untested; eager PyTorch means it will *run*, slowly. The delta-rule
  Cholesky needs a batched-solve backend — CPU works for small tests.

## VRAM and performance

The base model dominates VRAM; VDN adds ~4.3 GB of branch weights (streamed per
block in `stream` mode, so the working-set increase is roughly one block's ~86 MB,
plus transient raw q/k copies inside attention of about `2 x seq_len x 7168 x 2`
bytes).

Measured on RTX 5090 (int8 convrot base, `stream` mode, sage2 patch): 1280x736,
145 frames, 8 steps, euler/simple, seed 42, ~17 s/it (~2:15 sampling), audio
included. `grouped` vs `flex` attention backends measured parity at 34.5k tokens
— the grouped path issues only ~6 dense SDPA calls per block per step at this
length, so flex's fusion buys nothing yet; grouped stays the default. Reference
points from the official VDN report: a single B200 runs the dense 50-step model
in 13.95 min and the optimized VDN-H3 in 5.34 min (~2.6x from the hybrid alone);
the headline 74.5x combines 8xB200 parallelism, 8-step distillation, fp8 linears,
and FA4/flex kernels. Expect single-GPU gains on this port to track the ~2.6x
architectural figure, scaled by which attention backend your windows dispatch to.
Full measurement data and verification status: [Benchmarks.md](Benchmarks.md).

## Troubleshooting

- **`VDN checkpoint ... not found`** — the stage dir must sit under
  `models/vdn/` and contain `linear_branch/model.safetensors` and `model_spec.json`.
- **"checkpoint has N blocks but the loaded model has M"** — the VDN stage and the
  loaded base do not belong together (e.g. a 50-block stage on a different-depth
  model). Load the matching MiniMax-H3 base.
- **"This MODEL already has VDN-H3 applied"** — chain the node once.
- **OOM** — use `branch_weights: stream` (default), `lora_mode: merge`, shorter
  clips, smaller resolution.
- **Wrong-looking motion at 8 steps** — make sure `apply_turbo_adapter` is ON with
  8 steps, or OFF with ~50 steps; mixing the two schedules degrades output.
- **Video renders but looks like the plain model** — check `verbose` and look for
  `[vdn] layout:` in the console; on clips with <= 15 latent frames the window
  covers everything and VDN correctly falls back to dense attention.

## License & citation

This port is Apache-2.0 (see LICENSE). The VDN-H3 architecture, training, and
checkpoints are by [OpenVDN](https://github.com/OpenVDN/vdn-minimax-h3)
(Apache-2.0); the MiniMax-H3 weights are under the MiniMax H3 Community License.
If you use VDN-H3, cite the authors:

```bibtex
@misc{xi2026videodeltanet,
  title  = {VideoDeltaNet on MiniMax H3},
  author = {Haocheng Xi and Yiming Xie and Hexu Zhao and Yiwen Zhang and Michael Liu and Thomas Creavin and Kurt Keutzer and Xiuyu Li and Zhaoyang Lv and Chenfeng Xu and Haiwen Feng},
  year   = {2026},
  url    = {https://openvdn.github.io/}
}
```

