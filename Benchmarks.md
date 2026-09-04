# Benchmarks — ComfyUI-VDN-H3

## Optimization round (2026-09-04, this branch)

Rig: RTX 5090 (sm_120, 32 GB), Windows 11, torch 2.10.0+cu130, triton 3.6.0.
Base `minimax_h3_fl2va_int8_convrot.safetensors`, stage `stage-dmd-step-250`
(bf16 branch), turbo adapter ON, **merge**, grouped windows, er_sde/beta, 8
steps, seed 42. Peaks are `torch.cuda.max_memory_allocated()` reset right after
the model chain — i.e. the sampling-phase peak, excluding the one-time model
load and the VAE decode.

### 1280x736, 145 frames (F=47 latent, seq 34,487) — headline config

Fresh server per run, first run of each, stream mode, seed 42, prompt 0. ALL
rows measured with the identical methodology: `VDNResetPeak` in the model chain
(`torch.cuda.reset_peak_memory_stats()` right after the model chain), peak read
at the post-sampling latent dump — i.e. the sampling-phase peak, excluding the
one-time model load and the VAE decode. The baseline row ran the v1.3.1
checkout (git-verified) with the same workflow generator.

| build | allocation mode | s/it | sampling peak |
|---|---|---|---|
| v1.3.1 baseline | per-call (only pattern) | 16.30 (cold file; warm repeats 15.0–15.6) | 11.94 GiB |
| optimized | retained | 13.87 (−15% / −9% vs cold/warm baseline) | 13.08 GiB |
| optimized | transient (`retain_buffers off`) | 14.11 (−13% / −8%) | 13.08 GiB |

Findings, stated plainly:

- **Speed**: retained mode is ~15% faster than the same-protocol baseline run
  (the baseline's first-run figure includes a cold page cache; against its warm
  repeats the gain is ~9%). Transient mode keeps most of it (the prefetcher,
  not the buffers, is the larger part of the remaining gap).
- **Peak VRAM: allocator snapshots show the peak working set is UNCHANGED.**
  CUDA allocator history was recorded for baseline and optimized runs (native
  allocator, `--disable-cuda-malloc`, identical 145f workflow) and the traces
  replayed: baseline peak live set 11.875 GiB vs optimized 11.926 GiB
  (+0.05 GiB), with every major allocation group matching 1:1 (4.1 GiB
  readout working set, 2.19 GiB samplers, 3x0.55 GiB conv features, streamed
  weights — all equal; the optimized run's peak set merely has more/smaller
  blocks, 332 vs 79, same bytes). The ~+1.0 GiB that
  `torch.cuda.max_memory_allocated()` reports for the optimized code does not
  correspond to any allocation stack in the traces — it is a counter-level
  effect of the mid-run peak reset + backend accounting, not retained memory.
  Snapshots: `output/base145.pickle` / `new145.pickle` (viewable in
  pytorch.org/memory_viz). Release line: **~15% faster at the headline config,
  peak working set unchanged at the allocation level (counter reads ~+1 GiB
  with no corresponding allocation), output bit-identical.**
- **Parity at the headline config: bit-identical.** Both modes vs baseline:
  latent PSNR ∞ (zero MSE), frame LPIPS 0.0000 over all 158 decoded frames,
  seed 42.

### 512x320, 56 frames (F=17, S=160, seq 2,938) — smoke A/B vs v1.3.1

| build | branch_weights | mode | s/it | sampling peak |
|---|---|---|---|---|
| v1.3.1 baseline | stream | — | 1.66 | 2.03 GiB |
| optimized | stream | retained | 1.66–1.74 | 2.45 GiB |
| optimized | stream | transient | ~2.5 (cold run) | 2.34 GiB |
| optimized | auto → cache_gpu | retained | 1.55–1.56 | 6.28–6.36 GiB |
| optimized | fast_kernels (stream) | retained | 1.61 | 3.30 GiB |

At this tiny size the grouped window issues few SDPA calls and the scan loop is
short, so s/it is flat in stream mode and the buffers cost ~0.1 GiB retained.
Optimized-side cache_gpu numbers were taken on a warm server; the transient
smoke figure is from a cold first run and is not speed-comparable.

### Parity gate (PASS)

Seed 42, pre-VAE-decode latents + decoded frames, both sizes:

| A/B | size | latent PSNR | frame LPIPS (max) |
|---|---|---|---|
| v1.3.1 vs optimized `auto` (3 prompts) | 56f | **∞ (bit-identical)** | **0.0000** |
| v1.3.1 vs optimized stream+prefetch | 56f | **∞** | **0.0000** |
| v1.3.1 vs retained, after act-scratch fix | 56f | **∞** | **0.0000** |
| v1.3.1 vs transient (`retain_buffers off`) | 56f + **145f** | **∞** | **0.0000** |
| v1.3.1 vs retained | **145f** | **∞** | **0.0000** |

Gate thresholds (≥ 50 dB, ≤ 0.01) passed with maximum margin at both sizes and
in every buffer mode — the optimized defaults are exactly output-preserving,
not just within bf16 reduction-order noise.

**fast_kernels caveat (pre-existing, opt-in):** the same-seed fast_kernels run
diverges visibly (latent PSNR 17.3 dB, LPIPS 0.104). Root cause is NOT the new
compiled scan — that measures exact (max err 0.0 on GPU, cudagraph replays
included) — but the pre-existing fused epilogue/gather/q-store kernels, whose
1–2 ulp bf16 rounding differences (max 0.06 elementwise) the 8-step DMD
amplifies (the same sensitivity documented for bypass mode in v1.2.0).
fast_kernels stays off by default, now logs a warning on DMD stages, and is
documented as ablation-only in the README.

**auto policy validated live:** with VRAM crowded by earlier runs' caches, auto
downshifted to `stream` mid-queue (2.2/1.3 GiB free observed) and recovered;
on a fresh 32 GB card it picks `cache_gpu`. Synthetic threshold checks: cache_gpu
≥ 1.5× stage + 4 GiB headroom, else stream; int8_convrot stage file preferred
under pressure when both branch files coexist in one stage dir;
`retain_buffers=auto` retains when free ≥ stage + 10 GiB, else transient.

Unit tests: 12/12 (10 existing + compiled-scan parity + transient/retained
buffer parity).

## Rig

- GPU: NVIDIA GeForce RTX 5090 (sm_120, 32 GB), Windows 11
- torch 2.10.0+cu130, triton 3.6.0
- Base model: `minimax_h3_fl2va_int8_convrot.safetensors` (int8 convrot)
- VDN checkpoint: `stage-dmd-step-250` (8-step model), `apply_turbo_adapter` ON
- Node settings: `strength 1.0`, `lora_mode bypass`, `branch_weights stream`
  (settings as measured; `bypass` is no longer recommended — see the `lora_mode`
  note in the README)
- Sampler: euler / simple, **8 steps**, seed **42**, CFG 1, t2v + audio

## 1280x736, 145 frames (F=37, S=920, seq 34,487 tokens)

| attention_backend | s/it | sampling time | notes |
|---|---|---|---|
| grouped (default) | 16.92–16.95 | ~2:15 | cold + warm runs |
| flex (FlexAttention + BlockMask) | 17.10 | ~2:17 | compiled successfully on triton-windows, warm run |

**Result: parity on the 5090 at 34.5k tokens.** The grouped path issues only
~6 dense SDPA calls per block per step at this length (interior chunks share one
window), so the flex kernel's fusion buys nothing yet and carries compile cost.
flex stays available as an opt-in for much longer clips; `grouped` remains the
default.

## 512x320, 56 frames (F=17, S=160, seq 2,939) — smoke

8-step render completed end-to-end with audio; used for correctness bring-up
(per-frame pixel statistics verified healthy).

## Verification status

### v1.2.0

- Unit tests (10/10): existing coverage plus the `vdn_scaled` delta rule (scan +
  closed form vs `(I + A/S)^-1`), fused-vs-eager state-gather parity, frame-major
  q-store parity, and an end-to-end `LinearBranch.readout` fast_kernels-vs-eager
  parity check. Pytest harness hardened (session `conftest.py` path setup, the
  attention-dispatch stub scoped to its test, parametrized anchor loop).
- E2E A/B matrix (headless server, seed 42, 512x320/56f, 8 steps): full base
  merge/bypass x fast_kernels off/on, pruned base merge/bypass — all six outputs
  visually verified good. The fast_kernels runs confirmed the compiled path
  actually executed (epilogue + state gather + frame-major q store: the expected
  small single-rounding divergence from eager, quality intact).
- **merge vs bypass at a fixed seed gives different videos** (~26/255 mean
  per-pixel diff). Later measurement showed the two modes are not equivalent in
  quality: bypass applies the deltas in activation space, where bf16 rounding
  noise is amplified by the deep blocks (~10% of feature magnitude by block 49)
  and visibly degrades 8-step DMD checkpoints, while merge reproduces the
  validated weights exactly. v1.2.0 therefore defaults to `merge`, which is
  required for `stage-dmd-*` checkpoints (see the `lora_mode` note in the
  README).

### v1.0.0

- Unit tests: window partition (grouped vs reference, 6 geometry/anchor combos),
  delta-rule scans and vdn_solve vs step-by-step recurrence, bridge/gather vs
  naive construction, adapter re-keying (qkv block-diag fold, swiglu swap),
  bypass hook reload stability (3 inject/eject cycles), flex mask partition vs
  grouped, attention-dispatch routing. All pass.
- E2E: repeated back-to-back renders on a live server, including after a model
  unload/reload cycle (the recursion regression fixed in v1.0.0).
