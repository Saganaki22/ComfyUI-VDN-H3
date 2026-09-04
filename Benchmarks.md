# Benchmarks — ComfyUI-VDN-H3

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
