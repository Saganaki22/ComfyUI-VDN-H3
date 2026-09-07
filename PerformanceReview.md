# Performance and correctness review — 7 September 2026

Reviewed local baseline `b49130c26a70d12c542601c5bc4f7ee0f112ee2e` against
[OpenVDN commit 57edaf6](https://github.com/OpenVDN/vdn-minimax-h3/tree/57edaf696f19f5c0997d2dc63e14863f926dfeee).
Validation used the existing ComfyUI venv, PyTorch 2.10.0+cu130, and RTX 5090.

## Follow-up: long-clip allocation failure

The supplied crash log has 76,019 packed rows, 72 latent frames and 1,032
tokens/frame. It ends in a native abort during FP32 frame-statistics preparation,
not a caught CUDA out-of-memory exception. Memory pressure is plausible, but the
log alone does not establish the native failure's cause.

Frame-statistics preparation now processes independent frame batches with an
estimated 1 GiB preparation-workspace budget. At the logged dimensions that is
10 frames per batch, instead of all 70 non-anchor frames simultaneously. Output
statistics and backend workspaces are additional memory. Every frame still uses
its complete token reduction, original precision and original recurrence.

Initial validation was **CPU-only: 38 passed, 8 GPU tests skipped** while the
user rendered. After the GPU became available, isolated CUDA comparisons at the
crash geometry also matched bit-for-bit, with finite outputs and maximum error
zero. The reference was the same implementation with frame batching bypassed;
this comparison isolates the new memory fix from the earlier changes below.

Statistics tests used strided BF16 K/V inputs, 56 heads and head dimension 128.
The branch tests used actual block-0 INT8 ConvRot weights resolved to BF16,
synthetic BF16 activations, 72 frames, 1,032 tokens/frame, a 24×43 grid, 905 text
tokens, anchor exclusion (70 statistics frames), and the eager branch path.
Times are medians of eight synchronized wall-clock measurements per variant,
alternating execution order after warmup. Peak additional allocated memory
excludes inputs, weights and warmed persistent buffers; it is not total VRAM.

| Operation | Unbatched time | Batched time | Unbatched peak extra | Batched peak extra |
|---|---:|---:|---:|---:|
| Statistics, 14 frames × 920 tokens | 2.728 ms | 3.153 ms | 1,203.6 MiB | 1,043.7 MiB |
| Statistics, 35 frames × 920 tokens | 7.288 ms | 7.617 ms | 3,008.9 MiB | 1,190.7 MiB |
| Statistics, 70 frames × 1,032 tokens | 18.551 ms | 18.906 ms | 6,663.0 MiB | 1,441.6 MiB |
| Statistics, 72 frames × 1,032 tokens | 17.355 ms | 17.163 ms | 6,852.0 MiB | 1,455.6 MiB |
| INT8 branch readout, 72 frames | 106.274 ms | 106.733 ms | 9,634.0 MiB | 6,971.4 MiB |
| Same branch including final projection | 115.759 ms | 115.907 ms | 9,634.0 MiB | 6,971.4 MiB |

All six comparisons were bit-identical. The branch's measured additional peak
fell by **2.60 GiB (27.6%)**, with effectively unchanged timing in this run.
Batching introduces some launch/copy overhead, visible particularly in the
smaller statistics-only cases; this change reduces memory pressure rather than
establishing a speedup. The whole workflow and native-abort cause remain
unverified: no sampling job or video render was run.

The final full regression run with CUDA available passed **46 tests, no skips**
in 18.84 seconds. This includes CPU/CUDA FP32/BF16 frame-batching parity for both
statistics precision modes and a final partial batch. The 17 warnings were
dependency deprecations. Command (using the existing ComfyUI venv):
`python -B -m pytest tests -q -p no:cacheprovider`.

## Changes made

- **Restore missing text-refiner attention adapters.** The released default and
  Turbo files use `token_refiner.refiner_blocks.N.attn.to_q/k/v` and
  `attn.to_out.0`. The converter recognized only the main blocks' `attn.orig.*`
  spelling. Refiner attention deltas consequently targeted nonexistent ComfyUI
  parameters. They now map to the refiner's fused QKV and output projection.
  This changes output by restoring trained weights; it is a correctness fix,
  not a bit-identical performance change.
- **Fix the gated dense fallback for short clips.** ComfyUI attention returns
  flattened heads, while the softmax gate expects `[tokens, heads, head_dim]`.
  Restore that shape before applying the gate. The branch still correctly
  stays inactive when the local window covers the entire clip.
- **Release QKV and projection inputs earlier.** `q4` and `k4` kept the complete
  QKV allocation alive after `del q, k, v`. The `flat` input to the output
  projection also survived throughout the linear branch. Both lifetimes are
  shortened without changing any arithmetic. At 34,500 rows and 7,168 channels,
  these BF16 allocations total approximately **1.84 GiB**. This is calculated
  storage released before the branch, not a measured reduction in whole-run
  peak VRAM; another stage may determine that peak.
- **Remove the per-block GPU epsilon readback.** Creating epsilon on the GPU
  and calling `.item()` forced synchronization once per active branch block.
  Cache a Python scalar per dtype instead. Preserve the previous rounded
  epsilon exactly: for BF16 it is `9.98377799987793e-7`, not exactly `1e-6`.
  Changing this numerical convention is deliberately outside the speed fix.
- **Reuse the already-contiguous K operand in frame statistics.** The B GEMM
  now consumes the repack already made for A, as upstream does, instead of
  returning to the original strided K view.
- **Release dead branch intermediates and raw copies.** Feature K/V and frame
  statistics are released after the scans; raw Q/K/V copies are released before
  the final branch projection. Temporal convolution reuses its local sum buffer
  without fusing multiply/add or changing intermediate rounding.
- **Fix compiler-workaround scope.** MiniMax's outer `forward` starts its
  allocation graph before `DIFFUSION_MODEL` wrappers execute. Disabling the
  compiler there was too late on subsequent steps. An `APPLY_MODEL` wrapper
  now covers the entire call and restores the previous flag in `finally`.
  Applying a node no longer changes the process-wide setting before sampling.
- **Honor query short convolution when a checkpoint requests it.** The config
  accepted `q`, but the feature path ignored its convolution weights. Released
  K/V-only stages are unaffected.
- **Bound and clear FlexAttention masks.** Masks now have a bounded LRU cache,
  include the complete device in their key, and are released on interruption.
- **Reject incomplete INT8 weights.** A missing scale previously made the
  loader silently treat stored integers as unquantized weights.
- **Allow prefetch to recover after cancellation.** Reset now clears queued
  in-flight markers. Markers include the generation, so an old worker cannot
  remove a newer request for the same block.

## Measurements and mathematical checks

**38 tests passed**, including new CUDA parity checks, epsilon graph-capture
checks, refiner mapping, dense gated attention, buffer lifetime, and compiler
restoration on success and failure. Existing tests alone were mostly small CPU
checks; their compiled-path success was not proof of CUDA or visual parity.

Direct CUDA comparisons with downloaded upstream code matched bit-for-bit for
frame statistics (with the same TF32 policy), VDN transition/injection,
inference scans, and alpha-bridge gathering. The source comparison also checked
the key-channel orientation of alpha, half-scaled prompt state in both scans,
anchor exclusion, and window bounds. Sources:
[delta rule](https://github.com/OpenVDN/vdn-minimax-h3/blob/57edaf696f19f5c0997d2dc63e14863f926dfeee/src/models/linear_attention/delta_rule.py),
[scan and gather](https://github.com/OpenVDN/vdn-minimax-h3/blob/57edaf696f19f5c0997d2dc63e14863f926dfeee/src/models/linear_attention/scan.py),
[hybrid attention](https://github.com/OpenVDN/vdn-minimax-h3/blob/57edaf696f19f5c0997d2dc63e14863f926dfeee/src/models/hybrid_attention.py).

Branch-readout benchmarks used **actual block-0 weights** from both local stages,
synthetic BF16 activations, 56 heads, head dimension 128, hidden width 5,376,
920 tokens/frame, a 23×40 grid, 256 text tokens, trained windows and anchor
exclusion, and `fast_kernels=False`. Medians of 12 synchronized wall-clock
measurements per variant, alternating execution order after warmup:

| Branch weights | Latent frames | Before | After | Time reduction |
|---|---:|---:|---:|---:|
| BF16 | 16 | 20.733 ms | 20.790 ms | -0.3% |
| BF16 | 37 | 48.734 ms | 47.811 ms | 1.9% |
| INT8 ConvRot | 16 | 21.254 ms | 20.954 ms | 1.4% |
| INT8 ConvRot | 37 | 50.189 ms | 49.730 ms | 0.9% |

All four before/after readouts were **bit-identical**, maximum error zero.
These timings exclude the final branch projection, backbone, sampling loop,
and loading; they are not an INT8-vs-BF16 end-to-end comparison. Small timing
differences may include measurement noise; these results do not establish a
universal speedup. The isolated statistics function measured 3.701 → 3.369 ms
at 16 frames and 9.125 → 7.747 ms at 37 frames in this run, but an earlier run
showed a small short-shape regression. Kernel timings vary with shape and run.

Measured peak additional allocations inside the warmed readout fell from
approximately **1,836 to 1,762 MiB** at 16 frames and **4,590 to 4,406 MiB** at
37 frames, for both weight formats. This excludes persistent buffers and input
tensors. The separate attention/raw-QKV lifetime fixes are outside this readout
microbenchmark and release additional large allocations earlier.

No full video A/B was run. The adapter-loading correction changes the complete
model output, so bit-identical branch measurements do not certify perceptual
quality of the entire corrected pipeline.

## Further optimization boundaries

The performance changes preserve the tested eager output. The restored
refiner adapters are a separate correctness change and will alter complete
model output. No full video A/B was run, so perceptual quality is not certified.

The existing `fast_kernels` option still combines numerically different fusions;
keep it off for strict comparisons. Upstream's frame-statistics operand
preparation is a promising separate fusion candidate because it combines copies,
widening casts and independent multiplies without reassociation. It needs its
own GPU benchmark and parity checks before becoming a default.

Retain the trained window, anchor, and text-state settings. Use resident branch
weights when activation memory permits; current free memory at node application
is not a measurement of the later sampler peak.

The existing pruned-base merge path still skips 51 incompatible Turbo AdaLN
targets. The inspected pruned input has 8 features, versus the adapter's trained
2,688-feature input. No approximation for this mismatch was introduced here.
