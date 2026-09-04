"""Pre-quantize a VDN stage's linear-branch weights to Comfy Kitchen INT8 ConvRot.

Produces a sibling stage directory (<stage>-int8_convrot_comfyui/) using the
same serialization ComfyUI itself ships for int8-convrot diffusion models.
Quantized files are renamed <original-stem>_int8_convrot_comfyui.safetensors;
files that pass through unmodified keep their original names.

  <layer>.weight         torch.int8  [out, in]
  <layer>.weight_scale   float32     [out, 1]
  <layer>.comfy_quant    uint8 tensor of UTF-8 JSON:
                         {"format": "int8_tensorwise", "convrot": true,
                          "convrot_groupsize": 256}

Only tensors whose names AND shapes match the branch's dispatched F.linear
weights are quantized; everything else is copied verbatim. Never writes to the
input directory.

Usage:
  python tools/quantize_vdn_branch_int8.py <path-to-stage-dir>
      [--out <dir>] [--overwrite] [--cpu]
"""
import argparse
import json
import os
import struct
import sys

import torch
from safetensors.torch import load_file, save_file

# Suffixes of branch weights consumed through F.linear (verified dispatch sites
# in vdn_h3/branch.py / hybrid.py). Anything else stays as-is.
QUANTIZE_SUFFIXES = (
    "to_out_linear.weight",     # hybrid.py readout epilogue
    "beta_proj.weight",         # branch.py _readout/_text_state
    "output_gate.down.weight",  # branch.py output gate (down leg only; the up
                                #   leg is an mm-decomposition that would
                                #   dequantize per call -- kept bf16)
)
CONVROT_GROUPSIZE = 256
MIN_IN_FEATURES = 1024  # don't bother quantizing small mats even if named right

STAGE_MARKERS = ("linear_branch",)  # a stage dir must contain these entries
FILE_SUFFIX = "_int8_convrot_comfyui"
DIR_SUFFIX = "-int8_convrot_comfyui"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def eligible(key, header):
    if not key.endswith(QUANTIZE_SUFFIXES):
        return False
    shape = header[key].get("shape")
    if not shape or len(shape) != 2:
        return False
    if shape[1] % CONVROT_GROUPSIZE or shape[1] < MIN_IN_FEATURES:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="quantize on CPU (slow)")
    args = ap.parse_args()

    src = os.path.abspath(args.stage_dir)
    if not os.path.isdir(src):
        sys.exit(f"not a directory: {src}")
    if os.path.basename(src).endswith(DIR_SUFFIX):
        sys.exit("refusing to quantize an already-quantized stage")
    for marker in STAGE_MARKERS:
        if not os.path.exists(os.path.join(src, marker)):
            sys.exit(f"{src} does not look like a VDN stage (missing {marker}/)")

    dst = args.out or src + DIR_SUFFIX
    if os.path.exists(dst) and not args.overwrite:
        sys.exit(f"output exists: {dst} (use --overwrite)")

    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA unavailable; pass --cpu to quantize on CPU")

    from comfy_kitchen.tensor import TensorWiseINT8Layout

    sfiles = []
    for root, _dirs, names in os.walk(src):
        for n in names:
            if n.endswith(".safetensors"):
                sfiles.append(os.path.join(root, n))
    if not sfiles:
        sys.exit("no safetensors files found")

    os.makedirs(dst, exist_ok=True)
    total_before = total_after = 0

    for path in sorted(sfiles):
        rel = os.path.relpath(path, src)
        header = read_header(path)
        file_meta = header.get("__metadata__")
        sd = load_file(path)
        out = {}
        n_q = 0
        bytes_before = bytes_after = 0

        for key, tensor in sd.items():
            if not eligible(key, header):
                out[key] = tensor
                continue

            w = tensor.to(device)
            qdata, params = TensorWiseINT8Layout.quantize(
                w, convrot=True, per_channel=True, is_weight=True,
                stochastic_rounding=0)
            layer = key[: -len(".weight")]
            out[key] = qdata.to("cpu", torch.int8)
            out[key + "_scale"] = params.scale.to("cpu", torch.float32)
            conf = json.dumps({"format": "int8_tensorwise", "convrot": True,
                               "convrot_groupsize": CONVROT_GROUPSIZE})
            out[layer + ".comfy_quant"] = torch.tensor(
                list(conf.encode("utf-8")), dtype=torch.uint8)
            n_q += 1
            bytes_before += tensor.numel() * tensor.element_size()
            bytes_after += (qdata.numel() + params.scale.numel() * 4
                            + len(conf))

        was_quantized = n_q > 0
        dest_rel = rel
        if was_quantized:
            stem, ext = os.path.splitext(rel)
            dest_rel = stem + FILE_SUFFIX + ext
        dest = os.path.join(dst, dest_rel)
        os.makedirs(os.path.dirname(dest) or dst, exist_ok=True)
        save_file(out, dest, metadata=file_meta)
        total_before += bytes_before
        total_after += bytes_after
        print(f"{dest_rel}: quantized {n_q} tensors, "
              f"{bytes_before/1e6:.1f} -> {bytes_after/1e6:.1f} MB (quantized set)")

    # copy everything that was not rewritten (configs, adapters, ...) verbatim

    for root, dirs, names in os.walk(src):
        for n in names:
            s = os.path.join(root, n)
            rel = os.path.relpath(s, src)
            d = os.path.join(dst, rel)
            if n.endswith(".safetensors"):
                stem, ext = os.path.splitext(rel)
                if os.path.exists(os.path.join(dst, stem + FILE_SUFFIX + ext)):
                    continue  # rewritten under the quantized name
            if os.path.exists(d):
                continue
            os.makedirs(os.path.dirname(d) or dst, exist_ok=True)
            with open(s, "rb") as fi, open(d, "wb") as fo:
                fo.write(fi.read())
            print(f"copied {rel}")

    print(f"\nquantized weight bytes: {total_before/1e9:.2f} GB -> "
          f"{total_after/1e9:.2f} GB")
    print(f"output stage: {dst}")


if __name__ == "__main__":
    main()
