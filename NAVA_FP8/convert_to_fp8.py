"""
Convert a bf16/fp32 NAVA checkpoint into an fp8-quantized safetensors file.

Reads the original NAVA.safetensors / NAVA.ckpt, runs `quantize_state_dict`
over it (whitelisted block-Linear weights -> fp8_e4m3fn + per-row bf16 scale,
the rest -> bf16), and writes a new safetensors file.

Usage:
    python -m NAVA_FP8.convert_to_fp8 \
        --input  /path/to/NAVA.safetensors \
        --output /path/to/NAVA_fp8.safetensors

    # Dry run: just print what would be quantized + compression ratio.
    python -m NAVA_FP8.convert_to_fp8 --input NAVA.safetensors --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Allow running as either `python -m NAVA_FP8.convert_to_fp8 ...`
# or `python NAVA_FP8/convert_to_fp8.py ...` (no parent package).
if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PARENT = os.path.dirname(_HERE)
    if _PARENT not in sys.path:
        sys.path.insert(0, _PARENT)
    from NAVA_FP8.quantize import quantize_state_dict
else:
    from .quantize import quantize_state_dict


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    obj = torch.load(path, map_location="cpu", mmap=True)
    # NAVA training checkpoints wrap the weights under "state_dict".
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def _save_state_dict(sd: dict[str, torch.Tensor], path: str) -> None:
    if not path.endswith(".safetensors"):
        raise ValueError(
            f"Output must end in .safetensors (got: {path}). "
            f"fp8 weights only round-trip cleanly through safetensors."
        )
    from safetensors.torch import save_file
    # safetensors needs contiguous tensors.
    sd = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in sd.items()}
    save_file(sd, path)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantize a NAVA checkpoint to fp8_e4m3fn.")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to NAVA.safetensors (or NAVA.ckpt).")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (.safetensors). "
                             "Defaults to <input_stem>_fp8.safetensors next to input.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write output; just report what would happen.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print every key as it is quantized / passed through.")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"[fp8] ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    out_path = args.output
    if out_path is None:
        stem, _ = os.path.splitext(args.input)
        out_path = f"{stem}_fp8.safetensors"

    print(f"[fp8] loading: {args.input}")
    sd = _load_state_dict(args.input)
    in_bytes = sum(t.numel() * t.element_size() for t in sd.values() if isinstance(t, torch.Tensor))
    print(f"[fp8] input: {len(sd)} keys, {_fmt_bytes(in_bytes)}")

    print("[fp8] quantizing...")
    out_sd, stats = quantize_state_dict(sd, verbose=args.verbose)

    out_bytes = sum(
        t.numel() * t.element_size() for t in out_sd.values() if isinstance(t, torch.Tensor)
    )
    print(f"[fp8] quantized layers : {stats['quantized_n']}")
    print(f"[fp8] passthrough keys : {stats['passthrough_n']}")
    print(f"[fp8] input  size      : {_fmt_bytes(in_bytes)}")
    print(f"[fp8] output size      : {_fmt_bytes(out_bytes)}")
    if in_bytes > 0:
        print(f"[fp8] compression     : {in_bytes / max(out_bytes, 1):.2f}x  "
              f"({100 * (1 - out_bytes / in_bytes):.1f}% smaller)")

    if args.dry_run:
        print("[fp8] dry-run: not writing output.")
        return 0

    print(f"[fp8] writing: {out_path}")
    _save_state_dict(out_sd, out_path)
    print("[fp8] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
