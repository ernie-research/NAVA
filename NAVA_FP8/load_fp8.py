"""
Load an fp8-quantized NAVA checkpoint into a model.

The flow is:
  1) Patch the model in place: every block-Linear in `backbone.*_blocks.<i>.
     (self_attn|cross_attn|ffn).*` is replaced with an `FP8Linear` that owns
     `weight` (fp8) + `weight_scale` (bf16) buffers.
  2) load_state_dict(strict=False): the fp8 file has matching `<key>` and
     `<key>_scale` entries that line up with the patched modules; everything
     else (norms, modulation, biases, embeddings, ...) is bf16 passthrough
     and lands on the original modules untouched.

Usage:
    from NAVA_FP8 import load_fp8_checkpoint
    n_patched, missing, unexpected = load_fp8_checkpoint(pipe.model, "NAVA_fp8.safetensors")
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PARENT = os.path.dirname(_HERE)
    if _PARENT not in sys.path:
        sys.path.insert(0, _PARENT)
    from NAVA_FP8.patching import patch_model_to_fp8
else:
    from .patching import patch_model_to_fp8


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    obj = torch.load(path, map_location="cpu", mmap=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def load_fp8_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    *,
    verbose: bool = False,
) -> tuple[int, list[str], list[str]]:
    """
    Patch `model` to fp8 and load the fp8 checkpoint at `ckpt_path`.

    Args:
        model:     the bf16/empty NAVA model (typically `pipe.model`).
        ckpt_path: path to NAVA_fp8.safetensors.
        verbose:   print each module replaced and load_state_dict diagnostics.

    Returns:
        (num_patched, missing_keys, unexpected_keys) — same semantics as
        nn.Module.load_state_dict's return value, plus the patch count.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"fp8 checkpoint not found: {ckpt_path}")

    n_patched = patch_model_to_fp8(model, verbose=verbose)
    if verbose:
        print(f"[fp8] patched {n_patched} Linear modules -> FP8Linear")

    sd = _load_state_dict(ckpt_path)
    if verbose:
        n_fp8 = sum(1 for v in sd.values() if v.dtype == torch.float8_e4m3fn)
        print(f"[fp8] checkpoint: {len(sd)} keys ({n_fp8} fp8 tensors)")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[fp8] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"[fp8]   first missing: {missing[:5]}")
        if unexpected:
            print(f"[fp8]   first unexpected: {unexpected[:5]}")

    return n_patched, list(missing), list(unexpected)
