"""
Core FP8 quantization functions.

- `quantize_tensor`: per-channel symmetric quantization for a 2D weight.
- `is_quantizable`: whitelist filter — decides which state-dict keys get fp8.
- `quantize_state_dict`: walk a state dict, quantize whitelisted weights,
  emit `<key>` (fp8) and `<key>_scale` (bf16) entries.
"""

from __future__ import annotations

import re
from typing import Callable

import torch

from .fp8_linear import FP8_E4M3_MAX


# Default whitelist: 2D weights inside transformer blocks (self_attn / cross_attn / ffn).
# Excludes norms, modulation, biases, and anything outside `*_blocks.<i>.`.
_QUANTIZE_PATTERN = re.compile(
    r"^backbone\.(double|single|double_final)_blocks\.\d+\."
    r"(self_attn|cross_attn|ffn)\.[^.]+\.weight$"
)


def is_quantizable(key: str, tensor: torch.Tensor) -> bool:
    """Return True if this state-dict entry should be quantized to fp8."""
    if not key.endswith(".weight"):
        return False
    if tensor.ndim != 2:
        return False
    if "norm" in key or "modulation" in key:
        return False
    return bool(_QUANTIZE_PATTERN.match(key))


def quantize_tensor(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-output-channel symmetric quantization to fp8_e4m3fn.

    Args:
        w: weight tensor of shape [out_features, in_features], any float dtype.

    Returns:
        w_fp8 : float8_e4m3fn, shape [out, in]
        scale : float32, shape [out]   — recover via w_bf16 ≈ w_fp8 * scale[:, None]
    """
    if w.ndim != 2:
        raise ValueError(f"quantize_tensor expects 2D, got shape {tuple(w.shape)}")

    w = w.detach().to(torch.float32)
    # Per-row max-abs; eps to avoid div-by-zero on a zero row.
    row_amax = w.abs().amax(dim=1).clamp(min=1e-12)
    scale = row_amax / FP8_E4M3_MAX
    w_scaled = w / scale.unsqueeze(1)
    w_fp8 = w_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale


def quantize_state_dict(
    sd: dict[str, torch.Tensor],
    filter_fn: Callable[[str, torch.Tensor], bool] = is_quantizable,
    *,
    bf16_passthrough: bool = True,
    verbose: bool = False,
) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Walk a state dict, quantize whitelisted entries, downcast the rest to bf16.

    Returns:
        out_sd : dict where each quantized key `K` becomes:
                   K          -> fp8_e4m3fn  [out, in]
                   K_scale    -> bfloat16    [out]
                 non-quantized keys are bf16 (if bf16_passthrough) or untouched.
        stats  : {"quantized_n", "quantized_bytes", "passthrough_n",
                  "passthrough_bytes", "quantized_keys": list[str]}
    """
    out_sd: dict[str, torch.Tensor] = {}
    stats = {
        "quantized_n": 0,
        "quantized_bytes": 0,
        "passthrough_n": 0,
        "passthrough_bytes": 0,
        "quantized_keys": [],
    }

    for key, tensor in sd.items():
        if filter_fn(key, tensor):
            w_fp8, scale = quantize_tensor(tensor)
            out_sd[key] = w_fp8
            out_sd[f"{key}_scale"] = scale.to(torch.bfloat16)
            stats["quantized_n"] += 1
            stats["quantized_bytes"] += w_fp8.numel() + scale.numel() * 2
            stats["quantized_keys"].append(key)
            if verbose:
                print(f"  [fp8]   {key}  shape={tuple(tensor.shape)}")
        else:
            t = tensor.to(torch.bfloat16) if bf16_passthrough and tensor.is_floating_point() else tensor
            out_sd[key] = t
            stats["passthrough_n"] += 1
            stats["passthrough_bytes"] += t.numel() * t.element_size()
            if verbose:
                print(f"  [keep]  {key}  shape={tuple(tensor.shape)}  dtype={t.dtype}")

    return out_sd, stats
