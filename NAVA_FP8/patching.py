"""
Patch a model in-place: replace nn.Linear modules with FP8Linear.

Usage:
    from NAVA_FP8.patching import patch_model_to_fp8
    patch_model_to_fp8(pipe.model)              # patches based on default name filter
    # then load_state_dict(fp8_sd) to populate weights
"""

from __future__ import annotations

import re
from typing import Callable

import torch
import torch.nn as nn

from .fp8_linear import FP8Linear


# Patch any Linear whose dotted module path matches this — i.e. the linear
# is inside a transformer block's self_attn / cross_attn / ffn submodule.
_PATCH_NAME_PATTERN = re.compile(
    r"^backbone\.(double|single|double_final)_blocks\.\d+\."
    r"(self_attn|cross_attn|ffn)\."
)


def _default_should_patch(module_path: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    return bool(_PATCH_NAME_PATTERN.match(module_path))


def patch_model_to_fp8(
    model: nn.Module,
    *,
    should_patch: Callable[[str, nn.Module], bool] = _default_should_patch,
    quantize_from_existing: bool = False,
    verbose: bool = False,
) -> int:
    """
    Walk `model` and replace each matching nn.Linear with an FP8Linear in place.

    Args:
        should_patch: predicate(module_path, module) -> bool.
        quantize_from_existing: if True, the new FP8Linear inherits weights from
                                the original Linear (per-channel quantized on the
                                fly). Use this when patching a model that already
                                has bf16 weights loaded. If False, FP8Linear is
                                created with empty buffers — the caller is
                                expected to load_state_dict afterwards.
        verbose: print each replacement.

    Returns:
        number of modules replaced.
    """
    replaced = 0
    # Collect first to avoid mutating during iteration.
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent_path, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full_path = f"{parent_path}.{child_name}" if parent_path else child_name
            if should_patch(full_path, child):
                targets.append((parent, child_name, child))

    for parent, child_name, child in targets:
        if quantize_from_existing:
            new_mod = FP8Linear.from_linear(child)
        else:
            new_mod = FP8Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
            )
        # Preserve device of the original Linear's params (if any).
        try:
            device = next(child.parameters()).device
            new_mod = new_mod.to(device)
        except StopIteration:
            pass
        setattr(parent, child_name, new_mod)
        replaced += 1
        if verbose:
            print(f"  [patch] {child_name} -> FP8Linear")

    return replaced
