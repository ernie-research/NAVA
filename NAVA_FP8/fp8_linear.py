"""
FP8Linear: drop-in replacement for nn.Linear with fp8_e4m3fn weights.

Phase 1 design (weight-only quantization):
- Weight stored as torch.float8_e4m3fn, shape [out, in]
- Per-output-channel scale stored as bf16, shape [out]
- Bias (if present) stored as bf16
- forward dequantizes weight to activation dtype on the fly, then F.linear

Memory: weight is 1 byte/elem (vs 2 bytes for bf16, 4 bytes for fp32).
Compute: matmul is still in bf16 — no FP8 tensor-core speedup yet.
         Phase 2 will swap this for torch._scaled_mm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Maximum representable magnitude in float8_e4m3fn.
# (e4m3fn: 4-bit exp, 3-bit mantissa, no inf, no NaN encoding.)
FP8_E4M3_MAX = 448.0


class FP8Linear(nn.Module):
    """
    Linear layer with fp8_e4m3fn weight + per-channel bf16 scale.

    Mirrors nn.Linear's interface so the rest of the model is unaffected.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        # Per-output-channel scale (one per row of weight).
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, dtype=torch.bfloat16),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8Linear":
        """Build an FP8Linear from a bf16/fp32 nn.Linear, quantizing per-channel."""
        from .quantize import quantize_tensor

        out_features, in_features = linear.weight.shape
        fp8_lin = cls(in_features, out_features, bias=linear.bias is not None)

        w_fp8, scale = quantize_tensor(linear.weight.detach())
        fp8_lin.weight.data = w_fp8
        fp8_lin.weight_scale.data = scale.to(torch.bfloat16)
        if linear.bias is not None:
            fp8_lin.bias.data = linear.bias.detach().to(torch.bfloat16)
        return fp8_lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight into the activation dtype, then standard matmul.
        # weight: [out, in] fp8 ; weight_scale: [out] bf16
        # Result: [out, in] in x.dtype
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, dtype=float8_e4m3fn (per-channel scale)"
        )
