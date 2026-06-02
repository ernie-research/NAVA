# NAVA_FP8

Weight-only **FP8** quantization for the NAVA DiT backbone.

This package converts a standard NAVA checkpoint (`NAVA.safetensors`, fp32/bf16
weights) into an fp8-quantized variant whose weights take **half the memory**
of bf16. It targets ComfyUI / single-GPU inference where VRAM is the
bottleneck.

| Format       | Per-Linear bytes | NAVA backbone (~6 B params) |
| ------------ | ---------------- | --------------------------- |
| fp32         | 4                | ~24 GB                      |
| bf16         | 2                | ~12 GB                      |
| **fp8_e4m3** | **1 (+ scale)**  | **~6.1 GB**                 |

Phase 1 (this package) is **weight-only**: weights are stored as
`float8_e4m3fn` and dequantized to bf16 on-the-fly inside each layer's
`forward`. There is no compute speedup yet — the win is purely VRAM.
Phase 2 will swap the dequant path for `torch._scaled_mm` to get real
H100 / RTX 4090 FP8 tensor-core acceleration.

## What gets quantized

Only the 2-D `Linear` weights inside transformer blocks:

```
backbone.{double|single|double_final}_blocks.<i>.{self_attn|cross_attn|ffn}.*.weight
```

Norms, modulation parameters, biases, embeddings, and everything outside the
DiT blocks are kept in bf16. This whitelist is defined in
`NAVA_FP8/quantize.py` (`_QUANTIZE_PATTERN`).

Per-output-channel symmetric quantization is used:

```
w_fp8[i, :] = clip(w[i, :] / scale[i], ±448) -> float8_e4m3fn
scale[i]    = max(|w[i, :]|) / 448           (bf16)
```

## Install

The package depends only on `torch>=2.1` (for `float8_e4m3fn`) and
`safetensors`, which NAVA already requires.

```bash
# from the NAVA repo root
python -c "import NAVA_FP8; print(NAVA_FP8.__all__)"
```

## Convert a checkpoint

```bash
# Default: writes <stem>_fp8.safetensors next to the input.
python -m NAVA_FP8.convert_to_fp8 \
    --input  /path/to/NAVA.safetensors \
    --output /path/to/NAVA_fp8.safetensors

# Dry run: just print compression ratio, don't write.
python -m NAVA_FP8.convert_to_fp8 -i NAVA.safetensors --dry-run
```

Expected output:

```
[fp8] loading: NAVA.safetensors
[fp8] input: 1052 keys, 23.6 GiB
[fp8] quantizing...
[fp8] quantized layers : 360
[fp8] passthrough keys : 692
[fp8] input  size      : 23.6 GiB
[fp8] output size      : 6.1 GiB
[fp8] compression     : 3.87x  (74.2% smaller)
[fp8] writing: NAVA_fp8.safetensors
[fp8] done.
```

## Load a quantized checkpoint

```python
from NAVA_FP8 import load_fp8_checkpoint

# pipe.model is the bf16/empty NAVA model built from config.
n_patched, missing, unexpected = load_fp8_checkpoint(
    pipe.model, "NAVA_fp8.safetensors", verbose=True,
)
```

`load_fp8_checkpoint` does two things in order:
1. **Patch** every block-Linear in `pipe.model` to an `FP8Linear` (which
   owns `weight: fp8_e4m3fn` and `weight_scale: bf16` buffers).
2. **`load_state_dict(strict=False)`** the fp8 file. Quantized keys go into
   `FP8Linear`, everything else goes into the original modules in bf16.

## Verify correctness

```bash
# Synthetic Linear shapes only (fast, ~10 s on a GPU).
python -m NAVA_FP8.tests.verify_numerics

# Plus real layers from your NAVA checkpoint.
python -m NAVA_FP8.tests.verify_numerics --ckpt /path/to/NAVA.safetensors
```

Healthy output: cosine similarity ≥ 0.999, relative L2 < 1% on every layer.

## ComfyUI integration

(Coming next.) The plan is a `weight_dtype` dropdown on `NAVAModelLoader`
with values `bf16` (default) and `fp8_e4m3fn`. When fp8 is selected, the
loader calls `load_fp8_checkpoint` instead of the standard `load_state_dict`
path. No other ComfyUI nodes change — fp8 is opaque to the sampler.

## Files

| File                      | Purpose                                                |
| ------------------------- | ------------------------------------------------------ |
| `fp8_linear.py`           | `FP8Linear` module (drop-in `nn.Linear` replacement)   |
| `quantize.py`             | Per-channel quantization + state-dict whitelist        |
| `patching.py`             | Walk a model and replace block-Linears in place        |
| `convert_to_fp8.py`       | CLI: bf16 checkpoint → fp8 safetensors                 |
| `load_fp8.py`             | One-shot: patch model + load fp8 state dict            |
| `tests/verify_numerics.py`| Cosine / L2 sanity checks against bf16 reference       |
