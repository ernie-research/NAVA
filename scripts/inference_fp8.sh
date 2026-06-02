#!/bin/bash
# ============================================================
# NAVA Inference — FP8 (E4M3) weight-only quantization
#
# Loads NAVA_fp8.safetensors instead of NAVA.safetensors. The DiT
# backbone block-Linears are stored as float8_e4m3fn (~half the
# bf16 footprint, ~6 GB vs ~12 GB). Compute path is unchanged —
# weights are dequantized to bf16 on the fly inside FP8Linear.
#
# Memory layout at peak:
#   - DiT denoising peak: ~28 GB  (fp8 backbone + activations + KV)
#   - VAE decode peak  : ~22 GB   (with --vae_tiling, backbone evicted to CPU)
# Without --vae_tiling decode peak shoots to ~35 GB on 704×1280×145 frames.
#
# Generate the fp8 checkpoint first:
#   python -m NAVA_FP8.convert_to_fp8 -i NAVA.safetensors -o NAVA_fp8.safetensors
#
# Override defaults with env vars:
#   CKPT, CONFIG, DATA_FILE, OUT_DIR, NPROC, VAE_TILE_H, VAE_TILE_W
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA_fp8.safetensors}"
CONFIG="${CONFIG:-configs/nava.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/fp8}"
DATA_FILE="${DATA_FILE:-infer_cases/general/prompts.jsonl}"

# VAE tiling: latent-space tile size (H, W). Smaller → less VRAM, more tiles.
# Defaults (22, 40) → 352×640-pixel tiles for 704×1280 output.
# Drop to (16, 30) if you need to push peak below 18 GB.
VAE_TILE_H="${VAE_TILE_H:-22}"
VAE_TILE_W="${VAE_TILE_W:-40}"
VAE_STRIDE_H="${VAE_STRIDE_H:-14}"
VAE_STRIDE_W="${VAE_STRIDE_W:-26}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29509}"
NPROC="${NPROC:-8}"

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] FP8 checkpoint not found: $CKPT" >&2
    echo "        Run: python -m NAVA_FP8.convert_to_fp8 -i NAVA.safetensors -o $CKPT" >&2
    exit 1
fi
if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] DATA_FILE not found: $DATA_FILE" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "[INFO] Repo:        $REPO_ROOT"
echo "[INFO] Config:      $CONFIG"
echo "[INFO] Ckpt:        $CKPT  (fp8_e4m3fn)"
echo "[INFO] Data:        $DATA_FILE"
echo "[INFO] Out dir:     $OUT_DIR"
echo "[INFO] Mode:        FP8 weight-only + T5 offload + VAE tiling"
echo "[INFO] VAE tile:    ${VAE_TILE_H}x${VAE_TILE_W}  stride ${VAE_STRIDE_H}x${VAE_STRIDE_W}"

SETUPTOOLS_USE_DISTUTILS=stdlib torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    inference_nava.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --weight_dtype fp8_e4m3fn \
    --out_dir "$OUT_DIR" \
    --data_format json \
    --data_file "$DATA_FILE" \
    --width 1280 \
    --height 704 \
    --frames 37 \
    --fps 24 \
    --steps 50 \
    --save_sample \
    --gen_turn 1 \
    --use_sp \
    --t5_offload \
    --vae_tiling \
    --vae_tile_size "$VAE_TILE_H" "$VAE_TILE_W" \
    --vae_tile_stride "$VAE_STRIDE_H" "$VAE_STRIDE_W"
