#!/bin/bash
# ============================================================
# NAVA Inference — T5 + DiT Group Offload + VAE Tiled Decode
#
# Combines three memory optimisations:
#   - T5 (~11 GB) offloaded to CPU after text encoding
#   - DiT backbone blocks kept on CPU, moved to GPU one group
#     at a time during denoising (pinned memory + async stream)
#   - VAE decode spatial tiling: H×W latent split into overlapping
#     tiles decoded individually, blended on CPU
#
# Tune OFFLOAD_GROUP_SIZE to trade speed vs. VRAM:
#   1  — one block at a time (minimum VRAM, maximum transfers)
#   2  — two blocks at a time (balanced)
#   5+ — fewer transfers, higher peak VRAM per step
#
# VAE tile params (latent space):
#   VAE_TILE_SIZE_H/W   — tile size  (default 44×80 → 352×640 px)
#   VAE_TILE_STRIDE_H/W — tile stride (default 28×52 → 224×416 px)
#   Smaller tiles → lower peak VRAM, more tiles to process
#
# Override defaults with env vars:
#   CKPT, CONFIG, DATA_FILE, OUT_DIR, NPROC
#   OFFLOAD_GROUP_SIZE
#   VAE_TILE_SIZE_H, VAE_TILE_SIZE_W
#   VAE_TILE_STRIDE_H, VAE_TILE_STRIDE_W
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA.safetensors}"
CONFIG="${CONFIG:-configs/nava.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/group_offload}"
DATA_FILE="${DATA_FILE:-infer_cases/general/prompts.jsonl}"
OFFLOAD_GROUP_SIZE="${OFFLOAD_GROUP_SIZE:-10}"

VAE_TILE_SIZE_H="${VAE_TILE_SIZE_H:-22}"
VAE_TILE_SIZE_W="${VAE_TILE_SIZE_W:-40}"
VAE_TILE_STRIDE_H="${VAE_TILE_STRIDE_H:-14}"
VAE_TILE_STRIDE_W="${VAE_TILE_STRIDE_W:-26}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29507}"
NPROC="${NPROC:-8}"

if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] DATA_FILE not found: $DATA_FILE" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "[INFO] Repo:               $REPO_ROOT"
echo "[INFO] Config:             $CONFIG"
echo "[INFO] Ckpt:               $CKPT"
echo "[INFO] Data:               $DATA_FILE"
echo "[INFO] Out dir:            $OUT_DIR"
echo "[INFO] Mode:               T5 offload + DiT group offload + VAE tiled decode"
echo "[INFO] Offload group size: $OFFLOAD_GROUP_SIZE"
echo "[INFO] VAE tile size:      ${VAE_TILE_SIZE_H}x${VAE_TILE_SIZE_W}"
echo "[INFO] VAE tile stride:    ${VAE_TILE_STRIDE_H}x${VAE_TILE_STRIDE_W}"

SETUPTOOLS_USE_DISTUTILS=stdlib torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    inference_nava.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
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
    --group_offload \
    --offload_group_size "$OFFLOAD_GROUP_SIZE" \
    --vae_tiling \
    --vae_tile_size "$VAE_TILE_SIZE_H" "$VAE_TILE_SIZE_W" \
    --vae_tile_stride "$VAE_TILE_STRIDE_H" "$VAE_TILE_STRIDE_W"
