#!/bin/bash
# ============================================================
# NAVA Gradio Demo Launcher (SP=8) — defaults match scripts/inference_fp8_vl_rewrite.sh
#
# FP8 weight-only quant + T5 offload + VAE tiling are ON by default. The default
# CKPT is NAVA_fp8.safetensors — generate it once with:
#     python -m NAVA_FP8.convert_to_fp8 -i NAVA.safetensors -o NAVA_fp8.safetensors
#
# Override any knob via env var (CKPT, WEIGHT_DTYPE, T5_OFFLOAD, VAE_TILING,
# GROUP_OFFLOAD, OFFLOAD_GROUP_SIZE, VAE_TILE_H, VAE_TILE_W, …) or CLI flag.
#
# Usage:
#   bash start_gradio.sh
#   bash start_gradio.sh --ckpt /path/to/ckpt --port 7860 --share
#   CKPT=NAVA.safetensors WEIGHT_DTYPE=bf16 bash start_gradio.sh   # bf16 fallback
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Default paths (modify for your environment) ----
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29508

CONFIG="${CONFIG:-configs/nava.yaml}"
CKPT="${CKPT:-NAVA_fp8.safetensors}"
REWRITE_MODEL="${REWRITE_MODEL:-pe_src/Qwen3-4B-Instruct-2507}"
VL_MODEL="${VL_MODEL:-pe_src/Qwen3-VL-4B-Instruct}"
PORT="${PORT:-8000}"
NPROC="${NPROC:-8}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
FRAMES="${FRAMES:-37}"

# ---- Inference acceleration knobs (defaults align with inference_fp8_vl_rewrite.sh) ----
WEIGHT_DTYPE="${WEIGHT_DTYPE:-fp8_e4m3fn}"
T5_OFFLOAD="${T5_OFFLOAD:-1}"
VAE_TILING="${VAE_TILING:-1}"
VAE_TILE_H="${VAE_TILE_H:-22}"
VAE_TILE_W="${VAE_TILE_W:-40}"
VAE_TILE_STRIDE_H="${VAE_TILE_STRIDE_H:-14}"
VAE_TILE_STRIDE_W="${VAE_TILE_STRIDE_W:-26}"
GROUP_OFFLOAD="${GROUP_OFFLOAD:-0}"
OFFLOAD_GROUP_SIZE="${OFFLOAD_GROUP_SIZE:-1}"

# ---- Parse CLI arguments ----
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --rewrite_model) REWRITE_MODEL="$2"; shift 2 ;;
        --vl_model) VL_MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --width) WIDTH="$2"; shift 2 ;;
        --frames) FRAMES="$2"; shift 2 ;;
        --share) EXTRA_ARGS="$EXTRA_ARGS --share"; shift ;;
        --weight_dtype) WEIGHT_DTYPE="$2"; shift 2 ;;
        --t5_offload) T5_OFFLOAD=1; shift ;;
        --no_t5_offload) T5_OFFLOAD=0; shift ;;
        --vae_tiling) VAE_TILING=1; shift ;;
        --no_vae_tiling) VAE_TILING=0; shift ;;
        --vae_tile_size) VAE_TILE_H="$2"; VAE_TILE_W="$3"; shift 3 ;;
        --vae_tile_stride) VAE_TILE_STRIDE_H="$2"; VAE_TILE_STRIDE_W="$3"; shift 3 ;;
        --group_offload) GROUP_OFFLOAD=1; shift ;;
        --offload_group_size) OFFLOAD_GROUP_SIZE="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---- Build acceleration flags from bool toggles ----
[ "$T5_OFFLOAD" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --t5_offload"
[ "$VAE_TILING" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --vae_tiling --vae_tile_size $VAE_TILE_H $VAE_TILE_W --vae_tile_stride $VAE_TILE_STRIDE_H $VAE_TILE_STRIDE_W"
[ "$GROUP_OFFLOAD" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --group_offload --offload_group_size $OFFLOAD_GROUP_SIZE"

echo "============================================"
echo " NAVA Gradio Demo"
echo " Config:        $CONFIG"
echo " Checkpoint:    $CKPT"
echo " Weight dtype:  $WEIGHT_DTYPE"
echo " Rewrite Model: $REWRITE_MODEL"
echo " VL Model:      $VL_MODEL"
echo " SP Size:       $NPROC"
echo " Resolution:    ${WIDTH}x${HEIGHT}"
echo " Frames:        $FRAMES"
echo " T5 offload:    $T5_OFFLOAD"
echo " VAE tiling:    $VAE_TILING (tile ${VAE_TILE_H}x${VAE_TILE_W} stride ${VAE_TILE_STRIDE_H}x${VAE_TILE_STRIDE_W})"
echo " Group offload: $GROUP_OFFLOAD (group_size=$OFFLOAD_GROUP_SIZE)"
echo " Port:          $PORT"
echo "============================================"

# Friendly hint when fp8 ckpt is missing — give the exact one-liner to fix.
if [ "$WEIGHT_DTYPE" = "fp8_e4m3fn" ] && [ ! -f "$SCRIPT_DIR/../$CKPT" ] && [ ! -f "$CKPT" ]; then
    echo "[WARN] fp8 mode but $CKPT not found in $SCRIPT_DIR/.."
    echo "       Generate it with:"
    echo "       (cd $SCRIPT_DIR/.. && python -m NAVA_FP8.convert_to_fp8 -i NAVA.safetensors -o NAVA_fp8.safetensors)"
    echo "       Or fall back to bf16:"
    echo "       CKPT=NAVA.safetensors WEIGHT_DTYPE=bf16 bash $0"
fi

# Add project paths
 export PYTHONPATH="./:${SCRIPT_DIR}:${PYTHONPATH}"

# Run from NAVA root so relative paths (e.g. ./Wan2.2-TI2V-5B/) resolve correctly
cd "$SCRIPT_DIR/.."

SETUPTOOLS_USE_DISTUTILS=stdlib torchrun \
    --nproc_per_node=$NPROC \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    gradio_demo/gradio_server.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --weight_dtype "$WEIGHT_DTYPE" \
    --rewrite_model "$REWRITE_MODEL" \
    --vl_model "$VL_MODEL" \
    --port "$PORT" \
    --height $HEIGHT \
    --width $WIDTH \
    --frames $FRAMES \
    $EXTRA_ARGS
