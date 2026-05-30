#!/bin/bash
# ============================================================
# NAVA Gradio Demo Launcher (SP=8)
#
# Usage:
#   bash start_gradio.sh
#   bash start_gradio.sh --ckpt /path/to/ckpt --port 7860 --share
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Default paths (modify for your environment) ----
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29508

CONFIG="${CONFIG:-configs/nava.yaml}"
CKPT="${CKPT:-NAVA.safetensors}"
REWRITE_MODEL="${REWRITE_MODEL:-pe_src/Qwen3-4B-Thinking-2507}"
PORT="${PORT:-8000}"
NPROC="${NPROC:-8}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
FRAMES="${FRAMES:-37}"

# ---- Parse CLI arguments ----
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --rewrite_model) REWRITE_MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --width) WIDTH="$2"; shift 2 ;;
        --frames) FRAMES="$2"; shift 2 ;;
        --share) EXTRA_ARGS="$EXTRA_ARGS --share"; shift ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "============================================"
echo " NAVA Gradio Demo"
echo " Config:        $CONFIG"
echo " Checkpoint:    $CKPT"
echo " Rewrite Model: $REWRITE_MODEL"
echo " SP Size:       $NPROC"
echo " Resolution:    ${WIDTH}x${HEIGHT}"
echo " Frames:        $FRAMES"
echo " Port:          $PORT"
echo "============================================"

# Add project paths
export PYTHONPATH="/root/paddlejob/workspace/env_run/NAVA:${SCRIPT_DIR}:${PYTHONPATH}"

# Run from NAVA root so relative paths (e.g. ./Wan2.2-TI2V-5B/) resolve correctly
cd /root/paddlejob/workspace/env_run/NAVA

SETUPTOOLS_USE_DISTUTILS=stdlib torchrun \
    --nproc_per_node=$NPROC \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    gradio_demo/gradio_server.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --rewrite_model "$REWRITE_MODEL" \
    --port "$PORT" \
    --height $HEIGHT \
    --width $WIDTH \
    --frames $FRAMES \
    $EXTRA_ARGS
