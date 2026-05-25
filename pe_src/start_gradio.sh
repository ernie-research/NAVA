#!/bin/bash
# ============================================================
# NAVA Gradio Server Launcher (SP=8)
# Usage: bash start_gradio.sh [--port 7860] [--share]
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default paths - modify these for your environment
CONFIG="${CONFIG:-/root/paddlejob/workspace/env_run/ernie-one-av/configs/baseline_t2av_demo_mmdit_no_split_ltx_control_unipc.yaml}"
CKPT="${CKPT:-/root/paddlejob/workspace/env_run/ernie-one-av/checkpoints/latest.pt}"
REWRITE_MODEL="${REWRITE_MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-7860}"
NPROC="${NPROC:-8}"

# Parse extra arguments (passed to gradio_server.py)
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --rewrite_model) REWRITE_MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --share) EXTRA_ARGS="$EXTRA_ARGS --share"; shift ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "============================================"
echo " NAVA Gradio Server"
echo " Config:        $CONFIG"
echo " Checkpoint:    $CKPT"
echo " Rewrite Model: $REWRITE_MODEL"
echo " SP Size:       $NPROC"
echo " Port:          $PORT"
echo "============================================"

# Add project root to PYTHONPATH
export PYTHONPATH="/root/paddlejob/workspace/env_run/NAVA:${SCRIPT_DIR}:${PYTHONPATH}"

torchrun \
    --nproc_per_node=$NPROC \
    --master_port=29500 \
    gradio_server.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --rewrite_model "$REWRITE_MODEL" \
    --port "$PORT" \
    $EXTRA_ARGS
