#!/bin/bash
# ============================================================
# NAVA Inference — General T2AV example (Sequence Parallel, 8 GPU)
#
# Reads prompts from $DATA_FILE (defaults to
# eval_results/general/prompts.jsonl) and generates synchronized
# audio-video samples under $OUT_DIR.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA.safetensors}"
if [ ! -f "$CKPT" ]; then
    SF="${CKPT%.ckpt}.safetensors"
    if [ -f "$SF" ]; then
        echo "[INFO] $CKPT not found, falling back to $SF"
        CKPT="$SF"
    fi
fi
CONFIG="${CONFIG:-configs/nava.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/general}"
DATA_FILE="${DATA_FILE:-infer_cases/general/prompts.jsonl}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29507}"
NPROC="${NPROC:-8}"

if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] DATA_FILE not found: $DATA_FILE" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "[INFO] Repo:    $REPO_ROOT"
echo "[INFO] Config:  $CONFIG"
echo "[INFO] Ckpt:    $CKPT"
echo "[INFO] Data:    $DATA_FILE"
echo "[INFO] Out dir: $OUT_DIR"

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
    --use_sp
