#!/bin/bash
# ============================================================
# NAVA Inference — T5 CPU Offload (single GPU friendly)
#
# T5 encoder (~11 GB) is moved back to CPU after text encoding,
# freeing GPU memory for DiT denoising. Suitable for running on
# fewer GPUs or GPUs with limited VRAM.
#
# Override defaults with env vars:
#   CKPT, CONFIG, DATA_FILE, OUT_DIR, NPROC
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA.safetensors}"
CONFIG="${CONFIG:-configs/nava.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/offload_t5}"
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
echo "[INFO] Mode:    T5 CPU offload"

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
    --t5_offload
