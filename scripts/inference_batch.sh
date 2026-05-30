#!/bin/bash
# ============================================================
# NAVA Inference — Batch mode (8 GPU, DDP, no Sequence Parallel)
#
# Each GPU independently processes a slice of $DATA_FILE for
# maximum throughput across many prompts. For single-sample
# fastest latency, use scripts/inference.sh (SP=8) instead.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA.safetensors}"
CONFIG="${CONFIG:-configs/nava.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/batch}"
DATA_FILE="${DATA_FILE:-infer_cases/batch_infer_prompts.jsonl}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29507}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
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
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC" \
    --node_rank="$NODE_RANK" \
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
    --gen_turn 1
