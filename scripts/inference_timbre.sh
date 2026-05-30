#!/bin/bash
# ============================================================
# NAVA Inference — I2AV + Timbre Control example (SP=8)
#
# Reads JSONL records (each may carry image_path + spk_wavs) from
# $DATA_FILE (defaults to eval_results/timbre/prompts.jsonl) and runs
# I2AV with timbre control. I2V mode auto-engages per-sample when the
# record contains `image_path`. `--timbre_cfg` makes the speaker
# reference actually steer the voice.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA.safetensors}"
CONFIG="${CONFIG:-configs/nava.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/timbre}"
DATA_FILE="${DATA_FILE:-infer_cases/timbre/prompts.jsonl}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29508}"
NPROC="${NPROC:-8}"
TIMBRE_SCALE="${TIMBRE_SCALE:-1.0}"

if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] DATA_FILE not found: $DATA_FILE" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "[INFO] Repo:     $REPO_ROOT"
echo "[INFO] Config:   $CONFIG"
echo "[INFO] Ckpt:     $CKPT"
echo "[INFO] Data:     $DATA_FILE"
echo "[INFO] Out dir:  $OUT_DIR"
echo "[INFO] Timbre cfg scale: $TIMBRE_SCALE"

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
    --timbre_cfg \
    --timbre_align_guidance_scale "$TIMBRE_SCALE"
