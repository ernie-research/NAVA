#!/bin/bash
# ============================================================
# NAVA Inference — T2A (Text-to-Audio, audio-only mode)
#
# Supports both pure T2A and timbre-controlled speech generation.
# Reads JSONL from $DATA_FILE; each record may optionally carry
# "spk_wavs" for timbre reference — works with or without it.
#
# Override via env vars, e.g.:
#   NPROC=1 bash scripts/inference_t2a.sh
#   DATA_FILE=/path/to/prompts.jsonl bash scripts/inference_t2a.sh
#   DURATION=5.0 TIMBRE_SCALE=3.0 bash scripts/inference_t2a.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA.safetensors}"
CONFIG="${CONFIG:-configs/nava_seedtts.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/t2a}"
DATA_FILE="${DATA_FILE:-infer_cases/t2a/prompts.jsonl}"
DURATION="${DURATION:-6.0}"
TIMBRE_SCALE="${TIMBRE_SCALE:-3.0}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29509}"
NPROC="${NPROC:-8}"

if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] DATA_FILE not found: $DATA_FILE" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "[INFO] Repo:         $REPO_ROOT"
echo "[INFO] Config:       $CONFIG"
echo "[INFO] Ckpt:         $CKPT"
echo "[INFO] Data:         $DATA_FILE"
echo "[INFO] Out dir:      $OUT_DIR"
echo "[INFO] Duration:     ${DURATION}s"
echo "[INFO] Timbre scale: $TIMBRE_SCALE"
echo "[INFO] Num GPUs:     $NPROC"

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
    --duration "$DURATION" \
    --steps 50 \
    --save_sample \
    --gen_turn 1 \
    --timbre_cfg \
    --timbre_align_guidance_scale "$TIMBRE_SCALE" \
    --use_sp
