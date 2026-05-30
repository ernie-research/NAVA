#!/bin/bash
# ============================================================
# NAVA Inference — SeedTTS Benchmark Evaluation (Sequence Parallel, 8 GPU)
#
# Reads a SeedTTS meta.lst (utt_id|prompt_text|prompt_wav|infer_text per line)
# and produces zero-shot speech .wav files under $OUT_DIR.
#
# Override via env vars, e.g.:
#   LANG=en bash scripts/inference_seedtts.sh
#   DATA_FILE=/path/to/meta.lst bash scripts/inference_seedtts.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LANG="${LANG:-zh}"  # zh | en

CKPT="${CKPT:-NAVA.safetensors}"
CONFIG="${CONFIG:-configs/nava_seedtts.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/seedtts/${LANG}}"
DATA_FILE="${DATA_FILE:-infer_cases/meta.lst}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29508}"
NPROC="${NPROC:-8}"

if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] DATA_FILE not found: $DATA_FILE" >&2
    echo "        See infer_cases/seedtts/README.md for the expected layout." >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "[INFO] Repo:    $REPO_ROOT"
echo "[INFO] Config:  $CONFIG"
echo "[INFO] Ckpt:    $CKPT"
echo "[INFO] Lang:    $LANG"
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
    --data_format meta \
    --data_file "$DATA_FILE" \
    --steps 50 \
    --save_sample \
    --gen_turn 1 \
    --seedtts_mode \
    --timbre_cfg \
    --use_sp
