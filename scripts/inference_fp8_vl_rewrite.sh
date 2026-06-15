#!/bin/bash
# ============================================================
# NAVA Inference — FP8 + Prompt Rewrite + VL Image Caption
#
# Same FP8 + rewrite path as inference_fp8_rewrite.sh, plus a VL
# captioner that runs on samples with image_path: the VL model
# describes the image scene, that scene is composed with the user's
# original prompt, and the composed text feeds the rewriter.
# Samples without image_path fall through to plain rewrite.
#
# Pipeline (rank 0):
#   image_path? ─yes─► VL caption ─► compose(scene, user_prompt)
#                                            │
#   user_prompt ─no──────────────────────────┴──► rewriter ─► broadcast
#
# Per-sample log lines (rank 0 only):
#   [Captioner] IN  image: <path>
#   [Captioner] Done in X.Xs (NN tokens)
#   [Captioner] OUT (XX chars): <场景描述>
#   [Compose]   OUT (XX chars): <场景描述> <用户 prompt>
#   [Rewriter]  IN  (XX chars): <composed>
#   [Rewriter]  Done in X.Xs (NN tokens)
#   [Rewriter]  OUT (XX chars): <最终 prompt>
#
# Override defaults with env vars:
#   CKPT, CONFIG, DATA_FILE, OUT_DIR, NPROC,
#   REWRITE_MODEL, VL_MODEL, VAE_TILE_H, VAE_TILE_W
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA_fp8.safetensors}"
CONFIG="${CONFIG:-configs/nava.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/fp8_vl_rewrite}"
DATA_FILE="${DATA_FILE:-infer_cases/general/prompts_simple_i2v.jsonl}"
REWRITE_MODEL="${REWRITE_MODEL:-${REPO_ROOT}/pe_src/Qwen3-4B-Instruct-2507}"
VL_MODEL="${VL_MODEL:-${REPO_ROOT}/pe_src/Qwen3-VL-4B-Instruct}"

VAE_TILE_H="${VAE_TILE_H:-22}"
VAE_TILE_W="${VAE_TILE_W:-40}"
VAE_STRIDE_H="${VAE_STRIDE_H:-14}"
VAE_STRIDE_W="${VAE_STRIDE_W:-26}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29509}"
NPROC="${NPROC:-8}"

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] FP8 checkpoint not found: $CKPT" >&2
    echo "        Run: python -m NAVA_FP8.convert_to_fp8 -i NAVA.safetensors -o $CKPT" >&2
    exit 1
fi
if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] DATA_FILE not found: $DATA_FILE" >&2
    exit 1
fi
if [ ! -d "$VL_MODEL" ] && [[ "$VL_MODEL" != */* || "$VL_MODEL" == http* ]]; then
    : # HF repo id, let transformers handle download
elif [ ! -d "$VL_MODEL" ]; then
    echo "[ERROR] VL_MODEL not found: $VL_MODEL" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "[INFO] Repo:          $REPO_ROOT"
echo "[INFO] Config:        $CONFIG"
echo "[INFO] Ckpt:          $CKPT  (fp8_e4m3fn)"
echo "[INFO] Data:          $DATA_FILE"
echo "[INFO] Out dir:       $OUT_DIR"
echo "[INFO] Mode:          FP8 + Prompt Rewrite + VL Caption + T5 offload + VAE tiling"
echo "[INFO] Rewrite model: $REWRITE_MODEL"
echo "[INFO] VL model:      $VL_MODEL"
echo "[INFO] VAE tile:      ${VAE_TILE_H}x${VAE_TILE_W}  stride ${VAE_STRIDE_H}x${VAE_STRIDE_W}"

source "$SCRIPT_DIR/_cfg_args.sh"

SETUPTOOLS_USE_DISTUTILS=stdlib torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    inference_nava.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --weight_dtype fp8_e4m3fn \
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
    --vae_tiling \
    --vae_tile_size "$VAE_TILE_H" "$VAE_TILE_W" \
    --vae_tile_stride "$VAE_STRIDE_H" "$VAE_STRIDE_W" \
    --rewrite \
    --rewrite_model "$REWRITE_MODEL" \
    --vl_rewrite \
    --vl_model "$VL_MODEL" \
    $CFG_EXTRA_ARGS
