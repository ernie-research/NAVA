#!/bin/bash
# ============================================================
# NAVA Inference — FP8 on Verse-Bench (3 subsets, no rewrite)
#
# The verse_bench_rewrite/*.jsonl prompts are already long Chinese
# captions, so we skip the rewriter + VL captioner entirely and feed
# the prompts straight to NAVA. Otherwise identical to
# inference_fp8_vl_rewrite.sh (FP8 ckpt, SP=8, T5 offload, VAE tiling).
#
# Output layout (one mp4 per sample, namespaced by subset):
#   $OUT_BASE/set1/<save_path>-av-0.mp4
#   $OUT_BASE/set2/<save_path>-av-0.mp4
#   $OUT_BASE/set3/<save_path>-av-0.mp4
#
# Override defaults with env vars:
#   CKPT, CONFIG, OUT_BASE, NPROC, VAE_TILE_H, VAE_TILE_W,
#   SUBSETS  (space-separated list; default "set1 set2 set3")
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA_fp8.safetensors}"
CONFIG="${CONFIG:-configs/nava.yaml}"
OUT_BASE="${OUT_BASE:-eval_results/verse_bench_fp8}"
DATA_DIR="${DATA_DIR:-infer_cases/verse_bench_rewrite}"
SUBSETS="${SUBSETS:-set1 set2 set3}"

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

# Validate every subset jsonl exists before launching anything — saves a half-
# hour SP=8 startup just to discover a missing file at sample time.
for s in $SUBSETS; do
    DATA_FILE="$DATA_DIR/${s}.jsonl"
    if [ ! -f "$DATA_FILE" ]; then
        echo "[ERROR] DATA_FILE not found: $DATA_FILE" >&2
        exit 1
    fi
done

mkdir -p "$OUT_BASE"

echo "[INFO] Repo:        $REPO_ROOT"
echo "[INFO] Config:      $CONFIG"
echo "[INFO] Ckpt:        $CKPT  (fp8_e4m3fn)"
echo "[INFO] Data dir:    $DATA_DIR"
echo "[INFO] Subsets:     $SUBSETS"
echo "[INFO] Out base:    $OUT_BASE"
echo "[INFO] Mode:        FP8 + T5 offload + VAE tiling (no rewrite — prompts already long)"
echo "[INFO] VAE tile:    ${VAE_TILE_H}x${VAE_TILE_W}  stride ${VAE_STRIDE_H}x${VAE_STRIDE_W}"

source "$SCRIPT_DIR/_cfg_args.sh"

for SUBSET in $SUBSETS; do
    DATA_FILE="$DATA_DIR/${SUBSET}.jsonl"
    OUT_DIR="$OUT_BASE/$SUBSET"
    mkdir -p "$OUT_DIR"

    echo
    echo "============================================"
    echo "[VerseBench] Subset: $SUBSET"
    echo "  Data : $DATA_FILE"
    echo "  Out  : $OUT_DIR"
    echo "============================================"

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
        $CFG_EXTRA_ARGS
done

echo
echo "============================================"
echo "[VerseBench] All subsets done → $OUT_BASE"
echo "============================================"
