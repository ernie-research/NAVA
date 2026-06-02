#!/bin/bash
# ============================================================
# NAVA Inference — I2AV + Timbre Control, FP8 (E4M3) weight-only
#
# Same I2AV + timbre control flow as inference_timbre.sh, but loads
# NAVA_fp8.safetensors so the DiT backbone block-Linears stay in
# float8_e4m3fn (~half the bf16 footprint). Compute path is unchanged
# — weights are dequantized to bf16 on the fly inside FP8Linear.
# Combined with --t5_offload + --vae_tiling, peak VRAM stays
# around ~18 GB on 704x1280x37 frames.
#
# Generate the fp8 checkpoint first:
#   python -m NAVA_FP8.convert_to_fp8 -i NAVA.safetensors -o NAVA_fp8.safetensors
#
# Override defaults with env vars:
#   CKPT, CONFIG, DATA_FILE, OUT_DIR, NPROC, TIMBRE_SCALE,
#   VAE_TILE_H, VAE_TILE_W, VAE_STRIDE_H, VAE_STRIDE_W
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-NAVA_fp8.safetensors}"
CONFIG="${CONFIG:-configs/nava.yaml}"
OUT_DIR="${OUT_DIR:-eval_results/timbre_fp8}"
DATA_FILE="${DATA_FILE:-infer_cases/timbre/prompts.jsonl}"

# VAE tiling: latent-space tile size (H, W). Smaller -> less VRAM, more tiles.
# Defaults (22, 40) -> 352x640-pixel tiles for 704x1280 output.
# Drop to (16, 30) if you need to push peak below 18 GB.
VAE_TILE_H="${VAE_TILE_H:-22}"
VAE_TILE_W="${VAE_TILE_W:-40}"
VAE_STRIDE_H="${VAE_STRIDE_H:-14}"
VAE_STRIDE_W="${VAE_STRIDE_W:-26}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29510}"
NPROC="${NPROC:-8}"
TIMBRE_SCALE="${TIMBRE_SCALE:-1.0}"

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] FP8 checkpoint not found: $CKPT" >&2
    echo "        Run: python -m NAVA_FP8.convert_to_fp8 -i NAVA.safetensors -o $CKPT" >&2
    exit 1
fi
if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] DATA_FILE not found: $DATA_FILE" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

echo "[INFO] Repo:        $REPO_ROOT"
echo "[INFO] Config:      $CONFIG"
echo "[INFO] Ckpt:        $CKPT  (fp8_e4m3fn)"
echo "[INFO] Data:        $DATA_FILE"
echo "[INFO] Out dir:     $OUT_DIR"
echo "[INFO] Mode:        FP8 weight-only + T5 offload + VAE tiling + timbre"
echo "[INFO] VAE tile:    ${VAE_TILE_H}x${VAE_TILE_W}  stride ${VAE_STRIDE_H}x${VAE_STRIDE_W}"
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
    --timbre_cfg \
    --timbre_align_guidance_scale "$TIMBRE_SCALE"
