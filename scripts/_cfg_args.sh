# ============================================================
# Shared CFG knobs for inference_nava.py.
# Source this file from any scripts/inference*.sh after defining
# (or inheriting) the env vars below; it appends matching CLI flags
# to $CFG_EXTRA_ARGS, which the caller then expands into the
# torchrun ... inference_nava.py invocation.
#
# Env vars (all empty by default → fall back to YAML):
#   VIDEO_CFG, AUDIO_CFG, VIDEO_ALIGN_CFG, AUDIO_ALIGN_CFG
#   ALIGN_3D_CFG ∈ {"on","off",""}
# ============================================================

CFG_EXTRA_ARGS="${CFG_EXTRA_ARGS:-}"

if [ -n "${VIDEO_CFG:-}" ]; then
    CFG_EXTRA_ARGS+=" --video_guidance_scale ${VIDEO_CFG}"
fi
if [ -n "${AUDIO_CFG:-}" ]; then
    CFG_EXTRA_ARGS+=" --audio_guidance_scale ${AUDIO_CFG}"
fi
if [ -n "${VIDEO_ALIGN_CFG:-}" ]; then
    CFG_EXTRA_ARGS+=" --video_align_guidance_scale ${VIDEO_ALIGN_CFG}"
fi
if [ -n "${AUDIO_ALIGN_CFG:-}" ]; then
    CFG_EXTRA_ARGS+=" --audio_align_guidance_scale ${AUDIO_ALIGN_CFG}"
fi
if [ -n "${ALIGN_3D_CFG:-}" ]; then
    case "$ALIGN_3D_CFG" in
        on|off) CFG_EXTRA_ARGS+=" --align_3d_cfg ${ALIGN_3D_CFG}" ;;
        *) echo "[WARN] ALIGN_3D_CFG must be 'on' or 'off' (got: $ALIGN_3D_CFG); ignoring." >&2 ;;
    esac
fi
