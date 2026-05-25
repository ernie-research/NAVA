#!/bin/bash
# Start vLLM server for Qwen3-4B-Thinking-2507 rewriter
# Usage:
#   bash start_server.sh                       # standalone (full GPU)
#   bash start_server.sh --gpu 0 --low-footprint
#                                              # share GPU with backbone (~12GB)
#   bash start_server.sh --gpu 0 --gpu-util 0.2 --enforce-eager --max-model-len 2048

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults from config
MODEL="./Qwen3-4B-Thinking-2507"
PORT=8000
HOST="0.0.0.0"
TP_SIZE=1
MAX_MODEL_LEN=8192
GPU_UTIL=0.9
QUANTIZATION=""
GPU_ID="0"
ENFORCE_EAGER=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --tp) TP_SIZE="$2"; shift 2;;
        --quantization) QUANTIZATION="--quantization $2"; shift 2;;
        --gpu) GPU_ID="$2"; shift 2;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
        --gpu-util) GPU_UTIL="$2"; shift 2;;
        --enforce-eager) ENFORCE_EAGER="--enforce-eager"; shift 1;;
        # Convenience preset for sharing a GPU with the 8-GPU backbone:
        # ~12GB ceiling, eager mode (no CUDA-graph capture, less stream contention),
        # smaller KV cache. Backbone on the same GPU sees ~10-15% slowdown.
        --low-footprint)
            # Bigger ceiling so thinking + rewrite both fit. KV cache grows
            # with max-model-len, but at gpu_util=0.18 on an 80GB card we
            # still stay well under the 8-GPU backbone's headroom.
            GPU_UTIL="0.18"
            MAX_MODEL_LEN="8192"
            ENFORCE_EAGER="--enforce-eager"
            shift 1;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "========================================="
echo " Starting vLLM Server"
echo " Model: $MODEL"
echo " Port: $PORT"
echo " GPU: $GPU_ID (TP=$TP_SIZE)"
echo " Max Model Len: $MAX_MODEL_LEN"
echo " GPU Util: $GPU_UTIL"
echo " Eager: ${ENFORCE_EAGER:-no}"
echo " Quantization: ${QUANTIZATION:-none}"
echo "========================================="

# Check if port is already in use
if lsof -i :"$PORT" >/dev/null 2>&1; then
    echo "[WARN] Port $PORT is already in use. Stop existing server first."
    echo "       Run: bash stop_server.sh"
    exit 1
fi

# Start server in background
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    $ENFORCE_EAGER \
    $QUANTIZATION \
    > "$SCRIPT_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$SCRIPT_DIR/server.pid"
echo "[INFO] Server PID: $SERVER_PID"
echo "[INFO] Log file: $SCRIPT_DIR/server.log"

# Wait for server to be ready
echo "[INFO] Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[INFO] Server is ready! (took ${i}s)"
        echo "[INFO] API endpoint: http://localhost:$PORT/v1/chat/completions"
        exit 0
    fi
    sleep 1
done

echo "[ERROR] Server failed to start within 120s. Check server.log"
exit 1
