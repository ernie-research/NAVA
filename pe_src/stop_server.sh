#!/bin/bash
# Stop vLLM server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[INFO] Stopping server (PID: $PID)..."
        kill "$PID"
        sleep 2
        # Force kill if still alive
        if kill -0 "$PID" 2>/dev/null; then
            echo "[WARN] Force killing..."
            kill -9 "$PID"
        fi
        echo "[INFO] Server stopped."
    else
        echo "[INFO] Server process not found (PID: $PID already exited)."
    fi
    rm -f "$PID_FILE"
else
    echo "[INFO] No PID file found. Trying to kill by port..."
    PORT=${1:-8000}
    PID=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "[INFO] Killing process on port $PORT (PID: $PID)..."
        kill "$PID"
        echo "[INFO] Done."
    else
        echo "[INFO] No process found on port $PORT."
    fi
fi
