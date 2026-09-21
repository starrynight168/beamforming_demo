#!/usr/bin/env bash
set -euo pipefail

SERVER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SERVER_DIR/lib/common.sh"
LOG="$TRAIN_ROOT/results/SI/logs/master_daemon.log"
mkdir -p "$(dirname -- "$LOG")"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Master SI Daemon..." >> "$LOG"

# Run all waves with explicit per-GPU slots; override GPU_SLOTS when needed.
if "$PYTHON" -u "$SERVER_DIR/si_master_pipeline.py" --gpu-slots "${GPU_SLOTS:-4:4,0:2}" >> "$LOG" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 13 modules trained successfully! Starting full evaluation..." >> "$LOG"
    "$PYTHON" -u "$SERVER_DIR/evaluate_all_si.py" >> "$LOG" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL MODELS AND EVALUATIONS COMPLETED SUCCESSFULLY!" >> "$LOG"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Master training failed" >> "$LOG"
    exit 1
fi
