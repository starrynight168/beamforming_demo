#!/usr/bin/env bash
# Daemon launcher for mainline study on amax
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

LOG_FILE="$TRAIN_ROOT/results/SI/logs/adc_bit/adc_bit_daemon.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=== [$(date)] Starting Mainline ADC Study Daemon ===" >> "$LOG_FILE"
exec "$PYTHON" -u "$SCRIPT_DIR/run_mainline_adc_study.py" >> "$LOG_FILE" 2>&1
