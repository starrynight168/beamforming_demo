#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

MODE="${1:-bias}"
TRACE_GPU="${TRACE_GPU:-7}"
TRACE_FRAMES="${TRACE_FRAMES:-10}"
TRACE_RUNS="${TRACE_RUNS:-10}"
SPLIT_FILE="${SPLIT_FILE:-}"
OUT_ROOT="$ROOT/train/results/SI/evaluation"

resolve_split
require_file "$DATA_H5"

trace() {
    local name="$1" output="$2"
    shift 2
    mkdir -p "$output"
    require_file "$ROOT/$1"
    shift
    CUDA_VISIBLE_DEVICES="$TRACE_GPU" "$PYTHON" -u train/evaluate_mban.py \
        --mode trace \
        "$@" \
        --split-file "$SPLIT_FILE" \
        --config "$CONFIG" \
        --eval-config "$HARDWARE_CONFIG" \
        --profile common \
        --output "$output" \
        --test-frames "$TRACE_FRAMES" \
        --runs "$TRACE_RUNS" \
        --seed 42 \
        --micro-batch 8192 \
        --device cuda:0 \
        --verification-atol 1.0e-6 \
        --verification-ssim-atol 1.0e-8
    echo "[DONE] trace=$name output=$output"
}

case "$MODE" in
    bias)
        require_file "$ROOT/train/results/SI/model/bias/QAT4/fc123/best_val.pth"
        trace bias "$OUT_ROOT/hardware_trace_comparison" \
            "train/results/SI/model/bias/QAT4/fc2/best_val.pth" \
            --model "QAT4_bias_fc2=$ROOT/train/results/SI/model/bias/QAT4/fc2/best_val.pth" \
            --model "QAT4_bias_fc123=$ROOT/train/results/SI/model/bias/QAT4/fc123/best_val.pth"
    ;;
    windows)
        windows_checkpoint="train/results/SI/model/windows/QAT4/fc2/best_val.pth"
        trace windows "$OUT_ROOT/evaluation/windows/hardware_trace_progression" "$windows_checkpoint" \
            --model "QAT4_windows=$ROOT/$windows_checkpoint"
        ;;
    *) echo "usage: $0 {bias|windows}" >&2; exit 2 ;;
esac
