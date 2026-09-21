#!/usr/bin/env bash
set -euo pipefail

SERVER_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PROJECT_ROOT:-$SERVER_JOBS_DIR/../..}"
ROOT="$(cd -- "$ROOT" && pwd)"
export PROJECT_ROOT="$ROOT"
if [[ -z "${PYTHON:-}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "[ERROR] Python not found; set PYTHON=/path/to/python" >&2
        exit 2
    fi
fi
export PYTHON
TRAIN_ROOT="$ROOT/train"
ROOT_CONFIG="$ROOT/config.yaml"
CONFIG="$TRAIN_ROOT/config.yaml"
HARDWARE_CONFIG="$TRAIN_ROOT/hardware_eval.yaml"
DATA_H5="${DATA_H5:-$ROOT/data/volunteer_005.h5}"

if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$ROOT"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "[ERROR] missing file: $1" >&2
        return 2
    fi
}

require_checkpoints() {
    local root="$1"
    shift
    local label
    for label in "$@"; do
        require_file "$root/$label/best_val.pth"
    done
}

resolve_split() {
    if [[ -n "${SPLIT_FILE:-}" ]]; then
        require_file "$SPLIT_FILE"
        return
    fi
    local candidate
    for candidate in \
        "$ROOT/train/results/SI/shared_split_seed42_0p7-0p2-0p1.csv" \
        "$ROOT/train/results/model_ablation/shared_split_seed42_0p7-0p2-0p1.csv"; do
        if [[ -f "$candidate" ]]; then
            SPLIT_FILE="$candidate"
            export SPLIT_FILE
            return
        fi
    done
    echo "[ERROR] no shared split file found; set SPLIT_FILE=/path/to/shared_split.csv" >&2
    return 2
}

wait_for_pids() {
    local status=0 pid
    for pid in "$@"; do
        wait "$pid" || status=1
    done
    return "$status"
}

require_file "$CONFIG"
require_file "$HARDWARE_CONFIG"
