#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

ACTION="${1:-all}"
RUN_ID="${RUN_ID:-matrix_l1_fc2_nolinf_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${RUN_ROOT_OVERRIDE:-$ROOT/train/results/model_ablation/$RUN_ID}"
LOG_ROOT="$OUT_ROOT/logs"
SPLIT_FILE="$OUT_ROOT/shared_split_seed42_0p7-0p2-0p1.csv"
H_VALUES=(32 48 64)
OC_VALUES=(8 16 24 32 48 64 160)
MODEL_GPUS="${MODEL_GPUS:-0,1,2,3,4,5,6,7}"
MODEL_MAX_PARALLEL="${MODEL_MAX_PARALLEL:-8}"
MODEL_EPOCHS="${MODEL_EPOCHS:-100}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

make_split() {
    if [[ -f "$SPLIT_FILE" && "${FORCE_SPLIT:-0}" != 1 ]]; then
        echo "[SPLIT_READY] $SPLIT_FILE"
        return
    fi
    SPLIT_FILE="$SPLIT_FILE" "$PYTHON" - <<'PY'
import os
from pathlib import Path

from train.mban_core.config import load_config, runtime
from train.mban_core.data import load_or_create_mixed_split

split_file = Path(os.environ["SPLIT_FILE"])
cfg = load_config(
    Path("train/config.yaml"),
    overrides=[
        "mode=software",
        "normalization=l1",
        "normalization_layers=[fc2]",
        "control_normalization=none",
        "inter_layer=none",
        f"output_directory={split_file.parent}",
        f"split_file={split_file}",
        "make_new_split=true",
    ],
    profile_name="ideal",
    hardware_config_path=Path("train/hardware_eval.yaml"),
)
runtime.args = cfg
load_or_create_mixed_split(
    cfg.h5_files,
    cfg.train_ratio,
    cfg.validation_ratio,
    cfg.test_ratio,
    split_file=split_file,
    seed=cfg.seed,
    make_new_split=True,
)
print(f"[SPLIT_READY] {split_file}")
PY
}

run_job() {
    local phase="$1" gpu="$2" hidden="$3" controls="$4"
    local output="$OUT_ROOT/$phase/H${hidden}/OC${controls}"
    local log="$LOG_ROOT/${phase}_H${hidden}_OC${controls}.log"
    mkdir -p "$output"
    {
        echo "[START] phase=$phase gpu=$gpu hidden=$hidden controls=$controls"
        if [[ "$phase" == fp32 ]]; then
            if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u train/mban.py \
                --config "$CONFIG" \
                --eval-config "$HARDWARE_CONFIG" \
                --mode software \
                --profile ideal \
                --inter-layer none \
                --set epochs="$MODEL_EPOCHS" \
                --set hidden_width="$hidden" \
                --set output_controls="$controls" \
                --set normalization=l1 \
                --set normalization_layers='[fc2]' \
                --set control_normalization=none \
                --set split_file="$SPLIT_FILE" \
                --set make_new_split=false \
                --set resume=none \
                --set initial_checkpoint=null \
                --set save_images_every_epochs=0 \
                --set output_directory="$output"; then
                status=0
            else
                status=$?
            fi
        elif [[ "$phase" == qat4common ]]; then
            local checkpoint="$OUT_ROOT/fp32/H${hidden}/OC${controls}/best_val.pth"
            if [[ ! -f "$checkpoint" ]]; then
                echo "[ERROR] missing checkpoint: $checkpoint"
                return 2
            fi
            if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u train/mban.py \
                --config "$CONFIG" \
                --eval-config "$HARDWARE_CONFIG" \
                --mode qat \
                --profile common \
                --inter-layer none \
                --set epochs="$MODEL_EPOCHS" \
                --set hidden_width="$hidden" \
                --set output_controls="$controls" \
                --set normalization=l1 \
                --set normalization_layers='[fc2]' \
                --set control_normalization=none \
                --set split_file="$SPLIT_FILE" \
                --set make_new_split=false \
                --set resume=none \
                --set initial_checkpoint="$checkpoint" \
                --set save_images_every_epochs=0 \
                --set output_directory="$output"; then
                status=0
            else
                status=$?
            fi
        else
            echo "[ERROR] unsupported phase: $phase"
            return 2
        fi
        echo "[DONE] phase=$phase gpu=$gpu hidden=$hidden controls=$controls exit=$status"
    } >"$log" 2>&1
    return "$status"
}

run_phase() {
    local phase="$1"
    local -a gpu_list pids=()
    local index=0 hidden controls gpu
    IFS=',' read -r -a gpu_list <<< "$MODEL_GPUS"
    for hidden in "${H_VALUES[@]}"; do
        for controls in "${OC_VALUES[@]}"; do
            gpu="${gpu_list[$((index % ${#gpu_list[@]}))]}"
            run_job "$phase" "$gpu" "$hidden" "$controls" &
            pids+=("$!")
            index=$((index + 1))
            if (( ${#pids[@]} >= MODEL_MAX_PARALLEL )); then
                wait "${pids[0]}"
                pids=("${pids[@]:1}")
            fi
        done
    done
    ((${#pids[@]} == 0)) || wait_for_pids "${pids[@]}"
}

case "$ACTION" in
    fp32)
        make_split
        run_phase fp32
        ;;
    qat4)
        [[ -f "$SPLIT_FILE" ]] || { echo "[ERROR] missing split: $SPLIT_FILE" >&2; exit 2; }
        run_phase qat4common
        ;;
    all)
        make_split
        run_phase fp32
        run_phase qat4common
        ;;
    *) echo "usage: $0 {fp32|qat4|all}" >&2; exit 2 ;;
esac
