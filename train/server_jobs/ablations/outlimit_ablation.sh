#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

ACTION="${1:-all}"
BASE_ROOT="$ROOT/train/results/SI"
MODEL_ROOT="$BASE_ROOT/model/outlimit"
EVAL_ROOT="$ROOT/train/results/SI/evaluation/outlimit"
SPLIT_FILE="${SPLIT_FILE:-}"
OUTLIMIT_GPUS="${OUTLIMIT_GPUS:-1,2,3,7}"
OUTLIMIT_EVAL_GPU="${OUTLIMIT_EVAL_GPU:-7}"
OUTLIMIT_TEST_FRAMES="${OUTLIMIT_TEST_FRAMES:-10}"
CASES=(hard_nonnegative hard_signed free_nonnegative free_signed)
DOMAINS=(nonnegative signed nonnegative signed)
UNITY=(hard hard free free)

resolve_split
require_file "$DATA_H5"
mkdir -p "$MODEL_ROOT" "$EVAL_ROOT"

run_case() {
    local index="$1" gpu="$2" name="${CASES[$index]}"
    local output="$MODEL_ROOT/$name" log="$MODEL_ROOT/logs/${name}.log"
    mkdir -p "$output" "$(dirname -- "$log")"
    {
        echo "[START] case=$name gpu=$gpu output_domain=${DOMAINS[$index]} unity_constraint=${UNITY[$index]}"
        if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u train/mban.py \
            --config "$CONFIG" \
            --eval-config "$HARDWARE_CONFIG" \
            --mode software \
            --set output_directory="$output" \
            --set resume=none \
            --set hidden_width=32 \
            --set hidden_layers=2 \
            --set output_controls=8 \
            --set normalization=none \
            --set normalization_layers=[] \
            --set centering=none \
            --set centering_layers=[] \
            --set control_normalization=none \
            --set bias_layers='[fc2]' \
            --set output_domain="${DOMAINS[$index]}" \
            --set unity_constraint="${UNITY[$index]}" \
            --set split_file="$SPLIT_FILE" \
            --set h5_files="[\"$DATA_H5\"]" \
            --set save_images_every_epochs=0 \
            --set epochs=100; then
            status=0
        else
            status=$?
        fi
        echo "[DONE] case=$name exit=$status"
    } >"$log" 2>&1
    return "$status"
}

train_cases() {
    local -a gpu_list pids=()
    local index gpu
    IFS=',' read -r -a gpu_list <<< "$OUTLIMIT_GPUS"
    for index in "${!CASES[@]}"; do
        gpu="${gpu_list[$((index % ${#gpu_list[@]}))]}"
        run_case "$index" "$gpu" &
        pids+=("$!")
    done
    wait_for_pids "${pids[@]}"
}

build_models() {
    local name path
    MODEL_ARGS=()
    for name in "${CASES[@]}"; do
        path="$MODEL_ROOT/$name/best_val.pth"
        require_file "$path"
        MODEL_ARGS+=(--model "$name=$path")
    done
}

evaluate_cases() {
    local log="$EVAL_ROOT/logs/mc10.log"
    build_models
    mkdir -p "$EVAL_ROOT/invivo" "$EVAL_ROOT/scenes" "$(dirname -- "$log")"
    {
        echo "[START] outlimit evaluation gpu=$OUTLIMIT_EVAL_GPU"
        CUDA_VISIBLE_DEVICES="$OUTLIMIT_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py \
            --mode test \
            --model-type fp32 \
            "${MODEL_ARGS[@]}" \
            --split-file "$SPLIT_FILE" \
            --config "$CONFIG" \
            --eval-config "$HARDWARE_CONFIG" \
            --profile common \
            --output "$EVAL_ROOT/invivo" \
            --test-frames "$OUTLIMIT_TEST_FRAMES" \
            --seed 42 \
            --micro-batch 8192 \
            --device cuda:0
        CUDA_VISIBLE_DEVICES="$OUTLIMIT_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py \
            --mode scenes \
            --model-type fp32 \
            "${MODEL_ARGS[@]}" \
            --config "$CONFIG" \
            --eval-config "$HARDWARE_CONFIG" \
            --profile common \
            --output "$EVAL_ROOT/scenes" \
            --scene-config config.yaml \
            --scenes experiments_contrast_speckle,experiments_resolution_distorsion,simulation_contrast_speckle,simulation_resolution_distorsion \
            --seed 42 \
            --micro-batch 8192 \
            --device cuda:0
        echo "[DONE] outlimit evaluation"
    } >"$log" 2>&1
}

case "$ACTION" in
    train) train_cases ;;
    eval) evaluate_cases ;;
    all) train_cases; evaluate_cases ;;
    *) echo "usage: $0 {train|eval|all}" >&2; exit 2 ;;
esac
