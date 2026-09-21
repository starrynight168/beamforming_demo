#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

ACTION="${1:-all}"
MODEL_ROOT="$ROOT/train/results/SI/model/windows"
EVAL_ROOT="$ROOT/train/results/SI/evaluation/windows"
SPLIT_FILE="${SPLIT_FILE:-}"
WINDOWS_GPU_K8="${WINDOWS_GPU_K8:-7}"
WINDOWS_GPU_DIRECT="${WINDOWS_GPU_DIRECT:-2}"
WINDOWS_EVAL_GPU="${WINDOWS_EVAL_GPU:-7}"
WINDOWS_TEST_FRAMES="${WINDOWS_TEST_FRAMES:-50}"

resolve_split
require_file "$DATA_H5"
mkdir -p "$MODEL_ROOT" "$EVAL_ROOT"

run_step() {
    local mode="$1" output="$2" gpu="$3" controls="$4" checkpoint="${5:-}"
    local -a args=(
        --config "$CONFIG"
        --eval-config "$HARDWARE_CONFIG"
        --mode "$mode"
        --set epochs=100
        --set learning_rate=0.001
        --set scheduler=none
        --set hidden_width=32
        --set hidden_layers=2
        --set normalization=none
        --set normalization_layers=[]
        --set centering=none
        --set centering_layers=[]
        --set control_normalization=none
        --set bias_layers='[fc2]'
        --set output_controls="$controls"
        --set resume=none
        --set split_file="$SPLIT_FILE"
        --set h5_files="[\"$DATA_H5\"]"
        --set output_directory="$output"
        --set save_images_every_epochs=0
    )
    if [[ "$mode" == qat ]]; then
        require_file "$checkpoint"
        args+=(--profile common --inter-layer none --set initial_checkpoint="$checkpoint")
    elif [[ "$mode" != software ]]; then
        echo "[ERROR] unsupported mode: $mode" >&2
        return 2
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u train/mban.py "${args[@]}"
}

run_pipeline() {
    local name="$1" controls="$2" gpu="$3"
    local log="$MODEL_ROOT/logs/${name}.log"
    mkdir -p "$MODEL_ROOT/FP32/$name" "$MODEL_ROOT/QAT4/$name" "$(dirname -- "$log")"
    {
        echo "[START] pipeline=$name gpu=$gpu output_controls=$controls"
        run_step software "$MODEL_ROOT/FP32/$name" "$gpu" "$controls"
        require_file "$MODEL_ROOT/FP32/$name/best_val.pth"
        run_step qat "$MODEL_ROOT/QAT4/$name" "$gpu" "$controls" "$MODEL_ROOT/FP32/$name/best_val.pth"
        echo "[DONE] pipeline=$name"
    } >"$log" 2>&1
}

train_pipelines() {
    run_pipeline fc2 8 "$WINDOWS_GPU_K8" &
    local pid_k8=$!
    run_pipeline direct_full_H32 0 "$WINDOWS_GPU_DIRECT" &
    local pid_direct=$!
    wait_for_pids "$pid_k8" "$pid_direct"
}

evaluate_group() {
    local group="$1" model_type="$2" invivo_mode="$3" runs="$4"
    local invivo_output="$EVAL_ROOT/invivo/$group" scenes_output="$EVAL_ROOT/scenes/$group"
    local -a model_args
    if [[ "$group" == fp32 ]]; then
        model_args=(
            --model "K8_FP32=$MODEL_ROOT/FP32/fc2/best_val.pth"
            --model "DF_FP32=$MODEL_ROOT/FP32/direct_full_H32/best_val.pth"
        )
        require_file "$MODEL_ROOT/FP32/fc2/best_val.pth"
        require_file "$MODEL_ROOT/FP32/direct_full_H32/best_val.pth"
    elif [[ "$group" == qat4 ]]; then
        model_args=(
            --model "K8_QAT4=$MODEL_ROOT/QAT4/fc2/best_val.pth"
            --model "DF_QAT4=$MODEL_ROOT/QAT4/direct_full_H32/best_val.pth"
        )
        require_file "$MODEL_ROOT/QAT4/fc2/best_val.pth"
        require_file "$MODEL_ROOT/QAT4/direct_full_H32/best_val.pth"
    else
        echo "[ERROR] unsupported evaluation group: $group" >&2
        return 2
    fi
    mkdir -p "$invivo_output" "$scenes_output"
    local -a invivo_args=(
        --mode "$invivo_mode" --model-type "$model_type" "${model_args[@]}"
        --split-file "$SPLIT_FILE" --config "$CONFIG" --eval-config "$HARDWARE_CONFIG"
        --profile common --output "$invivo_output" --test-frames "$WINDOWS_TEST_FRAMES"
        --seed 42 --micro-batch 8192 --device cuda:0
    )
    [[ "$invivo_mode" == mc ]] && invivo_args+=(--runs "$runs")
    CUDA_VISIBLE_DEVICES="$WINDOWS_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py "${invivo_args[@]}"
    local -a scenes_args=(
        --mode scenes --model-type "$model_type" "${model_args[@]}"
        --config "$CONFIG" --eval-config "$HARDWARE_CONFIG" --profile common
        --output "$scenes_output" --scene-config config.yaml
        --scenes experiments_contrast_speckle,experiments_resolution_distorsion,simulation_contrast_speckle,simulation_resolution_distorsion
        --seed 42 --micro-batch 8192 --device cuda:0
    )
    [[ "$runs" -gt 1 ]] && scenes_args+=(--monte-carlo-runs "$runs")
    CUDA_VISIBLE_DEVICES="$WINDOWS_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py "${scenes_args[@]}"
}

evaluate_models() {
    local log="$EVAL_ROOT/logs/evaluation.log"
    mkdir -p "$(dirname -- "$log")"
    {
        echo "[START] windows evaluation gpu=$WINDOWS_EVAL_GPU"
        evaluate_group fp32 fp32 test 1
        evaluate_group qat4 qat mc 30
        echo "[DONE] windows evaluation"
    } >"$log" 2>&1
}

case "$ACTION" in
    train) train_pipelines ;;
    eval) evaluate_models ;;
    all) train_pipelines; evaluate_models ;;
    *) echo "usage: $0 {train|eval|all}" >&2; exit 2 ;;
esac
