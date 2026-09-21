#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

ACTION="${1:-all}"
MODEL_ROOT="$ROOT/train/results/SI/model/loss"
FP32_ROOT="$MODEL_ROOT/FP32"
EVAL_ROOT="$ROOT/train/results/SI/evaluation/loss"
SPLIT_FILE="${SPLIT_FILE:-}"
LOSS_GPUS="${LOSS_GPUS:-0,1,2,3,4,5}"
LOSS_EVAL_GPU="${LOSS_EVAL_GPU:-3}"
LOSS_TEST_FRAMES="${LOSS_TEST_FRAMES:-10}"
LOSS_MC_RUNS="${LOSS_MC_RUNS:-10}"

LABELS=(none d1 d2 d1_d2 lrange d1_lrange)
D1_WEIGHTS=(0.0 0.01 0.0 0.01 0.0 0.01)
D2_WEIGHTS=(0.0 0.0 0.01 0.01 0.0 0.0)
LRANGE_WEIGHTS=(0.0 0.0 0.0 0.0 0.01 0.01)

resolve_split
require_file "$DATA_H5"
mkdir -p "$MODEL_ROOT" "$EVAL_ROOT"

run_training() {
    local stage="$1" index="$2" gpu="$3"
    local label="${LABELS[$index]}"
    local d1_w="${D1_WEIGHTS[$index]}"
    local d2_w="${D2_WEIGHTS[$index]}"
    local lr_w="${LRANGE_WEIGHTS[$index]}"

    # 核心原则：默认全流程采用 none；仅当显式对比 linf 时（如 QAT4linf）采用 linf
    local control_normalization=none
    [[ "$stage" == *"linf"* ]] && control_normalization=linf
    local loss_config="{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: 0.1}, unity: {weight: 0.0}, d1: {domain: shape, weight: $d1_w}, d2: {domain: shape, weight: $d2_w}, lrange: {limit: 8.0, weight: $lr_w}}}"

    local output="$MODEL_ROOT/$stage/$label" log="$MODEL_ROOT/$stage/logs/${label}.log"
    mkdir -p "$output" "$(dirname -- "$log")"

    local -a train_cmd
    if [[ "$stage" == "FP32" ]]; then
        train_cmd=(
            "$PYTHON" -u train/mban.py
            --config "$CONFIG"
            --eval-config "$HARDWARE_CONFIG"
            --mode software
            --profile common
            --inter-layer none
            --set output_directory="$output"
            --set resume=none
            --set hidden_width=32
            --set hidden_layers=2
            --set output_controls=8
            --set bias_layers='[fc2]'
            --set normalization=none
            --set normalization_layers=[]
            --set control_normalization="$control_normalization"
            --set "loss=$loss_config"
            --set split_file="$SPLIT_FILE"
            --set "h5_files=[\"$DATA_H5\"]"
            --set save_images_every_epochs=0
            --set epochs=100
        )
    elif [[ "$stage" == "QAT4" || "$stage" == "QAT4linf" ]]; then
        local ckpt="$FP32_ROOT/$label/best_val.pth"
        require_file "$ckpt"
        train_cmd=(
            "$PYTHON" -u train/mban.py
            --config "$CONFIG"
            --eval-config "$HARDWARE_CONFIG"
            --mode qat
            --profile common
            --inter-layer none
            --set output_directory="$output"
            --set resume=none
            --set initial_checkpoint="$ckpt"
            --set hidden_width=32
            --set hidden_layers=2
            --set output_controls=8
            --set bias_layers='[fc2]'
            --set normalization=none
            --set normalization_layers=[]
            --set control_normalization="$control_normalization"
            --set "loss=$loss_config"
            --set split_file="$SPLIT_FILE"
            --set save_images_every_epochs=0
            --set epochs=100
        )
    else
        echo "[ERROR] unknown training stage: $stage" >&2
        return 2
    fi

    {
        echo "[START] stage=$stage label=$label gpu=$gpu d1=$d1_w d2=$d2_w lrange=$lr_w ctrl_norm=$control_normalization"
        if CUDA_VISIBLE_DEVICES="$gpu" "${train_cmd[@]}"; then
            status=0
        else
            status=$?
        fi
        echo "[DONE] stage=$stage label=$label exit=$status"
    } >"$log" 2>&1
    return "$status"
}

train_stage() {
    local stage="$1"
    local -a gpu_list pids=()
    local index gpu
    IFS=',' read -r -a gpu_list <<< "$LOSS_GPUS"
    for index in "${!LABELS[@]}"; do
        gpu="${gpu_list[$((index % ${#gpu_list[@]}))]}"
        run_training "$stage" "$index" "$gpu" &
        pids+=("$!")
    done
    wait_for_pids "${pids[@]}"
}

build_models() {
    local root="$1" label path
    require_checkpoints "$root" "${LABELS[@]}"
    MODEL_ARGS=()
    for label in "${LABELS[@]}"; do
        path="$root/$label/best_val.pth"
        MODEL_ARGS+=(--model "$label=$path")
    done
}

evaluate_stage() {
    local stage="$1" model_root="$2"
    local output="$EVAL_ROOT/$stage" log="$EVAL_ROOT/logs/${stage}.log"
    build_models "$model_root"
    mkdir -p "$output/invivo" "$output/scenes" "$(dirname -- "$log")"
    {
        echo "[START] stage=$stage models=$((${#MODEL_ARGS[@]} / 2)) gpu=$LOSS_EVAL_GPU"
        local model_type invivo_mode
        if [[ "$stage" == "FP32" ]]; then
            model_type=fp32
            invivo_mode=test
        elif [[ "$stage" == "QAT4" || "$stage" == "QAT4linf" ]]; then
            model_type=qat
            invivo_mode=mc
        else
            echo "[ERROR] unknown evaluation stage: $stage" >&2
            return 2
        fi
        local -a invivo_args=(
            --mode "$invivo_mode" --model-type "$model_type" "${MODEL_ARGS[@]}"
            --split-file "$SPLIT_FILE" --config "$CONFIG" --eval-config "$HARDWARE_CONFIG"
            --profile common --output "$output/invivo" --test-frames "$LOSS_TEST_FRAMES"
            --seed 42 --micro-batch 8192 --device cuda:0
        )
        [[ "$invivo_mode" == mc ]] && invivo_args+=(--runs "$LOSS_MC_RUNS")
        CUDA_VISIBLE_DEVICES="$LOSS_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py "${invivo_args[@]}"
        local -a scenes_args=(
            --mode scenes --model-type "$model_type" "${MODEL_ARGS[@]}"
            --config "$CONFIG" --eval-config "$HARDWARE_CONFIG" --profile common
            --output "$output/scenes" --scene-config config.yaml
            --scenes experiments_contrast_speckle,experiments_resolution_distorsion,simulation_contrast_speckle,simulation_resolution_distorsion
            --seed 42 --micro-batch 8192 --device cuda:0
        )
        [[ "$model_type" != fp32 ]] && scenes_args+=(--monte-carlo-runs "$LOSS_MC_RUNS")
        CUDA_VISIBLE_DEVICES="$LOSS_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py "${scenes_args[@]}"
        echo "[DONE] stage=$stage"
    } >"$log" 2>&1
}

case "$ACTION" in
    fp32)
        train_stage FP32
        evaluate_stage FP32 "$MODEL_ROOT/FP32"
        ;;
    qat4)
        train_stage QAT4
        evaluate_stage QAT4 "$MODEL_ROOT/QAT4"
        ;;
    qat4_linf)
        train_stage QAT4linf
        evaluate_stage QAT4linf "$MODEL_ROOT/QAT4linf"
        ;;
    train)
        train_stage FP32
        train_stage QAT4
        train_stage QAT4linf
        ;;
    eval)
        evaluate_stage FP32 "$MODEL_ROOT/FP32"
        evaluate_stage QAT4 "$MODEL_ROOT/QAT4"
        evaluate_stage QAT4linf "$MODEL_ROOT/QAT4linf"
        ;;
    all)
        train_stage FP32
        evaluate_stage FP32 "$MODEL_ROOT/FP32"
        train_stage QAT4
        evaluate_stage QAT4 "$MODEL_ROOT/QAT4"
        train_stage QAT4linf
        evaluate_stage QAT4linf "$MODEL_ROOT/QAT4linf"
        ;;
    *)
        echo "usage: $0 {fp32|qat4|qat4_linf|train|eval|all}" >&2
        exit 2
        ;;
esac
