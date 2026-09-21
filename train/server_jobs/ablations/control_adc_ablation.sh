#!/usr/bin/env bash
set -euo pipefail

SERVER_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PROJECT_ROOT:-$(cd -- "$SERVER_JOBS_DIR/../.." && pwd)}"
source "$SERVER_JOBS_DIR/lib/common.sh"

ACTION="${1:-all}"
BASE_ROOT="$ROOT/train/results/SI"
MODEL_ROOT="$BASE_ROOT/model/control_adc"
EVAL_ROOT="$BASE_ROOT/evaluation/control_adc"
SPLIT_FILE="$BASE_ROOT/shared_split_seed42_0p7-0p2-0p1.csv"

# 4任务并行运行：默认分配到空闲卡（可覆盖，如 CONTROL_GPUS="0,1,2,3" 或 "2,2,2,2"）
CONTROL_GPUS="${CONTROL_GPUS:-0,1,2,3}"
CONTROL_EVAL_GPU="${CONTROL_EVAL_GPU:-0}"
CONTROL_TEST_FRAMES="${CONTROL_TEST_FRAMES:-10}"
CONTROL_MC_RUNS="${CONTROL_MC_RUNS:-10}"

LABELS=(none_full_scale none_observer linf_full_scale linf_observer)
CONTROL_NORMS=(none none linf linf)
CONTROL_RANGES=(full_scale observer full_scale observer)

resolve_split
require_file "$DATA_H5"
require_file "$SPLIT_FILE"
mkdir -p "$MODEL_ROOT" "$EVAL_ROOT"

run_training() {
    local stage="$1" index="$2" gpu="$3"
    local label="${LABELS[$index]}"
    local norm="${CONTROL_NORMS[$index]}"
    local adc_range="${CONTROL_RANGES[$index]}"
    local output="$MODEL_ROOT/$stage/$label"
    local log="$MODEL_ROOT/$stage/logs/${label}.log"

    mkdir -p "$output" "$(dirname -- "$log")"

    if [[ -f "$output/best_val.pth" ]]; then
        echo "[SKIP] stage=$stage label=$label checkpoint already exists: $output/best_val.pth"
        return 0
    fi

    local -a train_cmd
    if [[ "$stage" == "FP32" ]]; then
        train_cmd=(
            "$PYTHON" -u train/mban.py
            --config "$CONFIG"
            --eval-config "$HARDWARE_CONFIG"
            --mode software
            --profile common
            --inter-layer analog
            --set bias_implementation=ordinary
            --set output_directory="$output"
            --set resume=none
            --set hidden_width=32
            --set hidden_layers=2
            --set output_controls=8
            --set control_normalization="$norm"
            --set control_adc_range="$adc_range"
            --set qat_clip_outside_grad=0.0
            --set split_file="$SPLIT_FILE"
            --set h5_files="[\"$DATA_H5\"]"
            --set save_images_every_epochs=0
            --set epochs=100
        )
    elif [[ "$stage" == "QAT4" ]]; then
        local ckpt="$MODEL_ROOT/FP32/$label/best_val.pth"
        require_file "$ckpt"
        train_cmd=(
            "$PYTHON" -u train/mban.py
            --config "$CONFIG"
            --eval-config "$HARDWARE_CONFIG"
            --mode qat
            --profile common
            --inter-layer analog
            --set bias_implementation=ordinary
            --set output_directory="$output"
            --set resume=none
            --set initial_checkpoint="$ckpt"
            --set hidden_width=32
            --set hidden_layers=2
            --set output_controls=8
            --set control_normalization="$norm"
            --set control_adc_range="$adc_range"
            --set qat_clip_outside_grad=0.0
            --set split_file="$SPLIT_FILE"
            --set h5_files="[\"$DATA_H5\"]"
            --set save_images_every_epochs=0
            --set epochs=100
        )
    else
        echo "[ERROR] unknown stage: $stage" >&2
        return 2
    fi

    {
        echo "[START] stage=$stage label=$label gpu=$gpu norm=$norm adc_range=$adc_range lrange_weight=0.0(default) qat_clip=0.0"
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
    IFS=',' read -r -a gpu_list <<< "$CONTROL_GPUS"

    echo "========================================================================"
    echo "Starting Stage: $stage (4 Parallel Tasks on GPUs: ${CONTROL_GPUS})"
    echo "========================================================================"

    for index in "${!LABELS[@]}"; do
        gpu="${gpu_list[$((index % ${#gpu_list[@]}))]}"
        echo "Launching in parallel: [$stage] ${LABELS[$index]} on GPU $gpu ..."
        run_training "$stage" "$index" "$gpu" &
        pids+=("$!")
    done

    echo "Waiting for all 4 $stage training processes: ${pids[*]} ..."
    wait_for_pids "${pids[@]}"
    echo "Stage $stage (all 4 parallel tasks) completed successfully!"
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

    local model_type invivo_mode
    if [[ "$stage" == "FP32" ]]; then
        model_type=fp32
        invivo_mode=test
    elif [[ "$stage" == "QAT4" ]]; then
        model_type=qat
        invivo_mode=mc
    else
        echo "[ERROR] unknown evaluation stage: $stage" >&2
        return 2
    fi

    local -a invivo_args=(
        --mode "$invivo_mode"
        --model-type "$model_type"
        "${MODEL_ARGS[@]}"
        --split-file "$SPLIT_FILE"
        --config "$CONFIG"
        --eval-config "$HARDWARE_CONFIG"
        --profile common
        --output "$output/invivo"
        --test-frames "$CONTROL_TEST_FRAMES"
        --seed 42
        --micro-batch 8192
        --device cuda:0
    )
    [[ "$invivo_mode" == "mc" ]] && invivo_args+=(--runs "$CONTROL_MC_RUNS")

    local -a scenes_args=(
        --mode scenes
        --model-type "$model_type"
        "${MODEL_ARGS[@]}"
        --config "$CONFIG"
        --eval-config "$HARDWARE_CONFIG"
        --profile common
        --output "$output/scenes"
        --scene-config config.yaml
        --scenes experiments_contrast_speckle,experiments_resolution_distorsion,simulation_contrast_speckle,simulation_resolution_distorsion
        --seed 42
        --micro-batch 8192
        --device cuda:0
    )
    [[ "$model_type" != "fp32" ]] && scenes_args+=(--monte-carlo-runs "$CONTROL_MC_RUNS")

    {
        echo "[START EVALUATION] stage=$stage models=$((${#MODEL_ARGS[@]} / 2)) gpu=$CONTROL_EVAL_GPU"
        CUDA_VISIBLE_DEVICES="$CONTROL_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py "${invivo_args[@]}"
        CUDA_VISIBLE_DEVICES="$CONTROL_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py "${scenes_args[@]}"
        echo "[DONE EVALUATION] stage=$stage"
    } >"$log" 2>&1
    echo "Stage $stage evaluation finished. Output in: $output"
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
    eval)
        evaluate_stage FP32 "$MODEL_ROOT/FP32"
        evaluate_stage QAT4 "$MODEL_ROOT/QAT4"
        ;;
    all)
        train_stage FP32
        evaluate_stage FP32 "$MODEL_ROOT/FP32"
        train_stage QAT4
        evaluate_stage QAT4 "$MODEL_ROOT/QAT4"
        echo "========================================================================"
        echo "All 4 combinations (FP32 + QAT4 + Evaluation) completed successfully!"
        echo "========================================================================"
        ;;
    *)
        echo "usage: $0 {fp32|qat4|eval|all}" >&2
        exit 2
        ;;
esac
