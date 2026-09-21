#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

ACTION="${1:-all}"
MODEL_ROOT="$ROOT/train/results/SI/model/bias"
EVAL_ROOT="$ROOT/train/results/SI/evaluation/bias"
SPLIT_FILE="${SPLIT_FILE:-}"
BIAS_FP32_GPU="${BIAS_FP32_GPU:-5}"
BIAS_FP32_PARALLEL="${BIAS_FP32_PARALLEL:-3}"
BIAS_QAT_GPU="${BIAS_QAT_GPU:-7}"
BIAS_QAT4_GPUS="${BIAS_QAT4_GPUS:-7,3,5,0,1,2,6}"
BIAS_EVAL_GPU="${BIAS_EVAL_GPU:-7}"
BIAS_TEST_FRAMES="${BIAS_TEST_FRAMES:-10}"
BIAS_MC_RUNS="${BIAS_MC_RUNS:-10}"

LABELS=(nobias fc1 fc2 fc12 fc23 fc123 fc3)
LAYERS=('[]' '[fc1]' '[fc2]' '[fc1, fc2]' '[fc2, fc3]' '[fc1, fc2, fc3]' '[fc3]')

resolve_split
require_file "$DATA_H5"
mkdir -p "$MODEL_ROOT" "$EVAL_ROOT"

run_training() {
    local stage="$1" label="$2" layers="$3" gpu="$4" eval_config="$5" profile="${6:-common}"
    local output="$MODEL_ROOT/$stage/$label" log="$MODEL_ROOT/$stage/logs/${label}.log"
    local checkpoint=""
    mkdir -p "$output" "$(dirname -- "$log")"
    if [[ "$stage" != "FP32" ]]; then
        checkpoint="$MODEL_ROOT/FP32/$label/best_val.pth"
        require_file "$checkpoint"
    fi
    {
        echo "[START] stage=$stage label=$label gpu=$gpu bias_layers=$layers"
        if [[ "$stage" == "FP32" ]]; then
            if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u train/mban.py \
                --config "$CONFIG" \
                --eval-config "$eval_config" \
                --mode software \
                --set output_directory="$output" \
                --set resume=none \
                --set hidden_width=32 \
                --set hidden_layers=2 \
                --set output_controls=8 \
                --set bias_layers="$layers" \
                --set split_file="$SPLIT_FILE" \
                --set save_images_every_epochs=0 \
                --set epochs=100; then
                status=0
            else
                status=$?
            fi
        elif [[ "$stage" == "QAT" ]]; then
            if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u train/mban.py \
                --config "$CONFIG" \
                --eval-config "$eval_config" \
                --mode qat \
                --profile "$profile" \
                --inter-layer none \
                --set output_directory="$output" \
                --set resume=none \
                --set initial_checkpoint="$checkpoint" \
                --set hidden_width=32 \
                --set hidden_layers=2 \
                --set output_controls=8 \
                --set bias_layers="$layers" \
                --set bias_bits=32 \
                --set normalization=none \
                --set normalization_layers=[] \
                --set control_normalization=none \
                --set split_file="$SPLIT_FILE" \
                --set save_images_every_epochs=0 \
                --set epochs=100; then
                status=0
            else
                status=$?
            fi
        elif [[ "$stage" == "QAT4" ]]; then
            if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u train/mban.py \
                --config "$CONFIG" \
                --eval-config "$eval_config" \
                --mode qat \
                --profile "$profile" \
                --inter-layer none \
                --set output_directory="$output" \
                --set resume=none \
                --set initial_checkpoint="$checkpoint" \
                --set hidden_width=32 \
                --set hidden_layers=2 \
                --set output_controls=8 \
                --set bias_layers="$layers" \
                --set normalization=none \
                --set normalization_layers=[] \
                --set control_normalization=none \
                --set split_file="$SPLIT_FILE" \
                --set save_images_every_epochs=0 \
                --set epochs=100; then
                status=0
            else
                status=$?
            fi
        else
            echo "[ERROR] unknown training stage: $stage" >&2
            status=2
        fi
        echo "[DONE] stage=$stage label=$label exit=$status"
    } >"$log" 2>&1
    return "$status"
}

train_fp32() {
    local pids=() index label
    for index in "${!LABELS[@]}"; do
        label="${LABELS[$index]}"
        run_training FP32 "$label" "${LAYERS[$index]}" "$BIAS_FP32_GPU" "$HARDWARE_CONFIG" common &
        pids+=("$!")
        if (( ${#pids[@]} >= BIAS_FP32_PARALLEL )); then
            wait_for_pids "${pids[@]}"
            pids=()
        fi
    done
    ((${#pids[@]} == 0)) || wait_for_pids "${pids[@]}"
}

train_qat() {
    local stage="$1" eval_config="$2" profile="$3" gpu_spec="$4"
    local -a gpu_list pids=()
    local index label gpu
    IFS=',' read -r -a gpu_list <<< "$gpu_spec"
    for index in "${!LABELS[@]}"; do
        label="${LABELS[$index]}"
        gpu="${gpu_list[$((index % ${#gpu_list[@]}))]}"
        run_training "$stage" "$label" "${LAYERS[$index]}" "$gpu" "$eval_config" "$profile" &
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
    local stage="$1" model_root="$2" eval_config="$3" profile="$4" mode="$5" model_type="$6"
    local output="$EVAL_ROOT/$stage" log="$EVAL_ROOT/logs/${stage}.log"
    local scenes_runs="$BIAS_MC_RUNS"
    [[ "$model_type" == fp32 ]] && scenes_runs=1
    local -a invivo_args
    build_models "$model_root"
    mkdir -p "$output/invivo" "$output/scenes" "$(dirname -- "$log")"
    invivo_args=(
        --mode "$mode"
        --model-type "$model_type"
        "${MODEL_ARGS[@]}"
        --split-file "$SPLIT_FILE"
        --config "$CONFIG"
        --eval-config "$eval_config"
        --profile "$profile"
        --output "$output/invivo"
        --test-frames "$BIAS_TEST_FRAMES"
        --seed 42
        --micro-batch 8192
        --device cuda:0
    )
    [[ "$mode" == mc ]] && invivo_args+=(--runs "$BIAS_MC_RUNS")
    {
        echo "[START] stage=$stage models=$((${#MODEL_ARGS[@]} / 2)) gpu=$BIAS_EVAL_GPU"
        CUDA_VISIBLE_DEVICES="$BIAS_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py "${invivo_args[@]}"
        CUDA_VISIBLE_DEVICES="$BIAS_EVAL_GPU" "$PYTHON" -u train/evaluate_mban.py \
            --mode scenes \
            --model-type "$model_type" \
            "${MODEL_ARGS[@]}" \
            --config "$CONFIG" \
            --eval-config "$eval_config" \
            --profile "$profile" \
            --output "$output/scenes" \
            --scene-config config.yaml \
            --scenes experiments_contrast_speckle,experiments_resolution_distorsion,simulation_contrast_speckle,simulation_resolution_distorsion \
            --seed 42 \
            --micro-batch 8192 \
            --device cuda:0 \
            --monte-carlo-runs "$scenes_runs"
        echo "[DONE] stage=$stage"
    } >"$log" 2>&1
}

evaluate_all() {
    local stage
    for stage in ${BIAS_EVAL_STAGES:-FP32 QAT QAT4 PTQ}; do
        case "$stage" in
            FP32) evaluate_stage FP32 "$MODEL_ROOT/FP32" "$HARDWARE_CONFIG" common test fp32 ;;
            QAT) evaluate_stage QAT "$MODEL_ROOT/QAT" "$HARDWARE_CONFIG" ideal mc qat ;;
            QAT4) evaluate_stage QAT4 "$MODEL_ROOT/QAT4" "$HARDWARE_CONFIG" common mc qat ;;
            PTQ) evaluate_stage PTQ "$MODEL_ROOT/FP32" "$HARDWARE_CONFIG" common ptq ptq ;;
            *) echo "[ERROR] unknown BIAS_EVAL_STAGES value: $stage" >&2; return 2 ;;
        esac
    done
}

case "$ACTION" in
    train)
        train_fp32
        train_qat QAT "$HARDWARE_CONFIG" ideal "$BIAS_QAT_GPU"
        train_qat QAT4 "$HARDWARE_CONFIG" common "$BIAS_QAT4_GPUS"
        ;;
    eval) evaluate_all ;;
    all)
        train_fp32
        train_qat QAT "$HARDWARE_CONFIG" ideal "$BIAS_QAT_GPU"
        train_qat QAT4 "$HARDWARE_CONFIG" common "$BIAS_QAT4_GPUS"
        evaluate_all
        ;;
    *) echo "usage: $0 {train|eval|all}" >&2; exit 2 ;;
esac
