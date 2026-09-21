#!/usr/bin/env bash
set -euo pipefail

SERVER_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SERVER_JOBS_DIR/lib/common.sh"

resolve_split

SI_ROOT="${SI_ROOT:-$TRAIN_ROOT/results/SI}"
FP32_MODELS="$SI_ROOT/model/lrange/FP32"
QAT4_MODELS="$SI_ROOT/model/lrange/QAT4"
LOG_DIR="$SI_ROOT/model/lrange/logs"
FP32_EVAL="$SI_ROOT/evaluation/lrange/FP32"
QAT4_EVAL="$SI_ROOT/evaluation/lrange/QAT4"

mkdir -p "$FP32_MODELS" "$QAT4_MODELS" "$LOG_DIR" "$FP32_EVAL/invivo" "$FP32_EVAL/scenes" "$QAT4_EVAL/invivo" "$QAT4_EVAL/scenes"

NAMES=(none limit_1 limit_2 limit_4 limit_8)
LIMITS=(8.0 1.0 2.0 4.0 8.0)
WEIGHTS=(0.0 0.1 0.1 0.1 0.1)
GPUS=(1 1 0 6 3)

echo "=================================================================="
echo "[STAGE 1] Checking / Training FP32 for 5 models..."
echo "=================================================================="
PIDS=()
for i in 0 1 2 3 4; do
    name="${NAMES[$i]}"
    limit="${LIMITS[$i]}"
    w="${WEIGHTS[$i]}"
    gpu="${GPUS[$i]}"
    out="$FP32_MODELS/$name"
    log="$LOG_DIR/fp32_${name}.log"
    mkdir -p "$out"
    if [ -f "$out/best_val.pth" ]; then
        echo "[SKIP FP32] $name already has best_val.pth"
        continue
    fi
    loss_json="{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: 0.1}, unity: {weight: 0.0}, d1: {domain: shape, weight: 0.0}, d2: {domain: shape, weight: 0.0}, lrange: {limit: $limit, weight: $w}}}"
    
    echo "[START FP32] $name (limit=$limit, weight=$w) on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$TRAIN_ROOT/mban.py" \
        --mode software \
        --config "$CONFIG" \
        --eval-config "$HARDWARE_CONFIG" \
        --inter-layer analog \
        --set bias_implementation="ordinary" \
        --set resume=none \
        --set initial_checkpoint=none \
        --set output_directory="$out" \
        --set split_file="$SPLIT_FILE" \
        --set h5_files="[\"$DATA_H5\"]" \
        --set epochs=100 \
        --set save_images_every_epochs=0 \
        --set loss="$loss_json" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

if [ ${#PIDS[@]} -gt 0 ]; then
    wait_for_pids "${PIDS[@]}"
fi
for name in "${NAMES[@]}"; do
    require_file "$FP32_MODELS/$name/best_val.pth"
done
echo "[STAGE 1 DONE] All 5 FP32 models ready."

echo "=================================================================="
echo "[STAGE 2] Evaluating FP32 models on GPU 1..."
echo "=================================================================="
FP32_ARGS=()
for name in "${NAMES[@]}"; do
    FP32_ARGS+=(--model "${name}=$FP32_MODELS/${name}/best_val.pth")
done

if [ ! -f "$FP32_EVAL/invivo/test_ssim_summary.csv" ]; then
    echo "[STAGE 2] Running FP32 Invivo evaluation..."
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u "$TRAIN_ROOT/evaluate_mban.py" \
        --mode test \
        --model-type fp32 \
        "${FP32_ARGS[@]}" \
        --split-file "$SPLIT_FILE" \
        --config "$CONFIG" \
        --output "$FP32_EVAL/invivo" \
        --micro-batch 8192 \
        --device cuda:0 \
        > "$FP32_EVAL/invivo/eval.log" 2>&1
else
    echo "[STAGE 2] FP32 Invivo evaluation already complete."
fi

echo "[STAGE 2] Running FP32 Physical Scenes evaluation..."
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u "$TRAIN_ROOT/evaluate_mban.py" \
    --mode scenes \
    --model-type fp32 \
    "${FP32_ARGS[@]}" \
    --config "$CONFIG" \
    --output "$FP32_EVAL/scenes" \
    --scene-config "$ROOT_CONFIG" \
    --scenes experiments_contrast_speckle,experiments_resolution_distorsion,simulation_contrast_speckle,simulation_resolution_distorsion \
    --seed 42 \
    --micro-batch 8192 \
    --device cuda:0 \
    > "$FP32_EVAL/scenes/eval.log" 2>&1
echo "[STAGE 2 DONE] FP32 evaluation finished."

echo "=================================================================="
echo "[STAGE 3] Launching QAT4 Training warm-started from FP32..."
echo "=================================================================="
PIDS=()
for i in 0 1 2 3 4; do
    name="${NAMES[$i]}"
    limit="${LIMITS[$i]}"
    w="${WEIGHTS[$i]}"
    gpu="${GPUS[$i]}"
    out="$QAT4_MODELS/$name"
    log="$LOG_DIR/qat4_${name}.log"
    ckpt="$FP32_MODELS/$name/best_val.pth"
    require_file "$ckpt"
    mkdir -p "$out"
    loss_json="{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: 0.1}, unity: {weight: 0.0}, d1: {domain: shape, weight: 0.0}, d2: {domain: shape, weight: 0.0}, lrange: {limit: $limit, weight: $w}}}"
    
    echo "[START QAT4] $name (limit=$limit, weight=$w) on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$TRAIN_ROOT/mban.py" \
        --mode qat \
        --profile common \
        --config "$CONFIG" \
        --eval-config "$HARDWARE_CONFIG" \
        --inter-layer analog \
        --set bias_implementation="ordinary" \
        --set resume=none \
        --set initial_checkpoint="$ckpt" \
        --set output_directory="$out" \
        --set split_file="$SPLIT_FILE" \
        --set h5_files="[\"$DATA_H5\"]" \
        --set epochs=100 \
        --set qat_clip_outside_grad=0.0 \
        --set save_images_every_epochs=0 \
        --set loss="$loss_json" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

wait_for_pids "${PIDS[@]}"
echo "[STAGE 3 DONE] All 5 QAT4 models trained successfully."

echo "=================================================================="
echo "[STAGE 4] Evaluating QAT4 models on GPU 1..."
echo "=================================================================="
QAT4_ARGS=()
for name in "${NAMES[@]}"; do
    require_file "$QAT4_MODELS/$name/best_val.pth"
    QAT4_ARGS+=(--model "${name}=$QAT4_MODELS/${name}/best_val.pth")
done

CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u "$TRAIN_ROOT/evaluate_mban.py" \
    --mode mc \
    --model-type qat \
    "${QAT4_ARGS[@]}" \
    --split-file "$SPLIT_FILE" \
    --config "$CONFIG" \
    --eval-config "$HARDWARE_CONFIG" \
    --profile common \
    --output "$QAT4_EVAL/invivo" \
    --test-frames 10 \
    --seed 42 \
    --runs 10 \
    --micro-batch 8192 \
    --device cuda:0 \
    > "$QAT4_EVAL/invivo/eval.log" 2>&1

CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u "$TRAIN_ROOT/evaluate_mban.py" \
    --mode scenes \
    --model-type qat \
    "${QAT4_ARGS[@]}" \
    --config "$CONFIG" \
    --eval-config "$HARDWARE_CONFIG" \
    --profile common \
    --output "$QAT4_EVAL/scenes" \
    --scene-config "$ROOT_CONFIG" \
    --scenes experiments_contrast_speckle,experiments_resolution_distorsion,simulation_contrast_speckle,simulation_resolution_distorsion \
    --seed 42 \
    --micro-batch 8192 \
    --device cuda:0 \
    --monte-carlo-runs 10 \
    > "$QAT4_EVAL/scenes/eval.log" 2>&1
echo "[STAGE 4 DONE] QAT4 evaluation finished."
echo "[ALL FINISHED] Full pipeline completed successfully!"
