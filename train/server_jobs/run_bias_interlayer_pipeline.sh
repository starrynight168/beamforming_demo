#!/usr/bin/env bash
set -euo pipefail

SERVER_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SERVER_JOBS_DIR/lib/common.sh"

resolve_split

MODEL_BASE="${MODEL_BASE:-$TRAIN_ROOT/results/SI/model/bias_interlayer}"
EVAL_BASE="${EVAL_BASE:-$TRAIN_ROOT/results/SI/evaluation/bias_interlayer}"
LOG_DIR="${LOG_DIR:-$MODEL_BASE/logs}"
QAT_RUN_TAG="${QAT_RUN_TAG:-QAT4}"
BIAS_INTERLAYER_EVAL_GPU="${BIAS_INTERLAYER_EVAL_GPU:-1}"
QAT_MODEL_BASE="$MODEL_BASE/$QAT_RUN_TAG"
QAT_EVAL_BASE="$EVAL_BASE/$QAT_RUN_TAG"

mkdir -p "$MODEL_BASE/FP32" "$QAT_MODEL_BASE" "$EVAL_BASE/FP32" "$QAT_EVAL_BASE" "$LOG_DIR"

MODELS=(
    "ordinary_analog"
    "ordinary_digital"
    "ordinary_none"
    "array_analog"
    "array_digital"
    "array_none"
)

BIAS_MODES=(
    "ordinary"
    "ordinary"
    "ordinary"
    "array"
    "array"
    "array"
)

INTER_LAYERS=(
    "analog"
    "digital"
    "none"
    "analog"
    "digital"
    "none"
)

GPUS=(0 1 4 5 6 7)

LOSS_JSON="{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: 0.1}, unity: {weight: 0.0}, d1: {domain: shape, weight: 0.0}, d2: {domain: shape, weight: 0.0}, lrange: {limit: 2.0, weight: 0.1}}}"

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 1: Checking FP32 Models"
echo "=================================================================="

for name in "${MODELS[@]}"; do
    if [ ! -f "$MODEL_BASE/FP32/$name/best_val.pth" ]; then
        echo "[ERROR] Missing $name FP32 best_val.pth"
        exit 1
    fi
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 1 DONE: All 6 FP32 models exist."

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 2: Evaluating 6 FP32 Models on GPU $BIAS_INTERLAYER_EVAL_GPU"
echo "=================================================================="

FP32_ARGS=()
for name in "${MODELS[@]}"; do
    FP32_ARGS+=(--model "${name}=$MODEL_BASE/FP32/${name}/best_val.pth")
done

mkdir -p "$EVAL_BASE/FP32/invivo" "$EVAL_BASE/FP32/scenes"
if [ ! -f "$EVAL_BASE/FP32/invivo/test_ssim_summary.csv" ]; then
    echo "[STAGE 2] Running FP32 In Vivo evaluation..."
    CUDA_VISIBLE_DEVICES="$BIAS_INTERLAYER_EVAL_GPU" "$PYTHON" -u "$TRAIN_ROOT/evaluate_mban.py" \
        --mode test \
        --model-type fp32 \
        "${FP32_ARGS[@]}" \
        --split-file "$SPLIT_FILE" \
        --config "$CONFIG" \
        --output "$EVAL_BASE/FP32/invivo" \
        --micro-batch 8192 \
        --device cuda:0 \
        > "$EVAL_BASE/FP32/invivo/eval.log" 2>&1
fi

echo "[STAGE 2] Running FP32 Physical Scenes evaluation..."
CUDA_VISIBLE_DEVICES="$BIAS_INTERLAYER_EVAL_GPU" "$PYTHON" -u "$TRAIN_ROOT/evaluate_mban.py" \
    --mode scenes \
    --model-type fp32 \
    "${FP32_ARGS[@]}" \
    --config "$CONFIG" \
    --output "$EVAL_BASE/FP32/scenes" \
    --scene-config "$ROOT_CONFIG" \
    --scenes experiments_contrast_speckle,experiments_resolution_distorsion,simulation_contrast_speckle,simulation_resolution_distorsion \
    --seed 42 \
    --micro-batch 8192 \
    --device cuda:0 \
    > "$EVAL_BASE/FP32/scenes/eval.log" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 2 DONE: FP32 evaluation finished."

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 3: Training 6 QAT4 Models (Warm-started from FP32, OMP=16)"
echo "=================================================================="

PIDS=()
for i in 0 1 2 3 4 5; do
    name="${MODELS[$i]}"
    bias_m="${BIAS_MODES[$i]}"
    inter_m="${INTER_LAYERS[$i]}"
    gpu="${GPUS[$i]}"
    out="$QAT_MODEL_BASE/$name"
    log="$LOG_DIR/${QAT_RUN_TAG}_${name}.log"
    ckpt="$MODEL_BASE/FP32/$name/best_val.pth"

    if [ -e "$out" ]; then
        echo "[ERROR] Existing QAT output: $out"
        echo "[ERROR] Set a new QAT_RUN_TAG; existing checkpoints are never reused by this pipeline."
        exit 1
    fi
    mkdir -p "$out"

    echo "[START QAT4] $name (bias=$bias_m, inter_layer=$inter_m) on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$TRAIN_ROOT/mban.py" \
        --mode qat \
        --profile common \
        --config "$CONFIG" \
        --eval-config "$HARDWARE_CONFIG" \
        --inter-layer "$inter_m" \
        --set bias_implementation="$bias_m" \
        --set bias_layers='[fc2, fc3]' \
        --set resume=none \
        --set initial_checkpoint="$ckpt" \
        --set output_directory="$out" \
        --set split_file="$SPLIT_FILE" \
        --set h5_files="[\"$DATA_H5\"]" \
        --set epochs=100 \
        --set qat_clip_outside_grad=0.0 \
        --set save_images_every_epochs=0 \
        --set loss="$LOSS_JSON" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

if [ ${#PIDS[@]} -gt 0 ]; then
    wait_for_pids "${PIDS[@]}"
fi
for name in "${MODELS[@]}"; do
    require_file "$QAT_MODEL_BASE/$name/best_val.pth"
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 3 DONE: All 6 QAT4 models trained successfully."

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 4: Evaluating 6 QAT4 Models on GPU $BIAS_INTERLAYER_EVAL_GPU (MC-10 + Scenes)"
echo "=================================================================="

QAT4_ARGS=()
for name in "${MODELS[@]}"; do
    require_file "$QAT_MODEL_BASE/$name/best_val.pth"
    QAT4_ARGS+=(--model "${name}=$QAT_MODEL_BASE/${name}/best_val.pth")
done

mkdir -p "$QAT_EVAL_BASE/invivo" "$QAT_EVAL_BASE/scenes"

echo "[STAGE 4] Running QAT4 In Vivo Monte Carlo (10 runs) evaluation..."
CUDA_VISIBLE_DEVICES="$BIAS_INTERLAYER_EVAL_GPU" "$PYTHON" -u "$TRAIN_ROOT/evaluate_mban.py" \
    --mode mc \
    --model-type qat \
    "${QAT4_ARGS[@]}" \
    --split-file "$SPLIT_FILE" \
    --config "$CONFIG" \
    --eval-config "$HARDWARE_CONFIG" \
    --profile common \
    --output "$QAT_EVAL_BASE/invivo" \
    --test-frames 10 \
    --seed 42 \
    --runs 10 \
    --micro-batch 8192 \
    --device cuda:0 \
    > "$QAT_EVAL_BASE/invivo/eval.log" 2>&1

echo "[STAGE 4] Running QAT4 Physical Scenes Monte Carlo (10 runs) evaluation..."
CUDA_VISIBLE_DEVICES="$BIAS_INTERLAYER_EVAL_GPU" "$PYTHON" -u "$TRAIN_ROOT/evaluate_mban.py" \
    --mode scenes \
    --model-type qat \
    "${QAT4_ARGS[@]}" \
    --config "$CONFIG" \
    --eval-config "$HARDWARE_CONFIG" \
    --profile common \
    --output "$QAT_EVAL_BASE/scenes" \
    --scene-config "$ROOT_CONFIG" \
    --scenes experiments_contrast_speckle,experiments_resolution_distorsion,simulation_contrast_speckle,simulation_resolution_distorsion \
    --seed 42 \
    --monte-carlo-runs 10 \
    --micro-batch 8192 \
    --device cuda:0 \
    > "$QAT_EVAL_BASE/scenes/eval.log" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 4 DONE: QAT4 evaluation finished."
echo "=================================================================="
echo "ALL STAGES (FP32 + QAT4 + EVALUATION) COMPLETED SUCCESSFULLY!"
echo "=================================================================="
