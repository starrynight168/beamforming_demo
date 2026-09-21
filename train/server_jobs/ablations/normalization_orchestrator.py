"""Normalization Ablation Orchestrator.

Enforces control_normalization="none" by default across all training stages (FP32 -> QAT).
Only enables control_normalization="linf" when explicitly performing the linf comparison.

Usage:
    python train/server_jobs/ablations/normalization_orchestrator.py --stage all
    python train/server_jobs/ablations/normalization_orchestrator.py --stage fp32
    python train/server_jobs/ablations/normalization_orchestrator.py --stage qat
    python train/server_jobs/ablations/normalization_orchestrator.py --stage qat_linf
    python train/server_jobs/ablations/normalization_orchestrator.py --stage eval
"""

import argparse
import os
import subprocess
import time

from ablation_common import (
    BASE_CONFIG_PATH,
    DATA_H5,
    HARDWARE_CONFIG_PATH,
    PYTHON_BIN,
    PROJECT_ROOT,
    SPLIT_FILE,
    TRAIN_ROOT,
    evaluate_models,
    model_arguments,
    require_inputs,
    task_environment,
    wait_for_processes,
)

MODEL_ROOT = TRAIN_ROOT / "results/SI/model/normalization"
EVAL_ROOT = TRAIN_ROOT / "results/SI/evaluation/normalization"

MODELS = [
    {"label": "L1_fc2", "norm": "l1", "layers": ["fc2"]},
    {"label": "L2_fc2", "norm": "l2", "layers": ["fc2"]},
    {"label": "BR_fc2", "norm": "batch_renorm", "layers": ["fc2"]},
    {"label": "None", "norm": "none", "layers": []},
    {"label": "RZ_fc2", "norm": "running_zscore", "layers": ["fc2"]},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Normalization Ablation Orchestrator (Hardware none baseline)")
    parser.add_argument(
        "--stage",
        choices=["fp32", "qat", "qat_linf", "train", "eval", "all"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("NORM_GPUS", "0,1,2,3,5"),
        help="Comma-separated GPU indices for training (default: 0,1,2,3,5)",
    )
    parser.add_argument(
        "--eval-gpu",
        default=os.environ.get("NORM_EVAL_GPU", "3"),
        help="GPU index for evaluation (default: 3)",
    )
    parser.add_argument(
        "--test-frames",
        type=int,
        default=int(os.environ.get("NORM_TEST_FRAMES", 10)),
        help="Number of in-vivo test frames (default: 10)",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=int(os.environ.get("NORM_MC_RUNS", 10)),
        help="Monte Carlo runs for evaluation (default: 10)",
    )
    return parser.parse_args()


def run_stage_training(stage_name: str, mode: str, ctrl_norm: str, gpus: list[str], warmstart_stage: str | None = None):
    if not gpus:
        raise ValueError("at least one GPU is required")
    out_dir = MODEL_ROOT / stage_name
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    print(f"\n{'='*70}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STARTING STAGE: {stage_name} (mode={mode}, control_normalization={ctrl_norm})")
    print(f"{'='*70}")

    for idx, m in enumerate(MODELS):
        gpu = gpus[idx % len(gpus)]
        label = m["label"]
        norm = m["norm"]
        layers = str(m["layers"]).replace(" ", "").replace("'", '"')
        model_out = out_dir / label
        model_out.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{label}.log"

        cmd = [
            PYTHON_BIN,
            "-u",
            str(TRAIN_ROOT / "mban.py"),
            "--config", str(BASE_CONFIG_PATH),
            "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--mode", mode,
            "--profile", "common",
            "--inter-layer", "none",
            "--set", f"output_directory={model_out}",
            "--set", "resume=none",
            "--set", "hidden_width=32",
            "--set", "hidden_layers=2",
            "--set", "output_controls=8",
            "--set", f"normalization={norm}",
            "--set", f"normalization_layers={layers}",
            "--set", f"control_normalization={ctrl_norm}",
            "--set", f"split_file={SPLIT_FILE}",
            "--set", f'h5_files=["{DATA_H5}"]',
            "--set", "save_images_every_epochs=0",
        ]

        if mode == "software":
            cmd.extend(["--set", "epochs=100"])
        elif mode == "qat":
            if warmstart_stage is None:
                raise ValueError(f"{stage_name} requires a warmstart stage")
            warm_ckpt = MODEL_ROOT / warmstart_stage / label / "best_val.pth"
            if not warm_ckpt.is_file():
                raise FileNotFoundError(f"Missing warmstart checkpoint: {warm_ckpt}")
            cmd.extend(["--set", f"initial_checkpoint={warm_ckpt}"])
            cmd.extend(["--set", "epochs=100"])
        else:
            raise ValueError(f"unsupported training mode: {mode}")

        env = task_environment(int(gpu))
        f_log = open(log_file, "w", encoding="utf-8")  # noqa: SIM115
        print(f"  -> Launching [{label}] on GPU {gpu} (ctrl_norm={ctrl_norm}) ...")
        p = subprocess.Popen(cmd, env=env, stdout=f_log, stderr=subprocess.STDOUT)
        procs.append((label, p, f_log))

    wait_for_processes(procs)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STAGE {stage_name} FINISHED SUCCESSFULLY.")


def evaluate_stage(stage_name: str, model_type: str, eval_gpu: str, test_frames: int, mc_runs: int):
    stage_model_dir = MODEL_ROOT / stage_name
    stage_eval_dir = EVAL_ROOT / stage_name
    model_args = model_arguments(
        (str(model["label"]), stage_model_dir / str(model["label"]) / "best_val.pth") for model in MODELS
    )
    print(f"\n{'='*70}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EVALUATING STAGE: {stage_name} on GPU {eval_gpu}")
    print(f"{'='*70}")
    evaluate_models(
        model_args,
        stage_eval_dir,
        eval_gpu,
        model_type=model_type,
        test_frames=test_frames,
        mc_runs=mc_runs,
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EVALUATION FOR {stage_name} COMPLETE. Results saved to {stage_eval_dir}")


def main():
    args = parse_args()
    require_inputs()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus and args.stage != "eval":
        raise ValueError("--gpus must contain at least one GPU")
    if args.test_frames <= 0:
        raise ValueError("--test-frames must be positive")
    if args.mc_runs <= 1:
        raise ValueError("--mc-runs must be greater than one")

    print("Normalization Orchestrator Initialized.")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Split File  : {SPLIT_FILE}")
    print(f"Data H5     : {DATA_H5}")
    print(f"Available GPUs: {gpus}")
    print(f"Execution Stage: {args.stage}")

    # Stage execution
    if args.stage in {"fp32", "train", "all"}:
        # FP32: Standard none baseline from scratch
        run_stage_training("FP32", mode="software", ctrl_norm="none", gpus=gpus)
        if args.stage in {"fp32", "all"}:
            evaluate_stage("FP32", "fp32", args.eval_gpu, args.test_frames, args.mc_runs)

    if args.stage in {"qat", "train", "all"}:
        # QAT: Clean none baseline warmstarted from clean FP32
        run_stage_training("QAT", mode="qat", ctrl_norm="none", gpus=gpus, warmstart_stage="FP32")
        if args.stage in {"qat", "all"}:
            evaluate_stage("QAT", "qat", args.eval_gpu, args.test_frames, args.mc_runs)

    if args.stage in {"qat_linf", "train", "all"}:
        # QATlinf: Explicit linf comparison stage warmstarted from clean FP32
        run_stage_training("QATlinf", mode="qat", ctrl_norm="linf", gpus=gpus, warmstart_stage="FP32")
        if args.stage in {"qat_linf", "all"}:
            evaluate_stage("QATlinf", "qat", args.eval_gpu, args.test_frames, args.mc_runs)

    if args.stage == "eval":
        for s in ["FP32", "QAT", "QATlinf"]:
            evaluate_stage(s, "fp32" if s == "FP32" else "qat", args.eval_gpu, args.test_frames, args.mc_runs)

    print("\nPipeline execution finished successfully.")


if __name__ == "__main__":
    main()
