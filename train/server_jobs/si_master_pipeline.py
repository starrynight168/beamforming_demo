#!/usr/bin/env python3
"""
Master End-to-End Orchestrator for All SI Ablations.
Sequences Wave 1 -> Wave 2 -> Wave 3 -> Wave 4 -> Full Evaluation.
Features:
- Splits tasks across target GPUs (default: GPU 3 and GPU 5, 4 tasks each = 8 concurrent tasks).
- Dynamic task queue supporting continuous slot refilling.
- Strict v40 semantics compliance and robust error tracking.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
TRAIN_ROOT = PROJECT_ROOT / "train"
PYTHON_BIN = os.environ.get("PYTHON", sys.executable)
SI_ROOT = TRAIN_ROOT / "results/SI"
MODEL_ROOT = SI_ROOT / "model"
LOG_DIR = SI_ROOT / "logs"
CONFIG_PATH = TRAIN_ROOT / "config.yaml"
HARDWARE_CONFIG_PATH = TRAIN_ROOT / "hardware_eval.yaml"
SPLIT_FILE = Path(
    os.environ.get(
        "SPLIT_FILE",
        SI_ROOT / "shared_split_seed42_0p7-0p2-0p1.csv",
    )
).expanduser().resolve()

BASE_ENV = os.environ.copy()
if conda_prefix := os.environ.get("CONDA_PREFIX"):
    conda_lib = Path(conda_prefix) / "lib"
    if conda_lib.is_dir():
        BASE_ENV["LD_LIBRARY_PATH"] = f"{conda_lib}:{BASE_ENV.get('LD_LIBRARY_PATH', '')}"
BASE_ENV["OMP_NUM_THREADS"] = "2"
BASE_ENV["MKL_NUM_THREADS"] = "2"
BASE_ENV["OPENBLAS_NUM_THREADS"] = "2"
BASE_ENV["NUMEXPR_NUM_THREADS"] = "2"
BASE_ENV["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
BASE_ENV["PYTHONUNBUFFERED"] = "1"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)


def is_model_done(out_dir):
    """Check if model checkpoint exists and training log marked [DONE]."""
    out_dir = Path(out_dir)
    ckpt = out_dir / "best_val.pth"
    log = out_dir / "training_log.txt"
    if not ckpt.exists():
        return False
    if not log.is_file():
        return False
    content = log.read_text(encoding="utf-8", errors="ignore")
    return "[DONE]" in content or "训练完成" in content


def launch_training(task_name, out_dir, gpu_id, extra_args=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{task_name}.log"

    cmd = [
        PYTHON_BIN, "-u", str(TRAIN_ROOT / "mban.py"),
        "--config", str(CONFIG_PATH),
        "--set", f"output_directory={out_dir}",
        "--set", f"split_file={SPLIT_FILE}",
        "--set", "save_images_every_epochs=0",
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc_env = BASE_ENV.copy()
    proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_handle = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(TRAIN_ROOT),
        env=proc_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT
    )
    print(f"[{time.strftime('%H:%M:%S')}] [LAUNCHED] {task_name} | PID: {proc.pid} | GPU: {gpu_id} | Log: {log_file}")
    return {
        "name": task_name,
        "pid": proc.pid,
        "proc": proc,
        "gpu": gpu_id,
        "out_dir": out_dir,
        "log_file": log_file,
        "log_handle": log_handle,
        "start_time": time.time(),
    }


def run_dynamic_queue(specs, gpu_slots=None, target_gpus=(3, 5), slots_per_gpu=4, stage_desc=""):
    """
    Executes a list of task specs evenly across target_gpus with dynamic queueing.
    gpu_slots: dict mapping gpu_id -> max_concurrent_tasks.
    """
    if gpu_slots is None:
        gpu_slots = {gpu: slots_per_gpu for gpu in target_gpus}
    if not gpu_slots or any(gpu < 0 or slots <= 0 for gpu, slots in gpu_slots.items()):
        raise ValueError("gpu_slots must contain positive capacity for non-negative GPU ids")

    pending = []
    for s in specs:
        name, out_dir, args = s
        if is_model_done(out_dir):
            print(f"[SKIP] {name} is already completed.")
        else:
            pending.append(s)

    if not pending:
        print(f"[QUEUE] All {len(specs)} tasks in {stage_desc} are already completed. Proceeding...")
        return

    max_concurrent = sum(gpu_slots.values())
    print("\n==================================================================")
    print(f"=== [DYNAMIC QUEUE with GPU Slots {gpu_slots}] {stage_desc} ===")
    print(f"=== Total Tasks: {len(pending)} | Max Concurrency: {max_concurrent} ===")
    print("==================================================================")

    active_tasks = []
    failed_tasks = []
    last_reported_epoch = {}

    def get_available_gpu():
        current_counts = {gpu: 0 for gpu in gpu_slots}
        for t in active_tasks:
            if t["gpu"] in current_counts:
                current_counts[t["gpu"]] += 1
        candidates = [g for g, max_s in gpu_slots.items() if current_counts[g] < max_s]
        if not candidates:
            return None
        # Prioritize GPU with lowest utilization fraction
        return min(candidates, key=lambda g: current_counts[g] / gpu_slots[g])

    while pending or active_tasks:
        # Fill available slots up to max_concurrent
        while len(active_tasks) < max_concurrent and pending:
            gpu = get_available_gpu()
            if gpu is None:
                break
            spec = pending.pop(0)
            name, out_dir, args = spec
            t = launch_training(name, out_dir, gpu, args)
            active_tasks.append(t)
            last_reported_epoch[name] = 0

        # Poll active tasks every 10 seconds
        time.sleep(10)

        for t in list(active_tasks):
            poll = t["proc"].poll()
            log_path = t["log_file"]

            latest_ep = last_reported_epoch.get(t["name"], 0)
            ep_time = None
            val_loss = None
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if "[EPOCH] Ep " in line:
                                try:
                                    ep_num = int(line.split("[EPOCH] Ep ")[1].split("/")[0])
                                    latest_ep = ep_num
                                    if "耗时=" in line:
                                        ep_time = line.split("耗时=")[1].split("s")[0].strip()
                                    if "Val=" in line:
                                        val_loss = line.split("Val=")[1].split(" |")[0].strip()
                                except Exception:
                                    pass
                except Exception:
                    pass

            if latest_ep > last_reported_epoch.get(t["name"], 0):
                last_reported_epoch[t["name"]] = latest_ep
                speed_warn = " ⚠️ SPEED > 20s!" if ep_time and float(ep_time) > 20.0 else ""
                print(f"[{time.strftime('%H:%M:%S')}] [{t['name']}] (GPU {t['gpu']}) Ep {latest_ep}/100 | Val: {val_loss} | Time: {ep_time}s{speed_warn}")

            if poll is not None:
                t["log_handle"].close()
                elapsed = time.time() - t["start_time"]
                if poll == 0:
                    print(f"\n[{time.strftime('%H:%M:%S')}] [SUCCESS] {t['name']} finished on GPU {t['gpu']} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
                else:
                    print(f"\n[{time.strftime('%H:%M:%S')}] [ERROR] {t['name']} failed on GPU {t['gpu']} with code {poll}! Log: {t['log_file']}")
                    failed_tasks.append(t["name"])
                active_tasks.remove(t)
                # Next while loop will immediately pop from pending and fill this freed slot!

    if failed_tasks:
        print(f"\n⚠️ [QUEUE WARNING] {len(failed_tasks)} tasks failed in {stage_desc}: {failed_tasks}")
        raise RuntimeError(f"tasks failed in {stage_desc}: {', '.join(failed_tasks)}")
    else:
        print(f"\n🎉 [QUEUE FINISHED] All tasks in {stage_desc} completed successfully!\n")


def build_all_specs():
    """Build specs for all waves."""
    # Wave 1 FP32
    w1_fp32 = [
        ("input_norm_FP32_none", MODEL_ROOT / "input_norm/FP32/none", ["--mode", "software", "--set", "input_normalization=none", "--set", "resume=none"]),
        ("input_norm_FP32_rms", MODEL_ROOT / "input_norm/FP32/rms", ["--mode", "software", "--set", "input_normalization=rms", "--set", "resume=none"]),
        ("input_norm_FP32_std", MODEL_ROOT / "input_norm/FP32/std", ["--mode", "software", "--set", "input_normalization=std", "--set", "resume=none"]),
        ("interp_nearest", MODEL_ROOT / "interpolation/nearest", ["--mode", "software", "--set", "output_interpolation=nearest", "--set", "resume=none"]),
        ("interp_linear", MODEL_ROOT / "interpolation/linear", ["--mode", "software", "--set", "output_interpolation=linear", "--set", "resume=none"]),
        ("interp_cubic", MODEL_ROOT / "interpolation/cubic", ["--mode", "software", "--set", "output_interpolation=cubic", "--set", "resume=none"]),
    ]

    # Wave 1 QAT
    w1_qat = [
        ("input_norm_QAT_none", MODEL_ROOT / "input_norm/QAT/none", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", "input_normalization=none", "--set", f"initial_checkpoint={MODEL_ROOT / 'input_norm/FP32/none/best_val.pth'}", "--set", "resume=none"
        ]),
        ("input_norm_QAT_rms", MODEL_ROOT / "input_norm/QAT/rms", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", "input_normalization=rms", "--set", f"initial_checkpoint={MODEL_ROOT / 'input_norm/FP32/rms/best_val.pth'}", "--set", "resume=none"
        ]),
        ("input_norm_QAT_std", MODEL_ROOT / "input_norm/QAT/std", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", "input_normalization=std", "--set", f"initial_checkpoint={MODEL_ROOT / 'input_norm/FP32/std/best_val.pth'}", "--set", "resume=none"
        ]),
    ]

    # Wave 2 FP32 (23 models)
    act_specs = [
        ("act_relu_single", MODEL_ROOT / "activation/relu_single", ["--mode", "software", "--set", "branch_mode=single", "--set", "activation={functions: {relu: {}}, layers: {fc1: relu, fc2: relu}}", "--set", "resume=none"]),
        ("act_relu_dual", MODEL_ROOT / "activation/relu_dual", ["--mode", "software", "--set", "branch_mode=dual", "--set", "activation={functions: {relu: {}}, layers: {fc1: relu, fc2: relu}}", "--set", "resume=none"]),
        ("act_pwl_single", MODEL_ROOT / "activation/pwl_single", ["--mode", "software", "--set", "branch_mode=single", "--set", "activation={functions: {pwl: {knee: 0.25, tail_slope: 0.2}}, layers: {fc1: pwl, fc2: pwl}}", "--set", "resume=none"]),
        ("act_pwl_dual", MODEL_ROOT / "activation/pwl_dual", ["--mode", "software", "--set", "branch_mode=dual", "--set", "activation={functions: {pwl: {knee: 0.25, tail_slope: 0.2}}, layers: {fc1: pwl, fc2: pwl}}", "--set", "resume=none"]),
        ("act_tanh_single", MODEL_ROOT / "activation/tanh_single", ["--mode", "software", "--set", "branch_mode=single", "--set", "activation={functions: {tanh: {beta: 3.0}}, layers: {fc1: tanh, fc2: tanh}}", "--set", "resume=none"]),
        ("act_tanh_dual", MODEL_ROOT / "activation/tanh_dual", ["--mode", "software", "--set", "branch_mode=dual", "--set", "activation={functions: {tanh: {beta: 3.0}}, layers: {fc1: tanh, fc2: tanh}}", "--set", "resume=none"]),
    ]
    outlim_specs = [
        ("outlim_hard_nonneg", MODEL_ROOT / "outlimit/hard_nonnegative", ["--mode", "software", "--set", "unity_constraint=hard", "--set", "output_domain=nonnegative", "--set", "resume=none"]),
        ("outlim_hard_signed", MODEL_ROOT / "outlimit/hard_signed", ["--mode", "software", "--set", "unity_constraint=hard", "--set", "output_domain=signed", "--set", "resume=none"]),
        ("outlim_free_nonneg", MODEL_ROOT / "outlimit/free_nonnegative", ["--mode", "software", "--set", "unity_constraint=free", "--set", "output_domain=nonnegative", "--set", "resume=none"]),
        ("outlim_free_signed", MODEL_ROOT / "outlimit/free_signed", ["--mode", "software", "--set", "unity_constraint=free", "--set", "output_domain=signed", "--set", "resume=none"]),
    ]
    bias_layers_map = {
        "nobias": "[]",
        "fc1": "[fc1]",
        "fc2": "[fc2]",
        "fc3": "[fc3]",
        "fc12": "[fc1, fc2]",
        "fc23": "[fc2, fc3]",
        "fc123": "[fc1, fc2, fc3]",
    }
    bias_fp32 = [
        (f"bias_FP32_{k}", MODEL_ROOT / f"bias/FP32/{k}", ["--mode", "software", "--set", f"bias_layers={v}", "--set", "resume=none"])
        for k, v in bias_layers_map.items()
    ]
    bi_modes = [
        ("ordinary_analog", "ordinary", "analog"),
        ("ordinary_digital", "ordinary", "digital"),
        ("ordinary_none", "ordinary", "none"),
        ("array_analog", "array", "analog"),
        ("array_digital", "array", "digital"),
        ("array_none", "array", "none"),
    ]
    bi_fp32 = [
        (f"bi_FP32_{name}", MODEL_ROOT / f"bias_interlayer/FP32/{name}", [
            "--mode", "software", "--inter-layer", im,
            "--set", f"bias_implementation={bm}", "--set", "bias_layers=[fc2, fc3]", "--set", "resume=none"
        ])
        for name, bm, im in bi_modes
    ]
    w2_fp32 = act_specs + outlim_specs + bias_fp32 + bi_fp32

    # Wave 2 QAT (13 models)
    bias_qat = [
        (f"bias_QAT4_{k}", MODEL_ROOT / f"bias/QAT4/{k}", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", f"bias_layers={v}", "--set", f"initial_checkpoint={MODEL_ROOT / f'bias/FP32/{k}/best_val.pth'}", "--set", "resume=none"
        ])
        for k, v in bias_layers_map.items()
    ]
    bi_qat = [
        (f"bi_QAT4_{name}", MODEL_ROOT / f"bias_interlayer/QAT4/{name}", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--inter-layer", im, "--set", f"bias_implementation={bm}", "--set", "bias_layers=[fc2, fc3]",
            "--set", f"initial_checkpoint={MODEL_ROOT / f'bias_interlayer/FP32/{name}/best_val.pth'}", "--set", "resume=none"
        ])
        for name, bm, im in bi_modes
    ]
    w2_qat = bias_qat + bi_qat

    # Wave 3 FP32 (25 models)
    loss_tmpl = "{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: %s}, unity: {weight: 0.0}, d1: {domain: shape, weight: %s}, d2: {domain: shape, weight: %s}, lrange: {limit: 2.0, weight: %s}}}"
    env_specs = [
        (f"env_{round(i * 0.1, 1)}", MODEL_ROOT / f"envelope/env_{round(i * 0.1, 1)}", ["--mode", "software", "--set", f"loss={loss_tmpl % (round(i * 0.1, 1), 0.0, 0.0, 0.0)}", "--set", "resume=none"])
        for i in range(11)
    ]
    loss_terms = {
        "none": loss_tmpl % (0.1, 0.0, 0.0, 0.0),
        "d1": loss_tmpl % (0.1, 0.001, 0.0, 0.0),
        "d2": loss_tmpl % (0.1, 0.0, 0.001, 0.0),
        "d1_d2": loss_tmpl % (0.1, 0.001, 0.001, 0.0),
        "lrange": loss_tmpl % (0.1, 0.0, 0.0, 0.1),
        "d1_lrange": loss_tmpl % (0.1, 0.001, 0.0, 0.1),
    }
    loss_fp32 = [
        (f"loss_FP32_{k}", MODEL_ROOT / f"loss/FP32/{k}", ["--mode", "software", "--set", f"loss={v}", "--set", "resume=none"])
        for k, v in loss_terms.items()
    ]
    lrange_settings = {
        "none": loss_tmpl % (0.1, 0.0, 0.0, 0.0),
        "limit_1": "{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: 0.1}, unity: {weight: 0.0}, d1: {domain: shape, weight: 0.0}, d2: {domain: shape, weight: 0.0}, lrange: {limit: 1.0, weight: 0.1}}}",
        "limit_2": "{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: 0.1}, unity: {weight: 0.0}, d1: {domain: shape, weight: 0.0}, d2: {domain: shape, weight: 0.0}, lrange: {limit: 2.0, weight: 0.1}}}",
        "limit_4": "{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: 0.1}, unity: {weight: 0.0}, d1: {domain: shape, weight: 0.0}, d2: {domain: shape, weight: 0.0}, lrange: {limit: 4.0, weight: 0.1}}}",
        "limit_8": "{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: 0.1}, unity: {weight: 0.0}, d1: {domain: shape, weight: 0.0}, d2: {domain: shape, weight: 0.0}, lrange: {limit: 8.0, weight: 0.1}}}",
    }
    lr_fp32 = [
        (f"lrange_FP32_{k}", MODEL_ROOT / f"lrange/FP32/{k}", ["--mode", "software", "--set", f"loss={v}", "--set", "resume=none"])
        for k, v in lrange_settings.items()
    ]
    qattext_fp32 = [
        ("qattext_FP32_none_30ep", MODEL_ROOT / "qattext/FP32/none_30ep", ["--mode", "software", "--set", "epochs=30", "--set", "scheduler=none", "--set", "learning_rate=0.0001", "--set", "resume=none"]),
        ("qattext_FP32_none_100ep", MODEL_ROOT / "qattext/FP32/none_100ep", ["--mode", "software", "--set", "epochs=100", "--set", "scheduler=none", "--set", "learning_rate=0.0001", "--set", "resume=none"]),
        ("qattext_FP32_OneCycle_100ep", MODEL_ROOT / "qattext/FP32/OneCycle_100ep", ["--mode", "software", "--set", "epochs=100", "--set", "scheduler=onecycle", "--set", "learning_rate=0.001", "--set", "resume=none"]),
    ]
    w3_fp32 = env_specs + loss_fp32 + lr_fp32 + qattext_fp32

    # Wave 3 QAT (14 models)
    loss_qat = [
        (f"loss_QAT4_{k}", MODEL_ROOT / f"loss/QAT4/{k}", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", f"loss={v}", "--set", f"initial_checkpoint={MODEL_ROOT / f'loss/FP32/{k}/best_val.pth'}", "--set", "resume=none"
        ])
        for k, v in loss_terms.items()
    ]
    lr_qat = [
        (f"lrange_QAT4_{k}", MODEL_ROOT / f"lrange/QAT4/{k}", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", f"loss={v}", "--set", f"initial_checkpoint={MODEL_ROOT / f'lrange/FP32/{k}/best_val.pth'}", "--set", "resume=none"
        ])
        for k, v in lrange_settings.items()
    ]
    qattext_qat = [
        ("qattext_QAT4_none_30ep", MODEL_ROOT / "qattext/QAT4/none_30ep", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", f"initial_checkpoint={MODEL_ROOT / 'qattext/FP32/none_30ep/best_val.pth'}", "--set", "resume=none"
        ]),
        ("qattext_QAT4_none_100ep", MODEL_ROOT / "qattext/QAT4/none_100ep", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", f"initial_checkpoint={MODEL_ROOT / 'qattext/FP32/none_100ep/best_val.pth'}", "--set", "resume=none"
        ]),
        ("qattext_QAT4_OneCycle_100ep", MODEL_ROOT / "qattext/QAT4/OneCycle_100ep", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", f"initial_checkpoint={MODEL_ROOT / 'qattext/FP32/OneCycle_100ep/best_val.pth'}", "--set", "resume=none"
        ]),
    ]
    w3_qat = loss_qat + lr_qat + qattext_qat

    # Wave 4 FP32 (7 models)
    norm_configs = {
        "None": ["--set", "normalization=none", "--set", "centering=none"],
        "L1_fc2": ["--set", "normalization=l1", "--set", "normalization_layers=[fc2]"],
        "L2_fc2": ["--set", "normalization=l2", "--set", "normalization_layers=[fc2]"],
        "BR_fc2": ["--set", "normalization=batch_renorm", "--set", "normalization_layers=[fc2]"],
        "RZ_fc2": ["--set", "normalization=running_zscore", "--set", "normalization_layers=[fc2]"],
    }
    norm_fp32 = [
        (f"norm_FP32_{k}", MODEL_ROOT / f"normalization/FP32/{k}", ["--mode", "software", *v, "--set", "resume=none"])
        for k, v in norm_configs.items()
    ]
    cadc_fp32 = [
        ("cadc_FP32", MODEL_ROOT / "control_adc/FP32", ["--mode", "software", "--set", "resume=none"])
    ]
    win_fp32 = [
        ("win_FP32_dynamic", MODEL_ROOT / "windows/FP32/dynamic", ["--mode", "software", "--set", "resume=none"])
    ]
    w4_fp32 = norm_fp32 + cadc_fp32 + win_fp32

    # Wave 4 QAT (15 models)
    norm_qat = [
        (f"norm_QAT_{k}", MODEL_ROOT / f"normalization/QAT/{k}", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            *v, "--set", "control_normalization=none", "--set", f"initial_checkpoint={MODEL_ROOT / f'normalization/FP32/{k}/best_val.pth'}", "--set", "resume=none"
        ])
        for k, v in norm_configs.items()
    ]
    norm_qatlinf = [
        (f"norm_QATlinf_{k}", MODEL_ROOT / f"normalization/QATlinf/{k}", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            *v, "--set", "control_normalization=linf", "--set", f"initial_checkpoint={MODEL_ROOT / f'normalization/FP32/{k}/best_val.pth'}", "--set", "resume=none"
        ])
        for k, v in norm_configs.items()
    ]
    init_cadc = MODEL_ROOT / "control_adc/FP32/best_val.pth"
    cadc_qat = [
        ("cadc_QAT4_none_full_scale", MODEL_ROOT / "control_adc/QAT4/none_full_scale", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", "control_normalization=none", "--set", "control_adc_range=full_scale",
            "--set", f"initial_checkpoint={init_cadc}", "--set", "resume=none"
        ]),
        ("cadc_QAT4_none_observer", MODEL_ROOT / "control_adc/QAT4/none_observer", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", "control_normalization=none", "--set", "control_adc_range=observer",
            "--set", f"initial_checkpoint={init_cadc}", "--set", "resume=none"
        ]),
        ("cadc_QAT4_linf_full_scale", MODEL_ROOT / "control_adc/QAT4/linf_full_scale", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", "control_normalization=linf", "--set", "control_adc_range=full_scale",
            "--set", f"initial_checkpoint={init_cadc}", "--set", "resume=none"
        ]),
        ("cadc_QAT4_linf_observer", MODEL_ROOT / "control_adc/QAT4/linf_observer", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", "control_normalization=linf", "--set", "control_adc_range=observer",
            "--set", f"initial_checkpoint={init_cadc}", "--set", "resume=none"
        ]),
    ]
    win_qat = [
        ("win_QAT4_dynamic", MODEL_ROOT / "windows/QAT4/dynamic", [
            "--mode", "qat", "--profile", "common", "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--set", f"initial_checkpoint={MODEL_ROOT / 'windows/FP32/dynamic/best_val.pth'}", "--set", "resume=none"
        ])
    ]
    w4_qat = norm_qat + norm_qatlinf + cadc_qat + win_qat

    return {
        "w1_fp32": w1_fp32,
        "w1_qat": w1_qat,
        "w2_fp32": w2_fp32,
        "w2_qat": w2_qat,
        "w3_fp32": w3_fp32,
        "w3_qat": w3_qat,
        "w4_fp32": w4_fp32,
        "w4_qat": w4_qat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="3,5", help="Comma-separated target GPUs (default: 3,5)")
    parser.add_argument("--slots-per-gpu", type=int, default=4, help="Concurrent tasks per GPU (default: 4)")
    parser.add_argument("--gpu-slots", type=str, default=None, help="GPU slots mapping, e.g. '4:4,0:3,3:1,6:1,7:1'")
    args = parser.parse_args()

    if args.gpu_slots:
        gpu_slots = {}
        for item in args.gpu_slots.split(","):
            if not item.strip():
                continue
            parts = item.strip().split(":")
            if len(parts) != 2:
                raise SystemExit(f"invalid --gpu-slots entry: {item}")
            gpu, slots = map(int, parts)
            if gpu < 0 or slots <= 0:
                raise SystemExit(f"invalid --gpu-slots entry: {item}")
            gpu_slots[gpu] = slots
    else:
        target_gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if not target_gpus or args.slots_per_gpu <= 0:
            raise SystemExit("--gpus and --slots-per-gpu must define positive capacity")
        if any(gpu < 0 for gpu in target_gpus):
            raise SystemExit("--gpus 只能包含非负整数")
        gpu_slots = {g: args.slots_per_gpu for g in target_gpus}

    if not gpu_slots:
        raise SystemExit("at least one GPU slot is required")

    total_slots = sum(gpu_slots.values())

    print("\n" + "=" * 80)
    print(f"=== MBAN SI MASTER TRAINING PIPELINE: GPU SLOTS {gpu_slots} (TOTAL {total_slots} CONCURRENT TASKS) ===")
    print("=" * 80)

    specs = build_all_specs()

    # Stage 1: Wave 1 FP32 (Already completed, will instant skip)
    run_dynamic_queue(specs["w1_fp32"], gpu_slots=gpu_slots, stage_desc="Wave 1 FP32 (6 models)")

    # Stage 2: Wave 1 QAT (3 models) + Wave 2 FP32 (23 models) combined!
    combined_w1_qat_w2_fp32 = specs["w1_qat"] + specs["w2_fp32"]
    run_dynamic_queue(combined_w1_qat_w2_fp32, gpu_slots=gpu_slots, stage_desc="Wave 1 QAT + Wave 2 FP32 Combined (26 models)")

    # Stage 3: Wave 2 QAT (13 models)
    run_dynamic_queue(specs["w2_qat"], gpu_slots=gpu_slots, stage_desc="Wave 2 QAT (13 models)")

    # Stage 4: Wave 3 FP32 (25 models)
    run_dynamic_queue(specs["w3_fp32"], gpu_slots=gpu_slots, stage_desc="Wave 3 FP32 (25 models)")

    # Stage 5: Wave 3 QAT (14 models)
    run_dynamic_queue(specs["w3_qat"], gpu_slots=gpu_slots, stage_desc="Wave 3 QAT (14 models)")

    # Stage 6: Wave 4 FP32 (7 models)
    run_dynamic_queue(specs["w4_fp32"], gpu_slots=gpu_slots, stage_desc="Wave 4 FP32 (7 models)")

    # Stage 7: Wave 4 QAT (15 models)
    run_dynamic_queue(specs["w4_qat"], gpu_slots=gpu_slots, stage_desc="Wave 4 QAT (15 models)")

    print("\n" + "=" * 80)
    print(f"=== ALL 13 SI ABLATION MODULES SUCCESSFULLY TRAINED ON {gpu_slots}! ===")
    print("=" * 80)


if __name__ == "__main__":
    main()
