import argparse
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

from ablation_common import (
    DATA_H5,
    HARDWARE_CONFIG_PATH,
    PYTHON_BIN,
    SPLIT_FILE,
    TRAIN_ROOT,
    evaluate_models,
    load_base_config,
    model_arguments,
    require_inputs,
    task_environment,
    wait_for_processes,
)


TASKS = (
    ("relu_single", 0, "relu", "single"),
    ("relu_dual", 1, "relu", "dual"),
    ("pwl_single", 2, "pwl", "single"),
    ("pwl_dual", 3, "pwl", "dual"),
    ("tanh_single", 4, "tanh", "single"),
    ("tanh_dual", 5, "tanh", "dual"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the activation × branch ablation matrix")
    parser.add_argument("--eval-gpu", default=os.environ.get("BRANCH_EVAL_GPU", "6"))
    return parser.parse_args()


def _write_configs(base_config: dict, model_root: Path) -> dict[str, Path]:
    config_dir = model_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    configs = {}
    for name, _, activation, branch in TASKS:
        output_dir = model_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        config = deepcopy(base_config)
        config["loss"]["terms"]["iq"]["weight"] = 1.0
        config["loss"]["terms"]["envelope"]["weight"] = 0.1
        config["loss"]["terms"]["lrange"]["weight"] = 0.0
        config["output"]["output_directory"] = str(output_dir)
        config["output"]["save_images_every_epochs"] = 0
        config["data"]["h5_files"] = [str(DATA_H5)]
        config["data"]["split_file"] = str(SPLIT_FILE)
        config["data"]["make_new_split"] = False
        config["training"]["epochs"] = 100
        config["training"]["resume"] = "none"
        config["backbone"]["branch_mode"] = branch
        config["backbone"]["layer_activations"] = {"fc1": activation, "fc2": activation}
        path = config_dir / f"config_{name}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        configs[name] = path
    return configs


def _train(configs: dict[str, Path], model_root: Path) -> None:
    processes = []
    for name, gpu, activation, branch in TASKS:
        log_path = model_root / "logs" / f"train_{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        command = [
            PYTHON_BIN,
            str(TRAIN_ROOT / "mban.py"),
            "--mode",
            "software",
            "--config",
            str(configs[name]),
            "--eval-config",
            str(HARDWARE_CONFIG_PATH),
            "--set",
            "resume=none",
        ]
        print(f"launching {name} ({activation}+{branch}) on GPU {gpu}")
        processes.append((name, subprocess.Popen(command, env=task_environment(gpu), stdout=log, stderr=subprocess.STDOUT), log))
    wait_for_processes(processes)


def main() -> None:
    args = parse_args()
    base_config = load_base_config()
    require_inputs()

    activation_model_root = TRAIN_ROOT / "results/SI/model/activation"
    activation_eval_root = TRAIN_ROOT / "results/SI/evaluation/activation"
    branch_eval_root = TRAIN_ROOT / "results/SI/evaluation/branch_mode"
    configs = _write_configs(base_config, activation_model_root)
    _train(configs, activation_model_root)

    activation_models = model_arguments(
        (name, activation_model_root / name / "best_val.pth") for name, _, _, _ in TASKS
    )
    evaluate_models(activation_models, activation_eval_root, args.eval_gpu)

    branch_models = model_arguments(
        (
            ("single", activation_model_root / "relu_single" / "best_val.pth"),
            ("dual", activation_model_root / "relu_dual" / "best_val.pth"),
        )
    )
    evaluate_models(branch_models, branch_eval_root, args.eval_gpu)
    print("ALL EVALUATIONS COMPLETE FOR 3x2 MATRIX!")


if __name__ == "__main__":
    main()
