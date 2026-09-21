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


WEIGHTS = tuple(round(index * 0.1, 1) for index in range(11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the envelope-loss weight sweep")
    parser.add_argument("--eval-gpu", default=os.environ.get("ENVELOPE_EVAL_GPU", "5"))
    return parser.parse_args()


def _write_configs(base_config: dict, model_root: Path) -> dict[float, Path]:
    config_dir = model_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    configs = {}
    for weight in WEIGHTS:
        output_dir = model_root / f"env_{weight}"
        output_dir.mkdir(parents=True, exist_ok=True)
        config = deepcopy(base_config)
        config["loss"]["terms"]["iq"]["weight"] = 1.0
        config["loss"]["terms"]["envelope"]["weight"] = weight
        config["loss"]["terms"]["lrange"]["weight"] = 0.0
        config["output"]["output_directory"] = str(output_dir)
        config["output"]["save_images_every_epochs"] = 0
        config["data"]["h5_files"] = [str(DATA_H5)]
        config["data"]["split_file"] = str(SPLIT_FILE)
        config["data"]["make_new_split"] = False
        config["training"]["epochs"] = 100
        config["training"]["resume"] = "none"
        path = config_dir / f"config_env_{weight}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        configs[weight] = path
    return configs


def _train(configs: dict[float, Path], model_root: Path) -> None:
    for batch_weights, batch_gpus in ((WEIGHTS[:8], range(8)), (WEIGHTS[8:], range(3))):
        processes = []
        for weight, gpu in zip(batch_weights, batch_gpus, strict=True):
            log_path = model_root / "logs" / f"train_env_{weight}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            command = [
                PYTHON_BIN,
                str(TRAIN_ROOT / "mban.py"),
                "--mode",
                "software",
                "--config",
                str(configs[weight]),
                "--eval-config",
                str(HARDWARE_CONFIG_PATH),
                "--set",
                "resume=none",
            ]
            print(f"launching env_{weight} on GPU {gpu}")
            processes.append(
                (
                    f"env_{weight}",
                    subprocess.Popen(command, env=task_environment(gpu), stdout=log, stderr=subprocess.STDOUT),
                    log,
                )
            )
        wait_for_processes(processes)


def _evaluate(model_root: Path, output_root: Path, gpu: str) -> None:
    model_args = model_arguments(
        (f"env_{weight}", model_root / f"env_{weight}" / "best_val.pth") for weight in WEIGHTS
    )
    evaluate_models(model_args, output_root, gpu)


def main() -> None:
    args = parse_args()
    base_config = load_base_config()
    require_inputs()
    model_root = TRAIN_ROOT / "results/SI/model/envelope_sweep/FP32"
    output_root = TRAIN_ROOT / "results/SI/evaluation/envelope_sweep/FP32"
    configs = _write_configs(base_config, model_root)
    _train(configs, model_root)
    _evaluate(model_root, output_root, args.eval_gpu)
    print(f"ALL DONE! Results stored in: {output_root}")


if __name__ == "__main__":
    main()
