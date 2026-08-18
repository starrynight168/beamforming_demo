"""Provide Python utilities for run_all."""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


ROOT = Path(__file__).resolve().parent


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description="Run all scenes in config.yaml.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--only",
        default="all",
        help="all or comma-separated scene ids",
    )
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Override algorithms, comma-separated",
    )
    parser.add_argument("--output_root", default="results")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--no_evaluate", action="store_true", default=False)
    parser.add_argument("--keep_existing", action="store_true", default=False)
    args, extra_algorithm_args = parser.parse_known_args()
    args.extra_algorithm_args = extra_algorithm_args
    return args


def load_config(path):
    """Load config."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read config.yaml. Install with: pip install pyyaml",
        )
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def main():
    """Run the command-line workflow."""
    args = parse_args()
    config = load_config(ROOT / args.config)
    scene_configs = config.get("scenes")
    if not isinstance(scene_configs, list) or not all(isinstance(scene, dict) for scene in scene_configs):
        raise ValueError("config.scenes 必须是场景字典列表")
    all_scenes = [scene.get("id") for scene in scene_configs]
    if (
        not all_scenes
        or any(not isinstance(scene_id, str) or not scene_id.strip() for scene_id in all_scenes)
        or len(all_scenes) != len(set(all_scenes))
    ):
        raise ValueError("config.scenes 必须包含非空且不重复的 id")
    scenes = all_scenes if args.only == "all" else [item.strip() for item in args.only.split(",") if item.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("--only 必须包含非空且不重复的场景 id")
    unknown = [scene for scene in scenes if scene not in all_scenes]
    if unknown:
        raise ValueError(f"Unknown scene ids: {unknown}")

    failed = []
    for idx, scene_id in enumerate(scenes, 1):
        cmd = [
            args.python_exe,
            str(ROOT / "run_one.py"),
            "--scene",
            scene_id,
            "--config",
            args.config,
            "--output_root",
            args.output_root,
            "--python_exe",
            args.python_exe,
        ]
        if args.algorithms:
            cmd.extend(["--algorithms", args.algorithms])
        if args.no_evaluate:
            cmd.append("--no_evaluate")
        if args.keep_existing:
            cmd.append("--keep_existing")
        cmd.extend(args.extra_algorithm_args)

        print("\n" + "#" * 70)
        print(f"[{idx}/{len(scenes)}] {scene_id}")
        print("#" * 70)
        result = subprocess.run(cmd, cwd=ROOT, text=True, check=False)
        if result.returncode != 0:
            failed.append(scene_id)

    if failed:
        print(f"Failed scenes: {failed}")
        raise SystemExit(1)
    print("All scenes completed.")


if __name__ == "__main__":
    main()
