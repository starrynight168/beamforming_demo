"""Provide Python utilities for run_ablation_cn."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

try:
    from . import run_wizard_cn as wizard
except ImportError:
    import run_wizard_cn as wizard

ALGORITHMS = wizard.ALGORITHMS
ExitCommandError = wizard.ExitCommandError
append_run_log = wizard.append_run_log
ask_choice = wizard.ask_choice
ask_text = wizard.ask_text
ask_yes_no = wizard.ask_yes_no
choose_h5 = wizard.choose_h5
choose_sample = wizard.choose_sample
inspect_h5 = wizard.inspect_h5
load_base_config = wizard.load_base_config
print_title = wizard.print_title
relative_to_root = wizard.relative_to_root
run_steps = wizard.run_steps

GLOBAL_PARAM_DESCRIPTIONS = {
    "select_angles": "角度选择。center=中心角,all=全部角度,也可以填 1/3/11 等数量。",
    "f_number": "动态孔径参数。越小孔径越大,分辨率可能更好但旁瓣更重;越大更平滑但可能变糊。",
    "dr": "显示动态范围。只影响显示/指标使用的 dB 范围,不改变算法物理过程。",
    "dynamic_aperture": "是否启用动态孔径。关闭后 f_number 基本没有意义。",
    "tgc": "是否启用深度增益补偿。关闭后深部通常更暗。",
    "tgc_alpha": "TGC 强度。只在 tgc=true 时有意义。",
    "window": "孔径窗函数。rect 锐但旁瓣明显;hann 平滑干净;tukey 折中。",
    "interp": "延迟插值方式。nearest 最快但粗糙,linear 较快,cubic 平滑,quintic/farrow/sinc 更高阶但通常更慢。",
}

ALGORITHM_PARAM_DESCRIPTIONS = {
    "aperture_mode": "DAS 接收孔径模式。discrete 为与 MV 一致的离散通道孔径;geometry 为 DAS 专用的连续几何孔径。",
    "mv_dl": "MV 对角加载。增大通常更稳定,但可能牺牲分辨率。",
    "fbss": "是否启用前后向空间平滑。通常提升稳健性。",
    "subarray_ratio": "子阵长度比例。越大孔径越大,分辨率/稳定性会变化。",
    "temporal_win": "时间平滑窗口。越大越平滑,但细节可能变钝。",
    "num_eig": "ESBMV 保留特征数。0 通常表示自动或禁用显式数量。",
    "eig_threshold": "ESBMV 特征阈值。影响信号子空间选择。",
    "gcf_low_bins": "GCF 低频 bin 数。影响相干因子估计。",
    "gcf_power": "GCF 权重幂次。越大,相干性影响越强。",
    "lmax_ratio": "CMSAW 权重参数。影响自适应加权强度。",
    "min_subarray_len": "CMSAW 最小子阵长度。",
    "delta_max": "CMSAW 权重变化上限。",
    "gamma": "CMSAW 权重曲线强度。",
    "clip_percentile": "CMSAW 裁剪百分位,影响异常值抑制。",
    "depth_smooth_rows": "CMSAW 深度方向平滑行数。",
}

PARAM_VALUE_LABELS = {
    "select_angles": "angles",
    "f_number": "fnumber",
    "dynamic_aperture": "dynamic_aperture",
    "dr": "dr",
    "tgc": "tgc",
    "tgc_alpha": "tgc_alpha",
    "window": "window",
    "interp": "interp",
    "aperture_mode": "aperture",
}

CATEGORICAL_PARAM_CHOICES = {
    "aperture_mode": ("discrete", "geometry"),
    "window": ("rect", "tukey", "hann", "hamming", "blackman", "kaiser"),
    "interp": ("nearest", "linear", "cubic", "quintic", "farrow", "sinc"),
}


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description="中文单参数消融向导(交互式)")
    return parser.parse_args()


def safe_name(value):
    """Execute safe name."""
    text = str(value).strip().replace("\\", "_").replace("/", "_").replace(" ", "")
    return text.replace(",", "_")


def value_scene_id(param_name, value):
    """Execute value scene id."""
    prefix = PARAM_VALUE_LABELS.get(param_name, param_name)
    return f"{prefix}{safe_name(value)}"


def parse_scalar(text):
    """Parse scalar."""
    raw = str(text).strip()
    low = raw.lower()
    if low in ("true", "yes", "y", "开", "是"):
        return True
    if low in ("false", "no", "n", "关", "否"):
        return False
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_value_list(raw):
    """Parse value list."""
    return [parse_scalar(item) for item in raw.split(",") if item.strip()]


def numeric_range_values():
    """Execute numeric range values."""
    start = float(ask_text("起始值"))
    stop = float(ask_text("结束值"))
    step = float(ask_text("步长"))
    if step == 0:
        raise ValueError("步长不能为 0。")
    values = []
    cur = start
    eps = abs(step) * 1e-9
    if step > 0:
        while cur <= stop + eps:
            values.append(round(cur, 10))
            cur += step
    else:
        while cur >= stop - eps:
            values.append(round(cur, 10))
            cur += step
    if not values:
        raise ValueError("范围没有生成任何值,请检查起始/结束/步长。")
    return values


def validate_values(param_name, default_value, values):
    """Validate values."""
    if param_name in CATEGORICAL_PARAM_CHOICES:
        allowed = set(CATEGORICAL_PARAM_CHOICES[param_name])
        normalized = [str(value).lower() for value in values]
        bad = [value for value in normalized if value not in allowed]
        if bad:
            raise ValueError(
                f"{param_name} 只能从 {', '.join(CATEGORICAL_PARAM_CHOICES[param_name])} 里选。",
            )
        return normalized

    if param_name == "select_angles":
        for value in values:
            if isinstance(value, (bool, float)):
                raise ValueError(
                    "select_angles 只能填 center/all 或整数角度数,例如 1,3,11;不能填 1.2 或 true。",
                )
            if isinstance(value, int) and value > 0:
                continue
            if isinstance(value, str) and value.lower() in ("center", "all"):
                continue
            raise ValueError(
                "select_angles 只能填 center/all 或整数角度数,例如 center,all,1,3,11。",
            )
        return values

    if isinstance(default_value, bool):
        if not all(isinstance(value, bool) for value in values):
            raise ValueError("这个参数是开关量,只能填 true/false 或 是/否。")
        return values

    if isinstance(default_value, (int, float)) and not isinstance(default_value, bool):
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            raise ValueError("这个参数是数值量,只能填数字。")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("数值必须是有限值。")
        if param_name in {"f_number", "dr"} and not all(
            float(value) > 0 for value in values
        ):
            raise ValueError(f"{param_name} 必须大于 0。")
        if param_name in {"tgc_alpha", "mv_dl", "gcf_power"} and not all(
            float(value) >= 0 for value in values
        ):
            raise ValueError(f"{param_name} 必须大于等于 0。")
        if param_name == "subarray_ratio" and not all(
            0 < float(value) <= 1 for value in values
        ):
            raise ValueError("subarray_ratio 必须位于 (0,1]。")
        if param_name in {"eig_threshold", "delta_max"} and not all(
            0 <= float(value) <= 1 for value in values
        ):
            raise ValueError(f"{param_name} 必须位于 [0,1]。")
        if param_name in {"gamma"} and not all(
            0 < float(value) <= 1 for value in values
        ):
            raise ValueError("gamma 必须位于 (0,1]。")
        if param_name == "lmax_ratio" and not all(
            0 < float(value) <= 0.5 for value in values
        ):
            raise ValueError("lmax_ratio 必须位于 (0,0.5]。")
        if param_name == "clip_percentile" and not all(
            0 < float(value) <= 100 for value in values
        ):
            raise ValueError("clip_percentile 必须位于 (0,100]。")
        if param_name in {"num_eig", "gcf_low_bins"} and not all(
            isinstance(value, int) and value >= 0 for value in values
        ):
            raise ValueError(f"{param_name} 必须是非负整数。")
        if param_name == "min_subarray_len" and not all(
            isinstance(value, int) and value >= 2 for value in values
        ):
            raise ValueError("min_subarray_len 必须是大于等于 2 的整数。")
        if param_name == "depth_smooth_rows" and not all(
            isinstance(value, int) and value >= 1 for value in values
        ):
            raise ValueError("depth_smooth_rows 必须是正整数。")
        if param_name == "temporal_win" and not all(
            isinstance(value, int) and value >= 1 and value % 2 == 1 for value in values
        ):
            raise ValueError("temporal_win 必须是正奇数。")
        return values

    return values


def available_params(config, algorithm):
    """Execute available params."""
    params = []
    for name, value in (config.get("params", {}) or {}).items():
        params.append(("global", name, value, GLOBAL_PARAM_DESCRIPTIONS.get(name, "")))
    method_params = (config.get("algorithm_params", {}) or {}).get(algorithm, {}) or {}
    for name, value in method_params.items():
        params.append(
            ("algorithm", name, value, ALGORITHM_PARAM_DESCRIPTIONS.get(name, "")),
        )
    return params


def choose_algorithm():
    """Execute choose algorithm."""
    options = [(key, f"{key:<7} {desc}") for key, desc in ALGORITHMS.items()]
    return ask_choice(
        "请选择要固定使用的算法(算法本身不参与消融)",
        options,
        default_index=0,
    )


def choose_param(config, algorithm):
    """Execute choose param."""
    params = available_params(config, algorithm)
    options = []
    for item in params:
        scope, name, default, desc = item
        prefix = "通用" if scope == "global" else algorithm
        text = f"[{prefix}] {name},当前默认={default}"
        if desc:
            text += f"。{desc}"
        options.append((item, text))
    return ask_choice("请选择只消融一个参数", options, default_index=0)


def choose_values(param_name, default_value):
    """Execute choose values."""
    if param_name == "aperture_mode":
        values = list(CATEGORICAL_PARAM_CHOICES[param_name])
        print(f"\n{param_name} 是固定枚举参数,将比较:{', '.join(values)}")
        return values

    if param_name == "select_angles":
        print("\nselect_angles 示例:center,all,1,3,11")
        print("注意:如果你想扫 1.2、1.5 这种小数,请返回上一步选择 f_number。")
    elif param_name in CATEGORICAL_PARAM_CHOICES:
        print(
            f"\n{param_name} 可选值:{', '.join(CATEGORICAL_PARAM_CHOICES[param_name])}",
        )
    if isinstance(default_value, bool):
        default_values = [True, False]
        print("\n布尔参数默认使用两个取值:true / false")
        if ask_yes_no("是否使用默认布尔取值", default=True):
            return default_values

    list_only = (
        param_name == "select_angles"
        or param_name in CATEGORICAL_PARAM_CHOICES
        or isinstance(default_value, bool)
    )
    while True:
        mode = (
            "list"
            if list_only
            else ask_choice(
                "请选择取值输入方式",
                [
                    ("list", "手动列举:例如 1.0,1.5,2.0"),
                    ("range", "数值范围:输入起始值、结束值、步长"),
                ],
                default_index=0,
            )
        )
        try:
            if mode == "range":
                values = numeric_range_values()
            else:
                raw = ask_text("请输入消融取值,多个用逗号分隔")
                values = parse_value_list(raw)
            if len(values) < 2:
                print("消融至少需要两个取值,请重新输入。")
                continue
            return validate_values(param_name, default_value, values)
        except ValueError as exc:
            print(f"取值不合法:{exc}")


def build_config(
    base_config,
    h5_path,
    sample_idx,
    algorithm,
    scope,
    param_name,
    value,
    scene_id,
):
    """Build config."""
    config = deepcopy(base_config)
    config["algorithms"] = [algorithm]
    config["scenes"] = [
        {
            "id": scene_id,
            "h5_path": str(relative_to_root(h5_path)).replace("\\", "/"),
            "sample_idx": int(sample_idx),
        },
    ]
    if scope == "global":
        config.setdefault("params", {})[param_name] = value
    else:
        config.setdefault("algorithm_params", {}).setdefault(algorithm, {})[
            param_name
        ] = value
    return config


def save_config(config, out_dir, param_name, value):
    """Save config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{value_scene_id(param_name, value)}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return path


def can_reuse_flat_result(config_path, scene_dir, algorithm):
    """Execute can reuse flat result."""
    output_path = scene_dir / f"{algorithm}.npy"
    params_path = scene_dir / "params.json"
    if not output_path.exists() or not params_path.exists():
        return False
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    with open(params_path, encoding="utf-8") as f:
        actual = json.load(f)
    expected = dict(config.get("params", {}) or {})
    expected.update((config.get("algorithm_params", {}) or {}).get(algorithm, {}) or {})
    scene = (config.get("scenes", []) or [{}])[0]
    expected["h5_path"] = str((ROOT / scene.get("h5_path", "")).resolve())
    expected["h5_sample_idx"] = int(scene.get("sample_idx", 0))
    expected["method"] = algorithm
    return all(actual.get(name) == value for name, value in expected.items())


def run_one(config_path, scene_id, output_root, keep_existing):
    """Execute run one."""
    scene_dir = Path(output_root) / scene_id
    algorithm = (
        yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    ).get(
        "algorithms",
        [""],
    )[0]
    if keep_existing and can_reuse_flat_result(config_path, scene_dir, algorithm):
        print(f"Reuse {scene_dir / f'{algorithm}.npy'}")
        return 0, []

    cmd = [
        sys.executable,
        str(ROOT / "run_one.py"),
        "--config",
        str(config_path),
        "--scene",
        scene_id,
        "--output_root",
        str(output_root),
    ]
    # 消融统一在根目录完成一次多方法评估,子目录只负责重建。
    cmd.append("--no_evaluate")
    if keep_existing:
        cmd.append("--keep_existing")

    scene_dir.mkdir(parents=True, exist_ok=True)
    log_path = scene_dir / "run.log"
    command_text = " ".join(
        ('"' + str(part) + '"') if " " in str(part) else str(part) for part in cmd
    )
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(command_text + "\n\n")
        log_file.flush()
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode, cmd


def method_output_path(scene_dir, algorithm, suffix):
    """Execute method output path."""
    direct = scene_dir / f"{algorithm}.{suffix}"
    if direct.exists():
        return direct
    return scene_dir / algorithm / f"{algorithm}.{suffix}"


def _backup_flatten_target(path: Path, backup_root: Path) -> None:
    backup_root.mkdir(parents=True, exist_ok=True)
    target = backup_root / path.name
    suffix = 1
    while target.exists():
        target = backup_root / f"{path.name}.{suffix}"
        suffix += 1
    shutil.move(str(path), str(target))


def flatten_ablation_result(scene_dir, algorithm):
    """Execute flatten ablation result."""
    method_dir = scene_dir / algorithm
    if not method_dir.exists():
        return
    backup_root = None

    def backup_existing(path: Path) -> None:
        nonlocal backup_root
        if backup_root is None:
            backup_root = (
                scene_dir
                / "_flatten_backup"
                / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
        _backup_flatten_target(path, backup_root)

    for source in method_dir.iterdir():
        target = scene_dir / source.name
        if target.exists():
            backup_existing(target)
        shutil.move(str(source), str(target))
    method_dir.rmdir()
    individual_dir = scene_dir / "individual_images"
    gt_source = individual_dir / "ground_truth.png"
    if gt_source.exists():
        target = scene_dir / "ground_truth.png"
        if target.exists():
            backup_existing(target)
        shutil.move(str(gt_source), str(target))
    for name in ("comparison.npy", "comparison.png", "run_params.json", "run.log"):
        path = scene_dir / name
        if path.exists():
            backup_existing(path)
    for name in ("metrics", "individual_images"):
        path = scene_dir / name
        if path.exists():
            backup_existing(path)


def preserve_ground_truth(scene_dir, out_dir):
    """Keep one GT panel before per-value comparison files are flattened away."""
    target = out_dir / "ground_truth.npy"
    if target.exists():
        return target
    comparison_path = scene_dir / "comparison.npy"
    if not comparison_path.exists():
        return None
    comparison = np.load(comparison_path).astype(np.float32)
    if comparison.ndim != 3 or comparison.shape[0] < 2:
        return None
    np.save(target, comparison[0])
    return target


def write_summary(rows, out_path):
    """Execute write summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "parameter", "value", "scene_id", "status", "result_dir"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def save_ablation_overview(rows, out_dir, algorithm, param_name):
    """Save ablation overview."""
    images = []
    titles = []
    display_drs = []
    gt_path = out_dir / "ground_truth.npy"
    gt_image = np.load(gt_path).astype(np.float32) if gt_path.exists() else None
    gt_dr = 60.0
    for row in rows:
        if row["status"] != "ok":
            continue
        scene_dir = Path(row["result_dir"])
        row_dr = 60.0
        config_path = out_dir / "configs" / f"{row['scene_id']}.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            row_dr = float((cfg.get("params", {}) or {}).get("dr", row_dr))
        else:
            params_path = scene_dir / "run_params.json"
            if params_path.exists():
                with open(params_path, encoding="utf-8") as f:
                    run_params = json.load(f)
                row_dr = float((run_params.get("params", {}) or {}).get("dr", row_dr))
        method_path = method_output_path(scene_dir, algorithm, "npy")
        if not method_path.exists():
            continue
        if gt_image is not None and not images:
            gt_dr = row_dr
        images.append(np.load(method_path).astype(np.float32))
        titles.append(f"{param_name}={row['value']}")
        display_drs.append(row_dr)

    if not images:
        return None
    if gt_image is not None:
        images = [gt_image, *images]
        titles = ["GT", *titles]
        display_drs = [gt_dr, *display_drs]

    stack = np.stack(images, axis=0)
    np.save(out_dir / "ablation_comparison.npy", stack)

    n_images = len(images)
    cols = min(4, n_images)
    rows_count = math.ceil(n_images / cols)
    fig, axes = plt.subplots(
        rows_count,
        cols,
        figsize=(3.4 * cols, 4.0 * rows_count),
        dpi=300,
        squeeze=False,
        constrained_layout=True,
    )
    for idx, (image, title, image_dr) in enumerate(
        zip(images, titles, display_drs, strict=True),
    ):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(image, cmap="gray", vmin=-image_dr, vmax=0.0, aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for idx in range(n_images, rows_count * cols):
        axes[idx // cols][idx % cols].axis("off")
    path = out_dir / "ablation_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def run_ablation_evaluation(rows, out_dir, h5_path, sample_idx, has_gt, dr):
    """Execute run ablation evaluation."""
    if not has_gt:
        return None
    comparison_path = out_dir / "ablation_comparison.npy"
    if not comparison_path.exists():
        return None
    valid_rows = [row for row in rows if row["status"] == "ok"]
    if not valid_rows:
        return None
    if any(row["parameter"] == "dr" for row in valid_rows):
        return None
    method_ids = [f"ablation_{row['index']:02d}" for row in valid_rows]
    method_labels = [f"{row['parameter']}={row['value']}" for row in valid_rows]
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluate_cmd = [
        sys.executable,
        str(ROOT / "evaluation" / "evaluate.py"),
        "--comparison_npy",
        str(comparison_path),
        "--h5_path",
        str(h5_path),
        "--h5_sample_idx",
        str(sample_idx),
        "--dr",
        str(dr),
        "--methods",
        ",".join(method_ids),
        "--method_labels",
        ",".join(method_labels),
        "--out_dir",
        str(metrics_dir),
        "--phantom_mode",
        "auto",
        "--phantom_source",
        "auto",
    ]
    result = subprocess.run(evaluate_cmd, cwd=ROOT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("消融统一评估失败")
    plot_cmd = [
        sys.executable,
        str(ROOT / "evaluation" / "plot_metrics.py"),
        "--metrics_dir",
        str(metrics_dir),
        "--reference-mode",
        "algorithms",
    ]
    result = subprocess.run(plot_cmd, cwd=ROOT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("消融统一指标绘图失败")
    return metrics_dir


def collect_ablation_individual_images(rows, out_dir, algorithm, param_name):
    """Execute collect ablation individual images."""
    output_dir = out_dir / "individual_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    copied_gt = False
    for row in rows:
        if row["status"] != "ok":
            continue
        scene_dir = Path(row["result_dir"])
        source = method_output_path(scene_dir, algorithm, "png")
        if not source.exists():
            continue
        filename = (
            f"{row['index']:02d}_{param_name}_{safe_name(row['value'])}_{algorithm}.png"
        )
        shutil.copy2(source, output_dir / filename)
        copied += 1

        if not copied_gt:
            for gt_source in (
                scene_dir / "ground_truth.png",
                scene_dir / "individual_images" / "ground_truth.png",
            ):
                if gt_source.exists():
                    shutil.copy2(gt_source, output_dir / "ground_truth.png")
                    copied_gt = True
                    break
    return output_dir if copied else None


def main():
    """Run the command-line workflow."""
    parse_args()
    print_title("中文单参数消融向导")
    print("用途:固定一个算法,只改变一个参数,自动批量运行并汇总指标。")
    print("建议:一次只扫一个变量,否则很难判断到底是谁造成画质变化。")
    print("提示:每一步都可以输入 b 返回上一步,输入 q 退出。")

    state = {"config": load_base_config()}

    def step_h5():
        """Execute step h5."""
        state["h5_path"] = choose_h5()

    def step_sample():
        """Execute step sample."""
        state["sample_idx"] = choose_sample(state["h5_path"], teaching=True)

    def step_algorithm():
        """Execute step algorithm."""
        state["algorithm"] = choose_algorithm()

    def step_param():
        """Execute step param."""
        scope, param_name, default_value, _ = choose_param(
            state["config"],
            state["algorithm"],
        )
        state["scope"] = scope
        state["param_name"] = param_name
        state["default_value"] = default_value

    def step_values():
        """Execute step values."""
        state["values"] = choose_values(state["param_name"], state["default_value"])

    def step_gt():
        """Execute step gt."""
        info = inspect_h5(state["h5_path"])
        state["has_gt"] = info["gt"] == "有"
        state["in_vivo"] = info["in_vivo"][state["sample_idx"]]

    def step_evaluate():
        """Execute step evaluate."""
        state["dr"] = float((state["config"].get("params") or {}).get("dr", 60.0))
        if state["param_name"] == "dr":
            state["evaluate"] = False
            print("指标: 跳过。dr 只影响显示动态范围,不对不同 dr 使用统一评估口径。")
        else:
            state["evaluate"] = state["has_gt"] and not state["in_vivo"]
        if state["param_name"] == "dr":
            return
        if state["in_vivo"]:
            print("指标: 活体样本自动跳过。")
        else:
            print(
                f"指标: {'H5 中存在 GT,将自动计算' if state['has_gt'] else 'H5 中不存在 GT,将跳过'}",
            )

    def step_keep_existing():
        """Execute step keep existing."""
        state["keep_existing"] = ask_yes_no(
            "如果某组结果已存在,是否复用",
            default=state.get("keep_existing", False),
        )

    def step_output():
        """Execute step output."""
        default_output = state.get("output_root")
        if default_output is None:
            h5_tag = Path(state["h5_path"]).stem
            default_output = Path("results") / f"{h5_tag}_sample{state['sample_idx']}"
        output_root = Path(
            ask_text("请输入消融结果根目录", default=str(default_output)),
        )
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        state["output_root"] = output_root

    def step_confirm():
        """Execute step confirm."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state["run_id"] = f"ablation_{timestamp}"
        state["run_root"] = state["output_root"] / state["run_id"]
        state["config_dir"] = state["run_root"] / "configs"
        print_title("消融计划")
        print(f"H5: {relative_to_root(state['h5_path'])}")
        print(f"样本: {state['sample_idx']}")
        print(f"算法: {state['algorithm']}")
        print(
            f"消融参数: {state['param_name']} ({'通用参数' if state['scope'] == 'global' else '算法参数'})",
        )
        print(f"取值: {state['values']}")
        print(f"计算指标: {'是' if state['evaluate'] else '否'}")
        print(f"输出目录: {state['run_root']}")
        state["execute"] = ask_yes_no("确认开始消融吗", default=True)

    try:
        run_steps(
            [
                step_h5,
                step_sample,
                step_algorithm,
                step_param,
                step_values,
                step_gt,
                step_evaluate,
                step_keep_existing,
                step_output,
                step_confirm,
            ],
        )
    except ExitCommandError:
        print("\n已退出。")
        return

    if not state["execute"]:
        print("已取消。")
        return

    rows = []
    for idx, value in enumerate(state["values"], 1):
        scene_id = value_scene_id(state["param_name"], value)
        cfg = build_config(
            state["config"],
            state["h5_path"],
            state["sample_idx"],
            state["algorithm"],
            state["scope"],
            state["param_name"],
            value,
            scene_id,
        )
        cfg_path = save_config(
            cfg,
            state["config_dir"],
            state["param_name"],
            value,
        )
        print_title(
            f"运行 {idx}/{len(state['values'])}: {state['param_name']} = {value}",
        )
        code, _cmd = run_one(
            cfg_path,
            scene_id,
            state["run_root"],
            state["keep_existing"],
        )
        scene_dir = state["run_root"] / scene_id
        status = "ok" if code == 0 else f"failed({code})"
        if code == 0:
            if state["has_gt"]:
                preserve_ground_truth(scene_dir, state["run_root"])
            flatten_ablation_result(scene_dir, state["algorithm"])
        rows.append(
            {
                "index": idx,
                "parameter": state["param_name"],
                "value": value,
                "scene_id": scene_id,
                "status": status,
                "result_dir": str(scene_dir),
            },
        )
        if code != 0:
            print("这一组运行失败,继续下一组。")

    summary_path = state["run_root"] / "ablation_summary.csv"
    write_summary(rows, summary_path)
    overview_path = save_ablation_overview(
        rows,
        state["run_root"],
        state["algorithm"],
        state["param_name"],
    )
    metrics_dir = None
    if state["evaluate"]:
        metrics_dir = run_ablation_evaluation(
            rows,
            state["run_root"],
            state["h5_path"],
            state["sample_idx"],
            state["has_gt"],
            state["dr"],
        )
    individual_dir = collect_ablation_individual_images(
        rows,
        state["run_root"],
        state["algorithm"],
        state["param_name"],
    )

    print_title("消融完成")
    print(f"汇总表: {summary_path}")
    if metrics_dir:
        print(f"指标目录: {metrics_dir}")
    if overview_path:
        print(f"总览图: {overview_path}")
    if individual_dir:
        print(f"单图目录: {individual_dir}")
    append_run_log(
        {
            "type": "ablation",
            "dir": str(relative_to_root(state["run_root"])).replace("\\", "/"),
            "h5_path": str(relative_to_root(state["h5_path"])).replace("\\", "/"),
            "sample_idx": int(state["sample_idx"]),
            "algorithm": state["algorithm"],
            "scope": state["scope"],
            "parameter": state["param_name"],
            "values": state["values"],
            "has_gt": state["has_gt"],
            "evaluate": state["evaluate"],
        },
    )


if __name__ == "__main__":
    main()
