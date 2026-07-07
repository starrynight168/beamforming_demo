import csv
import json
import math
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except Exception:
    yaml = None

from run_wizard_cn import (
    ALGORITHMS,
    ExitCommand,
    ROOT,
    append_run_log,
    ask_choice,
    ask_text,
    ask_yes_no,
    BackCommand,
    choose_h5,
    choose_sample,
    inspect_h5,
    print_title,
    relative_to_root,
    run_steps,
)


GLOBAL_PARAM_DESCRIPTIONS = {
    "select_angles": "角度选择。center=中心角，all=全部角度，也可以填 1/3/11 等数量。",
    "f_number": "动态孔径参数。越小孔径越大，分辨率可能更好但旁瓣更重；越大更平滑但可能变糊。",
    "dr": "显示动态范围。只影响显示/指标使用的 dB 范围，不改变算法物理过程。",
    "dynamic_aperture": "是否启用动态孔径。关闭后 f_number 基本没有意义。",
    "tgc": "是否启用深度增益补偿。关闭后深部通常更暗。",
    "tgc_alpha": "TGC 强度。只在 tgc=true 时有意义。",
    "window": "孔径窗函数。rect 锐但旁瓣明显；hann 平滑干净；tukey 折中。",
    "interp": "延迟插值方式。cubic 平滑，linear 较快，nearest 最粗糙。",
}

ALGORITHM_PARAM_DESCRIPTIONS = {
    "mv_dl": "MV 对角加载。增大通常更稳定，但可能牺牲分辨率。",
    "fbss": "是否启用前后向空间平滑。通常提升稳健性。",
    "subarray_ratio": "子阵长度比例。越大孔径越大，分辨率/稳定性会变化。",
    "temporal_win": "时间平滑窗口。越大越平滑，但细节可能变钝。",
    "num_eig": "ESBMV 保留特征数。0 通常表示自动或禁用显式数量。",
    "eig_threshold": "ESBMV 特征阈值。影响信号子空间选择。",
    "gcf_low_bins": "GCF 低频 bin 数。影响相干因子估计。",
    "gcf_power": "GCF 权重幂次。越大，相干性影响越强。",
    "lmax_ratio": "CMSAW 权重参数。影响自适应加权强度。",
    "min_subarray_len": "CMSAW 最小子阵长度。",
    "delta_max": "CMSAW 权重变化上限。",
    "gamma": "CMSAW 权重曲线强度。",
    "clip_percentile": "CMSAW 裁剪百分位，影响异常值抑制。",
    "depth_smooth_rows": "CMSAW 深度方向平滑行数。",
}

METRIC_OPTIONS = {
    "SSIM_vs_GT": ("越大越好", "max"),
    "PSNR_dB_vs_GT": ("越大越好", "max"),
    "MAE_dB_vs_GT": ("越小越好", "min"),
    "contrast_dB": ("通常绝对值/场景相关，不建议自动最优", "max"),
    "CNR": ("越大越好", "max"),
    "gCNR": ("越大越好", "max"),
    "FWHM_axial_mm": ("越小越好", "min"),
    "FWHM_lateral_mm": ("越小越好", "min"),
    "PSLR_dB": ("通常越低越好", "min"),
    "ISLR_dB": ("通常越低越好", "min"),
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
}


def load_base_config():
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法读取 config.yaml。请先安装：pip install pyyaml")
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(value):
    text = str(value).strip().replace("\\", "_").replace("/", "_").replace(" ", "")
    return text.replace(",", "_")


def value_scene_id(param_name, value):
    prefix = PARAM_VALUE_LABELS.get(param_name, param_name)
    return f"{prefix}{safe_name(value)}"


def parse_scalar(text):
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
    return [parse_scalar(item) for item in raw.replace("，", ",").split(",") if item.strip()]


def numeric_range_values():
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
        raise ValueError("范围没有生成任何值，请检查起始/结束/步长。")
    return values


def validate_values(param_name, default_value, values):
    if param_name == "select_angles":
        for value in values:
            if isinstance(value, bool) or isinstance(value, float):
                raise ValueError("select_angles 只能填 center/all 或整数角度数，例如 1,3,11；不能填 1.2 或 true。")
            if isinstance(value, int) and value > 0:
                continue
            if isinstance(value, str) and value.lower() in ("center", "all"):
                continue
            raise ValueError("select_angles 只能填 center/all 或整数角度数，例如 center,all,1,3,11。")
        return values

    if param_name in ("window", "interp"):
        allowed = {"window": {"rect", "hann", "tukey"}, "interp": {"cubic", "linear", "nearest"}}[param_name]
        bad = [value for value in values if not isinstance(value, str) or value.lower() not in allowed]
        if bad:
            raise ValueError(f"{param_name} 只能从 {', '.join(sorted(allowed))} 里选。")
        return [value.lower() for value in values]

    if isinstance(default_value, bool):
        if not all(isinstance(value, bool) for value in values):
            raise ValueError("这个参数是开关量，只能填 true/false 或 是/否。")
        return values

    if isinstance(default_value, (int, float)) and not isinstance(default_value, bool):
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            raise ValueError("这个参数是数值量，只能填数字。")
        return values

    return values


def available_params(config, algorithm):
    params = []
    for name, value in (config.get("params", {}) or {}).items():
        params.append(("global", name, value, GLOBAL_PARAM_DESCRIPTIONS.get(name, "")))
    method_params = (config.get("algorithm_params", {}) or {}).get(algorithm, {}) or {}
    for name, value in method_params.items():
        if name == "baseline_mv":
            continue
        params.append(("algorithm", name, value, ALGORITHM_PARAM_DESCRIPTIONS.get(name, "")))
    return params


def choose_algorithm():
    options = [(key, f"{key:<7} {desc}") for key, desc in ALGORITHMS.items()]
    return ask_choice("请选择要固定使用的算法（算法本身不参与消融）", options, default_index=0)


def choose_param(config, algorithm):
    params = available_params(config, algorithm)
    options = []
    for item in params:
        scope, name, default, desc = item
        prefix = "通用" if scope == "global" else algorithm
        text = f"[{prefix}] {name}，当前默认={default}"
        if desc:
            text += f"。{desc}"
        options.append((item, text))
    return ask_choice("请选择只消融一个参数", options, default_index=0)


def choose_values(param_name, default_value):
    if param_name == "select_angles":
        print("\nselect_angles 示例：center,all,1,3,11")
        print("注意：如果你想扫 1.2、1.5 这种小数，请返回上一步选择 f_number。")
    if isinstance(default_value, bool):
        default_values = [True, False]
        print("\n布尔参数默认使用两个取值：true / false")
        if ask_yes_no("是否使用默认布尔取值", default=True):
            return default_values
    while True:
        mode = ask_choice(
            "请选择取值输入方式",
            [
                ("list", "手动列举：例如 1.0,1.5,2.0 或 rect,hann,tukey"),
                ("range", "数值范围：输入起始值、结束值、步长"),
            ],
            default_index=0,
        )
        try:
            if mode == "range":
                if isinstance(default_value, bool) or param_name in ("select_angles", "window", "interp"):
                    raise ValueError("这个参数不适合用数值范围，请使用手动列举。")
                values = numeric_range_values()
            else:
                raw = ask_text("请输入消融取值，多个用逗号分隔")
                values = parse_value_list(raw)
            if len(values) < 2:
                print("消融至少需要两个取值，请重新输入。")
                continue
            return validate_values(param_name, default_value, values)
        except ValueError as exc:
            print(f"取值不合法：{exc}")


def build_config(base_config, h5_path, sample_idx, algorithm, scope, param_name, value, scene_id, has_gt):
    config = deepcopy(base_config)
    config["algorithms"] = [algorithm]
    config["scenes"] = [{
        "id": scene_id,
        "h5_path": str(relative_to_root(h5_path)).replace("\\", "/"),
        "sample_idx": int(sample_idx),
        "phantom_mode": "auto",
        "phantom_source": "auto",
        "has_gt": bool(has_gt),
    }]
    if scope == "global":
        config.setdefault("params", {})[param_name] = value
    else:
        config.setdefault("algorithm_params", {}).setdefault(algorithm, {})[param_name] = value
    return config


def save_config(config, out_dir, idx, param_name, value):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{value_scene_id(param_name, value)}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return path


def run_one(config_path, scene_id, output_root, evaluate, keep_existing):
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
    if not evaluate:
        cmd.append("--no_evaluate")
    if keep_existing:
        cmd.append("--keep_existing")

    scene_dir = Path(output_root) / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    command_path = scene_dir / "run_command.txt"
    log_path = scene_dir / "run.log"
    command_text = " ".join(("\"" + str(part) + "\"") if " " in str(part) else str(part) for part in cmd)
    command_path.write_text(command_text + "\n", encoding="utf-8")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(command_text + "\n\n")
        log_file.flush()
        result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode, cmd


def metric_method_names(config, algorithm):
    label = (config.get("algorithm_labels", {}) or {}).get(algorithm, "")
    names = {algorithm.lower()}
    if label:
        names.add(str(label).lower())
    names.add(algorithm.replace("_", "-").lower())
    return names


def read_metric(scene_dir, method_names, metric):
    csv_path = scene_dir / "metrics" / "summary_metrics.csv"
    if not csv_path.exists():
        return math.nan
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method", "").lower() in method_names:
                raw = row.get(metric, "")
                try:
                    return float(raw)
                except ValueError:
                    return math.nan
    return math.nan


def write_summary(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "parameter", "value", "scene_id", "status", "metric", "metric_value", "result_dir"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_ablation_overview(rows, out_dir, algorithm, param_name):
    images = []
    titles = []
    display_drs = []
    gt_image = None
    gt_dr = 60.0
    for row in rows:
        if row["status"] != "ok":
            continue
        scene_dir = Path(row["result_dir"])
        row_dr = 60.0
        config_path = out_dir / "configs" / f"{row['scene_id']}.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            row_dr = float((cfg.get("params", {}) or {}).get("dr", row_dr))
        else:
            params_path = scene_dir / "run_params.json"
            if params_path.exists():
                with open(params_path, "r", encoding="utf-8") as f:
                    run_params = json.load(f)
                row_dr = float((run_params.get("params", {}) or {}).get("dr", row_dr))
        method_path = scene_dir / algorithm / f"{algorithm}.npy"
        if not method_path.exists():
            continue
        comparison_path = scene_dir / "comparison.npy"
        if gt_image is None and comparison_path.exists():
            comparison = np.load(comparison_path).astype(np.float32)
            if comparison.ndim >= 3 and comparison.shape[0] >= 2:
                gt_image = comparison[0]
                gt_dr = row_dr
        images.append(np.load(method_path).astype(np.float32))
        titles.append(f"{param_name}={row['value']}")
        display_drs.append(row_dr)

    if not images:
        return None
    if gt_image is not None:
        images = [gt_image] + images
        titles = ["GT"] + titles
        display_drs = [gt_dr] + display_drs

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
    for idx, (image, title, image_dr) in enumerate(zip(images, titles, display_drs)):
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


def choose_metric():
    options = []
    for key, (desc, _) in METRIC_OPTIONS.items():
        options.append((key, f"{key}：{desc}"))
    return ask_choice("请选择用于自动挑选最优的指标", options, default_index=0)


def best_row(rows, metric, direction):
    valid = [row for row in rows if row["status"] == "ok" and not math.isnan(row["metric_value"])]
    if not valid:
        return None
    reverse = direction == "max"
    return sorted(valid, key=lambda row: row["metric_value"], reverse=reverse)[0]


def main():
    print_title("中文单参数消融向导")
    print("用途：固定一个算法，只改变一个参数，自动批量运行并汇总指标。")
    print("建议：一次只扫一个变量，否则很难判断到底是谁造成画质变化。")
    print("提示：每一步都可以输入 b 返回上一步，输入 q 退出。")

    state = {"config": load_base_config()}

    def step_h5():
        state["h5_path"] = choose_h5()

    def step_sample():
        state["sample_idx"] = choose_sample(state["h5_path"], teaching=True)

    def step_algorithm():
        state["algorithm"] = choose_algorithm()

    def step_param():
        scope, param_name, default_value, _ = choose_param(state["config"], state["algorithm"])
        state["scope"] = scope
        state["param_name"] = param_name
        state["default_value"] = default_value

    def step_values():
        state["values"] = choose_values(state["param_name"], state["default_value"])

    def step_gt():
        has_gt_default = inspect_h5(state["h5_path"])["gt"] == "有"
        state["has_gt"] = ask_yes_no("是否把 H5 里的 GT 加入对比图", default=state.get("has_gt", has_gt_default)) if has_gt_default else False

    def step_evaluate():
        state["evaluate"] = ask_yes_no("是否计算指标", default=state.get("evaluate", state["has_gt"]))
        if state["evaluate"]:
            state["metric"] = choose_metric()
        else:
            state["metric"] = ""
        state["direction"] = METRIC_OPTIONS.get(state["metric"], ("", "max"))[1]

    def step_keep_existing():
        state["keep_existing"] = ask_yes_no("如果某组结果已存在，是否复用", default=state.get("keep_existing", False))

    def step_output():
        default_output = state.get("output_root")
        if default_output is None:
            h5_tag = Path(state["h5_path"]).stem
            default_output = Path("results") / f"{h5_tag}_sample{state['sample_idx']}"
        output_root = Path(ask_text("请输入消融结果根目录", default=str(default_output)))
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        state["output_root"] = output_root

    def step_confirm():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state["run_id"] = f"ablation_{timestamp}"
        state["run_root"] = state["output_root"] / state["run_id"]
        state["config_dir"] = state["run_root"] / "configs"
        print_title("消融计划")
        print(f"H5: {relative_to_root(state['h5_path'])}")
        print(f"样本: {state['sample_idx']}")
        print(f"算法: {state['algorithm']}")
        print(f"消融参数: {state['param_name']} ({'通用参数' if state['scope'] == 'global' else '算法参数'})")
        print(f"取值: {state['values']}")
        print(f"计算指标: {'是，最优指标=' + state['metric'] if state['evaluate'] else '否'}")
        print(f"输出目录: {state['run_root']}")
        state["execute"] = ask_yes_no("确认开始消融吗", default=True)

    try:
        run_steps([
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
        ])
    except ExitCommand:
        print("\n已退出。")
        return

    if not state["execute"]:
        print("已取消。")
        return

    rows = []
    method_names = metric_method_names(state["config"], state["algorithm"])
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
            state["has_gt"],
        )
        cfg_path = save_config(cfg, state["config_dir"], idx, state["param_name"], value)
        print_title(f"运行 {idx}/{len(state['values'])}: {state['param_name']} = {value}")
        code, cmd = run_one(cfg_path, scene_id, state["run_root"], state["evaluate"], state["keep_existing"])
        scene_dir = state["run_root"] / scene_id
        status = "ok" if code == 0 else f"failed({code})"
        metric_value = read_metric(scene_dir, method_names, state["metric"]) if state["evaluate"] else math.nan
        rows.append({
            "index": idx,
            "parameter": state["param_name"],
            "value": value,
            "scene_id": scene_id,
            "status": status,
            "metric": state["metric"],
            "metric_value": metric_value,
            "result_dir": str(scene_dir),
        })
        if code != 0:
            print("这一组运行失败，继续下一组。")

    summary_path = state["run_root"] / "ablation_summary.csv"
    write_summary(rows, summary_path)
    overview_path = save_ablation_overview(rows, state["run_root"], state["algorithm"], state["param_name"])

    print_title("消融完成")
    print(f"汇总表: {summary_path}")
    if overview_path:
        print(f"总览图: {overview_path}")
    if state["evaluate"]:
        winner = best_row(rows, state["metric"], state["direction"])
        if winner is None:
            print("没有可用于自动选择最优的有效指标。")
        else:
            print(f"按 {state['metric']}（{METRIC_OPTIONS[state['metric']][0]}）选择的最优值:")
            print(f"  {state['param_name']} = {winner['value']}")
            print(f"  指标值 = {winner['metric_value']}")
            print(f"  结果目录 = {winner['result_dir']}")

    append_run_log({
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
        "metric": state["metric"],
    })


if __name__ == "__main__":
    main()
