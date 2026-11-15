"""Shared input and interaction helpers for beamforming workflows."""

import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

from algorithms.shared_core.labels import target_label
from shared_core.h5_schema import require_h5_schema
from evaluation.context import load_picmus_context
from tools.shared.run_output import emit

SAMPLE_HELP = "一个 H5 里可以有多个样本。样本编号就是选择第几帧/第几个场景来成像。"

ROOT = Path(__file__).resolve().parents[2]
TOOLS_OUTPUT_ROOT = ROOT / "tools" / "output"

DATA_DIR = ROOT / "data"


def append_run_log(entry):
    """追加一条运行记录到工具统一输出目录。"""
    log_path = TOOLS_OUTPUT_ROOT / "run_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry.setdefault("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


ALGORITHMS = {
    "das": "DAS:最基础、最快,适合入门观察",
    "mv": "MV:自适应波束合成,速度较慢",
    "ccmv": "CC-MV:全配对子阵交叉协方差的 MV",
    "esbmv": "ESBMV:MV 的特征空间版本,速度较慢",
    "gcfmv": "GCF-MV:带相干因子的 MV",
    "cmsaw": "CMSAW:基于 MV 相位的内存加权方法",
    "fdmas": "F-DMAS:非线性延迟乘加方法",
}


def print_title(text):
    print()
    emit(None, "info", text)


class BackCommandError(Exception):
    """Represent BackCommandError."""


class ExitCommandError(Exception):
    """Represent ExitCommandError."""


def handle_nav_command(value):
    """Execute handle nav command."""
    low = value.strip().lower()
    if low in ("b", "back", "上一步", "返回"):
        raise BackCommandError
    if low in ("q", "quit", "exit", "退出"):
        raise ExitCommandError


def read_line(prompt):
    """Read line."""
    try:
        return input(prompt)
    except EOFError:
        print()
        raise SystemExit("输入已结束,向导退出。") from None


def ask_text(prompt, default=None):
    """Execute ask text."""
    suffix = f"(默认:{default})" if default not in (None, "") else ""
    value = read_line(f"{prompt}{suffix}: ").strip()
    handle_nav_command(value)
    if value == "" and default is not None:
        return str(default)
    return value


def ask_yes_no(prompt, default=True):
    """Execute ask yes no."""
    default_text = "Y/n" if default else "y/N"
    while True:
        value = read_line(f"{prompt} [{default_text}]: ").strip().lower()
        handle_nav_command(value)
        if value == "":
            return default
        if value in ("y", "yes", "是", "1"):
            return True
        if value in ("n", "no", "否", "0"):
            return False
        print("请输入 y 或 n。")


def ask_choice(prompt, options, default_index=0):
    """Execute ask choice."""
    if not options:
        raise ValueError("没有可选项。")
    print(f"\n{prompt}")
    for idx, (_, desc) in enumerate(options, 1):
        default_mark = "  ← 默认" if idx - 1 == default_index else ""
        print(f"  {idx}. {desc}{default_mark}")
    while True:
        raw = read_line("请输入序号: ").strip()
        handle_nav_command(raw)
        if raw == "":
            return options[default_index][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        for key, _ in options:
            if isinstance(key, str) and raw.lower() == key.lower():
                return key
        print("序号不合法,请重新输入。")


def configured_choice_index(options, value, setting_name):
    for index, (key, _) in enumerate(options):
        if key == value:
            return index
    raise ValueError(f"{setting_name} 配置值不受支持: {value}")


def ask_multi_choice(prompt, options, default_keys):
    """Execute ask multi choice."""
    option_keys = {key for key, _ in options}
    if (
        not isinstance(default_keys, (list, tuple))
        or not default_keys
        or any(not isinstance(key, str) for key in default_keys)
        or len(default_keys) != len(set(default_keys))
        or any(key not in option_keys for key in default_keys)
    ):
        raise ValueError("config.algorithms 必须是非空且仅含支持算法的列表")
    print(f"\n{prompt}")
    for idx, (key, desc) in enumerate(options, 1):
        print(f"  {idx}. {key:<7} {desc}")
    default_text = ",".join(default_keys)
    while True:
        raw = read_line(
            f"请输入序号或名称,多个用逗号分隔(默认:{default_text}): ",
        ).strip()
        handle_nav_command(raw)
        if raw == "":
            return default_keys
        selected = []
        ok = True
        for item in raw.split(","):
            token = item.strip().lower()
            if not token:
                continue
            if token.isdigit() and 1 <= int(token) <= len(options):
                selected.append(options[int(token) - 1][0])
            elif token in dict(options):
                selected.append(token)
            else:
                ok = False
                print(f"无法识别:{item}")
                break
        if ok and selected:
            if len(selected) != len(set(selected)):
                print("算法列表包含重复项,请重新选择。")
                continue
            return selected
        print("请选择至少一个合法算法。")


def relative_to_root(path):
    """Execute relative to root."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path.resolve()


def list_h5_files():
    """Execute list h5 files."""
    paths = sorted(DATA_DIR.rglob("*.h5"))
    return [p for p in paths if p.is_file()]


def inspect_h5(path):
    """Execute inspect h5."""
    with h5py.File(path, "r") as hf:
        require_h5_schema(hf)
        if "config_yaml" not in hf:
            raise ValueError("当前 H5 必须包含 config_yaml")
        raw_config = hf["config_yaml"][()]
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8", errors="replace")
        config = yaml.safe_load(raw_config)
        if not isinstance(config, dict):
            raise ValueError("config_yaml 顶层必须是映射")
        generation = config.get("generation", {})
        storage = generation.get("storage", {}) if isinstance(generation, dict) else {}
        declared = storage.get("inputs", {}) if isinstance(storage, dict) else {}
        if not isinstance(declared, dict):
            raise ValueError("generation.storage.inputs 必须是映射")
        available = []
        for domain in ("iq", "rf", "analytic_rf"):
            if domain not in declared:
                continue
            if domain == "iq":
                required = ("iq_i", "iq_q", "iq_fs", "iq_angles", "iq_t0", "iq_length")
            elif domain == "rf":
                required = ("rf", "rf_fs", "rf_angles", "rf_t0", "rf_length")
            else:
                required = ("ar_i", "ar_q", "ar_fs", "ar_angles", "ar_t0", "ar_length")
            missing = sorted(set(required) - set(hf.keys()))
            if missing:
                raise ValueError(f"storage.inputs.{domain} 声明的字段缺失:{', '.join(missing)}")
            available.append(domain)
        stored_domains = list(available)
        if "rf" in available:
            if "analytic_rf" not in available:
                available.append("analytic_rf")
            if "iq" not in available and isinstance(generation.get("input_iq"), dict):
                available.append("iq")
        if not available:
            raise ValueError("H5 未保存可用的 rf、analytic_rf 或 iq 输入")
        default_domain = next((d for d in ("iq", "analytic_rf", "rf") if d in available), available[0])
        data_domain = next((d for d in ("iq", "analytic_rf", "rf") if d in stored_domains), stored_domains[0])
        if data_domain == "iq":
            data_name, imag_name = "iq_i", "iq_q"
            fs_name, angles_name, starts_name, length_name = "iq_fs", "iq_angles", "iq_t0", "iq_length"
        elif data_domain == "rf":
            data_name, imag_name = "rf", None
            fs_name, angles_name, starts_name, length_name = "rf_fs", "rf_angles", "rf_t0", "rf_length"
        else:
            data_name, imag_name = "ar_i", "ar_q"
            fs_name, angles_name, starts_name, length_name = "ar_fs", "ar_angles", "ar_t0", "ar_length"
        shape = hf[data_name].shape
        n = len(hf["acquisition_id"]) if "acquisition_id" in hf else shape[0]
        if len(shape) != 4 or shape[0] != n or min(shape[1], shape[3]) < 1 or shape[2] < 2:
            raise ValueError(f"{data_name} 必须是非空 [N,A,T,C] 数组")
        if imag_name and hf[data_name].shape != hf[imag_name].shape:
            raise ValueError(f"{data_name}/{imag_name} 形状必须一致")
        _, a, t, c = shape
        valid_time_dataset = hf[length_name]
        if valid_time_dataset.shape != (n,) or valid_time_dataset.dtype.kind not in "iu":
            raise ValueError(f"{length_name} 必须是长度为 N 的整数数组")
        valid_time = [int(value) for value in valid_time_dataset[:]]
        if any(value < 2 or value > t for value in valid_time):
            raise ValueError(f"{length_name} 必须为每个样本提供有效时间长度")
        if hf[starts_name].shape != (n, a) or hf[angles_name].shape != (a,):
            raise ValueError(f"{data_domain} 起始时间或角度形状与输入不匹配")
        if not np.isfinite(hf[angles_name][:]).all() or not np.isfinite(hf[starts_name][:]).all():
            raise ValueError(f"{data_domain} 角度或时间起点必须有限")
        required = {"c", "fc", "pitch", "num_channels", "z_grid", "x_grid"}
        missing = sorted(required - set(hf.keys()))
        if missing:
            raise ValueError(f"H5 缺少公共成像字段:{', '.join(missing)}")
        for name in ("z_grid", "x_grid"):
            values = np.asarray(hf[name][:], dtype=np.float32)
            if (
                values.ndim != 1
                or values.size < 2
                or not np.isfinite(values).all()
                or not np.all(np.diff(values) > 0)
            ):
                raise ValueError(f"{name} 必须是一维有限严格递增坐标")
        scalar_values = {}
        for name in ("c", "fc", fs_name, "pitch"):
            value = float(hf[name][()])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
            scalar_values[name] = value
        if int(hf["num_channels"][()]) != c:
            raise ValueError(f"num_channels 必须与 {data_domain} 通道维度一致")
        targets = generation.get("training_targets", {}) if isinstance(generation, dict) else {}
        ground_truth = generation.get("ground_truth", {}) if isinstance(generation, dict) else {}
        target_domains = targets.get("domain_targets", {}) if isinstance(targets, dict) else {}
        if not isinstance(ground_truth, dict) or not ground_truth.get("algorithm") or ground_truth.get("signal_domain") not in ("rf", "analytic_rf", "iq"):
            raise ValueError("config_yaml.generation.ground_truth 必须明确 algorithm 和 signal_domain")
        gt_by_domain = {}
        for domain in ("rf", "analytic_rf", "iq"):
            algorithm = ground_truth["algorithm"]
            entry = target_domains.get(domain, {}).get(algorithm, {}) if isinstance(target_domains, dict) else {}
            paths = [entry.get(name) for name in ("db", "envelope") if isinstance(entry, dict) and entry.get(name)]
            if not paths and domain != "rf" and isinstance(entry, dict):
                signal = entry.get("signal", {})
                if isinstance(signal, dict):
                    paths = [signal.get("real"), signal.get("imag")]
            gt_by_domain[domain] = bool(paths and all(path and path in hf for path in paths))
        main_domain = ground_truth["signal_domain"]
        gt = "有" if gt_by_domain.get(main_domain, False) else "无"
        angles = a
        fs = scalar_values[fs_name]
        fc = scalar_values["fc"]
        samples = config.get("source_samples", [])
        if isinstance(samples, list):
            if len(samples) != n:
                raise ValueError("config_yaml.source_samples 数量必须与 H5 样本数一致")
            if not all(isinstance(sample, dict) for sample in samples):
                raise ValueError("config_yaml.source_samples 必须是对象列表")
            names = [
                sample.get("id") or sample.get("acquisition_id") for sample in samples
            ]
            if any(name is None or not str(name).strip() for name in names):
                raise ValueError(
                    "config_yaml.source_samples 每个样本都必须包含 id 或 acquisition_id"
                )
            names = [str(name) for name in names]
            in_vivo = [
                sample.get("phantom_mode") == "in_vivo"
                or sample.get("phantom_source") == "in_vivo"
                for sample in samples
            ]
        elif (
            isinstance(samples, dict)
            and samples.get("encoding") in ("root_labels", "root_labels_with_path_template")
            and samples.get("count") == n
        ):
            id_dataset = str(samples["acquisition_id_dataset"])
            if id_dataset not in hf or hf[id_dataset].shape != (n,):
                raise ValueError("EPFL 紧凑样本元数据缺少有效 acquisition_id 标签")
            names = [
                str(value.decode("utf-8") if isinstance(value, bytes) else value)
                for value in hf[id_dataset][:]
            ]
            is_in_vivo = (
                samples.get("phantom_mode") == "in_vivo"
                or samples.get("phantom_source") == "in_vivo"
            )
            in_vivo = [is_in_vivo] * n
        else:
            raise ValueError("H5 必须声明每个样本的来源标识")
    return {
        "n": n,
        "a": a,
        "t": t,
        "c": c,
        "gt": gt,
        "gt_label": target_label(config, main_domain, ground_truth["algorithm"]) if gt == "有" else "无",
        "angles": angles,
        "fs": fs,
        "fc": fc,
        "names": names,
        "valid_time": valid_time,
        "in_vivo": in_vivo,
        "domains": available,
        "default_domain": default_domain,
        "gt_by_domain": gt_by_domain,
    }


def sample_can_be_evaluated(h5_path, sample_idx):
    context = load_picmus_context(h5_path, sample_idx)
    return context is not None and context["meta"]["has_gt"]


def choose_h5():
    """Execute choose h5."""
    paths = list_h5_files()
    if paths:
        options = []
        for path in paths:
            rel = relative_to_root(path)
            try:
                info = inspect_h5(path)
                desc = f"{rel}  | 样本 {info['n']},角度 {info['a']},T={info['t']},通道={info['c']},GT={info['gt']}"
            except (
                OSError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                yaml.YAMLError,
            ) as exc:
                desc = f"{rel}  | 无法读取:{exc}"
            options.append((str(path), desc))
        options.append(("manual", "手动输入 H5 路径"))
        choice = ask_choice("请选择要成像的 H5 文件", options, default_index=0)
        if choice != "manual":
            return Path(choice)
    while True:
        raw = ask_text("请输入 H5 路径")
        path = Path(raw.strip('"').strip("'"))
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return path
        print("文件不存在,请重新输入。")


def choose_sample(path, teaching=True):
    """Execute choose sample."""
    info = inspect_h5(path)
    if teaching:
        print(f"\n说明:{SAMPLE_HELP}")
    print("\n这个 H5 的基本信息:")
    print(f"  样本数: {info['n']}")
    print(f"  输入形状: [N={info['n']}, A={info['a']}, T={info['t']}, C={info['c']}]")
    print(f"  可选信号域: {', '.join(info['domains'])}")
    if teaching:
        print("  形状说明: N=样本数,A=角度数,T=时间采样点,C=阵元/通道数")
    print(f"  H5 角度数: {info['angles']}")
    if info["fs"] is not None:
        print(f"  fs: {info['fs'] / 1e6:.3f} MHz")
    if info["fc"] is not None:
        print(f"  fc: {info['fc'] / 1e6:.3f} MHz")
    print(f"  GT: {info['gt_label']}")

    if info["n"] <= 20:
        options = []
        for idx, name in enumerate(info["names"]):
            options.append((idx, f"{idx}: {name}"))
        return int(ask_choice("请选择样本编号", options, default_index=0))

    preview_count = 5
    print("\n样本较多，以下仅显示首尾预览:")
    for idx, name in list(enumerate(info["names"][:preview_count])) + list(
        enumerate(info["names"][-preview_count:], info["n"] - preview_count),
    ):
        print(f"  {idx}: {name}")
    print("输入 0 起始的样本编号，或输入样本名/名称片段（直接回车选 0）。")
    while True:
        raw = ask_text("请选择样本", default="0")
        if raw.isdigit() and 0 <= int(raw) < info["n"]:
            return int(raw)
        matched = [
            (idx, name)
            for idx, name in enumerate(info["names"])
            if raw.lower() in name.lower()
        ]
        if len(matched) == 1:
            idx, name = matched[0]
            print(f"已匹配: {idx}: {name}")
            return idx
        if matched:
            print("匹配到多个样本，请输入更精确的名称或编号:")
            for idx, name in matched[:20]:
                print(f"  {idx}: {name}")
            if len(matched) > 20:
                print(f"  …其余 {len(matched) - 20} 个结果未显示")
        else:
            print(f"未找到样本 {raw!r}，请重新输入。")


def load_base_config():
    """Load base config."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("config.yaml 顶层必须是映射")
    return config


def run_steps(steps):
    """Execute run steps."""

    def execute_step(step):
        """Execute execute step."""
        try:
            step()
            return True
        except BackCommandError:
            return False

    index = 0
    while index < len(steps):
        if execute_step(steps[index]):
            index += 1
        elif index == 0:
            print("已经是第一步,输入 q 可退出。")
        else:
            index -= 1
            print("\n已返回上一步。")
