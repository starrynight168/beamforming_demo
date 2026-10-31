"""Provide Python utilities for run_wizard_cn."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate import load_picmus_context


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="超声波束合成中文向导")
    return parser.parse_args()


def append_run_log(entry):
    """追加一条运行记录到 results/run_log.jsonl。."""
    log_path = ROOT / "results" / "run_log.jsonl"
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

ALGORITHM_PARAM_HELP = {
    "aperture_mode": "接收孔径模式",
    "mv_dl": "MV 对角加载系数(非负)",
    "fbss": "是否启用前后向空间平滑",
    "subarray_ratio": "子阵长度比例,范围 (0, 1]",
    "tx_subarray_ratio": "多角度 Tx 子阵比例;回车沿用 subarray_ratio",
    "temporal_win": "时间平滑窗口,必须为正奇数",
    "num_eig": "ESBMV 保留特征数,0 表示按阈值自动选择",
    "eig_threshold": "ESBMV 特征阈值,范围 [0, 1]",
    "gcf_low_bins": "GCF 低频 bin 数,必须为非负整数",
    "gcf_tx_low_bins": "多角度 Tx GCF 低频 bin 数;回车沿用 gcf_low_bins",
    "gcf_power": "GCF 权重幂次(非负)",
    "lmax_ratio": "CMSAW 最大子阵比例,范围 (0, 0.5]",
    "min_subarray_len": "CMSAW 最小子阵长度,至少为 2",
    "delta_max": "CMSAW 权重变化上限,范围 [0, 1]",
    "gamma": "CMSAW 权重曲线强度,范围 (0, 1]",
    "clip_percentile": "CMSAW 裁剪百分位,范围 (0, 100]",
    "depth_smooth_rows": "CMSAW 深度平滑行数,必须为正奇数",
}

ALGORITHM_PARAM_TYPES = {
    "aperture_mode": str,
    "mv_dl": float,
    "fbss": bool,
    "subarray_ratio": float,
    "tx_subarray_ratio": float,
    "temporal_win": int,
    "num_eig": int,
    "eig_threshold": float,
    "gcf_low_bins": int,
    "gcf_tx_low_bins": int,
    "gcf_power": float,
    "lmax_ratio": float,
    "min_subarray_len": int,
    "delta_max": float,
    "gamma": float,
    "clip_percentile": float,
    "depth_smooth_rows": int,
}

HELP_TEXT = {
    "h5": "H5 是已经整理好的成像数据包,里面通常包含输入 IQ、角度、探头参数、成像网格,有些还包含 GT 参考图。",
    "sample": "一个 H5 里可以有多个样本。样本编号就是选择第几帧/第几个场景来成像。",
    "algorithm": "成像方法决定如何把通道数据合成为图像。刚入门建议先选 DAS,速度快、结果最容易理解。",
    "angles": "角度表示使用哪些平面波发射角。center 只用中心角,all 用全部角度。角度越多通常图像越稳定,但速度越慢。",
    "dynamic_aperture": "动态孔径会随深度改变接收孔径。开启后浅层更稳、深层分辨率更合理;关闭后实现更简单,但浅层可能更容易出现旁瓣/伪影。",
    "f_number": "F-Number 只在动态孔径开启时生效。数值越小,孔径越大,横向分辨率可能更好但旁瓣/噪声也可能更重;数值越大,图像更平滑但可能变糊。常用 1.5。",
    "dr": "动态范围只影响显示灰度,不改变算法原始输出。60 dB 常用;数值小会让图像对比更强但暗部细节少,数值大会保留暗部但图像可能发灰。",
    "tgc": "TGC 是深度增益补偿。开启后深部不会太暗,更接近常见 B-mode 显示;关闭后可以观察原始衰减,但深部通常偏暗。",
    "tgc_alpha": "TGC_ALPHA 只在 TGC 开启时生效。数值越大,深部越亮;太大会让深部噪声和伪影也被放大。默认 0.5。",
    "window": "窗函数影响旁瓣、斑点和分辨率。rect 分辨率较锐但旁瓣更明显;hann 更干净平滑但会牺牲一点横向分辨率;tukey 介于两者之间。",
    "interp": "插值影响延迟取样精度。nearest 最快但最粗糙;linear 更快;cubic/catmull_rom 更平滑;quintic、farrow、sinc 精度更高但通常更慢。",
    "gt": "GT 是参考图像。加入 GT 后,对比图会把算法结果和参考图放在一起,便于肉眼比较。",
    "evaluate": "指标会计算算法结果和 GT 的差异,并生成 CSV/图片。没有 GT 时不建议计算。",
    "keep_existing": "复用已有输出可以节省时间,但如果你改了参数,应关闭复用重新计算。",
    "execute": "如果暂不执行,向导只会显示最终配置和命令,不会写入临时配置。",
    "output": "结果目录会保存各算法输出图、comparison.png、运行参数和可选指标。",
}


def explain(enabled, key):
    """Execute explain."""
    if enabled:
        print(f"\n说明:{HELP_TEXT[key]}")


def print_title(text):
    """Execute print title."""
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


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


def ask_float(prompt, default, minimum=None, strictly_greater=False):
    """Execute ask float."""
    while True:
        raw = ask_text(prompt, default=default)
        try:
            value = float(raw)
        except ValueError:
            print("请输入合法数字。")
            continue
        if not math.isfinite(value):
            print("请输入有限数字。")
            continue
        if minimum is not None:
            valid = value > minimum if strictly_greater else value >= minimum
            if not valid:
                relation = "大于" if strictly_greater else "大于等于"
                print(f"数值必须{relation} {minimum}。")
                continue
        return value


def validate_algorithm_param(name, value):
    """Validate one method-specific parameter and return its normalized value."""
    if name in {"num_eig", "gcf_low_bins", "gcf_tx_low_bins"} and value < 0:
        raise ValueError("必须是非负整数。")
    if name == "min_subarray_len" and value < 2:
        raise ValueError("必须是大于等于 2 的整数。")
    if name == "depth_smooth_rows" and (value < 1 or value % 2 == 0):
        raise ValueError("必须是正奇数。")
    if name == "temporal_win" and (value < 1 or value % 2 != 1):
        raise ValueError("必须是正奇数。")
    if name in {"mv_dl", "gcf_power"} and value < 0:
        raise ValueError("必须大于等于 0。")
    if name in {"subarray_ratio", "tx_subarray_ratio"} and not 0 < value <= 1:
        raise ValueError("必须位于 (0, 1]。")
    if name in {"eig_threshold", "delta_max"} and not 0 <= value <= 1:
        raise ValueError("必须位于 [0, 1]。")
    if name == "gamma" and not 0 < value <= 1:
        raise ValueError("必须位于 (0, 1]。")
    if name == "lmax_ratio" and not 0 < value <= 0.5:
        raise ValueError("必须位于 (0, 0.5]。")
    if name == "clip_percentile" and not 0 < value <= 100:
        raise ValueError("必须位于 (0, 100]。")
    return value


def ask_algorithm_param(algorithm, name, default):
    """Ask for one method-specific parameter using its algorithm CLI type."""
    label = ALGORITHM_PARAM_HELP.get(name, name)
    prompt = f"[{algorithm}] {name} - {label}"
    if name == "mv_axis":
        options = [("rx", "rx:接收端 MV"), ("tx", "tx:接收DAS后发射端 MV"), ("joint", "joint:收发联合 MV")]
        return ask_choice(prompt, options, default_index=configured_choice_index(options, default, f"{algorithm}.{name}"))
    if name == "aperture_mode":
        options = [
            ("discrete", "discrete:离散通道孔径,适合公平对照"),
            ("geometry", "geometry:连续几何孔径,适合 DAS 单独研究"),
        ]
        default_index = configured_choice_index(options, default, f"{algorithm}.{name}")
        return ask_choice(prompt, options, default_index=default_index)
    value_type = ALGORITHM_PARAM_TYPES.get(name)
    if value_type is None:
        raise ValueError(f"不支持的算法参数: {algorithm}.{name}")
    if value_type is bool:
        if not isinstance(default, bool):
            raise ValueError(f"{algorithm}.{name} 配置必须是布尔值")
        return ask_yes_no(prompt, default=default)

    while True:
        if default is None and name in {"tx_subarray_ratio", "gcf_tx_low_bins"}:
            raw = ask_text(f"{prompt}（回车沿用关联参数）", default="")
            if not raw:
                return None
        else:
            raw = ask_text(prompt, default=default)
        try:
            if value_type is int:
                value = int(raw)
                if str(value) != raw.strip() and raw.strip() not in {f"+{value}"}:
                    raise ValueError
            else:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError
            return validate_algorithm_param(name, value)
        except (TypeError, ValueError) as exc:
            detail = str(exc) or (
                "请输入整数。" if value_type is int else "请输入合法数字。"
            )
            print(detail)


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


def validate_select_angles(value):
    """Validate select angles."""
    text = str(value).strip().lower()
    if text in ("center", "all"):
        return text
    raw_parts = [part.strip() for part in text.split(",")]
    if len(raw_parts) == 2 and raw_parts[0].isdigit() and raw_parts[1] == "":
        return f"{raw_parts[0]},"
    parts = raw_parts
    if len(parts) == 1 and parts[0].isdigit() and int(parts[0]) > 0:
        return parts[0]
    if len(parts) > 1 and all(part.isdigit() for part in parts):
        return ",".join(parts)
    raise ValueError(
        "角度只能填 center、all、正整数角度数,或 0 基整数索引列表,例如 1、3、11、0,37,74；单个索引以逗号结尾,例如 0,。",
    )


def ask_select_angles(default="1"):
    """Execute ask select angles."""
    while True:
        raw = ask_text(
            "请输入角度数量 N 或整数索引列表,例如 1、3、11、0,37,74；单个索引以逗号结尾,例如 0,",
            default=default,
        )
        try:
            return validate_select_angles(raw)
        except ValueError as exc:
            print(exc)


def list_h5_files():
    """Execute list h5 files."""
    paths = sorted(DATA_DIR.rglob("*.h5"))
    return [p for p in paths if p.is_file()]


def inspect_h5(path):
    """Execute inspect h5."""
    with h5py.File(path, "r") as hf:
        if "config_yaml" not in hf or "acquisition_id" not in hf:
            raise ValueError("当前 H5 必须包含 config_yaml 和 acquisition_id")
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
        definitions = {
            "iq": ("all_multi_I", "all_multi_Q", "fs", "angles", "time_start_vector", "valid_time_samples"),
            "analytic_rf": ("ar_i", "ar_q", "ar_fs", "ar_angles", "ar_t0", "ar_len"),
            "rf": ("rf", None, "rf_fs", "rf_angles", "rf_t0", "rf_length"),
        }
        available = []
        for domain, fields in definitions.items():
            if domain not in declared:
                continue
            required = [field for field in fields if field is not None]
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
        data_name, imag_name, fs_name, angles_name, starts_name, length_name = definitions[data_domain]
        n = len(hf["acquisition_id"])
        shape = hf[data_name].shape
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
        required = {"body_region", "c", "fc", "pitch", "num_channels", "z_grid", "x_grid"}
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
        gt_by_domain = {}
        for domain in ("rf", "analytic_rf", "iq"):
            algorithm = ground_truth.get("algorithm") if isinstance(ground_truth, dict) else None
            entry = target_domains.get(domain, {}).get(algorithm, {}) if isinstance(target_domains, dict) else {}
            paths = [entry.get(name) for name in ("db", "envelope") if isinstance(entry, dict) and entry.get(name)]
            if not paths and domain != "rf" and isinstance(entry, dict):
                signal = entry.get("signal", {})
                if isinstance(signal, dict):
                    paths = [signal.get("real"), signal.get("imag")]
            gt_by_domain[domain] = bool(paths and all(path and path in hf for path in paths))
        main_domain = ground_truth.get("signal_domain", default_domain) if isinstance(ground_truth, dict) else default_domain
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
            and samples.get("encoding") == "root_labels_with_path_template"
            and samples.get("count") == n
        ):
            id_dataset = str(samples.get("acquisition_id_dataset", "/acquisition_id"))
            if id_dataset not in hf or hf[id_dataset].shape != (n,):
                raise ValueError("EPFL 紧凑样本元数据缺少有效 acquisition_id 标签")
            names = [
                str(value.decode("utf-8") if isinstance(value, bytes) else value)
                for value in hf[id_dataset][:]
            ]
            dataset_id = str((config.get("dataset") or {}).get("id", "")).lower()
            is_in_vivo = "invivo" in dataset_id or "volunteer" in dataset_id
            in_vivo = [is_in_vivo] * n
        else:
            raise ValueError("config_yaml.source_samples 数量必须与 H5 样本数一致")
    return {
        "n": n,
        "a": a,
        "t": t,
        "c": c,
        "gt": gt,
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
    explain(teaching, "sample")
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
    print(f"  GT: {info['gt']}")

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


def build_config(base_config, h5_path, sample_idx, algorithms, params):
    """Build config."""
    h5_rel = str(relative_to_root(h5_path)).replace("\\", "/")
    scene_id = f"{Path(h5_path).stem}_sample{sample_idx}"
    scene = {
        "id": scene_id,
        "h5_path": h5_rel,
        "sample_idx": int(sample_idx),
    }
    config = {
        "algorithms": algorithms,
        "algorithm_labels": base_config.get("algorithm_labels", {}),
        "params": params,
        "algorithm_params": deepcopy(base_config.get("algorithm_params", {})),
        "scenes": [scene],
    }
    return config, scene_id


def reusable_scene_id(config):
    """Return a stable scene ID for an identical reusable wizard configuration."""
    scene = config["scenes"][0]
    algorithms = config["algorithms"]
    identity = {
        "h5_path": scene["h5_path"],
        "sample_idx": scene["sample_idx"],
        "algorithms": algorithms,
        "params": config["params"],
        "algorithm_params": {
            algorithm: (config.get("algorithm_params", {}) or {}).get(algorithm, {})
            for algorithm in algorithms
        },
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"wizard_{hashlib.sha256(encoded).hexdigest()[:16]}"


def load_base_config():
    """Load base config."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("config.yaml 顶层必须是映射")
    return config


def make_temp_config_path(output_root, scene_id):
    """Execute make temp config path."""
    return output_root / scene_id / "config.yaml"


def write_config(config, path):
    """Execute write config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return path


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


def main():
    """Run the command-line workflow."""
    parse_args()
    print_title("超声波束合成中文向导")
    print(
        "这个向导会一步一步问你问题,然后调用现有 run_one.py 完成成像、对比和可选指标计算。",
    )
    print("说明:本向导只新增入口,不会修改原来的 config.yaml、算法脚本或数据文件。")
    print("提示:每一步都可以输入 b 返回上一步,输入 q 退出。")

    state = {"base_config": load_base_config()}

    def step_teaching():
        """Execute step teaching."""
        state["teaching"] = ask_yes_no(
            "是否开启小白说明模式(每一步解释参数含义)",
            default=state.get("teaching", True),
        )

    def step_h5():
        """Execute step h5."""
        explain(state["teaching"], "h5")
        state["h5_path"] = choose_h5()

    def step_sample():
        """Execute step sample."""
        state["sample_idx"] = choose_sample(
            state["h5_path"],
            teaching=state["teaching"],
        )

    def step_signal_domain():
        info = inspect_h5(state["h5_path"])
        labels = {"iq": "iq: 基带 IQ", "rf": "rf: 实数 RF", "analytic_rf": "analytic_rf: 解析 RF"}
        options = [(domain, labels[domain]) for domain in info["domains"]]
        default = state.get(
            "signal_domain",
            (state["base_config"].get("params", {}) or {}).get("signal_domain", info["default_domain"]),
        )
        if default not in info["domains"]:
            default = info["default_domain"]
        state["signal_domain"] = ask_choice(
            "请选择算法输入信号域",
            options,
            default_index=next(i for i, (domain, _) in enumerate(options) if domain == default),
        )

    def step_algorithms():
        """Execute step algorithms."""
        algorithm_options = list(ALGORITHMS.items())
        explain(state["teaching"], "algorithm")
        state["algorithms"] = ask_multi_choice(
            "请选择成像方法",
            algorithm_options,
            default_keys=state.get(
                "algorithms",
                state["base_config"].get("algorithms", ["das"]),
            ),
        )

    def step_algorithm_params():
        """Ask for every parameter owned by each selected algorithm."""
        base_params = state["base_config"].get("algorithm_params", {}) or {}
        previous_params = state.get("algorithm_params", {})
        selected_params = {}
        for algorithm in state["algorithms"]:
            defaults = deepcopy(base_params.get(algorithm, {}) or {})
            editable = list(defaults)
            if not editable:
                print(f"\n[{algorithm}] 没有独有的可调参数。")
                selected_params[algorithm] = defaults
                continue
            print(f"\n请设置 {algorithm} 的方法专属参数:")
            current = previous_params.get(algorithm, {})
            for name in editable:
                default = current.get(name, defaults[name])
                defaults[name] = ask_algorithm_param(algorithm, name, default)
            selected_params[algorithm] = defaults
        state["algorithm_params"] = selected_params

    def step_angles():
        """Execute step angles."""
        explain(state["teaching"], "angles")
        default = str(
            state.get(
                "select_angles",
                (state["base_config"].get("params", {}) or {}).get(
                    "select_angles",
                    "center",
                ),
            ),
        )
        default_index = 0 if default == "center" else 1 if default == "all" else 2
        value = ask_choice(
            "请选择使用哪些角度",
            [
                ("center", "center:只用中心角,最适合入门和单角度数据"),
                ("all", "all:使用 H5 中保存的全部角度"),
                ("custom", "输入一个数字 N:使用 N 个角度"),
            ],
            default_index=default_index,
        )
        if value == "custom":
            value = ask_select_angles(default=default)
        state["select_angles"] = value

    def step_dr():
        """Execute step dr."""
        explain(state["teaching"], "dr")
        state["dr"] = ask_float(
            "请输入显示动态范围 dB",
            default=state.get(
                "dr",
                (state["base_config"].get("params", {}) or {}).get("dr", "60"),
            ),
            minimum=0,
            strictly_greater=True,
        )

    def step_aperture():
        """Execute step aperture."""
        explain(state["teaching"], "dynamic_aperture")
        state["dynamic_aperture"] = ask_yes_no(
            "是否启用动态孔径",
            default=state.get(
                "dynamic_aperture",
                (state["base_config"].get("params", {}) or {}).get(
                    "dynamic_aperture",
                    True,
                ),
            ),
        )
        if state["dynamic_aperture"]:
            explain(state["teaching"], "f_number")
            state["f_number"] = ask_float(
                "请输入 F-Number",
                default=state.get(
                    "f_number",
                    (state["base_config"].get("params", {}) or {}).get(
                        "f_number",
                        "1.5",
                    ),
                ),
                minimum=0,
                strictly_greater=True,
            )
        else:
            state["f_number"] = (state["base_config"].get("params", {}) or {}).get(
                "f_number",
                1.5,
            )

    def step_tgc():
        """Execute step tgc."""
        explain(state["teaching"], "tgc")
        state["tgc"] = ask_yes_no(
            "是否启用 TGC 深度补偿",
            default=state.get(
                "tgc",
                (state["base_config"].get("params", {}) or {}).get("tgc", True),
            ),
        )
        if state["tgc"]:
            explain(state["teaching"], "tgc_alpha")
            state["tgc_alpha"] = ask_float(
                "请输入 TGC_ALPHA",
                default=state.get(
                    "tgc_alpha",
                    (state["base_config"].get("params", {}) or {}).get(
                        "tgc_alpha",
                        "0.5",
                    ),
                ),
                minimum=0,
            )
        else:
            state["tgc_alpha"] = (state["base_config"].get("params", {}) or {}).get(
                "tgc_alpha",
                0.5,
            )

    def step_window():
        """Execute step window."""
        explain(state["teaching"], "window")
        options = [
            ("rect", "rect:矩形窗,默认"),
            ("tukey", "tukey:折中"),
            ("hann", "hann:更平滑"),
            ("hamming", "hamming:平滑"),
            ("blackman", "blackman:旁瓣抑制更强"),
            ("kaiser", "kaiser:可调折中窗"),
        ]
        default = state.get(
            "window",
            (state["base_config"].get("params", {}) or {}).get("window", "rect"),
        )
        state["window"] = ask_choice(
            "请选择孔径窗函数",
            options,
            configured_choice_index(options, default, "params.window"),
        )

    def step_interp():
        """Execute step interp."""
        explain(state["teaching"], "interp")
        options = [
            ("cubic", "cubic:默认,较平滑"),
            ("catmull_rom", "catmull_rom:埃尔米特样条,平滑且无过冲"),
            ("linear", "linear:较快"),
            ("nearest", "nearest:最快但较粗糙"),
            ("quintic", "quintic:高阶插值"),
            ("farrow", "farrow:更宽核高阶插值"),
            ("sinc", "sinc:窗化 sinc 插值"),
        ]
        default = state.get(
            "interp",
            (state["base_config"].get("params", {}) or {}).get("interp", "cubic"),
        )
        state["interp"] = ask_choice(
            "请选择插值方式",
            options,
            configured_choice_index(options, default, "params.interp"),
        )

    def step_gt_evaluate():
        """Execute step gt evaluate."""
        info = inspect_h5(state["h5_path"])
        has_gt_default = info["gt_by_domain"].get(state["signal_domain"], False)
        state["in_vivo"] = info["in_vivo"][state["sample_idx"]]
        sample_name = info["names"][state["sample_idx"]]
        explain(state["teaching"], "gt")
        state["has_gt"] = has_gt_default
        state["evaluate"] = sample_can_be_evaluated(
            state["h5_path"], state["sample_idx"]
        )
        if state["in_vivo"]:
            print("活体样本:保留重建与对比图,自动跳过评估指标。")
        elif has_gt_default and not state["evaluate"]:
            print(f"样本 {sample_name} 不属于 PICMUS 模体:保留 GT 对比,跳过指标。")
        else:
            print(
                f"GT: {'H5 中存在,将自动加入对比并计算指标' if has_gt_default else 'H5 中不存在,将跳过对比和指标'}",
            )

    def step_keep_existing():
        """Execute step keep existing."""
        explain(state["teaching"], "keep_existing")
        state["keep_existing"] = ask_yes_no(
            "如果结果已存在,是否复用已有算法输出",
            default=state.get("keep_existing", False),
        )

    def step_output():
        """Execute step output."""
        explain(state["teaching"], "output")
        default_output = state.get("output_root")
        if default_output is None:
            h5_tag = Path(state["h5_path"]).stem
            default_output = Path("results") / f"{h5_tag}_sample{state['sample_idx']}"
        output_root = Path(
            ask_text("请输入结果输出根目录(场景子目录)", default=str(default_output)),
        )
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        state["output_root"] = output_root

    def step_confirm():
        """Execute step confirm."""
        params = {
            "signal_domain": state["signal_domain"],
            "select_angles": state["select_angles"],
            "f_number": state["f_number"],
            "dr": state["dr"],
            "dynamic_aperture": state["dynamic_aperture"],
            "tgc": state["tgc"],
            "tgc_alpha": state["tgc_alpha"],
            "window": state["window"],
            "interp": state["interp"],
        }
        config, _ = build_config(
            state["base_config"],
            state["h5_path"],
            state["sample_idx"],
            state["algorithms"],
            params,
        )
        for algorithm, method_params in state["algorithm_params"].items():
            config.setdefault("algorithm_params", {})[algorithm] = deepcopy(
                method_params
            )
        if state["keep_existing"]:
            run_id = reusable_scene_id(config)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_id = f"wizard_{timestamp}"
        config["scenes"][0]["id"] = run_id
        config_path = make_temp_config_path(state["output_root"], run_id)
        cmd = [
            sys.executable,
            str(ROOT / "run_one.py"),
            "--config",
            str(config_path),
            "--scene",
            run_id,
            "--algorithms",
            ",".join(state["algorithms"]),
            "--output_root",
            str(state["output_root"]),
        ]
        if not state["evaluate"]:
            cmd.append("--no_evaluate")
        if state["keep_existing"]:
            cmd.append("--keep_existing")
        state["scene_id"] = run_id
        state["config"] = config
        state["config_path"] = config_path
        state["cmd"] = cmd

        print_title("即将执行的配置")
        print(f"H5 文件: {relative_to_root(state['h5_path'])}")
        print(f"样本编号: {state['sample_idx']}")
        print(f"算法: {', '.join(state['algorithms'])}")
        print(f"输入信号域: {state['signal_domain']}")
        print("方法专属参数:")
        for algorithm in state["algorithms"]:
            method_params = state["algorithm_params"].get(algorithm, {})
            print(f"  {algorithm}: {method_params or '无'}")
        print(f"角度选择: {state['select_angles']}")
        print(f"F-Number: {state['f_number']}")
        print(f"动态范围: {state['dr']} dB")
        print(f"动态孔径: {'开' if state['dynamic_aperture'] else '关'}")
        print(f"TGC: {'开' if state['tgc'] else '关'}")
        print(f"窗函数: {state['window']}")
        print(f"插值: {state['interp']}")
        print(f"计算指标: {'是' if state['evaluate'] else '否'}")
        print(f"临时配置: {relative_to_root(config_path)}(确认执行后才写入)")
        print("\n命令:")
        print(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))

        explain(state["teaching"], "execute")
        state["execute"] = ask_yes_no("确认按以上配置执行吗", default=True)

    try:
        run_steps(
            [
                step_teaching,
                step_h5,
                step_sample,
                step_signal_domain,
                step_algorithms,
                step_algorithm_params,
                step_angles,
                step_dr,
                step_aperture,
                step_tgc,
                step_window,
                step_interp,
                step_gt_evaluate,
                step_keep_existing,
                step_output,
                step_confirm,
            ],
        )
    except ExitCommandError:
        print("\n已退出。")
        return

    if not state["execute"]:
        print("\n已按你的选择只展示配置和命令,没有写入临时配置,也没有执行。")
        return

    print_title("开始执行")
    write_config(state["config"], state["config_path"])
    result = subprocess.run(state["cmd"], cwd=ROOT, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"流程执行失败,退出码:{result.returncode}")

    scene_dir = state["output_root"] / state["scene_id"]
    print_title("完成")
    print(f"结果目录: {scene_dir}")
    print(f"对比图: {scene_dir / 'comparison.png'}")
    print(f"运行参数: {scene_dir / 'run_params.json'}")
    if state["evaluate"]:
        print(f"指标目录: {scene_dir / 'metrics'}")

    h5_rel = str(relative_to_root(state["h5_path"])).replace("\\", "/")
    append_run_log(
        {
            "type": "wizard",
            "dir": str(
                relative_to_root(state["output_root"] / state["scene_id"]),
            ).replace("\\", "/"),
            "h5_path": h5_rel,
            "sample_idx": int(state["sample_idx"]),
            "algorithms": state["algorithms"],
            "algorithm_params": {
                algorithm: state["algorithm_params"].get(algorithm, {})
                for algorithm in state["algorithms"]
            },
            "params": {
                "signal_domain": state["signal_domain"],
                "select_angles": state["select_angles"],
                "f_number": state["f_number"],
                "dr": state["dr"],
                "dynamic_aperture": state["dynamic_aperture"],
                "tgc": state["tgc"],
                "tgc_alpha": state["tgc_alpha"],
                "window": state["window"],
                "interp": state["interp"],
            },
            "has_gt": state["has_gt"],
            "evaluate": state["evaluate"],
        },
    )


if __name__ == "__main__":
    main()
