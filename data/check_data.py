"""打包 H5 文件简洁体检脚本。."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

COMPARISON_VALUE_1024 = 1024
COMPARISON_VALUE_2 = 2
COMPARISON_VALUE_4096 = 4096

# 脚本位于 data/;默认检查同目录的打包 H5,和启动时的工作目录无关。
DATA_DIR = Path(__file__).resolve().parent
ROOT = DATA_DIR.parent
LOG_DIR = DATA_DIR / "logs"

REQUIRED_FIELDS = [
    "all_multi_I",
    "all_multi_Q",
    "time_start_vector",
    "fs",
    "c",
    "fc",
    "pitch",
    "num_channels",
    "z_grid",
    "x_grid",
    "angles",
    "all_scale_ref",
    "all_norm_ref",
    "valid_time_samples",
    "config_yaml",
]

KEY_FIELDS = [
    "all_multi_I",
    "all_multi_Q",
    "all_envdb_norm",
    "all_scale_ref",
    "all_norm_ref",
    "fs",
    "c",
    "fc",
    "pitch",
    "num_channels",
    "z_grid",
    "x_grid",
    "angles",
    "time_start_vector",
    "valid_time_samples",
    "config_yaml",
]

MOVED_TO_CONFIG_FIELDS = {
    "sample_names",
    "phantom_mode",
    "phantom_source",
    "iq_path",
    "scan_path",
    "phantom_path",
    "gt_path",
}


class Tee:
    """Represent Tee."""

    def __init__(self, *streams):
        """Initialize the instance."""
        self.streams = streams

    def write(self, text: str) -> None:
        """Execute write."""
        for stream in self.streams:
            stream.write(text)

    def flush(self) -> None:
        """Execute flush."""
        for stream in self.streams:
            stream.flush()


def default_log_path() -> Path:
    """Execute default log path."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"check_data_{stamp}.txt"


def decode_value(value):
    """Execute decode value."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return item
    return value


def display_width(value) -> int:
    """Execute display width."""
    width = 0
    for char in str(value):
        width += 2 if unicodedata.east_asian_width(char) in {"F", "width"} else 1
    return width


def shorten_text(text, limit: int = 110) -> str:
    """Execute shorten text."""
    text = str(text).replace("\n", " | ")
    return text if display_width(text) <= limit else text[:limit] + " ..."


def format_value(value, max_items: int = 1) -> str:
    """Execute format value."""
    decoded = decode_value(value)
    if isinstance(decoded, np.ndarray):
        if decoded.dtype.kind in {"S", "O", "U"}:
            flat = [decode_value(item) for item in decoded.reshape(-1)]
            return " | ".join(shorten_text(item) for item in flat[:max_items])
        if decoded.ndim == 0:
            return shorten_text(decode_value(decoded[()]))
        return np.array2string(decoded, threshold=decoded.size, max_line_width=120)
    return shorten_text(decoded)


def dataset_preview(ds: h5py.Dataset) -> str:
    """Execute dataset preview."""
    if ds.shape == ():
        return format_value(ds[()])
    if ds.size == 0:
        return "空"
    if ds.dtype.kind in {"S", "U", "O"}:
        return format_value(ds[:])

    if ds.size <= COMPARISON_VALUE_4096:
        arr = np.asarray(ds[:], dtype=np.float64)
    else:
        index = (0, *tuple(slice(None) for _ in range(ds.ndim - 1)))
        arr = np.asarray(ds[index], dtype=np.float64)
    if arr.size == 0:
        return "空"
    if not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return "非有限值"
        return f"[{finite.min():.4g}, {finite.max():.4g}] + 非有限值"
    return f"[{arr.min():.4g}, {arr.max():.4g}]"


def dataset_storage_bytes(ds: h5py.Dataset) -> int:
    """Execute dataset storage bytes."""
    try:
        return int(ds.id.get_storage_size())
    except Exception:
        return 0


def format_bytes(num_bytes: int) -> str:
    """Execute format bytes."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < COMPARISON_VALUE_1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.3f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def format_shape(shape) -> str:
    """Execute format shape."""
    return "标量" if shape == () else "x".join(str(dim) for dim in shape)


def pad_display(value, width: int) -> str:
    """Execute pad display."""
    text = str(value)
    return text + " " * max(width - display_width(text), 0)


def print_table(rows, headers) -> None:
    """Execute print table."""
    widths = [display_width(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], display_width(cell))
    print(
        "  " + "  ".join(pad_display(cell, widths[idx]) for idx, cell in enumerate(headers)),
    )
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  " + "  ".join(pad_display(cell, widths[idx]) for idx, cell in enumerate(row)),
        )


def scalar_float(hf: h5py.File, key: str) -> float | None:
    """Execute scalar float."""
    if key not in hf:
        return None
    try:
        return float(decode_value(hf[key][()]))
    except Exception:
        return None


def decode_compact_sequence(value) -> np.ndarray:
    """Decode list or arithmetic-sequence config into an array."""
    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, dict) or value.get("encoding") != "arithmetic_sequence":
        raise ValueError("必须是数组或 arithmetic_sequence")
    count = int(value["count"])
    return np.asarray(value["start"]) + np.arange(count) * np.asarray(value["step"])


def validate_config_schema(config: dict, hf: h5py.File, problems: list[str]) -> None:
    """Validate config schema."""
    if config.get("schema_version") != 1:
        problems.append(
            f"config_yaml.schema_version 应为 1,实际为 {config.get('schema_version')}",
        )

    dataset = config.get("dataset", {}) or {}
    if dataset.get("purpose") != "algorithm_evaluation":
        problems.append("config_yaml.dataset.purpose 应为 algorithm_evaluation")

    schema = config.get("h5_schema", {}) or {}
    if schema.get("iq_layout") != ["sample", "angle", "time", "channel"]:
        problems.append("config_yaml.h5_schema.iq_layout 与实际 IQ 布局不一致")
    if schema.get("ground_truth_layout") != ["sample", "image_channel", "z", "x"]:
        problems.append(
            "config_yaml.h5_schema.ground_truth_layout 与实际 GT 布局不一致",
        )
    padding = schema.get("padding", {}) or {}
    if padding.get("value") != 0.0:
        problems.append("config_yaml.h5_schema.padding.value 应为 0")
    if padding.get("time_valid_length_dataset") != "/valid_time_samples":
        problems.append(
            "config_yaml.h5_schema.padding.time_valid_length_dataset 应指向 /valid_time_samples",
        )
    if schema.get("sample_metadata") != "config_yaml.source_samples":
        problems.append(
            "config_yaml.h5_schema.sample_metadata 应指向 config_yaml.source_samples",
        )
    units = schema.get("units", {}) or {}
    problems.extend(
        f"config_yaml.h5_schema.units 缺少 {key}"
        for key in (
            "fs",
            "c",
            "fc",
            "pitch",
            "x_grid",
            "z_grid",
            "angles",
            "time_start_vector",
        )
        if key not in units
    )

    generation = config.get("generation", {}) or {}
    input_iq = generation.get("input_iq", {}) or {}
    ground_truth = generation.get("ground_truth", {}) or {}
    if not isinstance(input_iq, dict) or not input_iq:
        problems.append("config_yaml.generation.input_iq 必须是非空字典")
    if not isinstance(ground_truth, dict) or not ground_truth:
        problems.append("config_yaml.generation.ground_truth 必须是非空字典")
    if isinstance(ground_truth, dict) and ground_truth.get("dataset") != "/all_envdb_norm":
        problems.append(
            "config_yaml.generation.ground_truth.dataset 应指向 /all_envdb_norm",
        )
    reference_fs = ground_truth.get("reference_sampling_frequency_hz") if isinstance(ground_truth, dict) else None
    if not isinstance(reference_fs, (int, float)) or not np.isfinite(reference_fs) or reference_fs <= 0:
        problems.append("ground_truth.reference_sampling_frequency_hz 必须是有限正数")
    input_norm = (input_iq.get("normalization", {}) or {}) if isinstance(input_iq, dict) else {}
    if not isinstance(input_norm, dict) or input_norm.get("scale_reference_dataset") != "/all_scale_ref":
        problems.append(
            "input_iq.normalization.scale_reference_dataset 应指向 /all_scale_ref",
        )
    decimation = input_iq.get("decimation_factor")
    source_fs = input_iq.get("source_sampling_frequency_hz")
    packed_fs = input_iq.get("packed_sampling_frequency_hz")
    root_fs = scalar_float(hf, "fs")
    if not isinstance(decimation, int) or decimation < 1:
        problems.append(f"input_iq.decimation_factor 非法: {decimation}")
    elif source_fs is None or packed_fs is None or not np.isclose(
        float(source_fs) / decimation, float(packed_fs), rtol=1e-5, atol=1.0
    ):
        problems.append("input_iq source_sampling_frequency_hz/decimation_factor 与 packed fs 不一致")
    elif root_fs is None or not np.isclose(float(packed_fs), root_fs, rtol=1e-5, atol=1.0):
        problems.append("input_iq.packed_sampling_frequency_hz 与根数据集 fs 不一致")
    elif isinstance(reference_fs, (int, float)) and not np.isclose(
        float(reference_fs), float(source_fs), rtol=1e-5, atol=1.0
    ):
        problems.append("ground_truth.reference_sampling_frequency_hz 应与 input_iq.source_sampling_frequency_hz 一致")
    try:
        all_angles = decode_compact_sequence(input_iq.get("all_steering_angles_rad", [])).astype(
            np.float64,
        )
        selected_indices = decode_compact_sequence(input_iq.get("selected_angle_indices", [])).astype(
            np.int64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"config_yaml 输入角度编码非法: {exc}")
    else:
        if input_iq.get("input_angle_selection") != "all_available":
            problems.append("config_yaml input_angle_selection 应为 all_available")
        if np.any((selected_indices < 0) | (selected_indices >= len(all_angles))):
            problems.append("config_yaml selected_angle_indices 越界")
        elif "angles" in hf and (
            selected_indices.shape != hf["angles"].shape
            or not np.allclose(
                all_angles[selected_indices],
                hf["angles"][:],
                rtol=1e-5,
                atol=1e-6,
            )
        ):
            problems.append("config_yaml 输入角度记录与根数据集 angles 不一致")
    gt_norm = (ground_truth.get("normalization", {}) or {}) if isinstance(ground_truth, dict) else {}
    if not isinstance(gt_norm, dict) or gt_norm.get("norm_reference_dataset") != "/all_norm_ref":
        problems.append(
            "ground_truth.normalization.norm_reference_dataset 应指向 /all_norm_ref",
        )


def run_checks(hf: h5py.File) -> tuple[str, list[str]]:
    """Execute run checks."""
    problems: list[str] = []
    missing = [key for key in REQUIRED_FIELDS if key not in hf]
    if missing:
        problems.append("缺少字段: " + ", ".join(missing))

    duplicated = sorted(MOVED_TO_CONFIG_FIELDS.intersection(hf.keys()))
    if duplicated:
        problems.append(
            "以下逐样本元数据应只存在于 config_yaml: " + ", ".join(duplicated),
        )

    if "all_multi_I" in hf and "all_multi_Q" in hf and hf["all_multi_I"].shape != hf["all_multi_Q"].shape:
        problems.append("all_multi_I / all_multi_Q 形状不一致")

    for key in ("all_multi_I", "all_multi_Q", "all_envdb_norm"):
        if key in hf and hf[key].dtype != np.float16:
            problems.append(f"{key} 类型应为 float16，实际为 {hf[key].dtype}")
    for key in ("fs", "c", "fc", "pitch"):
        if key in hf and hf[key].dtype != np.float32:
            problems.append(f"{key} 类型应为 float32，实际为 {hf[key].dtype}")
    if "num_channels" in hf and hf["num_channels"].dtype != np.int32:
        problems.append(f"num_channels 类型应为 int32，实际为 {hf['num_channels'].dtype}")

    if {"all_multi_I", "all_envdb_norm"} <= set(hf.keys()) and hf["all_multi_I"].shape[0] != hf["all_envdb_norm"].shape[0]:
        problems.append("输入 IQ 与 GT 帧数不一致")

    if {"all_multi_I", "time_start_vector"} <= set(hf.keys()) and hf["time_start_vector"].shape[:2] != hf["all_multi_I"].shape[:2]:
        problems.append("time_start_vector 形状应匹配 [N,A]")

    if {"all_multi_I", "angles"} <= set(hf.keys()) and hf["angles"].shape[0] != hf["all_multi_I"].shape[1]:
        problems.append("angles 数量应匹配输入角度维度")

    if {"all_multi_I", "num_channels"} <= set(hf.keys()):
        channels = int(decode_value(hf["num_channels"][()]))
        if channels != hf["all_multi_I"].shape[-1]:
            problems.append(
                f"num_channels={channels}, 输入通道数={hf['all_multi_I'].shape[-1]}",
            )

    if {"all_multi_I", "valid_time_samples"} <= set(hf.keys()):
        valid = np.asarray(hf["valid_time_samples"][:], dtype=np.int64).reshape(-1)
        n_samples, _, n_time, _ = hf["all_multi_I"].shape
        if len(valid) != n_samples:
            problems.append(
                f"valid_time_samples 长度 {len(valid)} 与样本数 {n_samples} 不一致",
            )
        elif np.any((valid < COMPARISON_VALUE_2) | (valid > n_time)):
            problems.append(
                f"valid_time_samples 必须位于 [2,{n_time}],实际={valid.tolist()}",
            )
        else:
            for sample_idx, valid_time in enumerate(valid):
                if valid_time >= n_time:
                    continue
                i_tail = hf["all_multi_I"][sample_idx, :, valid_time:, :]
                q_tail = hf["all_multi_Q"][sample_idx, :, valid_time:, :]
                if np.any(i_tail != 0) or np.any(q_tail != 0):
                    problems.append(
                        f"样本 {sample_idx} 在 valid_time_samples={valid_time} 之后包含非零 IQ",
                    )

    for key in ("x_grid", "z_grid"):
        if key in hf:
            grid = np.asarray(hf[key][:], dtype=np.float64)
            if grid.ndim != 1 or grid.size < COMPARISON_VALUE_2:
                problems.append(f"{key} 必须是至少含 2 点的一维数组")
            elif not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
                problems.append(f"{key} 必须全部有限且严格递增")

    for key in ("fs", "c", "fc", "pitch"):
        if key in hf:
            value = scalar_float(hf, key)
            if value is None or not np.isfinite(value) or value <= 0:
                problems.append(f"{key} 必须是有限正数")

    if "config_yaml" in hf:
        value = decode_value(hf["config_yaml"][()])
        try:
            config = yaml.safe_load(value) or {}
        except yaml.YAMLError as exc:
            problems.append(f"config_yaml 不是合法 YAML: {exc}")
        else:
            if not isinstance(config, dict):
                problems.append("config_yaml 顶层必须是字典")
            else:
                problems.extend(
                    f"config_yaml 缺少顶层字段 {key}"
                    for key in (
                        "schema_version",
                        "dataset",
                        "provenance",
                        "h5_schema",
                        "generation",
                        "source_samples",
                    )
                    if key not in config
                )
                validate_config_schema(config, hf, problems)
                samples = config.get("source_samples")
                n_samples = hf["all_multi_I"].shape[0] if "all_multi_I" in hf else 0
                if not isinstance(samples, list) or len(samples) != n_samples:
                    problems.append(
                        "config_yaml.source_samples 数量必须与 H5 样本数一致",
                    )
                else:
                    for index, sample in enumerate(samples):
                        if not isinstance(sample, dict):
                            problems.append(f"source_samples[{index}] 必须是字典")
                            continue
                        if sample.get("sample_index") != index:
                            problems.append(
                                f"source_samples[{index}].sample_index 应为 {index}",
                            )
                        for key in (
                            "id",
                            "phantom_mode",
                            "phantom_source",
                            "iq_path",
                            "scan_path",
                            "phantom_path",
                            "gt_path",
                        ):
                            if key not in sample:
                                problems.append(f"source_samples[{index}] 缺少 {key}")
                                continue
                            path = str(sample[key] or "")
                            if key.endswith("_path") and path and Path(path).is_absolute():
                                problems.append(
                                    f"source_samples[{index}].{key} 必须是相对路径",
                                )
                        expected_path_roles = {
                            "iq_path": "read_input",
                            "scan_path": "read_input",
                            "phantom_path": "reference_only_not_read",
                            "gt_path": (
                                "generated_not_read"
                                if str(sample.get("gt_path") or "").startswith("generated:")
                                else "read_input"
                            ),
                        }
                        if sample.get("path_roles") != expected_path_roles:
                            problems.append(
                                f"source_samples[{index}].path_roles 未准确记录路径用途",
                            )
                convention = config.get("path_convention", {}) or {}
                if convention.get("type") != "relative":
                    problems.append("config_yaml.path_convention.type 应为 relative")
                if convention.get("base") != "PICMUS_ROOT":
                    problems.append("config_yaml.path_convention.base 应为 PICMUS_ROOT")

    for key in ("all_multi_I", "all_multi_Q", "all_envdb_norm"):
        if key in hf and hf[key].shape:
            for sample_idx in range(hf[key].shape[0]):
                arr = np.asarray(hf[key][sample_idx])
                if not np.isfinite(arr).all():
                    problems.append(f"{key} 第 {sample_idx} 帧包含 NaN/Inf")

    return ("通过" if not problems else "失败"), problems


def inspect_file(path: Path) -> bool:
    """Execute inspect file."""
    print()
    print("=" * 100)
    print(f"文件: {path.name}")
    print(f"路径: {path}")

    if not path.exists():
        print("状态: 失败")
        print("原因: 文件不存在")
        return False

    try:
        with h5py.File(path, "r") as hf:
            status, problems = run_checks(hf)
            file_size = path.stat().st_size / 1024**2
            n_frames = hf["all_multi_I"].shape[0] if "all_multi_I" in hf else "?"
            input_shape = format_shape(hf["all_multi_I"].shape) if "all_multi_I" in hf else "缺失"
            gt_shape = format_shape(hf["all_envdb_norm"].shape) if "all_envdb_norm" in hf else "无"
            fs = scalar_float(hf, "fs")
            gt_reference_fs = None
            decimation = None
            if "config_yaml" in hf:
                try:
                    config = yaml.safe_load(decode_value(hf["config_yaml"][()])) or {}
                    generation = config.get("generation", {}) or {}
                    gt_reference_fs = (generation.get("ground_truth", {}) or {}).get(
                        "reference_sampling_frequency_hz"
                    )
                    decimation = (generation.get("input_iq", {}) or {}).get("decimation_factor")
                except yaml.YAMLError:
                    pass

            summary_rows = [
                ("检查状态", status),
                ("文件大小", f"{file_size:.3f} MiB"),
                ("样本帧数", n_frames),
                ("输入 IQ", input_shape),
                ("主 GT", gt_shape),
                ("输入采样率", f"{fs / 1e6:.6g} MHz" if fs else "缺失"),
                ("GT 参考采样率", f"{float(gt_reference_fs) / 1e6:.6g} MHz" if gt_reference_fs else "缺失"),
                ("降采样倍数", int(decimation) if decimation else "缺失"),
                ("全部字段数", len(hf.keys())),
            ]
            print_table(summary_rows, ("项目", "值"))

            field_rows = []
            keys = [key for key in KEY_FIELDS if key in hf]
            keys.extend(sorted(key for key in hf if key not in keys))
            for key in keys:
                ds = hf[key]
                if not isinstance(ds, h5py.Dataset):
                    continue
                storage_bytes = dataset_storage_bytes(ds)
                row = (
                    key,
                    format_shape(ds.shape),
                    str(ds.dtype),
                    format_bytes(storage_bytes),
                    dataset_preview(ds),
                )
                field_rows.append((storage_bytes, row))
            field_rows.sort(key=lambda item: item[0], reverse=True)

            print()
            print("[字段占用]")
            print_table(
                [row for _, row in field_rows],
                ("字段", "形状", "类型", "占用空间", "范围/示例"),
            )

            if "config_yaml" in hf:
                config_text = str(decode_value(hf["config_yaml"][()]))
                print()
                print("[config_yaml 完整内容]")
                print(config_text.rstrip())

            if problems:
                print()
                print("[问题]")
                for item in problems:
                    print(f"  - {item}")

            return status == "通过"
    except Exception as exc:
        print("状态: 失败")
        print(f"原因: {exc}")
        return False


def discover_default_files() -> list[Path]:
    """Execute discover default files."""
    return sorted(DATA_DIR.glob("*.h5"))


def resolve_files(values: list[str] | None) -> list[Path]:
    """Execute resolve files."""
    if not values:
        return discover_default_files()
    return [Path(value).resolve() for value in values]


def main() -> None:
    """Run the command-line workflow."""
    parser = argparse.ArgumentParser(description="打包 H5 文件简洁体检。")
    parser.add_argument("files", nargs="*", help="H5 文件;默认检查 data/*.h5")
    parser.add_argument(
        "--files",
        dest="files_opt",
        nargs="*",
        default=None,
        help="H5 文件列表。",
    )
    parser.add_argument("--h5_path", nargs="*", default=None, help="等价于 --files。")
    parser.add_argument(
        "--log",
        default=None,
        help="日志路径;默认 data/logs/check_data_*.txt",
    )
    parser.add_argument("--no_log", action="store_true", help="只打印到命令行。")
    args = parser.parse_args()

    file_args = args.h5_path or args.files_opt or args.files
    files = resolve_files(file_args)
    if not files:
        raise SystemExit("未找到 H5 文件。请把 H5 放到 data/ 下,或显式传入路径。")

    def run() -> list[bool]:
        """Execute run."""
        return [inspect_file(path) for path in files]

    if args.no_log:
        results = run()
    else:
        log_path = Path(args.log) if args.log else default_log_path()
        log_path = log_path if log_path.is_absolute() else (ROOT / log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as f:
            old_stdout = sys.stdout
            sys.stdout = Tee(old_stdout, f)
            try:
                results = run()
                print()
                print(f"日志: {log_path}")
            finally:
                sys.stdout = old_stdout

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
