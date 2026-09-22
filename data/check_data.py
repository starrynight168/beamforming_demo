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

BYTES_PER_UNIT = 1024
MIN_DATA_POINTS = 2
PREVIEW_MAX_ITEMS = 4096

# 脚本位于 data/;默认检查同目录的打包 H5,和启动时的工作目录无关。
DATA_DIR = Path(__file__).resolve().parent
ROOT = DATA_DIR.parent
LOG_DIR = DATA_DIR / "logs"

REQUIRED_FIELDS = [
    "all_multi_I",
    "all_multi_Q",
    "all_envdb_norm",
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
    "time_start_vector",
    "valid_time_samples",
    "angles",
    "z_grid",
    "x_grid",
    "c",
    "fc",
    "pitch",
    "num_channels",
    "config_yaml",
]

PREFERRED_FIELD_ORDER = {name: idx for idx, name in enumerate(KEY_FIELDS)}

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
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
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

    if ds.size <= PREVIEW_MAX_ITEMS:
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
    except (OSError, RuntimeError, ValueError):
        return 0


def format_bytes(num_bytes: int) -> str:
    """Execute format bytes."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < BYTES_PER_UNIT or unit == units[-1]:
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
        "  "
        + "  ".join(pad_display(cell, widths[idx]) for idx, cell in enumerate(headers)),
    )
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  "
            + "  ".join(pad_display(cell, widths[idx]) for idx, cell in enumerate(row)),
        )


def scalar_float(hf: h5py.File, key: str) -> float | None:
    """Execute scalar float."""
    if key not in hf:
        return None
    try:
        return float(decode_value(hf[key][()]))
    except (OverflowError, TypeError, ValueError):
        return None


def dataset_shape(hf: h5py.File, key: str):
    """Return a dataset shape, or None for missing/non-dataset fields."""
    dataset = hf.get(key)
    return dataset.shape if isinstance(dataset, h5py.Dataset) else None


def decode_compact_sequence(value) -> np.ndarray:
    """Decode list or arithmetic-sequence config into an array."""
    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, dict) or value.get("encoding") != "arithmetic_sequence":
        raise ValueError("必须是数组或 arithmetic_sequence")
    count_raw = value["count"]
    count = int(count_raw)
    if float(count_raw) != count or count < 1:
        raise ValueError("count 必须是正整数")
    start = np.asarray(value["start"])
    step = np.asarray(value["step"])
    if (
        start.shape != ()
        or step.shape != ()
        or not np.isfinite(start)
        or not np.isfinite(step)
    ):
        raise ValueError("start/step 必须是有限标量")
    return start + np.arange(count) * step


def validate_config_schema(config: dict, hf: h5py.File, problems: list[str]) -> None:
    """Validate config schema."""
    if config.get("schema_version") != 1:
        problems.append(
            f"config_yaml.schema_version 应为 1,实际为 {config.get('schema_version')}",
        )

    dataset = config.get("dataset", {}) or {}
    if not isinstance(dataset, dict):
        problems.append("config_yaml.dataset 必须是字典")
        dataset = {}
    if dataset.get("purpose") != "algorithm_evaluation":
        problems.append("config_yaml.dataset.purpose 应为 algorithm_evaluation")

    schema = config.get("h5_schema", {}) or {}
    if not isinstance(schema, dict):
        problems.append("config_yaml.h5_schema 必须是字典")
        schema = {}
    if schema.get("iq_layout") != ["sample", "angle", "time", "channel"]:
        problems.append("config_yaml.h5_schema.iq_layout 与实际 IQ 布局不一致")
    if schema.get("ground_truth_layout") != ["sample", "image_channel", "z", "x"]:
        problems.append(
            "config_yaml.h5_schema.ground_truth_layout 与实际 GT 布局不一致",
        )
    padding = schema.get("padding", {}) or {}
    if not isinstance(padding, dict):
        problems.append("config_yaml.h5_schema.padding 必须是字典")
        padding = {}
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
    if not isinstance(units, dict):
        problems.append("config_yaml.h5_schema.units 必须是字典")
        units = {}
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
    if not isinstance(generation, dict):
        problems.append("config_yaml.generation 必须是字典")
        generation = {}
    input_iq = generation.get("input_iq", {}) or {}
    ground_truth = generation.get("ground_truth", {}) or {}
    if not isinstance(input_iq, dict) or not input_iq:
        problems.append("config_yaml.generation.input_iq 必须是非空字典")
        input_iq = {}
    if not isinstance(ground_truth, dict) or not ground_truth:
        problems.append("config_yaml.generation.ground_truth 必须是非空字典")
        ground_truth = {}
    if (
        isinstance(ground_truth, dict)
        and ground_truth.get("dataset") != "/all_envdb_norm"
    ):
        problems.append(
            "config_yaml.generation.ground_truth.dataset 应指向 /all_envdb_norm",
        )
    reference_fs = (
        ground_truth.get("reference_sampling_frequency_hz")
        if isinstance(ground_truth, dict)
        else None
    )
    if (
        not isinstance(reference_fs, (int, float))
        or not np.isfinite(reference_fs)
        or reference_fs <= 0
    ):
        problems.append("ground_truth.reference_sampling_frequency_hz 必须是有限正数")
    input_norm = (
        (input_iq.get("normalization", {}) or {}) if isinstance(input_iq, dict) else {}
    )
    if (
        not isinstance(input_norm, dict)
        or input_norm.get("scale_reference_dataset") != "/all_scale_ref"
    ):
        problems.append(
            "input_iq.normalization.scale_reference_dataset 应指向 /all_scale_ref",
        )
    decimation = input_iq.get("decimation_factor")
    source_fs = input_iq.get("source_sampling_frequency_hz")
    packed_fs = input_iq.get("packed_sampling_frequency_hz")
    root_fs = scalar_float(hf, "fs")
    if type(decimation) is not int or decimation < 1:
        problems.append(f"input_iq.decimation_factor 非法: {decimation}")
    else:
        try:
            source_fs_value = float(source_fs)
            packed_fs_value = float(packed_fs)
        except (TypeError, ValueError):
            problems.append("input_iq 的 source/packed sampling frequency 必须是数值")
        else:
            if not all(
                np.isfinite(value) and value > 0
                for value in (source_fs_value, packed_fs_value)
            ):
                problems.append(
                    "input_iq 的 source/packed sampling frequency 必须是有限正数"
                )
            elif not np.isclose(
                source_fs_value / decimation, packed_fs_value, rtol=1e-5, atol=1.0
            ):
                problems.append(
                    "input_iq source_sampling_frequency_hz/decimation_factor 与 packed fs 不一致"
                )
            elif root_fs is None or not np.isclose(
                packed_fs_value, root_fs, rtol=1e-5, atol=1.0
            ):
                problems.append(
                    "input_iq.packed_sampling_frequency_hz 与根数据集 fs 不一致"
                )
            elif isinstance(reference_fs, (int, float)) and not np.isclose(
                float(reference_fs), source_fs_value, rtol=1e-5, atol=1.0
            ):
                problems.append(
                    "ground_truth.reference_sampling_frequency_hz 应与 input_iq.source_sampling_frequency_hz 一致"
                )
    try:
        all_angles = decode_compact_sequence(
            input_iq.get("all_steering_angles_rad", [])
        ).astype(np.float64)
        selected_raw = decode_compact_sequence(
            input_iq.get("selected_angle_indices", [])
        ).astype(np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"config_yaml 输入角度编码非法: {exc}")
    else:
        angles_valid = (
            all_angles.ndim == 1
            and all_angles.size > 0
            and np.isfinite(all_angles).all()
        )
        if not angles_valid:
            problems.append(
                "config_yaml all_steering_angles_rad 必须是非空有限一维数组"
            )
        indices_valid = not (
            selected_raw.ndim != 1
            or selected_raw.size == 0
            or not np.isfinite(selected_raw).all()
            or not np.all(selected_raw == np.floor(selected_raw))
        )
        if not indices_valid:
            problems.append("config_yaml selected_angle_indices 必须是非空有限整数数组")
            selected_indices = np.empty(0, dtype=np.int64)
        else:
            selected_indices = selected_raw.astype(np.int64)
            if np.unique(selected_indices).size != selected_indices.size:
                problems.append("config_yaml selected_angle_indices 不能重复")
        if input_iq.get("input_angle_selection") != "all_available":
            problems.append("config_yaml input_angle_selection 应为 all_available")
        if (
            angles_valid
            and indices_valid
            and np.any(
                (selected_indices < 0) | (selected_indices >= len(all_angles)),
            )
        ):
            problems.append("config_yaml selected_angle_indices 越界")
        elif (
            angles_valid
            and indices_valid
            and "angles" in hf
            and (
                selected_indices.shape != hf["angles"].shape
                or not np.allclose(
                    all_angles[selected_indices],
                    hf["angles"][:],
                    rtol=1e-5,
                    atol=1e-6,
                )
            )
        ):
            problems.append("config_yaml 输入角度记录与根数据集 angles 不一致")
    gt_norm = (
        (ground_truth.get("normalization", {}) or {})
        if isinstance(ground_truth, dict)
        else {}
    )
    if (
        not isinstance(gt_norm, dict)
        or gt_norm.get("norm_reference_dataset") != "/all_norm_ref"
    ):
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

    invalid_datasets = [
        key
        for key in REQUIRED_FIELDS
        if key in hf and not isinstance(hf[key], h5py.Dataset)
    ]
    problems.extend(f"{key} 必须是 dataset" for key in invalid_datasets)
    if invalid_datasets:
        return "失败", problems

    iq_dataset = hf.get("all_multi_I")
    q_dataset = hf.get("all_multi_Q")
    iq_shape = None
    if isinstance(iq_dataset, h5py.Dataset):
        iq_shape = iq_dataset.shape
        if (
            len(iq_shape) != 4
            or min(iq_shape[0], iq_shape[1], iq_shape[3]) < 1
            or iq_shape[2] < 2
        ):
            problems.append(f"all_multi_I 应为非空 [N,A,T,C] 且 T>=2，实际 {iq_shape}")
            iq_shape = None

    if (
        isinstance(iq_dataset, h5py.Dataset)
        and isinstance(q_dataset, h5py.Dataset)
        and iq_dataset.shape != q_dataset.shape
    ):
        problems.append("all_multi_I / all_multi_Q 形状不一致")

    for key in ("all_multi_I", "all_multi_Q", "all_envdb_norm"):
        if key in hf and hf[key].dtype != np.float16:
            problems.append(f"{key} 类型应为 float16，实际为 {hf[key].dtype}")
    for key in ("fs", "c", "fc", "pitch"):
        if key in hf:
            if hf[key].shape != ():
                problems.append(f"{key} 必须是标量 dataset")
            if hf[key].dtype != np.float32:
                problems.append(f"{key} 类型应为 float32，实际为 {hf[key].dtype}")
    if "num_channels" in hf and hf["num_channels"].dtype != np.int32:
        problems.append(
            f"num_channels 类型应为 int32，实际为 {hf['num_channels'].dtype}"
        )
    if "num_channels" in hf and hf["num_channels"].shape != ():
        problems.append("num_channels 必须是标量 dataset")
    for key in ("all_scale_ref", "all_norm_ref"):
        if key in hf and hf[key].dtype != np.float32:
            problems.append(f"{key} 类型应为 float32，实际为 {hf[key].dtype}")
    for key in ("time_start_vector", "angles", "z_grid", "x_grid"):
        if key in hf and hf[key].dtype != np.float32:
            problems.append(f"{key} 类型应为 float32，实际为 {hf[key].dtype}")
    if "valid_time_samples" in hf and hf["valid_time_samples"].dtype != np.int32:
        problems.append(
            f"valid_time_samples 类型应为 int32，实际为 {hf['valid_time_samples'].dtype}",
        )

    if "all_envdb_norm" in hf:
        gt_shape = hf["all_envdb_norm"].shape
        if (
            len(gt_shape) != 4
            or gt_shape[1] != 1
            or min(gt_shape[0], gt_shape[2], gt_shape[3]) < 1
        ):
            problems.append(f"all_envdb_norm 应为 [N,1,H,W]，实际 {gt_shape}")
        elif iq_shape is not None and gt_shape[0] != iq_shape[0]:
            problems.append("输入 IQ 与 GT 帧数不一致")
        elif (
            "z_grid" in hf
            and "x_grid" in hf
            and gt_shape[2:] != (hf["z_grid"].size, hf["x_grid"].size)
        ):
            problems.append("GT 空间形状必须匹配 z_grid/x_grid")

    if iq_shape is not None and "time_start_vector" in hf:
        time_start = np.asarray(hf["time_start_vector"][:])
        if time_start.shape != iq_shape[:2] or not np.isfinite(time_start).all():
            problems.append("time_start_vector 必须是匹配 IQ [N,A] 的有限数组")

    if iq_shape is not None and "angles" in hf:
        angles = np.asarray(hf["angles"][:])
        if (
            angles.ndim != 1
            or angles.size != iq_shape[1]
            or not np.isfinite(angles).all()
        ):
            problems.append("angles 必须是匹配输入角度维的有限一维数组")

    if iq_shape is not None and "num_channels" in hf:
        try:
            channels = int(decode_value(hf["num_channels"][()]))
        except (OverflowError, TypeError, ValueError):
            problems.append("num_channels 必须是整数标量")
        else:
            if channels < 1 or channels != iq_shape[-1]:
                problems.append(
                    f"num_channels={channels}, 输入通道数={iq_shape[-1]}",
                )

    if (
        iq_shape is not None
        and isinstance(iq_dataset, h5py.Dataset)
        and isinstance(q_dataset, h5py.Dataset)
        and "valid_time_samples" in hf
    ):
        valid_ds = hf["valid_time_samples"]
        valid = np.asarray(valid_ds[:], dtype=np.int64).reshape(-1)
        n_samples, _, n_time, _ = iq_shape
        if (
            valid_ds.ndim != 1
            or valid_ds.dtype.kind not in "iu"
            or len(valid) != n_samples
        ):
            problems.append(
                f"valid_time_samples 必须是长度 {n_samples} 的一维整数数组",
            )
        elif np.any((valid < MIN_DATA_POINTS) | (valid > n_time)):
            problems.append(
                f"valid_time_samples 必须位于 [{MIN_DATA_POINTS},{n_time}],实际={valid.tolist()}",
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

    if iq_shape is not None:
        for key in ("all_scale_ref", "all_norm_ref"):
            if key not in hf:
                continue
            values = np.asarray(hf[key][:])
            if values.ndim != 1 or values.size != iq_shape[0]:
                problems.append(f"{key} 必须是长度 {iq_shape[0]} 的一维数组")
            elif not np.isfinite(values).all() or np.any(values <= 0):
                problems.append(f"{key} 必须全部为有限正数")

    for key in ("x_grid", "z_grid"):
        if key in hf:
            grid = np.asarray(hf[key][:], dtype=np.float64)
            if grid.ndim != 1 or grid.size < MIN_DATA_POINTS:
                problems.append(f"{key} 必须是至少含 2 点的一维数组")
            elif not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
                problems.append(f"{key} 必须全部有限且严格递增")

    for key in ("fs", "c", "fc", "pitch"):
        if key in hf:
            value = scalar_float(hf, key)
            if value is None or not np.isfinite(value) or value <= 0:
                problems.append(f"{key} 必须是有限正数")

    if "all_envdb_norm" in hf and hf["all_envdb_norm"].size:
        gt_min = float(np.min(hf["all_envdb_norm"][:]))
        gt_max = float(np.max(hf["all_envdb_norm"][:]))
        if (
            not np.isfinite(gt_min)
            or not np.isfinite(gt_max)
            or gt_min < -1e-3
            or gt_max > 1.0 + 1e-3
        ):
            problems.append(
                f"all_envdb_norm 必须位于 [0,1]，实际范围 [{gt_min:g},{gt_max:g}]"
            )

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
                n_samples = iq_shape[0] if iq_shape is not None else 0
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
                        if not str(sample.get("id") or "").strip():
                            problems.append(f"source_samples[{index}].id 不能为空")
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
                            if key in {"iq_path", "scan_path", "gt_path"} and not path:
                                problems.append(
                                    f"source_samples[{index}].{key} 不能为空"
                                )
                            if (
                                key.endswith("_path")
                                and path
                                and Path(path).is_absolute()
                            ):
                                problems.append(
                                    f"source_samples[{index}].{key} 必须是相对路径",
                                )
                        expected_path_roles = {
                            "iq_path": "read_input",
                            "scan_path": "read_input",
                            "phantom_path": "reference_only_not_read",
                            "gt_path": (
                                "generated_not_read"
                                if str(sample.get("gt_path") or "").startswith(
                                    "generated:"
                                )
                                else "read_input"
                            ),
                        }
                        if sample.get("path_roles") != expected_path_roles:
                            problems.append(
                                f"source_samples[{index}].path_roles 未准确记录路径用途",
                            )
                convention = config.get("path_convention", {}) or {}
                if not isinstance(convention, dict):
                    problems.append("config_yaml.path_convention 必须是字典")
                    convention = {}
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

    if not path.is_file():
        print("状态: 失败")
        print("原因: 文件不存在")
        return False

    try:
        with h5py.File(path, "r") as hf:
            status, problems = run_checks(hf)
            file_size = path.stat().st_size / 1024**2
            input_shape = dataset_shape(hf, "all_multi_I")
            gt_shape = dataset_shape(hf, "all_envdb_norm")
            n_frames = input_shape[0] if input_shape else "?"
            input_shape = format_shape(input_shape) if input_shape else "缺失"
            gt_shape = format_shape(gt_shape) if gt_shape else "无"
            fs = scalar_float(hf, "fs")
            gt_reference_fs = None
            decimation = None
            config_dataset = hf.get("config_yaml")
            if isinstance(config_dataset, h5py.Dataset):
                try:
                    config = yaml.safe_load(decode_value(config_dataset[()])) or {}
                    generation = config.get("generation", {}) or {}
                    gt_reference_fs = (generation.get("ground_truth", {}) or {}).get(
                        "reference_sampling_frequency_hz"
                    )
                    decimation = (generation.get("input_iq", {}) or {}).get(
                        "decimation_factor"
                    )
                except (AttributeError, TypeError, yaml.YAMLError):
                    gt_reference_fs = None
                    decimation = None

            summary_rows = [
                ("检查状态", status),
                ("文件大小", f"{file_size:.3f} MiB"),
                ("样本帧数", n_frames),
                ("输入 IQ", input_shape),
                ("主 GT", gt_shape),
                ("输入采样率", f"{fs / 1e6:.6g} MHz" if fs else "缺失"),
                (
                    "GT 参考采样率",
                    f"{float(gt_reference_fs) / 1e6:.6g} MHz"
                    if gt_reference_fs
                    else "缺失",
                ),
                ("降采样倍数", int(decimation) if decimation else "缺失"),
                ("全部字段数", len(hf.keys())),
            ]
            print_table(summary_rows, ("项目", "值"))

            field_rows = []
            keys = sorted(
                hf.keys(),
                key=lambda key: (PREFERRED_FIELD_ORDER.get(key, 10_000), key),
            )
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
            field_rows.sort(
                key=lambda item: (
                    PREFERRED_FIELD_ORDER.get(item[1][0], 10_000),
                    -item[0],
                    item[1][0],
                ),
            )

            print()
            print("[字段占用]")
            print_table(
                [row for _, row in field_rows],
                ("字段", "形状", "类型", "占用空间", "范围/示例"),
            )

            if isinstance(config_dataset, h5py.Dataset):
                config_text = str(decode_value(config_dataset[()]))
                print()
                print("[config_yaml 完整内容]")
                print(config_text.rstrip())

            if problems:
                print()
                print("[问题]")
                for item in problems:
                    print(f"  - {item}")

            return status == "通过"
    except (
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        RuntimeError,
        h5py.Error,
        yaml.YAMLError,
    ) as exc:
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
        "--log",
        default=None,
        help="日志路径;默认 data/logs/check_data_*.txt",
    )
    parser.add_argument("--no_log", action="store_true", help="只打印到命令行。")
    args = parser.parse_args()

    file_args = args.files
    files = resolve_files(file_args)
    if not files:
        raise SystemExit("未找到 H5 文件。请把 H5 放到 data/ 下,或显式传入路径。")

    if args.no_log:
        results = [inspect_file(path) for path in files]
    else:
        log_path = Path(args.log) if args.log else default_log_path()
        log_path = log_path if log_path.is_absolute() else (ROOT / log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as f:
            old_stdout = sys.stdout
            sys.stdout = Tee(old_stdout, f)
            try:
                results = [inspect_file(path) for path in files]
                print()
                print(f"日志: {log_path}")
            finally:
                sys.stdout = old_stdout

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
