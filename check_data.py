"""打包 H5 文件简洁体检脚本。"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
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
    "sample_names",
    "phantom_mode",
    "phantom_source",
    "iq_path",
    "scan_path",
    "phantom_path",
    "gt_path",
    "has_gt",
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def default_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"check_data_{stamp}.txt"


def decode_value(value):
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
    width = 0
    for char in str(value):
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def shorten_text(text, limit: int = 110) -> str:
    text = str(text).replace("\n", " | ")
    return text if display_width(text) <= limit else text[:limit] + " ..."


def format_value(value, max_items: int = 1) -> str:
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
    if ds.shape == ():
        return format_value(ds[()])
    if ds.size == 0:
        return "空"
    if ds.dtype.kind in {"S", "U", "O"}:
        return format_value(ds[:])

    if ds.size <= 4096:
        arr = np.asarray(ds[:], dtype=np.float64)
    else:
        index = (0,) + tuple(slice(None) for _ in range(ds.ndim - 1))
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
    try:
        return int(ds.id.get_storage_size())
    except Exception:
        return 0


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.3f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def format_shape(shape) -> str:
    return "标量" if shape == () else "x".join(str(dim) for dim in shape)


def pad_display(value, width: int) -> str:
    text = str(value)
    return text + " " * max(width - display_width(text), 0)


def print_table(rows, headers) -> None:
    widths = [display_width(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], display_width(cell))
    print("  " + "  ".join(pad_display(cell, widths[idx]) for idx, cell in enumerate(headers)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(pad_display(cell, widths[idx]) for idx, cell in enumerate(row)))


def scalar_float(hf: h5py.File, key: str) -> float | None:
    if key not in hf:
        return None
    try:
        return float(decode_value(hf[key][()]))
    except Exception:
        return None


def run_checks(hf: h5py.File) -> tuple[str, list[str]]:
    problems: list[str] = []
    missing = [key for key in REQUIRED_FIELDS if key not in hf]
    if missing:
        problems.append("缺少字段: " + ", ".join(missing))

    if "all_multi_I" in hf and "all_multi_Q" in hf:
        if hf["all_multi_I"].shape != hf["all_multi_Q"].shape:
            problems.append("all_multi_I / all_multi_Q 形状不一致")

    if {"all_multi_I", "all_envdb_norm"} <= set(hf.keys()):
        if hf["all_multi_I"].shape[0] != hf["all_envdb_norm"].shape[0]:
            problems.append("输入 IQ 与 GT 帧数不一致")

    if {"all_multi_I", "time_start_vector"} <= set(hf.keys()):
        if hf["time_start_vector"].shape[:2] != hf["all_multi_I"].shape[:2]:
            problems.append("time_start_vector 形状应匹配 [N,A]")

    if {"all_multi_I", "angles"} <= set(hf.keys()):
        if hf["angles"].shape[0] != hf["all_multi_I"].shape[1]:
            problems.append("angles 数量应匹配输入角度维度")

    if {"all_multi_I", "num_channels"} <= set(hf.keys()):
        channels = int(decode_value(hf["num_channels"][()]))
        if channels != hf["all_multi_I"].shape[-1]:
            problems.append(f"num_channels={channels}, 输入通道数={hf['all_multi_I'].shape[-1]}")

    for key in ("all_multi_I", "all_multi_Q", "all_envdb_norm"):
        if key in hf and hf[key].shape:
            arr = np.asarray(hf[key][0])
            if not np.isfinite(arr).all():
                problems.append(f"{key} 第 0 帧包含 NaN/Inf")

    return ("通过" if not problems else "失败"), problems


def inspect_file(path: Path) -> bool:
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
            fc = scalar_float(hf, "fc")

            summary_rows = [
                ("检查状态", status),
                ("文件大小", f"{file_size:.3f} MiB"),
                ("样本帧数", n_frames),
                ("输入 IQ", input_shape),
                ("GT 图像", gt_shape),
                ("采样率 fs", f"{fs / 1e6:.6g} MHz" if fs else "缺失"),
                ("中心频率 fc", f"{fc / 1e6:.6g} MHz" if fc else "缺失"),
                ("全部字段数", len(hf.keys())),
            ]
            print_table(summary_rows, ("项目", "值"))

            field_rows = []
            keys = [key for key in KEY_FIELDS if key in hf]
            keys.extend(sorted(key for key in hf.keys() if key not in keys))
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
            print_table([row for _, row in field_rows], ("字段", "形状", "类型", "占用空间", "范围/示例"))

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
    return sorted(DATA_DIR.glob("*.h5"))


def resolve_files(values: list[str] | None) -> list[Path]:
    if not values:
        return discover_default_files()
    return [Path(value).resolve() for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 H5 文件简洁体检。")
    parser.add_argument("files", nargs="*", help="H5 文件；默认检查 data/*.h5")
    parser.add_argument("--files", dest="files_opt", nargs="*", default=None, help="H5 文件列表。")
    parser.add_argument("--h5_path", nargs="*", default=None, help="等价于 --files。")
    parser.add_argument("--log", default=None, help="日志路径；默认 data/logs/check_data_*.txt")
    parser.add_argument("--no_log", action="store_true", help="只打印到命令行。")
    args = parser.parse_args()

    file_args = args.h5_path or args.files_opt or args.files
    files = resolve_files(file_args)
    if not files:
        raise SystemExit("未找到 H5 文件。请把 H5 放到 data/ 下，或显式传入路径。")

    def run() -> list[bool]:
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
