"""Provide Python utilities for common_params."""

import argparse
import math
import os

try:
    from beamforming_utils import INTERP_CHOICES, WINDOW_CHOICES
except ModuleNotFoundError:
    from .beamforming_utils import INTERP_CHOICES, WINDOW_CHOICES

try:
    import yaml
except Exception:
    yaml = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COMMON_PARAMS = {
    "select_angles": "1",
    "f_number": 1.5,
    "dr": 60.0,
    "dynamic_aperture": True,
    "tgc": True,
    "tgc_alpha": 0.5,
    "window": "rect",
    "interp": "cubic",
}


def positive_float(value):
    """Execute positive float."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return value


def unit_interval_float(value):
    """Execute unit interval float."""
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in (0, 1]")
    return value


def nonnegative_float(value):
    """Execute nonnegative float."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than or equal to 0",
        )
    return value


def closed_unit_interval_float(value):
    """Execute closed unit interval float."""
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return value


def nonnegative_int(value):
    """Execute nonnegative int."""
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return value


def positive_int(value):
    """Parse a positive integer command-line value."""
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def positive_odd_int(value):
    """Execute positive odd int."""
    value = int(value)
    if value < 1 or value % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return value


def load_common_params(config_path=None):
    """Load common params."""
    params = COMMON_PARAMS.copy()
    config_path = config_path or os.path.join(PROJECT_ROOT, "config.yaml")
    if yaml is None or not os.path.exists(config_path):
        return params

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    params.update(config.get("params", {}) or {})
    params["select_angles"] = str(
        params.get("select_angles", COMMON_PARAMS["select_angles"]),
    )
    return params


def add_common_arguments(
    parser,
    params=None,
    select_help="角度选择: all, center, N(数量)",
    window_help="窗函数",
):
    """Execute add common arguments."""
    params = params or load_common_params()
    parser.add_argument(
        "--select_angles",
        type=str,
        default=params["select_angles"],
        help=select_help,
    )
    parser.add_argument(
        "--f_number",
        type=positive_float,
        default=positive_float(params["f_number"]),
        help="F-Number",
    )
    parser.add_argument(
        "--dr",
        type=positive_float,
        default=positive_float(params["dr"]),
        help="动态范围 dB",
    )
    parser.add_argument(
        "--dynamic_aperture",
        action="store_true",
        default=bool(params["dynamic_aperture"]),
        help="启用动态孔径",
    )
    parser.add_argument(
        "--no_dynamic_aperture",
        dest="dynamic_aperture",
        action="store_false",
        help="禁用动态孔径",
    )
    parser.add_argument(
        "--tgc",
        action="store_true",
        default=bool(params["tgc"]),
        help="启用 TGC",
    )
    parser.add_argument("--no_tgc", dest="tgc", action="store_false", help="禁用 TGC")
    parser.add_argument(
        "--tgc_alpha",
        type=nonnegative_float,
        default=nonnegative_float(params["tgc_alpha"]),
        help="TGC 衰减系数",
    )
    parser.add_argument(
        "--window",
        type=str,
        default=params["window"],
        choices=WINDOW_CHOICES,
        help=window_help,
    )
    parser.add_argument(
        "--interp",
        type=str,
        default=params["interp"],
        choices=INTERP_CHOICES,
        help="插值方式",
    )


def add_io_arguments(parser, save_gt_help="保存GT图像"):
    """Execute add io arguments."""
    parser.add_argument(
        "--h5_path",
        type=str,
        default="data/simulation.h5",
        help="H5 数据文件路径",
    )
    parser.add_argument("--h5_sample_idx", type=int, default=0, help="H5 样本索引")
    parser.add_argument("--output_dir", type=str, default="results", help="输出目录")
    parser.add_argument(
        "--save_gt",
        action="store_true",
        default=False,
        help=save_gt_help,
    )
