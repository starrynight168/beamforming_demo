import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import yaml
from shared_core.h5_schema import require_h5_schema
from .workflow_common import TOOLS_OUTPUT_ROOT
from .run_output import emit, run_process

ROOT = Path(__file__).resolve().parents[2]


def ask(prompt, default=None):
    hint = f" [默认: {default}]" if default is not None else ""
    raw = input(f"{prompt}{hint}: ").strip()
    if raw.lower() in {"q", "quit", "exit", "退出"}:
        raise SystemExit("已退出。")
    return str(default) if not raw and default is not None else raw


def path_input(prompt, default=None, kind="file"):
    while True:
        raw = ask(prompt, default).strip().strip('"').strip("'")
        if not raw:
            print("请输入路径。")
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if kind == "output" or (kind == "file" and path.is_file()) or (kind == "dir" and path.is_dir()) or (kind == "either" and path.exists()):
            return path.resolve()
        print("路径不存在或类型不正确，请重新输入。")


def number(prompt, default, cast=float, valid=lambda value: True):
    while True:
        raw = ask(prompt, default)
        try:
            value = cast(raw)
            if math.isfinite(value) and valid(value):
                return value
        except ValueError:
            pass
        print("输入不合法，请重新输入。")


def choice(prompt, choices, default=0):
    print(prompt)
    for index, (_, label) in enumerate(choices, 1):
        print(f"  {index}. {label}" + ("（默认）" if index - 1 == default else ""))
    selected = number("请输入序号", default + 1, int, lambda value: 1 <= value <= len(choices))
    return choices[selected - 1][0]


def sample_input(h5_path):
    with h5py.File(h5_path, "r") as f:
        require_h5_schema(f)
        config = yaml.safe_load(f["config_yaml"][()])
        inputs = config["generation"]["storage"]["inputs"]
        domain = next((name for name in ("iq", "analytic_rf", "rf") if name in inputs), None)
        if domain is None:
            raise ValueError("H5 未声明已保存的输入域。")
        field = {"iq": "iq_i", "analytic_rf": "ar_i", "rf": "rf"}[domain]
        count = f[field].shape[0]
        print(f"H5 共 {count} 帧，输入范围 0–{count - 1}。")
        selected = number("样本序号", 0, int, lambda value: 0 <= value < count)
        if "acquisition_id" in f:
            item = f["acquisition_id"][selected]
            print("样本:", item.decode("utf-8") if isinstance(item, bytes) else item)
        return selected


def output_input(default_name):
    default = TOOLS_OUTPUT_ROOT / default_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    return path_input("输出目录，可拖入文件夹或直接回车", default, "output")


def run(script, args):
    command = [sys.executable, "-m", f"tools.code.{Path(script).stem}", *map(str, args)]
    emit(None, "run", subprocess.list2cmdline(command))
    return_code = run_process(command, ROOT, None)
    if return_code:
        emit(None, "error", f"进程退出码 {return_code}。")
        raise SystemExit(return_code)
    emit(None, "done", "工具运行结束。")
