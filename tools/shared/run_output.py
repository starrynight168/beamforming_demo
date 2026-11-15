import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS_OUTPUT_ROOT = ROOT / "tools" / "output"
STAGES = {
    "prepare": "准备",
    "run": "运行",
    "progress": "进度",
    "evaluate": "评估",
    "warning": "提示",
    "error": "错误",
    "done": "完成",
    "info": "信息",
}


def resolve_output_dir(tool_name, output_dir=None):
    if output_dir is None:
        path = TOOLS_OUTPUT_ROOT / tool_name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    else:
        path = Path(output_dir).expanduser()
        if not path.is_absolute():
            path = ROOT / path
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_tree(run_dir):
    run_dir = Path(run_dir)
    for name in ("figures", "metrics", "data"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


def emit(run_dir, stage, message, *, flush=True):
    label = STAGES.get(stage, stage)
    line = f"[{datetime.now().strftime('%H:%M:%S')}] [{label}] {message}"
    print(line, flush=flush)
    if run_dir is not None:
        path = Path(run_dir) / "run.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def run_process(command, cwd, run_dir):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            message = line.rstrip("\r\n")
            if message:
                emit(run_dir, "progress", message)
    return process.wait()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    columns = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def set_plot_style(*, font_family="Times New Roman", font_size=8):
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.55,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "axes.prop_cycle": mpl.cycler(
                color=("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
            ),
        }
    )


def save_figure(fig, path, *, dpi=300, formats=None, close=True):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = []
    if formats is None:
        formats = (path.suffix.lstrip(".") or "png",)
    for extension in formats:
        target = path.with_suffix(f".{extension}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
        output_paths.append(target)
    if close:
        plt.close(fig)
    return output_paths[0] if len(output_paths) == 1 else output_paths


class ToolOutput:
    def __init__(self, tool_name, output_dir=None):
        self.root = prepare_output_tree(resolve_output_dir(tool_name, output_dir))

    def path(self, area, name):
        if area not in {"figures", "metrics", "data"}:
            raise ValueError(f"Unknown output area: {area}")
        path = self.root / area / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, stage, message, *, flush=True):
        emit(self.root, stage, message, flush=flush)

    def config(self, payload):
        return write_json(self.root / "run_config.json", payload)

    def json(self, area, name, payload):
        return write_json(self.path(area, name), payload)

    def csv(self, area, name, rows, fieldnames=None):
        return write_csv(self.path(area, name), rows, fieldnames=fieldnames)

    def figure(self, name, fig, *, dpi=300, formats=None):
        return save_figure(
            fig,
            self.path("figures", name),
            dpi=dpi,
            formats=formats,
        )
