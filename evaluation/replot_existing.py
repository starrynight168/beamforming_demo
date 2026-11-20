"""Replot stored training curves and comparison figures with the shared style.

The helper rebuilds supported figures without loading a model or re-running an
evaluation.

Usage:
    python -m evaluation.replot_existing --root train/results
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.plot_metrics import (
    MAX_CURVE_METHODS_PER_PAGE,
    build_page_method_colors,
)
from shared_core.file_sets import replace_files
from shared_core.paths import PROJECT_ROOT as PROJECT_ROOT
from shared_core.paths import portable_path
from shared_core.plot_style import (
    DIAGNOSTIC,
    apply_style,
    save_figure,
    style_axes,
)

def display_name(name: str) -> str:
    return "MBAN" if name == "roi05" else name


def replace_pages(pending: list[tuple[Path, Path]]) -> None:
    """Install rendered pages through the shared rollback-safe file helper."""
    if not pending:
        return
    targets = [target for _, target in pending]
    parents = {target.absolute().parent for target in targets}
    if len(parents) != 1:
        raise ValueError("all replacement pages must share one target directory")
    backup_root = parents.pop()
    replace_files(pending, targets, backup_root)


def replot_training_curves(directory: Path) -> bool:
    path = directory / "training_curves.csv"
    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    if frame.empty or "method" not in frame:
        return False
    methods = list(dict.fromkeys(frame["method"].astype(str)))
    page_size = MAX_CURVE_METHODS_PER_PAGE
    pages = [methods[start : start + page_size] for start in range(0, len(methods), page_size)]
    apply_style(DIAGNOSTIC)
    pending: list[tuple[Path, Path]] = []
    try:
        for page_index, page_methods in enumerate(pages, start=1):
            suffix = "" if page_index == 1 else f"_page{page_index}"
            target = directory / f"training_curves{suffix}.png"
            temporary = target.with_name(
                f".{target.stem}.{uuid.uuid4().hex}{target.suffix}"
            )
            height = max(4.2, 3.0 + 0.35 * len(page_methods))
            fig, axes = plt.subplots(1, 2, figsize=(12.0, height))
            colors = build_page_method_colors(page_methods)
            for method in page_methods:
                subset = frame[frame["method"].astype(str) == method].sort_values("epoch")
                color = colors[method]
                label = display_name(method)
                axes[0].plot(subset["epoch"], subset["train_loss"], color=color, linewidth=1.1, label=label)
                validation = pd.to_numeric(subset["validation_loss"], errors="coerce")
                best = validation[np.isfinite(validation)].min()
                validation_label = f"{label} (best={best:.4g})" if np.isfinite(best) else label
                axes[1].plot(subset["epoch"], validation, color=color, linewidth=1.1, label=validation_label)
            axes[0].set_title("Training loss")
            axes[1].set_title("Validation loss")
            for axis in axes:
                axis.set_xlabel("Epoch")
                axis.set_ylabel("Loss")
                style_axes(axis)
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2)
            fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
            save_figure(fig, temporary, profile=DIAGNOSTIC)
            pending.append((temporary, target))
        replace_pages(pending)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
    expected = {target.name for _, target in pending}
    for stale in directory.glob("training_curves*.png"):
        if stale.name not in expected:
            stale.unlink(missing_ok=True)
    return True


def comparison_meta(directory: Path) -> dict | None:
    for candidate in (
        directory / "metrics" / "evaluation_meta.json",
        directory / "evaluation_meta.json",
    ):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def replot_comparison(directory: Path) -> bool:
    array_path = directory / "comparison.npy"
    meta = comparison_meta(directory)
    if meta is None or not array_path.is_file():
        return False
    h5_path = portable_path(meta.get("h5_path", ""))
    if not h5_path.is_file():
        return False
    methods = [str(name) for name in meta.get("methods", [])]
    images = np.load(array_path)
    if not methods or images.ndim != 3 or images.shape[0] != len(methods):
        return False
    with h5py.File(h5_path, "r") as handle:
        x_mm = np.asarray(handle["x_grid"][:], dtype=float) * 1000.0
        z_mm = np.asarray(handle["z_grid"][:], dtype=float) * 1000.0
    if (
        x_mm.ndim != 1 or z_mm.ndim != 1
        or x_mm.size < 2 or z_mm.size < 2
        or not np.isfinite(x_mm).all() or not np.isfinite(z_mm).all()
        or not np.all(np.diff(x_mm) > 0) or not np.all(np.diff(z_mm) > 0)
        or images.shape[1:] != (z_mm.size, x_mm.size)
    ):
        raise ValueError(f"比较图尺寸或坐标网格无效: {array_path}, {h5_path}")
    extent = [float(x_mm[0]), float(x_mm[-1]), float(z_mm[-1]), float(z_mm[0])]
    titles = [display_name(name) for name in methods]
    import run_one

    run_one.save_comparison(
        [images[index] for index in range(len(methods))],
        titles,
        extent,
        directory / "comparison.png",
        float(meta.get("dr", 60.0)),
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("train/results"))
    parser.add_argument("--only", choices=("all", "curves", "comparisons"), default="all")
    args = parser.parse_args(argv)
    root = args.root
    training = comparisons = skipped = 0
    if args.only in ("all", "curves"):
        for path in sorted(root.rglob("training_curves.csv")):
            training += int(replot_training_curves(path.parent))
    if args.only in ("all", "comparisons"):
        for path in sorted(root.rglob("comparison.npy")):
            if replot_comparison(path.parent):
                comparisons += 1
            else:
                skipped += 1
    print(
        f"replotted training={training} comparisons={comparisons} skipped={skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
