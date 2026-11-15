"""可视化中心化、标准化与归一化（Min-Max）的区别。

运行：
    python -m tools.code.normalization_demo

输出：
    tools/output/normalization_demo/normalization_demo.png

可用 --output 指定输出文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tools.shared.run_output import ToolOutput, save_figure


def main(argv=None):
    parser = argparse.ArgumentParser(description="可视化中心化、标准化与归一化")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output_dir is not None:
        parser.error("--output 与 --output-dir 不能同时使用")
    if args.output is not None:
        custom_output = args.output.expanduser().resolve()
        output = ToolOutput("normalization_demo", custom_output.parent)
        figure_path = custom_output
    else:
        output = ToolOutput("normalization_demo", args.output_dir)
        figure_path = output.path("figures", "normalization_demo.png")
    # ---------------------------------------------------------------------------
    # 1. 造一批二维点：两个特征量纲不同、且正相关
    # ---------------------------------------------------------------------------
    rng = np.random.default_rng(20261104)
    N = 1500
    MEAN_X, STD_X = 62.0, 13.5
    MEAN_Y, STD_Y = 29.0, 4.5
    RHO = 0.78

    z1 = rng.standard_normal(N)
    z2 = rng.standard_normal(N)
    x = MEAN_X + STD_X * z1
    y = MEAN_Y + STD_Y * (RHO * z1 + np.sqrt(1.0 - RHO**2) * z2)
    raw = np.column_stack([x, y])

    mu = raw.mean(axis=0)
    sigma = raw.std(axis=0, ddof=0)
    lo = raw.min(axis=0)
    hi = raw.max(axis=0)
    span = hi - lo

    transforms = {
        "raw": raw.copy(),
        "center": raw - mu,
        "standard": (raw - mu) / sigma,
        "minmax": (raw - lo) / span,
    }

    # 每一列共用的横轴范围：让“中心化前后跨度不变”肉眼可比
    raw_xlim = (lo[0] - 0.08 * span[0], hi[0] + 0.08 * span[0])
    raw_ylim = (lo[1] - 0.08 * span[1], hi[1] + 0.08 * span[1])
    xlims = {
        "raw": raw_xlim,
        "center": (
            -0.5 * (raw_xlim[1] - raw_xlim[0]),
            0.5 * (raw_xlim[1] - raw_xlim[0]),
        ),
        "standard": (-4.1, 4.1),
        "minmax": (-0.09, 1.09),
    }
    ylims = {
        "raw": raw_ylim,
        "center": (
            -0.5 * (raw_ylim[1] - raw_ylim[0]),
            0.5 * (raw_ylim[1] - raw_ylim[0]),
        ),
        "standard": (-4.1, 4.1),
        "minmax": (-0.09, 1.09),
    }

    def one_decimal(v: float) -> str:
        """避免出现 -0.0 这种不好看的标注。"""
        return "0.0" if abs(v) < 0.05 else f"{v:.1f}"

    def two_decimals(v: float) -> str:
        return "0.00" if abs(v) < 0.005 else f"{v:.2f}"

    # ---------------------------------------------------------------------------
    # 2. 配色与版式
    # ---------------------------------------------------------------------------
    BG = "#F7F5EF"
    PANEL = "#FFFEFB"
    INK = "#1F2933"
    MUTED = "#667085"
    GRID = "#E4E7EC"
    BLUE = "#2563EB"
    ORANGE = "#D9480F"
    POINT = "#1F2937"

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": "#D0D5DD",
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )

    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[2.4, 1.15],
        left=0.046,
        right=0.975,
        bottom=0.160,
        top=0.830,
        wspace=0.17,
        hspace=0.30,
    )

    # ---------------------------------------------------------------------------
    # 3. 标题与图例
    # ---------------------------------------------------------------------------
    fig.text(
        0.046,
        0.968,
        "中心化 / 标准化 / 归一化：到底改变了什么？",
        fontsize=26,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.048,
        0.922,
        "同一组 1,500 个二维样本  ·  两个特征分别处理  ·  上排看点云，下排看分布",
        fontsize=11.5,
        color=MUTED,
        ha="left",
        va="top",
    )
    legend_handles = [
        Patch(facecolor=BLUE, alpha=0.45, edgecolor=BLUE, label="特征 X 的分布"),
        Patch(facecolor=ORANGE, alpha=0.45, edgecolor=ORANGE, label="特征 Y 的分布"),
        Line2D(
            [0],
            [0],
            marker="+",
            color=ORANGE,
            markersize=11,
            linestyle="None",
            label="特征均值",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.976, 0.958),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        columnspacing=1.5,
    )

    # ---------------------------------------------------------------------------
    # 4. 四个变换的标题、公式、统计量与一句话解释
    # ---------------------------------------------------------------------------
    columns = [
        {
            "key": "raw",
            "title": "原始数据",
            "formula": r"$x,\ y$  （保持原样）",
            "note": "量纲不同：X 跨度大、Y 跨度小，大尺度特征容易主导距离。",
            "stats": (
                f"特征 X   μ = {one_decimal(mu[0])}   σ = {one_decimal(sigma[0])}\n"
                f"特征 Y   μ = {one_decimal(mu[1])}   σ = {one_decimal(sigma[1])}"
            ),
        },
        {
            "key": "center",
            "title": "中心化",
            "formula": r"$x' = x-\mu$",
            "note": "只平移：均值归零，离散程度与分布形状原样保留。",
            "stats": (
                f"特征 X   μ = 0.0   σ = {one_decimal(sigma[0])}\n"
                f"特征 Y   μ = 0.0   σ = {one_decimal(sigma[1])}"
            ),
        },
        {
            "key": "standard",
            "title": "标准化",
            "formula": r"$z = (x-\mu)/\sigma$",
            "note": "再除以标准差：均值 0、标准差 1，两个特征尺度终于可比。",
            "stats": "特征 X   μ = 0.00   σ = 1.00\n特征 Y   μ = 0.00   σ = 1.00",
        },
        {
            "key": "minmax",
            "title": "归一化（Min–Max）",
            "formula": r"$x' = (x-\min)\,/\,(\max-\min)$",
            "note": "压到 [0, 1]：均值/标准差不再固定，而且结果受极值影响。",
            "stats": (
                f"特征 X   μ = {two_decimals(transforms['minmax'][:, 0].mean())}"
                f"   σ = {two_decimals(transforms['minmax'][:, 0].std(ddof=0))}\n"
                f"特征 Y   μ = {two_decimals(transforms['minmax'][:, 1].mean())}"
                f"   σ = {two_decimals(transforms['minmax'][:, 1].std(ddof=0))}"
            ),
        },
    ]

    for col, spec in enumerate(columns):
        key = spec["key"]
        data = transforms[key]

        # ---------------- 上排：二维点云 ----------------
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(PANEL)
        ax.set_xlim(xlims[key])
        ax.set_ylim(ylims[key])
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        # 零轴（中心化、标准化后尤其重要）
        if xlims[key][0] < 0 < xlims[key][1]:
            ax.axvline(
                0, color="#98A2B3", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1
            )
        if ylims[key][0] < 0 < ylims[key][1]:
            ax.axhline(
                0, color="#98A2B3", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1
            )

        ax.scatter(
            data[:, 0],
            data[:, 1],
            s=9,
            c=POINT,
            alpha=0.38,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )
        ax.scatter(
            data[:, 0].mean(),
            data[:, 1].mean(),
            marker="+",
            s=200,
            c=ORANGE,
            linewidths=2.0,
            zorder=3,
        )

        ax.set_title(
            spec["title"],
            loc="left",
            fontsize=16,
            fontweight="bold",
            color=INK,
            pad=31,
        )
        ax.text(
            0.0,
            1.012,
            spec["formula"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11.5,
            color=ORANGE,
        )
        ax.text(
            0.03,
            0.965,
            spec["stats"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.4,
            color="#344054",
            linespacing=1.55,
            bbox={
                "boxstyle": "round,pad=0.45,rounding_size=0.35",
                "facecolor": "white",
                "edgecolor": "#EAECF0",
                "alpha": 0.90,
            },
            zorder=4,
        )

        ax.set_xlabel("特征 X", fontsize=9.5, labelpad=6)
        if col == 0:
            ax.set_ylabel("特征 Y", fontsize=9.5, labelpad=7)

        # ---------------- 下排：两个特征的分布 ----------------
        hist_ax = fig.add_subplot(gs[1, col])
        hist_ax.set_facecolor(PANEL)
        hist_ax.set_xlim(xlims[key])
        hist_ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.9)
        hist_ax.set_axisbelow(True)
        for side in ("top", "right"):
            hist_ax.spines[side].set_visible(False)

        hx = hist_ax.hist(
            data[:, 0],
            bins=55,
            density=True,
            histtype="stepfilled",
            color=BLUE,
            alpha=0.30,
            edgecolor=BLUE,
            linewidth=1.0,
            zorder=2,
        )
        hy = hist_ax.hist(
            data[:, 1],
            bins=55,
            density=True,
            histtype="stepfilled",
            color=ORANGE,
            alpha=0.30,
            edgecolor=ORANGE,
            linewidth=1.0,
            zorder=3,
        )

        x_mean = data[:, 0].mean()
        y_mean = data[:, 1].mean()
        hist_ax.axvline(
            x_mean, color=BLUE, linestyle=(0, (4, 2)), linewidth=1.4, zorder=4
        )
        hist_ax.axvline(
            y_mean, color=ORANGE, linestyle=(0, (4, 2)), linewidth=1.4, zorder=4
        )

        # 归一化时标出 0 和 1 的边界
        if key == "minmax":
            for edge in (0.0, 1.0):
                hist_ax.axvline(
                    edge, color="#98A2B3", linewidth=0.8, linestyle=":", zorder=1
                )
            hist_ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # 给分布图留一点顶部空间
        ymax = max(hx[0].max(), hy[0].max())
        hist_ax.set_ylim(0, ymax * 1.18)

        hist_ax.text(
            0.0,
            1.105,
            spec["note"],
            transform=hist_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#475467",
            clip_on=False,
        )
        hist_ax.set_xlabel("特征值", fontsize=9, labelpad=5)
        if col == 0:
            hist_ax.set_ylabel("密度", fontsize=9, labelpad=7)

    # ---------------------------------------------------------------------------
    # 5. 底部总结
    # ---------------------------------------------------------------------------
    fig.add_artist(
        Line2D(
            [0.046, 0.975],
            [0.116, 0.116],
            transform=fig.transFigure,
            color="#D0D5DD",
            linewidth=0.8,
        )
    )
    fig.text(
        0.046,
        0.083,
        "一句话：中心化只改位置；标准化改位置和尺度；归一化把范围压到 [0, 1]。",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
    )
    fig.text(
        0.046,
        0.044,
        "注意：标准化不保证落在 [0, 1]；归一化不保证均值为 0、标准差为 1；两者都应逐特征（逐列）进行。",
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="center",
    )

    output.config({"tool": "normalization_demo", "seed": 20261104, "samples": N})
    save_figure(fig, figure_path, dpi=300)
    summary_rows = []
    for name, values in transforms.items():
        for feature_index, feature in enumerate(("X", "Y")):
            column = values[:, feature_index]
            summary_rows.append(
                {
                    "transform": name,
                    "feature": feature,
                    "mean": float(np.mean(column)),
                    "std": float(np.std(column)),
                    "min": float(np.min(column)),
                    "max": float(np.max(column)),
                },
            )
    output.csv("metrics", "normalization_summary.csv", summary_rows)
    output.log("done", f"说明图和统计表已保存到 {output.root}。")


if __name__ == "__main__":
    main()
