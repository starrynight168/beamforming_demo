"""Project-wide matplotlib style, derived from the ``simulation_profiles`` figure.

Every plotting entry point should call :func:`apply_style` once, style its axes
with :func:`style_axes`, and export with :func:`save_figure`.  Keeping the tokens
in one module prevents the per-script drift that previously produced five
different font/line/tick systems across the paper figures and evaluation plots.

The numeric values (8.41 pt text, 0.788 pt rules, inward ticks, framed legend,
600 dpi PNG/SVG/PDF) are copied from ``train/figure/code/simulation_profiles.py``
so that migrating that figure is visually neutral.
"""

from __future__ import annotations

import functools
import re
import unicodedata
import uuid
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.text import Text

# Fonts are required, never substituted.  A missing family raises at apply
# time so a figure cannot silently change typeface.
FONT_FAMILY = ("Times New Roman",)
CJK_FONT_FAMILY = ("Times New Roman", "SimSun")

FONT_SIZE = 8.41
AXES_LINE_WIDTH = 0.788
TICK_LENGTH = 3.5
LINE_WIDTH = 1.1
MARKER_SIZE = 3.2

# Canvas presets.  Figures with a bespoke layout may still pass an explicit
# ``figsize``; the presets are the default vocabulary for new plots.
FIGSIZE_SINGLE = (3.54, 2.60)
FIGSIZE_DOUBLE = (7.16, 3.40)
FIGSIZE_WIDE = (9.5, 3.4)
FIGSIZE_TALL = (7.16, 5.25)
CANVAS = {
    "single": FIGSIZE_SINGLE,
    "double": FIGSIZE_DOUBLE,
    "wide": FIGSIZE_WIDE,
    "tall": FIGSIZE_TALL,
}

# Method colours shared by paper and evaluation figures.
METHOD_COLORS = {
    "GT": "#000000",
    "DAS": "#0057b8",
    "MV": "#8e44ad",
    "MBAN": "#d62728",
    "FP32": "#d62728",
    "QAT": "#e69f00",
    "PTQ": "#548235",
    "roi05": "#d62728",
}

RASTER_DPI = 600
DIAGNOSTIC_DPI = 300
OUTPUT_KINDS = ("png", "svg", "pdf")


@dataclass(frozen=True)
class StyleProfile:
    """Export settings for one class of figures."""

    name: str
    dpi: int
    formats: tuple[str, ...]


PAPER = StyleProfile(name="paper", dpi=RASTER_DPI, formats=OUTPUT_KINDS)
DIAGNOSTIC = StyleProfile(name="diagnostic", dpi=DIAGNOSTIC_DPI, formats=("png",))


def require_font_family(candidates: tuple[str, ...]) -> list[str]:
    """Return the requested family list or fail if any family is missing."""

    installed = {font.name for font in font_manager.fontManager.ttflist}
    missing = [name for name in candidates if name not in installed]
    if missing:
        raise RuntimeError(
            "缺少必需字体: "
            + ", ".join(missing)
            + "。本项目不做字体回退；请安装后再绘图。"
        )
    return list(candidates)


# Mathtext is rendered through STIX, not through ``font.family``; only the
# non-math parts of a label are checked against the text-font chain.
_MATHTEXT = re.compile(r"(?<!\\)\$(?:[^$\\]|\\.)*\$", re.DOTALL)
_NON_GLYPH_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp", "Zs"})


@functools.lru_cache(maxsize=None)
def _font_face(family: str, weight: str, style: str) -> FT2Font:
    properties = font_manager.FontProperties(
        family=[family], weight=weight, style=style
    )
    try:
        path = font_manager.findfont(properties, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            f"缺少必需字体: {family}。本项目不做字体回退；请安装后再绘图。"
        ) from exc
    return FT2Font(path)


def _family_chain(families: Sequence[str] | str | None) -> tuple[str, ...]:
    if families is None:
        families = mpl.rcParams["font.family"]
    if isinstance(families, str):
        families = (families,)
    chain = tuple(str(family) for family in families)
    if not chain:
        raise RuntimeError("字体链为空，无法检查字形。")
    return chain


def _needs_glyph(char: str) -> bool:
    return not char.isspace() and unicodedata.category(char) not in _NON_GLYPH_CATEGORIES


def _glyph_label(char: str) -> str:
    return f"U+{ord(char):04X} {char!r}"


def find_missing_glyphs(
    text: str,
    families: Sequence[str] | str | None = None,
    *,
    weight: str = "normal",
    style: str = "normal",
) -> list[str]:
    """Return the unique characters ``text`` cannot draw with ``families``.

    A character is supported when any family in the chain contains a glyph for
    it, which matches matplotlib's per-glyph font fallback. Mathtext segments
    are omitted because Matplotlib renders them through its mathtext fonts.
    """

    chain = _family_chain(families)
    faces = tuple(_font_face(family, weight, style) for family in chain)
    checked_text = _MATHTEXT.sub("", text)
    missing: list[str] = []
    seen: set[str] = set()
    for char in checked_text:
        if char in seen or not _needs_glyph(char):
            continue
        if not any(face.get_char_index(ord(char)) != 0 for face in faces):
            seen.add(char)
            missing.append(char)
    return missing


def require_glyphs(
    text: str,
    families: Sequence[str] | str | None = None,
    *,
    weight: str = "normal",
    style: str = "normal",
    context: str | None = None,
) -> None:
    """Fail if ``text`` contains a character none of ``families`` can draw."""

    chain = _family_chain(families)
    missing = find_missing_glyphs(text, chain, weight=weight, style=style)
    if not missing:
        return
    where = f"（{context}）" if context else ""
    raise RuntimeError(
        f"缺少字形{where}: "
        + ", ".join(_glyph_label(char) for char in missing)
        + f"；字体链 {'、'.join(chain)} 无法绘制这些字符。"
        + "请安装覆盖这些字符的字体或删除相应文本。"
    )


def _iter_text_artists(figure: plt.Figure) -> Iterator[Text]:
    yield from figure.texts
    for axis in figure.axes:
        if not axis.get_visible():
            continue
        yield from axis.texts
        yield axis.title
        yield from (
            title
            for title in (
                getattr(axis, "_left_title", None),
                getattr(axis, "_right_title", None),
            )
            if title is not None
        )
        for axis_object in (axis.xaxis, axis.yaxis):
            if not axis.axison or not axis_object.get_visible():
                continue
            yield axis_object.label
            yield axis_object.get_offset_text()
            for tick in (*axis_object.get_major_ticks(), *axis_object.get_minor_ticks()):
                yield tick.label1
                yield tick.label2
        legend = axis.get_legend()
        if legend is not None and legend.get_visible():
            yield from legend.get_texts()
            yield legend.get_title()
        for table in getattr(axis, "tables", ()):
            if not table.get_visible():
                continue
            for cell in table.get_celld().values():
                yield cell.get_text()
    for legend in figure.legends:
        if legend.get_visible():
            yield from legend.get_texts()
            yield legend.get_title()


def check_figure_glyphs(
    figure: plt.Figure,
    families: Sequence[str] | str | None = None,
) -> None:
    """Fail if any visible text in ``figure`` contains an unrenderable glyph.

    The check runs at export time, so it covers tick labels generated by
    formatters (including the Unicode minus) and every text artist, not just
    the font families requested by :func:`apply_style`.
    """

    if figure.canvas is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"Glyph \d+ .* missing from font"
            )
            figure.canvas.draw()
    missing: dict[str, str] = {}
    chains: set[str] = set()
    for text in _iter_text_artists(figure):
        if not isinstance(text, Text) or not text.get_visible() or text.get_usetex():
            continue
        content = text.get_text()
        if not content:
            continue
        chain = tuple(text.get_fontfamily()) or _family_chain(families)
        chains.update(chain)
        for char in find_missing_glyphs(
            content,
            chain,
            weight=text.get_fontweight(),
            style=text.get_fontstyle(),
        ):
            missing.setdefault(char, content)
    if not missing:
        return
    samples = "；".join(
        f"{_glyph_label(char)} 来自 {content[:48]!r}"
        for char, content in list(missing.items())[:3]
    )
    raise RuntimeError(
        "缺少字形: "
        + samples
        + f"；字体链 {'、'.join(sorted(chains))} 无法绘制这些字符。"
        + "请安装覆盖这些字符的字体或删除相应文本。"
    )


TOKENS: dict[str, object] = {
    "font.family": list(FONT_FAMILY),
    "font.size": FONT_SIZE,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "axes.linewidth": AXES_LINE_WIDTH,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": AXES_LINE_WIDTH,
    "ytick.major.width": AXES_LINE_WIDTH,
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "xtick.top": False,
    "ytick.right": False,
    "axes.grid": False,
    "mathtext.fontset": "stix",
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "legend.handlelength": 1.6,
    "legend.labelspacing": 0.35,
    "legend.borderpad": 0.35,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": RASTER_DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
}


def apply_style(
    profile: StyleProfile = PAPER,
    *,
    cjk: bool = False,
    weight: str | None = None,
    overrides: dict[str, object] | None = None,
) -> None:
    """Apply the shared matplotlib tokens.

    Parameters
    ----------
    profile:
        Export profile; only the ``savefig.dpi`` default is taken from it here,
        the explicit ``dpi`` argument of :func:`save_figure` always wins.
    cjk:
        Append Chinese fonts to the family list and disable the Unicode minus
        (some CJK fonts lack U+2212).  Latin text keeps Times New Roman.
    weight:
        Optional override for ``font.weight``/``axes.labelweight``/
        ``axes.titleweight`` (the hardware table keeps a bold look).
    overrides:
        Last-minute rcParams overrides for a documented exception.
    """

    tokens = dict(TOKENS)
    if cjk:
        tokens["font.family"] = require_font_family(CJK_FONT_FAMILY)
        tokens["axes.unicode_minus"] = False
    else:
        tokens["font.family"] = require_font_family(FONT_FAMILY)
    if weight is not None:
        tokens["font.weight"] = weight
        tokens["axes.labelweight"] = weight
        tokens["axes.titleweight"] = weight
    tokens["savefig.dpi"] = profile.dpi
    if overrides:
        tokens.update(overrides)
    mpl.rcParams.update(tokens)


def style_axes(
    axis: plt.Axes,
    *,
    box: bool = True,
    ticks: bool = True,
    y_ticks: bool = True,
    grid: bool = False,
    labelsize: float | None = None,
) -> None:
    """Apply the shared spine/tick/grid treatment to one axes.

    ``y_ticks=False`` keeps the y tick labels but removes the tick marks, which
    is what the reference figure does to keep a two-panel dB axis uncluttered.
    """

    for spine in axis.spines.values():
        spine.set_visible(box)
        spine.set_linewidth(AXES_LINE_WIDTH)
        spine.set_color("black")

    size = FONT_SIZE if labelsize is None else labelsize
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=TICK_LENGTH,
        width=AXES_LINE_WIDTH,
        colors="black",
        labelsize=size,
    )
    axis.tick_params(axis="x", top=False, bottom=ticks, labelbottom=True)
    axis.tick_params(
        axis="y",
        direction="in",
        right=False,
        left=ticks and y_ticks,
        labelleft=True,
    )

    if grid:
        axis.grid(True, color="#d9d9d9", linewidth=0.4, alpha=0.6)
    else:
        axis.grid(False)
    axis.set_axisbelow(True)


def style_legend(legend, *, fontsize: float | None = None) -> None:
    """Normalise a legend created with per-call settings."""

    legend.set_frame_on(True)
    frame = legend.get_frame()
    frame.set_edgecolor("black")
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    if fontsize is not None:
        for text in legend.get_texts():
            text.set_fontsize(fontsize)


def panel_label(
    axis: plt.Axes,
    letter: str,
    title: str | None = None,
    *,
    x: float = 0.0,
    y: float = 1.03,
) -> None:
    """Draw the ``(a) Title`` panel heading used by the paper figures."""

    text = f"({letter})"
    if title:
        text = f"{text} {title}"
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="normal",
    )


def save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    profile: StyleProfile = PAPER,
    dpi: int | None = None,
    formats: tuple[str, ...] | None = None,
    close: bool = True,
) -> tuple[Path, ...]:
    """Write one or more formats with an atomic replace for each target.

    ``path`` may be a stem (``.../simulation_profiles``) or a file name
    (``.../standard_metrics.png``).  When ``formats`` is omitted, a stem uses
    ``profile.formats`` and a suffixed name writes only that suffix.
    """

    path = Path(path)
    if formats is None:
        suffix = path.suffix.lstrip(".")
        formats = (suffix,) if suffix else profile.formats
    if not formats:
        raise ValueError("no output format selected")

    path.parent.mkdir(parents=True, exist_ok=True)
    targets: list[Path] = []
    try:
        check_figure_glyphs(fig)
        for extension in formats:
            target = path.with_suffix(f".{extension.lstrip('.')}")
            temporary = target.with_name(
                f".{target.stem}.{uuid.uuid4().hex}{target.suffix}"
            )
            try:
                fig.savefig(
                    temporary,
                    dpi=profile.dpi if dpi is None else dpi,
                    bbox_inches="tight",
                    facecolor="white",
                )
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            targets.append(target)
    finally:
        if close:
            plt.close(fig)
    return tuple(targets)


__all__ = [
    "AXES_LINE_WIDTH",
    "CANVAS",
    "CJK_FONT_FAMILY",
    "DIAGNOSTIC",
    "DIAGNOSTIC_DPI",
    "FONT_FAMILY",
    "FONT_SIZE",
    "FIGSIZE_DOUBLE",
    "FIGSIZE_SINGLE",
    "FIGSIZE_TALL",
    "FIGSIZE_WIDE",
    "LINE_WIDTH",
    "MARKER_SIZE",
    "METHOD_COLORS",
    "OUTPUT_KINDS",
    "PAPER",
    "RASTER_DPI",
    "StyleProfile",
    "TICK_LENGTH",
    "TOKENS",
    "apply_style",
    "check_figure_glyphs",
    "find_missing_glyphs",
    "panel_label",
    "require_font_family",
    "require_glyphs",
    "save_figure",
    "style_axes",
    "style_legend",
]
