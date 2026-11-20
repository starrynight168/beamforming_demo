import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_core import plot_style
from tools.shared.run_output import ToolOutput


class PlotStyleTests(unittest.TestCase):
    def setUp(self):
        self._rcparams = matplotlib.rcParams.copy()
        self.addCleanup(matplotlib.rcParams.update, self._rcparams)

    def test_paper_profile_matches_reference_tokens(self):
        plot_style.apply_style()

        self.assertEqual(matplotlib.rcParams["font.family"][0], "Times New Roman")
        self.assertAlmostEqual(float(matplotlib.rcParams["font.size"]), 8.41)
        self.assertAlmostEqual(float(matplotlib.rcParams["axes.linewidth"]), 0.788)
        self.assertAlmostEqual(float(matplotlib.rcParams["xtick.major.size"]), 3.5)
        self.assertEqual(matplotlib.rcParams["xtick.direction"], "in")
        self.assertEqual(matplotlib.rcParams["ytick.direction"], "in")
        self.assertTrue(matplotlib.rcParams["axes.spines.top"])
        self.assertTrue(matplotlib.rcParams["axes.spines.right"])
        self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
        self.assertEqual(matplotlib.rcParams["svg.fonttype"], "none")
        self.assertEqual(matplotlib.rcParams["mathtext.fontset"], "stix")
        self.assertFalse(matplotlib.rcParams["axes.grid"])
        self.assertEqual(matplotlib.rcParams["savefig.dpi"], 600)
        self.assertEqual(plot_style.PAPER.formats, ("png", "svg", "pdf"))

    def test_diagnostic_profile_uses_single_format_and_300_dpi(self):
        plot_style.apply_style(plot_style.DIAGNOSTIC)

        self.assertEqual(matplotlib.rcParams["savefig.dpi"], 300)
        self.assertEqual(plot_style.DIAGNOSTIC.formats, ("png",))

    def test_cjk_profile_requires_times_and_simsun(self):
        plot_style.apply_style(cjk=True)

        families = list(matplotlib.rcParams["font.family"])
        self.assertEqual(families, ["Times New Roman", "SimSun"])
        self.assertFalse(matplotlib.rcParams["axes.unicode_minus"])

    def test_missing_font_raises_instead_of_falling_back(self):
        with self.assertRaisesRegex(RuntimeError, "Definitely Missing Font"):
            plot_style.require_font_family(("Definitely Missing Font",))

    def test_missing_font_family_raises_on_apply(self):
        previous = plot_style.FONT_FAMILY
        plot_style.FONT_FAMILY = ("Definitely Missing Font",)
        try:
            with self.assertRaisesRegex(RuntimeError, "Definitely Missing Font"):
                plot_style.apply_style()
        finally:
            plot_style.FONT_FAMILY = previous

    def test_style_axes_sets_box_inward_ticks_and_optional_y_marks(self):
        fig, axis = plt.subplots()
        self.addCleanup(plt.close, fig)
        plot_style.style_axes(axis, y_ticks=False)

        for spine in axis.spines.values():
            self.assertTrue(spine.get_visible())
            self.assertAlmostEqual(spine.get_linewidth(), 0.788)
        x_params = axis.xaxis.get_tick_params(which="major")
        self.assertEqual(x_params["direction"], "in")
        self.assertAlmostEqual(x_params["length"], 3.5)
        self.assertAlmostEqual(x_params["width"], 0.788)
        self.assertFalse(x_params["top"])
        y_tick = axis.yaxis.get_major_ticks()[0]
        self.assertFalse(y_tick.tick1line.get_visible())
        self.assertTrue(y_tick.label1.get_visible())

    def test_style_axes_without_box_hides_spines(self):
        fig, axis = plt.subplots()
        self.addCleanup(plt.close, fig)
        plot_style.style_axes(axis, box=False)

        for spine in axis.spines.values():
            self.assertFalse(spine.get_visible())

    def test_save_figure_writes_all_paper_formats_and_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            fig, axis = plt.subplots()
            axis.plot([0, 1], [0, 1])
            paths = plot_style.save_figure(fig, Path(directory) / "demo")

            self.assertEqual({path.suffix for path in paths}, {".png", ".svg", ".pdf"})
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            self.assertFalse(plt.fignum_exists(fig.number))

    def test_save_figure_suffixed_path_writes_only_that_format(self):
        with tempfile.TemporaryDirectory() as directory:
            fig, axis = plt.subplots()
            paths = plot_style.save_figure(fig, Path(directory) / "metric.png")

            self.assertEqual([path.suffix for path in paths], [".png"])
            self.assertTrue(paths[0].is_file())

    def test_tool_output_uses_diagnostic_dpi(self):
        plot_style.apply_style(plot_style.DIAGNOSTIC)
        with tempfile.TemporaryDirectory() as directory:
            fig, axis = plt.subplots(figsize=(2, 2))
            axis.plot([0, 1], [0, 1])
            path = ToolOutput("plot_test", directory).figure("plot.png", fig)
            with Image.open(path) as image:
                self.assertAlmostEqual(image.info["dpi"][0], 300, delta=1)

    def test_panel_label_uses_reference_typography(self):
        fig, axis = plt.subplots()
        self.addCleanup(plt.close, fig)
        plot_style.panel_label(axis, "a", "Image similarity")

        text = axis.texts[0]
        self.assertEqual(text.get_text(), "(a) Image similarity")
        self.assertAlmostEqual(text.get_fontsize(), 8.41)
        self.assertEqual(text.get_fontweight(), "normal")

    def test_plotting_entry_points_do_not_set_global_style(self):
        root = Path(__file__).resolve().parents[1]
        offenders = []
        candidates = list((root / "train" / "figure" / "code").glob("*.py"))
        candidates += [root / "evaluation" / "plot_metrics.py"]
        for path in candidates:
            text = path.read_text(encoding="utf-8")
            if "rcParams.update" in text or "rcParams[" in text:
                offenders.append(path.name)

        self.assertEqual(offenders, [])

    def test_require_glyphs_accepts_characters_covered_by_the_chain(self):
        plot_style.require_glyphs("ABC 0123", ("Times New Roman",))
        plot_style.require_glyphs("中文", ("Times New Roman", "SimSun"))

    def test_require_glyphs_rejects_a_character_missing_from_every_family(self):
        with self.assertRaisesRegex(RuntimeError, "U\\+10FFFD"):
            plot_style.require_glyphs("\U0010FFFD", ("Times New Roman", "SimSun"))

    def test_require_glyphs_uses_any_family_in_the_chain(self):
        with self.assertRaisesRegex(RuntimeError, "缺少字形"):
            plot_style.require_glyphs("中文", ("Times New Roman",))
        plot_style.require_glyphs("中文", ("Times New Roman", "SimSun"))

    def test_require_glyphs_ignores_mathtext_segments(self):
        plot_style.require_glyphs(r"$\alpha + \beta = \gamma$", ("Times New Roman",))
        with self.assertRaisesRegex(RuntimeError, "缺少字形"):
            plot_style.require_glyphs(r"$\alpha$ 中文", ("Times New Roman",))

    def test_check_figure_glyphs_scans_titles_labels_and_tick_text(self):
        plot_style.apply_style()
        fig, axis = plt.subplots()
        self.addCleanup(plt.close, fig)
        axis.set_title("中文")
        axis.set_xlabel("Lateral (mm)")
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["fine", "ok"])

        with self.assertRaisesRegex(RuntimeError, "缺少字形"):
            plot_style.check_figure_glyphs(fig)

    def test_check_figure_glyphs_skips_hidden_text(self):
        plot_style.apply_style()
        fig, axis = plt.subplots()
        self.addCleanup(plt.close, fig)
        axis.text(0.5, 0.5, "\U0010FFFD", visible=False)

        plot_style.check_figure_glyphs(fig)

    def test_check_figure_glyphs_skips_hidden_axes_and_legend(self):
        plot_style.apply_style()
        fig, (hidden, visible) = plt.subplots(1, 2)
        self.addCleanup(plt.close, fig)
        hidden.set_title("中文")
        hidden.set_visible(False)
        visible.plot([0, 1], [0, 1], label="中文")
        visible.legend().set_visible(False)
        visible.set_xlabel("中文")
        visible.xaxis.set_visible(False)
        visible.set_ylabel("中文")
        visible.set_axis_off()

        plot_style.check_figure_glyphs(fig)

    def test_save_figure_refuses_missing_glyph_before_writing(self):
        plot_style.apply_style()
        with tempfile.TemporaryDirectory() as directory:
            fig, axis = plt.subplots()
            axis.set_title("中文")
            with self.assertRaisesRegex(RuntimeError, "缺少字形"):
                plot_style.save_figure(fig, Path(directory) / "missing")

            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertFalse(plt.fignum_exists(fig.number))


if __name__ == "__main__":
    unittest.main()
