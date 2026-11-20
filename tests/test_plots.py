import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from cornea.plots.confusion import (
    _draw_matrix,
    plot_per_label_confusion_matrices,
    plot_sweep_confusion_matrices,
)
from cornea.plots.metrics import plot_experiment_comparison
from cornea.plots.style import save_figure


class PlotTests(unittest.TestCase):
    def test_comparison_averages_labels_within_folds(self):
        metrics = pd.DataFrame([
            {"backbone": "resnet18", "freeze_backbone": False, "fold": fold,
             "specificity": value, "sensitivity": value, "auroc": value}
            for fold, value in [(1, .6), (1, 1.0), (2, .2)]
        ])
        figures = []
        close = plt.close

        def capture(fig):
            if hasattr(fig, "axes"):
                figures.append(fig)
            close(fig)

        with tempfile.TemporaryDirectory() as directory, patch("matplotlib.pyplot.close", side_effect=capture):
            path = Path(directory) / "comparison.png"
            plot_experiment_comparison(metrics, path, threshold=.7)
            with Image.open(path) as image:
                image.verify()
            self.assertEqual(list(Path(directory).iterdir()), [path])
        fig = figures[-1]
        self.assertEqual(len(fig.axes), 3)
        for ax in fig.axes:
            np.testing.assert_allclose(ax.lines[0].get_xdata(), [.5])
        self.assertEqual([text.get_text() for text in fig.legends[0].texts], ["Unfrozen"])

    def test_comparison_exports_multiple_backbones_and_freeze_modes(self):
        metrics = pd.DataFrame([
            {"backbone": backbone, "freeze_backbone": freeze, "fold": fold,
             "specificity": score, "sensitivity": score - .1, "auroc": score + .05}
            for backbone, baseline in [("resnet18", .7), ("swin_t", .8)]
            for freeze, gain in [(True, .02), (False, .04)]
            for fold, score in [(1, baseline + gain), (2, baseline + gain + .01)]
        ])
        with tempfile.TemporaryDirectory() as directory, patch(
            "cornea.plots.metrics.save_figure", wraps=save_figure,
        ) as save:
            path = Path(directory) / "comparison.png"
            plot_experiment_comparison(metrics, path)
            with Image.open(path) as image:
                image.verify()
        fig, _ = save.call_args.args
        self.assertEqual({label.get_text() for label in fig.axes[0].get_yticklabels()}, {"resnet18", "swin_t"})
        self.assertEqual({label.get_text() for label in fig.legends[0].texts}, {"Frozen", "Unfrozen"})

    def test_confusion_matrix_counts_and_row_percentages(self):
        predictions = pd.DataFrame({
            "backbone": ["resnet18"] * 4,
            "freeze_backbone": [True] * 4,
            "fungal": [0, 0, 1, 1],
            "fungal_prediction": [0, 1, 1, 1],
            "bacterial": [0, 0, 0, 0],
            "bacterial_prediction": [0, 0, 1, 1],
        })
        figures = []
        close = plt.close

        def capture(fig):
            if hasattr(fig, "axes"):
                figures.append(fig)
            close(fig)

        with tempfile.TemporaryDirectory() as directory, patch("matplotlib.pyplot.close", side_effect=capture):
            plot_per_label_confusion_matrices(predictions, ["fungal", "bacterial"], directory)
            path = Path(directory) / "confusion_matrices_per_label_frozen.png"
            with Image.open(path) as image:
                image.verify()
            self.assertFalse((Path(directory) / "confusion_matrices_per_label_unfrozen.png").exists())
        fig = figures[-1]
        self.assertEqual([text.get_text() for text in fig.axes[0].texts],
                         ["1\n50.0%", "1\n50.0%", "0\n0.0%", "2\n100.0%"])
        self.assertEqual([text.get_text() for text in fig.axes[1].texts][-2:], ["0\nN/A", "0\nN/A"])
        self.assertEqual(fig.axes[0].images[0].get_clim(), (0, 1))

    def test_default_confusion_matrix_includes_mixed_and_every_prediction(self):
        labels = ["fungal", "bacterial", "viral_other"]
        predictions = pd.DataFrame(
            [(1, 0, 0, 1, 1, 0), (0, 1, 0, 0, 1, 1), (0, 0, 1, 0, 0, 0),
             (1, 1, 0, 1, 1, 0), (1, 1, 0, 0, 1, 0)],
            columns=labels + [f"{label}_prediction" for label in labels],
        ).assign(backbone="resnet18", freeze_backbone=False)
        with tempfile.TemporaryDirectory() as directory, patch(
            "cornea.plots.confusion._draw_matrix", wraps=_draw_matrix,
        ) as draw:
            plot_sweep_confusion_matrices(predictions, labels, directory)
            self.assertEqual(list(Path(directory).iterdir()), [Path(directory) / "confusion_matrices_unfrozen.png"])
        _, matrix, names = draw.call_args.args
        predicted_names = draw.call_args.kwargs["predicted_names"]
        self.assertEqual(names, ["Fungal", "Bacterial", "Viral / Other", "Mixed"])
        self.assertEqual(predicted_names, [*names, "No label", "Bacterial +\nViral / Other"])
        self.assertEqual(matrix.sum(), len(predictions))
        np.testing.assert_array_equal(matrix[names.index("Mixed")], [0, 1, 0, 1, 0, 0])
