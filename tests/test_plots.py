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

from cornea.plots import plot_experiment_comparison, plot_sweep_confusion_matrices


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
            self.assertTrue(path.with_suffix(".svg").is_file())
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            plot_experiment_comparison(metrics, path)
            with Image.open(path) as image:
                image.verify()
            svg = path.with_suffix(".svg").read_text()
        for label in ("resnet18", "swin_t", "Frozen", "Unfrozen"):
            self.assertIn(label, svg)

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
            plot_sweep_confusion_matrices(predictions, ["fungal", "bacterial"], directory)
            path = Path(directory) / "confusion_matrices_frozen.png"
            with Image.open(path) as image:
                image.verify()
            self.assertFalse((Path(directory) / "confusion_matrices_unfrozen.png").exists())
        fig = figures[-1]
        self.assertEqual([text.get_text() for text in fig.axes[0].texts],
                         ["1\n50.0%", "1\n50.0%", "0\n0.0%", "2\n100.0%"])
        self.assertEqual([text.get_text() for text in fig.axes[1].texts][-2:], ["0\nN/A", "0\nN/A"])
        self.assertEqual(fig.axes[0].images[0].get_clim(), (0, 1))
