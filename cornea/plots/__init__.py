from pathlib import Path

from .confusion import (
    BINARY_LABELS,
    MATRIX_COLORS,
    plot_class_confusion_matrix,
    plot_confusion_matrices,
    plot_sweep_confusion_matrices,
    plot_three_class_confusion_matrices,
)
from .metrics import plot_experiment_comparison, plot_specificity, plot_sweep_roc_curves
from .style import GRID, INK, MODE_NAMES, PAPER, SERIES_COLORS, STYLE, TEAL, TERRACOTTA, figure_grid, save_figure


def plot_sweep(metrics, predictions, labels, output_dir, threshold):
    output_dir = Path(output_dir)
    plot_experiment_comparison(metrics, output_dir / "experiment_comparison.png", threshold)
    plot_sweep_confusion_matrices(predictions, labels, output_dir, threshold)
    plot_sweep_roc_curves(predictions, labels, output_dir)
    plot_three_class_confusion_matrices(predictions, labels, output_dir)
