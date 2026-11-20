from pathlib import Path

from .confusion import (
    plot_per_label_confusion_matrices,
    plot_sweep_confusion_matrices,
)
from .metrics import plot_experiment_comparison, plot_sweep_roc_curves


def plot_sweep(metrics, predictions, labels, output_dir, threshold):
    output_dir = Path(output_dir)
    plot_experiment_comparison(metrics, output_dir / "experiment_comparison.png", threshold)
    plot_sweep_confusion_matrices(predictions, labels, output_dir, threshold)
    plot_per_label_confusion_matrices(predictions, labels, output_dir, threshold)
    plot_sweep_roc_curves(predictions, labels, output_dir)
