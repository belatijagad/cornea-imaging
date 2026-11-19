from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix

from .style import INK, MODE_NAMES, PAPER, STYLE, TEAL, figure_grid, save_figure


BINARY_LABELS = ("Absent", "Present")
MATRIX_COLORS = LinearSegmentedColormap.from_list(
    "cornea", [PAPER, "#aec7be", TEAL, "#183e38"]
)


def _draw_matrix(ax, counts, names, *, percentages=False):
    totals = counts.sum(axis=1, keepdims=True)
    if percentages:
        values = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    else:
        values = counts
    image = ax.imshow(
        values, cmap=MATRIX_COLORS, vmin=0, vmax=1 if percentages else None,
        interpolation="nearest",
    )
    for (row, col), count in np.ndenumerate(counts):
        text = str(count)
        value = values[row, col]
        if percentages:
            text += f"\n{value:.1%}" if totals[row, 0] else "\nN/A"
        ax.text(col, row, text, ha="center", va="center",
                color=PAPER if image.norm(value) >= 0.6 else INK)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="Predicted label", ylabel="True label")
    ax.spines[:].set_visible(False)
    return image


@plt.rc_context(STYLE)
def plot_confusion_matrices(matrices, labels, output_dir):
    fig, axes = figure_grid(1, len(labels), (3.2, 3.4))
    for ax, label, matrix in zip(axes.flat, labels, matrices):
        _draw_matrix(ax, matrix, BINARY_LABELS)
        ax.set_title(label)
    save_figure(fig, Path(output_dir) / "confusion_matrices.png")


@plt.rc_context(STYLE)
def plot_sweep_confusion_matrices(predictions, labels, output_dir, threshold=0.5):
    for freeze, group in predictions.groupby("freeze_backbone", sort=False):
        mode = MODE_NAMES[freeze]
        backbones = group.groupby("backbone", sort=False)
        fig, axes = figure_grid(
            len(backbones), len(labels), (3.55, 2.9),
            f"{mode} backbones · out-of-fold confusion matrices\nThreshold {threshold:g}",
        )
        for row, (backbone, part) in enumerate(backbones):
            for col, label in enumerate(labels):
                ax = axes[row, col]
                counts = confusion_matrix(part[label], part[f"{label}_prediction"], labels=[0, 1])
                image = _draw_matrix(ax, counts, BINARY_LABELS, percentages=True)
                if col == 0:
                    ax.set_ylabel(f"{backbone}\nTrue label", fontweight="bold")
                if row == 0:
                    ax.set_title(label)
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Share within true label",
                     shrink=.8, format=PercentFormatter(1))
        save_figure(fig, Path(output_dir) / f"confusion_matrices_{mode.lower()}.png", svg=True)


@plt.rc_context(STYLE)
def plot_three_class_confusion_matrices(predictions, labels, output_dir):
    output_dir = Path(output_dir)
    names = [label.replace("_", " / ").title() for label in labels]
    probabilities = [f"{label}_probability" for label in labels]
    for freeze, group in predictions.groupby("freeze_backbone", sort=False):
        mode = MODE_NAMES[freeze]
        backbones = group.groupby("backbone", sort=False)
        columns = min(3, len(backbones))
        rows = (len(backbones) + columns - 1) // columns
        fig, axes = figure_grid(rows, columns, (4.6, 4.1), f"{mode} backbones · multiclass confusion matrices")
        for ax, (backbone, part) in zip(axes.flat, backbones):
            cases = part.loc[part[labels].sum(axis=1) == 1]
            targets = cases[labels].to_numpy().argmax(axis=1)
            predicted = cases[probabilities].to_numpy().argmax(axis=1)
            counts = confusion_matrix(targets, predicted, labels=range(len(labels)))
            _draw_matrix(ax, counts, names, percentages=True)
            ax.set_title(backbone)
        for ax in axes.ravel()[len(backbones):]:
            ax.set_visible(False)
        save_figure(fig, output_dir / f"confusion_matrices_three_class_{mode.lower()}.png", svg=True)


@plt.rc_context(STYLE)
def plot_class_confusion_matrix(counts, labels, path, title):
    fig, axes = figure_grid(1, 1, (6.2, 5.6))
    ax = axes[0, 0]
    _draw_matrix(ax, counts, labels, percentages=True)
    ax.set_title(title)
    save_figure(fig, path, svg=True)
