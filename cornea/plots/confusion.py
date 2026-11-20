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


def _draw_matrix(ax, counts, names, *, percentages=False, predicted_names=None):
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
    predicted_names = names if predicted_names is None else predicted_names
    ax.set_xticks(range(len(predicted_names)), predicted_names)
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="Predicted label", ylabel="True label")
    ax.spines[:].set_visible(False)
    return image


@plt.rc_context(STYLE)
def plot_confusion_matrices(matrices, labels, output_dir):
    fig, axes = figure_grid(1, len(labels), (3.2, 3.4))
    for ax, label, matrix in zip(axes.flat, labels, matrices, strict=True):
        _draw_matrix(ax, matrix, BINARY_LABELS)
        ax.set_title(label)
    save_figure(fig, Path(output_dir) / "confusion_matrices.png")


@plt.rc_context(STYLE)
def plot_per_label_confusion_matrices(predictions, labels, output_dir, threshold=0.5):
    for freeze, group in predictions.groupby("freeze_backbone", sort=False):
        mode = MODE_NAMES[freeze]
        backbones = group.groupby("backbone", sort=False)
        fig, axes = figure_grid(
            len(backbones), len(labels), (3.55, 2.9),
            "Per-label confusion matrices",
        )
        fig.supxlabel(f"{mode}, threshold {threshold:g}", fontsize=9)
        for row, (backbone, part) in enumerate(backbones):
            for col, label in enumerate(labels):
                ax = axes[row, col]
                counts = confusion_matrix(part[label], part[f"{label}_prediction"], labels=[0, 1])
                image = _draw_matrix(ax, counts, BINARY_LABELS, percentages=True)
                if col == 0:
                    ax.set_ylabel(f"{backbone}\nTrue label", fontweight="bold")
                if row == 0:
                    ax.set_title(label.replace("_", " / ").title())
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Share within true label",
                     shrink=.8, format=PercentFormatter(1))
        save_figure(fig, Path(output_dir) / f"confusion_matrices_per_label_{mode.lower()}.png")


@plt.rc_context(STYLE)
def plot_sweep_confusion_matrices(predictions, labels, output_dir, threshold=0.5):
    output_dir = Path(output_dir)
    weights = 2 ** np.arange(len(labels))
    names = [label.replace("_", " / ").title() for label in labels]
    display = dict(zip(weights, names, strict=True))
    if {"fungal", "bacterial"}.issubset(labels):
        mixed = weights[labels.index("fungal")] + weights[labels.index("bacterial")]
        display[mixed] = "Mixed"
    decision_columns = [f"{label}_prediction" for label in labels]
    truth = predictions[labels].to_numpy(dtype=int) @ weights
    decisions = predictions[decision_columns].to_numpy(dtype=int) @ weights
    true_codes = [*display, *sorted(set(truth) - display.keys())]
    predicted_codes = [*display, *sorted(set(decisions) - display.keys())]
    for code in set(true_codes + predicted_codes) - display.keys():
        display[code] = " +\n".join(
            name for weight, name in zip(weights, names, strict=True) if code & weight
        ) or "No label"
    codes = list(dict.fromkeys(true_codes + predicted_codes))
    row_indices = [codes.index(code) for code in true_codes]
    column_indices = [codes.index(code) for code in predicted_codes]
    true_names = [display[code] for code in true_codes]
    predicted_names = [display[code] for code in predicted_codes]

    for freeze, group in predictions.groupby("freeze_backbone", sort=False):
        mode = MODE_NAMES[freeze]
        backbones = group.groupby("backbone", sort=False)
        columns = min(3, len(backbones))
        rows = (len(backbones) + columns - 1) // columns
        fig, axes = figure_grid(
            rows, columns, (max(5.2, 1.05 * len(predicted_codes)), 4.6),
            "Label-set confusion matrix" if len(backbones) == 1 else "Label-set confusion matrices",
        )
        fig.supxlabel(f"Decision threshold {threshold:g}", fontsize=9)
        for index, (backbone, part) in enumerate(backbones):
            ax = axes.flat[index]
            targets = part[labels].to_numpy(dtype=int) @ weights
            predicted = part[decision_columns].to_numpy(dtype=int) @ weights
            counts = confusion_matrix(targets, predicted, labels=codes)[np.ix_(row_indices, column_indices)]
            _draw_matrix(ax, counts, true_names, percentages=True, predicted_names=predicted_names)
            ax.tick_params(axis="x", labelsize=8)
            name = backbone.replace("resnet", "ResNet-") if backbone.startswith("resnet") else backbone
            ax.set_title(f"{name} ({mode.lower()})", fontsize=11)
        for ax in axes.ravel()[len(backbones):]:
            ax.set_visible(False)
        save_figure(fig, output_dir / f"confusion_matrices_{mode.lower()}.png")


@plt.rc_context(STYLE)
def plot_class_confusion_matrix(counts, labels, path, title):
    fig, axes = figure_grid(1, 1, (6.2, 5.6))
    ax = axes[0, 0]
    _draw_matrix(ax, counts, labels, percentages=True)
    ax.set_title(title)
    save_figure(fig, path)
