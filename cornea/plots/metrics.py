from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter
from sklearn.metrics import auc, roc_curve

from .style import (
    INK,
    MODE_NAMES,
    SERIES_COLORS,
    STYLE,
    TEAL,
    TERRACOTTA,
    figure_grid,
    save_figure,
)


@plt.rc_context(STYLE)
def plot_specificity(metrics, output_path, threshold=0.5, show=True):
    fig, axes = figure_grid(1, 1, (8, 4.8), "Specificity by epoch")
    ax = axes[0, 0]
    for label, rows in metrics.groupby("label", sort=False):
        ax.plot(rows["epoch"], rows["specificity"], marker="o", label=label)
    macro = metrics.groupby("epoch")["specificity"].mean()
    ax.plot(macro.index, macro, linestyle="--", color=INK, label="Macro average")
    ax.set(xlabel="Epoch", ylabel="Specificity", ylim=(0, 1.05), xticks=macro.index)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y")
    ax.legend(title=f"Decision threshold {threshold:g}")
    save_figure(fig, output_path, show=show)


@plt.rc_context(STYLE)
def plot_experiment_comparison(metrics, output_path, threshold=0.5):
    titles = {"specificity": "Specificity", "sensitivity": "Sensitivity", "auroc": "AUROC"}
    names = list(titles)
    folds = metrics.groupby(["backbone", "freeze_backbone", "fold"])[names].mean()
    summary = folds.groupby(level=["backbone", "freeze_backbone"])
    means, stds = summary.mean(), summary.std()
    order = means["specificity"].groupby(level="backbone").max().sort_values(ascending=False).index
    lower_bound = (means - stds.fillna(0)).min().min()
    lower_limit = np.clip(np.floor((lower_bound - .02) * 10) / 10, 0, .65) if pd.notna(lower_bound) else 0
    modes = means.index.get_level_values("freeze_backbone")
    fig, axes = figure_grid(1, len(titles), (4.3, max(3.8, .42 * len(order) + 1.8)),
                        "Cross-validation performance", sharey=True)
    positions = np.arange(len(order))
    for freeze, color, marker, offset in [(True, TEAL, "o", -.14), (False, TERRACOTTA, "D", .14)]:
        if freeze not in modes:
            continue
        centers = means.xs(freeze, level="freeze_backbone").reindex(order)
        spreads = stds.xs(freeze, level="freeze_backbone").reindex(order)
        for ax, name in zip(axes.flat, names, strict=True):
            ax.errorbar(
                centers[name].to_numpy(), positions + offset, xerr=spreads[name].to_numpy(),
                fmt=marker, color=color, capsize=3, label=MODE_NAMES[freeze],
            )
    axes[0, 0].set(yticks=positions, yticklabels=order, ylim=(len(order) - .5, -.5))
    for ax, title in zip(axes.flat, titles.values(), strict=True):
        ax.set(title=title, xlim=(lower_limit, 1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x")
        ax.spines["left"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    save_figure(fig, output_path)


@plt.rc_context(STYLE)
def plot_sweep_roc_curves(predictions, labels, output_dir):
    output_dir = Path(output_dir)
    modes = predictions.groupby("freeze_backbone", sort=True)
    backbones = predictions["backbone"].unique()
    colors = {name: SERIES_COLORS[i % len(SERIES_COLORS)] for i, name in enumerate(backbones)}
    fig, axes = figure_grid(len(modes), len(labels), (4.5, 4.5), "ROC curves")
    for row, (freeze, group) in enumerate(modes):
        parts = {name: part for name, part in group.groupby("backbone", sort=False)}
        for col, label in enumerate(labels):
            ax = axes[row, col]
            for backbone in backbones:
                part = parts.get(backbone)
                if part is None or part[label].nunique() < 2:
                    continue
                fpr, tpr, _ = roc_curve(part[label], part[f"{label}_probability"])
                area = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[backbone], label=f"{backbone} ({area:.3f})")
            if ax.lines:
                ax.legend(loc="lower right", fontsize=8, title="AUROC")
            ax.plot([0, 1], [0, 1], "--", color="#8d8578", linewidth=1, zorder=1)
            title = label.replace("_", " / ").title()
            ax.set(title=f"{title} ({MODE_NAMES[freeze].lower()})", xlabel="False positive rate",
                   ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1), aspect="equal")
            ax.grid()
    save_figure(fig, output_dir / "roc_curves.png")
