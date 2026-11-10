"""Plot generation for training histories and experiment comparisons."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_specificity(metrics, output_dir, threshold=0.5, show=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rows in metrics.groupby("label", sort=False):
        ax.plot(rows["epoch"], rows["specificity"], marker="o", label=label)
    macro = metrics.groupby("epoch")["specificity"].mean()
    ax.plot(macro.index, macro.values, linestyle="--", color="black", label="Macro average")
    ax.set(xlabel="Epoch", ylabel="Specificity", ylim=(0, 1.05),
           title=f"Validation specificity (threshold = {threshold:g})")
    ax.set_xticks(macro.index)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "specificity.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrices(matrices, labels, output_dir):
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), squeeze=False)
    for ax, label, matrix in zip(axes.ravel(), labels, matrices):
        ConfusionMatrixDisplay(matrix, display_labels=["Absent", "Present"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_experiment_comparison(metrics, output_path, threshold=0.5, dummy=False):
    """Compare fold results, averaging labels within each fold before folds."""
    metric_names = ["specificity", "sensitivity", "auroc"]
    data = metrics.groupby(["backbone", "freeze_backbone", "fold"], as_index=False)[metric_names].mean()
    data["mode"] = data["freeze_backbone"].map({True: "Frozen", False: "Unfrozen"})
    summary = data.groupby(["backbone", "mode"])[metric_names].agg(["mean", "std"])
    order = summary["specificity"]["mean"].groupby(level="backbone").max().sort_values(ascending=False).index.tolist()
    colors = {"Frozen": "#1f77b4", "Unfrozen": "#ff7f0e"}
    markers = {"Frozen": "o", "Unfrozen": "D"}
    modes = [mode for mode in colors if mode in set(data["mode"])]
    bounds = [(summary[name]["mean"] - summary[name]["std"].fillna(0)).min() for name in metric_names]
    lower_bound = min((value for value in bounds if np.isfinite(value)), default=0)
    lower_limit = max(0, min(.65, np.floor((lower_bound - .02) * 10) / 10))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "bold", "axes.edgecolor": "#ccd3da",
                         "text.color": "#243447", "axes.labelcolor": "#46576b",
                         "xtick.color": "#64748b", "ytick.color": "#334155",
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
        fig.subplots_adjust(left=.12, right=.97, top=.70, bottom=.20, wspace=.20)
        fig.text(.055, .935, "Backbone × freeze comparison", fontsize=25, fontweight="bold", va="top")
        fig.text(.055, .88, f"Macro scores · fold mean ± SD · threshold = {threshold:g}",
                 fontsize=12, color="#64748b")
        legend = [Line2D([], [], marker=markers[mode], color=colors[mode], linestyle="none",
                         markersize=8, label=mode) for mode in modes]
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.05, .845), ncol=2,
                   frameon=False, columnspacing=2)
        for ax, metric, title in zip(axes, metric_names, ["Specificity", "Sensitivity", "AUROC"]):
            ax.set_title(title, loc="left", pad=19, fontsize=15)
            for y in range(len(order)):
                if y % 2 == 0:
                    ax.axhspan(y-.46, y+.46, color="#f5f7fa", zorder=0)
            for mode, offset in [("Frozen", -.14), ("Unfrozen", .14)]:
                if mode not in modes:
                    continue
                means = summary[metric]["mean"].reindex([(name, mode) for name in order]).to_numpy()
                stds = summary[metric]["std"].reindex([(name, mode) for name in order]).to_numpy()
                ax.errorbar(means, np.arange(len(order)) + offset, xerr=stds,
                            fmt=markers[mode], color=colors[mode], markersize=7, capsize=4,
                            elinewidth=1.6, markeredgecolor="white", markeredgewidth=.7, zorder=3)
            ax.set(xlim=(lower_limit, 1.0), ylim=(len(order)-.55, -.55), xlabel="Score · higher is better")
            ax.set_xticks(np.arange(np.ceil(lower_limit * 10) / 10, 1.01, .1))
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(axis="x", alpha=.24, linewidth=.8)
            ax.tick_params(axis="both", length=0, pad=10)
            for side in ["top", "right", "left"]:
                ax.spines[side].set_visible(False)
            ax.set_yticks(range(len(order)), order)
        axes[0].text(0, 1.14, "PRIMARY COMPARISON", transform=axes[0].transAxes,
                     fontsize=9, color="#64748b", fontweight="bold")
        fig.savefig(output_path, dpi=180, facecolor="white")
        fig.savefig(output_path.with_suffix(".svg"), facecolor="white")
        plt.close(fig)


def plot_sweep_confusion_matrices(predictions, labels, output_dir, threshold=0.5):
    """Pool held-out predictions across folds, with a figure per freeze setting."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for freeze, group in predictions.groupby("freeze_backbone", sort=False):
        mode = "frozen" if freeze else "unfrozen"
        backbones = group["backbone"].drop_duplicates().tolist()
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                             "text.color": "#243447", "axes.labelcolor": "#46576b",
                             "svg.fonttype": "none"}):
            fig, axes = plt.subplots(
                len(backbones), len(labels), figsize=(3.8 * len(labels) + 1, 2.9 * len(backbones) + 1),
                squeeze=False, layout="constrained",
            )
            fig.suptitle(
                f"{mode.title()} backbones · out-of-fold confusion matrices\n"
                f"Count / row percentage · threshold = {threshold:g}",
                fontsize=15, fontweight="bold",
            )
            for row, backbone in enumerate(backbones):
                part = group[group["backbone"] == backbone]
                for col, label in enumerate(labels):
                    ax = axes[row, col]
                    matrix = confusion_matrix(part[label], part[f"{label}_prediction"], labels=[0, 1])
                    totals = matrix.sum(axis=1, keepdims=True)
                    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
                    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
                    for i in range(2):
                        for j in range(2):
                            percentage = f"{normalized[i, j]:.1%}" if totals[i, 0] else "N/A"
                            ax.text(j, i, f"{matrix[i, j]}\n{percentage}", ha="center", va="center",
                                    color="white" if normalized[i, j] > .5 else "#243447")
                    ax.set_xticks([0, 1], ["Absent", "Present"])
                    ax.set_yticks([0, 1], ["Absent", "Present"])
                    ax.set_xlabel("Predicted label")
                    ax.set_ylabel(f"{backbone}\nTrue label" if col == 0 else "True label")
                    if row == 0:
                        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
                    ax.tick_params(length=0)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
            fig.colorbar(im, ax=axes.ravel().tolist(), label="Proportion within true class", shrink=.75)
            path = output_dir / f"confusion_matrices_{mode}.png"
            fig.savefig(path, dpi=160, facecolor="white")
            fig.savefig(path.with_suffix(".svg"), facecolor="white")
            plt.close(fig)
