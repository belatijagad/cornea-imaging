import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from .embeddings import CLASS_COLORS
from .style import INK, PAPER, STYLE, figure_grid, save_figure


def _draw_boundaries(ax, xx, yy, regions, xyz, labels, names, title):
    colors = [CLASS_COLORS[name] for name in names]
    ax.contourf(xx, yy, regions, levels=np.arange(len(names) + 1) - .5,
                cmap=ListedColormap(colors), alpha=.18)
    for label, color in enumerate(colors):
        ax.scatter(*xyz[labels == label, :2].T, color=color, s=32, alpha=.9,
                   edgecolors=PAPER, linewidths=.4, label=names[label])
    ax.set(title=title, xlabel="PC1", ylabel="PC2", aspect="equal")


@plt.rc_context(STYLE)
def plot_boundary_slices(panels, names, backbone, output):
    folds = len(panels)
    columns = min(3, folds)
    rows = (folds + columns - 1) // columns
    fig, axes = figure_grid(rows, columns, (5.4, 4.8), f"{backbone} clustering boundaries")
    for fold, (ax, panel) in enumerate(zip(axes.flat, panels), 1):
        _draw_boundaries(ax, *panel, names, f"Fold {fold}")
        single, single_axes = figure_grid(1, 1, (7, 6))
        single_ax = single_axes[0, 0]
        _draw_boundaries(single_ax, *panel, names, f"{backbone} · fold {fold}")
        single_ax.legend(loc="upper right")
        save_figure(single, output / f"decision_boundary_fold_{fold}.png", svg=True)
    for ax in axes.ravel()[folds:]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=CLASS_COLORS[name], marker="o", linestyle="", label=name) for name in names]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(names), labelcolor=INK)
    save_figure(fig, output / "decision_boundaries.png", svg=True)
