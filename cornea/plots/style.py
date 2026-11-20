from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler

PAPER = "#faf8f2"
INK = "#282620"
GRID = "#ddd7cc"
TERRACOTTA = "#a54e37"
TEAL = "#2f665e"
SERIES_COLORS = (
    TEAL, TERRACOTTA, "#78638e", "#8b681e", "#456c91",
    "#91546b", "#526d3e", "#665348", "#3a777f", "#9b633f",
)
MODE_NAMES = {True: "Frozen", False: "Unfrozen"}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": PAPER,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "axes.facecolor": PAPER,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.prop_cycle": cycler(color=SERIES_COLORS),
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
}


def figure_grid(rows, columns, panel_size, title=None, **kwargs):
    width, height = panel_size
    fig, axes = plt.subplots(
        rows, columns, figsize=(width * columns, height * rows),
        layout="constrained", squeeze=False, **kwargs,
    )
    if title:
        fig.suptitle(title)
    return fig, axes


def save_figure(fig, path, *, show=False):
    path = Path(path).with_suffix(".png")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        if show:
            plt.show()
    finally:
        plt.close(fig)
