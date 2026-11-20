from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import INK, PAPER, SERIES_COLORS, STYLE, save_figure

COMPONENTS = ["PC1", "PC2", "PC3"]
CLASS_NAMES = ["Fungi", "Bacteria", "Other", "Mixed"]
CLASS_COLORS = dict(zip(CLASS_NAMES, SERIES_COLORS, strict=False))
CLUSTER_MARKERS = ("o", "^", "s", "D")


@plt.rc_context(STYLE)
def plot_projection(frame, metadata, out, dimensions):
    components = COMPONENTS[:dimensions]
    variance = metadata["explained_variance_ratio"]
    is_3d = dimensions == 3
    fig, ax = plt.subplots(
        figsize=(10, 7), layout="constrained",
        subplot_kw={"projection": "3d"} if is_3d else {},
    )
    label_colors = {label: CLASS_COLORS.get(label, color)
                    for label, color in zip(frame["label"].unique(), cycle(SERIES_COLORS), strict=False)}
    markers = dict(zip(range(1, metadata["clusters"] + 1), CLUSTER_MARKERS, strict=False))
    for (label, cluster), group in frame.groupby(["label", "cluster"], sort=False):
        ax.scatter(*group[components].to_numpy().T, color=label_colors[label], marker=markers[cluster],
                   s=30, alpha=.8, edgecolors=PAPER, linewidths=.3)
    backbone = metadata.get("backbone") or Path(metadata["checkpoint"]).parts[-4]
    source = "ImageNet" if metadata.get("model_source") == "ImageNet" else "Fine-tuned"
    axis_labels = [f"{name} ({value:.1%})" for name, value in zip(components, variance[:dimensions], strict=True)]
    ax.set(title=f"{backbone} embeddings · {source}", xlabel=axis_labels[0], ylabel=axis_labels[1])
    if is_3d:
        ax.set_zlabel(axis_labels[2])
        ax.view_init(elev=23, azim=40)
    else:
        ax.set_aspect("equal", adjustable="box")
        ax.grid()
    handles = [Line2D([], [], marker="o", color=color, linestyle="", label=label)
               for label, color in label_colors.items()]
    handles.extend(Line2D([], [], marker=marker, color=INK, linestyle="", label=f"Cluster {cluster}")
                   for cluster, marker in markers.items())
    fig.legend(handles=handles, loc="outside right upper")
    save_figure(fig, out / f"embeddings_{dimensions}d.png")
