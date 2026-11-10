"""Generate a standalone visual preview; all measurements are synthetic."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from cornea.plots import plot_experiment_comparison

OUT = Path(__file__).resolve().parents[1] / "outputs" / "preview"
OUT.mkdir(parents=True, exist_ok=True)
CENTERS = {
    "ResNet-18": [[.884, .705, .843], [.908, .762, .874]],
    "ResNet-34": [[.901, .731, .868], [.924, .789, .899]],
    "ResNet-50": [[.919, .758, .891], [.938, .816, .922]],
    "Swin-T":    [[.931, .774, .908], [.949, .835, .940]],
    "MaxViT-T":  [[.946, .790, .924], [.957, .849, .951]],
    "Swin-S":    [[.956, .781, .930], [.965, .832, .946]],
}
METRICS = ["specificity", "sensitivity", "auroc"]
rng = np.random.default_rng(42)
rows = []
for backbone, values in CENTERS.items():
    for mode, centers in zip(["Frozen", "Unfrozen"], values):
        noise = rng.normal(size=(5, 3))
        noise -= noise.mean(axis=0)
        noise *= np.array([.009, .020, .012]) / noise.std(axis=0, ddof=1)
        for fold, scores in enumerate(np.array(centers) + noise, start=1):
            rows.append({"backbone": backbone, "freeze_backbone": mode == "Frozen", "fold": fold,
                         **dict(zip(METRICS, scores))})
data = pd.DataFrame(rows)
data.to_csv(OUT / "dummy_fold_metrics.csv", index=False)

plot_experiment_comparison(data, OUT / "experiment_comparison_dummy.png", dummy=True)
print(OUT / "experiment_comparison_dummy.png")
