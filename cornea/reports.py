import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from cornea.config import load_config
from cornea.dataset import load_manifest
from cornea.plots import MODE_NAMES, plot_class_confusion_matrix, plot_sweep


def regenerate_plots(run):
    run = Path(run)
    config = load_config(run / "config.yml")
    output = run / "plots"
    plot_sweep(
        pd.read_csv(run / "fold_metrics.csv"), pd.read_csv(run / "out_of_fold_predictions.csv"),
        config.data.labels, output, config.trainer.threshold,
    )
    return output


def _macro_auroc(targets, scores):
    present = (targets.min(axis=0) != targets.max(axis=0))
    return roc_auc_score(targets[:, present], scores[:, present], average="macro") if present.any() else np.nan


def _classification_metrics(actual, predicted, targets, scores, classes):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", labels=classes, zero_division=0),
        "macro_auroc": _macro_auroc(targets, scores),
    }


def write_classification_report(actual, predicted, names, out, title):
    classes = range(len(names))
    report = classification_report(actual, predicted, labels=classes, target_names=names,
                                   output_dict=True, zero_division=0)
    report = pd.DataFrame(report).T
    report.to_csv(out / "classification_report.csv")
    matrix = confusion_matrix(actual, predicted, labels=classes)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(out / "confusion_matrix.csv")
    plot_class_confusion_matrix(matrix, names, out / "confusion_matrix.png", title)
    return report


def _write_multiclass(frame, labels, names, out, title):
    targets = frame[labels].to_numpy()
    scores = frame[[f"{label}_probability" for label in labels]].to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Multiclass probabilities must be finite")
    actual, predicted = targets.argmax(axis=1), scores.argmax(axis=1)
    classes = range(len(labels))
    out.mkdir(parents=True, exist_ok=True)
    frame.assign(true_class=names[actual], predicted_class=names[predicted]).to_csv(
        out / "out_of_fold_predictions.csv", index=False,
    )
    fold_metrics = []
    for fold, positions in frame.groupby("fold").indices.items():
        metrics = _classification_metrics(actual[positions], predicted[positions], targets[positions], scores[positions], classes)
        fold_metrics.append({"fold": int(fold), "images": len(positions), **metrics})
    fold_metrics = pd.DataFrame(fold_metrics)
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    fold_metrics.drop(columns=["fold", "images"]).agg(["mean", "std"]).to_csv(out / "summary.csv")
    write_classification_report(actual, predicted, names, out, title)
    pooled = {
        "images": len(frame), **_classification_metrics(actual, predicted, targets, scores, classes),
        "majority_baseline_accuracy": float(np.bincount(actual).max() / len(frame)),
        "prediction_rule": "Argmax of sigmoid probabilities on single-label images.",
        "evaluation": "Pooled held-out predictions from patient-grouped folds; checkpoints selected by validation loss.",
    }
    if np.isnan(pooled["macro_auroc"]):
        pooled["macro_auroc"] = None
    (out / "pooled_metrics.json").write_text(json.dumps(pooled, indent=2))
    print(fold_metrics.round(4).to_string(index=False))


def summarize_multiclass(run):
    run = Path(run)
    config = load_config(run / "config.yml")
    labels = config.data.labels
    names = np.array(["Other" if label == "viral_other" else label.replace("_", " ").title() for label in labels])
    manifest = load_manifest(run, config)
    if manifest.groupby("patient_id")["fold"].nunique().gt(1).any():
        raise ValueError("Multiclass evaluation requires patient groups confined to one fold")
    predictions = pd.read_csv(run / "out_of_fold_predictions.csv")
    groups = predictions.groupby(["backbone", "freeze_backbone"], sort=False)
    output = run / "multiclass"
    for (backbone, freeze), frame in groups:
        aligned = frame.merge(manifest[["image_path", "fold", *labels]], on="image_path",
                              validate="one_to_one", suffixes=("", "_manifest"))
        if len(aligned) != len(frame) or len(frame) != len(manifest) or not aligned["fold"].eq(aligned["fold_manifest"] + 1).all():
            raise ValueError(f"Predictions do not align with manifest folds for {backbone}")
        if any(not aligned[label].eq(aligned[f"{label}_manifest"]).all() for label in labels):
            raise ValueError(f"Prediction targets do not match the manifest for {backbone}")
        cases = frame.loc[frame[labels].sum(axis=1).eq(1)]
        if cases.empty:
            raise ValueError(f"No single-label cases for {backbone}")
        mode = MODE_NAMES[freeze]
        destination = output if len(groups) == 1 else output / backbone / mode.lower()
        _write_multiclass(cases, labels, names, destination, f"{backbone} · {mode}")
    return output
