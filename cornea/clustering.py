import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import normalize

from cornea.config import load_config
from cornea.dataset import load_manifest
from cornea.embeddings import checkpoint_path, load_embeddings
from cornea.plots.clustering import plot_boundary_slices
from cornea.plots.embeddings import CLASS_NAMES
from cornea.reports import write_classification_report


def clustering_cases(frame, config, exclude_other):
    labels = config.data.labels
    expected = ("fungal", "bacterial", "viral_other")
    if set(labels) != set(expected):
        raise ValueError("Clustering evaluation requires fungal, bacterial, and viral_other labels")
    patterns = {(1, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 2, (1, 1, 0): 3}
    try:
        targets = np.array([patterns[tuple(row)] for row in frame[list(expected)].to_numpy()])
    except KeyError as error:
        raise ValueError(f"Unsupported clustering target: {error.args[0]}") from error
    keep = targets != 2 if exclude_other else np.ones(len(frame), dtype=bool)
    if exclude_other:
        return keep, np.array([0, 1, -1, 2])[targets[keep]], ["Fungi", "Bacteria", "Mixed"]
    return keep, targets, CLASS_NAMES


def evaluate_clustering(run, backbone="resnet18", *, exclude_other=False):
    run = Path(run)
    config = load_config(run / "config.yml")
    original_frame = load_manifest(run, config)
    if original_frame["image_path"].duplicated().any() or original_frame.groupby("patient_id")["fold"].nunique().gt(1).any():
        raise ValueError("Clustering requires unique images and patient groups confined to one fold")
    keep, targets, names = clustering_cases(original_frame, config, exclude_other)
    frame = original_frame.loc[keep].reset_index(drop=True)
    n_classes = len(names)
    design = "k3_without_other" if exclude_other else "k4"
    output = run / "clustering_cv" / f"{backbone}_unfrozen_pca3_{design}"
    output.mkdir(parents=True, exist_ok=True)
    predictions = np.full(len(frame), -1, dtype=int)
    baseline = np.full(len(frame), -1, dtype=int)
    folds = []
    details = []
    assignments = frame['fold'].to_numpy()
    classes = range(n_classes)
    for fold in range(config.trainer.folds):
        train = assignments != fold
        held_out = ~train
        if np.unique(targets[train]).size != n_classes or not held_out.any():
            raise ValueError(f"Fold {fold + 1} needs all training classes and held-out cases")
        cache = run / "embeddings" / f"{backbone}_unfrozen_fold_{fold + 1}" / "embeddings.npy"
        checkpoint = checkpoint_path(run, backbone, "unfrozen", fold + 1)
        features = load_embeddings(original_frame, config, backbone, checkpoint, cache)
        features = normalize(features[keep])
        pca = PCA(n_components=3, svd_solver='full').fit(features[train])
        kmeans = KMeans(n_clusters=n_classes, n_init=30, random_state=config.trainer.seed).fit(
            pca.transform(features[train])
        )
        counts = np.zeros((n_classes, n_classes), dtype=int)
        np.add.at(counts, (kmeans.labels_, targets[train]), 1)
        clusters, matched_classes = linear_sum_assignment(-counts)
        mapping = np.empty(n_classes, dtype=int)
        mapping[clusters] = matched_classes
        guess = mapping[kmeans.predict(pca.transform(features[held_out]))]
        predictions[held_out] = guess
        baseline[held_out] = np.bincount(targets[train], minlength=n_classes).argmax()
        result = {'fold': fold + 1, 'train_images': int(train.sum()), 'test_images': int(held_out.sum()),
                  'accuracy': accuracy_score(targets[held_out], guess),
                  'balanced_accuracy': balanced_accuracy_score(targets[held_out], guess),
                  'macro_f1': f1_score(targets[held_out], guess, labels=classes, average='macro', zero_division=0),
                  'weighted_f1': f1_score(targets[held_out], guess, labels=classes, average='weighted', zero_division=0)}
        folds.append(result)
        details.append({'fold': fold + 1, 'checkpoint': str(checkpoint.resolve()),
                        'train_cluster_class_counts': counts.tolist(), 'cluster_to_class': mapping.tolist(),
                        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist()})
        np.savez(output / f'fold_{fold + 1}_transform.npz', pca_mean=pca.mean_,
                 pca_components=pca.components_, cluster_centers=kmeans.cluster_centers_, mapping=mapping)
        print(result, flush=True)
    if (predictions < 0).any():
        raise ValueError("Manifest folds do not cover all images")
    result_frame = frame.copy()
    result_frame['true_label'] = np.array(names)[targets]
    result_frame['predicted_label'] = np.array(names)[predictions]
    result_frame.to_csv(output / 'out_of_fold_predictions.csv', index=False)
    fold_metrics = pd.DataFrame(folds)
    fold_metrics.to_csv(output / 'fold_metrics.csv', index=False)
    fold_metrics[['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']].agg(['mean', 'std']).to_csv(output / 'summary.csv')
    report = write_classification_report(targets, predictions, names, output, f"{backbone} clustering")
    metadata = {'backbone': backbone, 'normalization': 'L2 per image', 'dimensions': 3,
                'clusters': n_classes, 'classes': names, 'excluded_other': exclude_other, 'seed': config.trainer.seed,
                'mapping': 'One-to-one Hungarian assignment maximizing training-label matches',
                'evaluation': 'Original patient-grouped folds; PCA, KMeans and mapping fit on training only',
                'limitation': 'Reuses supervised backbone checkpoints selected on these validation folds; not an independent test.',
                'majority_baseline_accuracy': accuracy_score(targets, baseline), 'folds': details}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(report.round(4).to_string(), flush=True)
    print(f'Majority baseline accuracy: {metadata["majority_baseline_accuracy"]:.4f}', flush=True)
    return output


def plot_cluster_boundaries(run, backbone="resnet18", *, exclude_other=False):
    run = Path(run)
    config = load_config(run / "config.yml")
    frame = load_manifest(run, config)
    keep, labels, names = clustering_cases(frame, config, exclude_other)
    design = "k3_without_other" if exclude_other else "k4"
    output = run / "clustering_cv" / f"{backbone}_unfrozen_pca3_{design}"
    folds = config.trainer.folds
    assignments = frame.loc[keep, 'fold'].to_numpy()
    panels = []
    slices = []
    for fold in range(1, folds + 1):
        cache = run / "embeddings" / f"{backbone}_unfrozen_fold_{fold}" / "embeddings.npy"
        features = normalize(np.load(cache)[keep])
        with np.load(output / f"fold_{fold}_transform.npz") as transform:
            xyz = (features - transform["pca_mean"]) @ transform["pca_components"].T
            centers, mapping = transform["cluster_centers"], transform["mapping"]
        held_out = assignments == fold - 1
        pc3 = float(np.median(xyz[~held_out, 2]))
        low, high = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
        padding = np.maximum(high - low, .01) * .08
        xx, yy = np.meshgrid(*(np.linspace(lo, hi, 350) for lo, hi in zip(low - padding, high + padding)))
        grid = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, pc3)])
        distances = ((grid[:, None] - centers) ** 2).sum(axis=2)
        regions = mapping[distances.argmin(axis=1)].reshape(xx.shape)
        panels.append((xx, yy, regions, xyz[held_out], labels[held_out]))
        slices.append({"fold": fold, "pc3_slice": pc3, "held_out_images": int(held_out.sum())})
    plot_boundary_slices(panels, names, backbone, output)
    pd.DataFrame(slices).to_csv(output / "decision_boundary_slices.csv", index=False)
    return output
