import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from cornea.checkpoints import fold_directory, load_checkpoint
from cornea.config import load_config
from cornea.dataset import CorneaDataset, load_manifest, make_transforms
from cornea.model import CorneaClassifier
from cornea.plots.embeddings import COMPONENTS, plot_projection


def checkpoint_path(run, backbone, mode, fold):
    modern = fold_directory(run, backbone, mode, fold) / "best.pt"
    legacy = run / backbone / mode / f"fold_{fold}" / "best.pt"
    return legacy if not modern.exists() and legacy.exists() else modern


def embedding_directory(run, backbone, mode, fold, *, base=False, clusters=3):
    name = f"{backbone}_imagenet_whole_data_k{clusters}" if base else f"{backbone}_{mode}_fold_{fold}"
    return run / "embeddings" / name


def load_embeddings(frame, config, backbone, checkpoint, cache):
    cached = cache.exists()
    if cached:
        features = np.load(cache)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CorneaClassifier(
            num_labels=len(config.data.labels), pretrained=checkpoint is None, backbone=backbone,
        )
        if checkpoint is not None:
            load_checkpoint(model, checkpoint)
        model.to(device).eval()
        loader = DataLoader(
            CorneaDataset(frame, make_transforms(config.data), config.data.labels),
            batch_size=config.data.batch_size, num_workers=config.data.num_workers,
        )
        print(f"Extracting {backbone} features on {device}", flush=True)
        with torch.inference_mode():
            features = np.concatenate([
                model.backbone(images.to(device)).float().cpu().numpy()
                for images, _ in loader
            ])
    if features.ndim != 2 or len(features) != len(frame) or not np.isfinite(features).all():
        raise ValueError(f"Embeddings must contain one finite feature vector per manifest row: {cache}")
    if not cached:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=cache.parent, delete=False) as file:
            temporary = Path(file.name)
            try:
                np.save(file, features)
                file.close()
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
    return features


def project_embeddings(run, backbone, mode="unfrozen", fold=1, *, base=False, clusters=3, plot_root=None):
    run = Path(run)
    config = load_config(run / "config.yml")
    frame = load_manifest(run, config)
    out = embedding_directory(run, backbone, mode, fold, base=base, clusters=clusters)
    checkpoint = None if base else checkpoint_path(run, backbone, mode, fold)
    features = load_embeddings(frame, config, backbone, checkpoint, out / "embeddings.npy")
    pca = PCA(n_components=3, svd_solver="full")
    coordinates = pca.fit_transform(normalize(features))
    kmeans = KMeans(n_clusters=clusters, n_init=30, random_state=42)
    frame["cluster"] = kmeans.fit_predict(coordinates) + 1
    labels = config.data.labels
    names = {
        "fungal": "Fungi", "bacterial": "Bacteria", "viral_other": "Other",
    }
    readable_labels = np.array([names.get(label, label.replace("_", " ").title()) for label in labels])
    targets = frame[labels].to_numpy()
    frame["label"] = np.where(targets.sum(axis=1) == 1, readable_labels[targets.argmax(axis=1)], "Mixed")
    frame[COMPONENTS] = coordinates
    label_table = pd.crosstab(frame["cluster"], frame["label"])
    frame.to_csv(out / "image_embeddings_3d.csv", index=False)
    label_table.to_csv(out / "cluster_labels.csv")
    metadata = {
        "checkpoint": None if base else str(checkpoint.resolve()),
        "images": len(frame),
        "backbone": backbone,
        "model_source": "ImageNet" if base else "Cornea fine-tuned",
        "embedding_dimensions": features.shape[1],
        "normalization": "L2 per image",
        "reduction": "PCA, 3 components",
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "clustering": "KMeans on 3D PCA coordinates",
        "clusters": clusters,
        "seed": 42,
        "centers": kmeans.cluster_centers_.tolist(),
        "interpretation": ("Exploratory; PCA and clustering fitted on the entire dataset."
                           if base else "Exploratory; includes training and held-out images for this checkpoint."),
        "mixed": "Images with multiple positive labels.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    plot_dir = out if plot_root is None else Path(plot_root) / out.name
    for dimensions in (2, 3):
        plot_projection(frame, metadata, plot_dir, dimensions)
    print(label_table.to_string())
    print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.1%}")
    return out


def replot_embeddings_2d(directory):
    directory = Path(directory)
    frame = pd.read_csv(directory / "image_embeddings_3d.csv")
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata.setdefault("clusters", 3)
    plot_projection(frame, metadata, directory, dimensions=2)
    return directory
