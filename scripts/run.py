import argparse
from datetime import datetime
import random
import re
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from cornea.config import DEFAULT_CONFIG, load_config
from cornea.dataset import CorneaDataset, make_transforms
from cornea.model import BACKBONES, CorneaClassifier
from cornea.plots import MODE_NAMES, plot_sweep
from cornea.preprocessing import (
    CORNEA_BOX_COLUMNS, attach_cornea_boxes, attach_preprocessed_images, generate_preprocessed_images,
)
from cornea.trainer import Trainer, TrainerConfig


def process_data(data_root, labels_csv, labels, folder_labels, image_extensions):
    if labels_csv is not None:
        frame = pd.read_csv(labels_csv, dtype={"patient_id": "string"})
    else:
        rows = []
        for folder, targets in folder_labels.items():
            for path in sorted((data_root / folder).rglob("*")):
                if path.is_file() and path.suffix.lower() in image_extensions:
                    prefix = re.split(
                        r"\d|\b(?:fu|follow|od|os|kanan|kiri|nrm|copy)\b", path.stem.lower(), maxsplit=1
                    )[0]
                    rows.append({
                        "image_path": str(path.relative_to(data_root)),
                        "patient_id": re.sub(r"[^a-z]+", " ", prefix).strip(),
                        **dict(zip(labels, targets)),
                    })
        frame = pd.DataFrame(rows)
    frame["image_path"] = frame["image_path"].map(lambda p: str((data_root / p).resolve()))
    return frame


def assign_folds(frame, labels, n_splits=5, seed=42):
    frame = frame.copy()
    strata = frame[labels].astype(int).astype(str).agg("".join, axis=1)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(frame, strata, frame["patient_id"])):
        frame.loc[frame.index[val_idx], "fold"] = fold
    frame["fold"] = frame["fold"].astype(int)
    return frame


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="Run cross-validation and generate plots.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    option_groups = [
        ("data", Path, ("data-root", "labels-csv", "cornea-boxes-csv", "preprocessed-images-csv")),
        ("trainer", Path, ("output-dir",)),
        ("trainer", int, ("epochs", "seed", "validation-interval", "folds", "batch-size", "num-workers")),
        ("transforms", int, ("image-size",)),
        ("trainer", float, ("learning-rate", "weight-decay", "threshold")),
    ]
    for section, value_type, options in option_groups:
        for option in options:
            parser.add_argument(f"--{option}", type=value_type,
                                dest=f"{section}.{option.replace('-', '_')}",
                                default=argparse.SUPPRESS)
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES,
                        dest="model.backbones", default=argparse.SUPPRESS)
    parser.add_argument("--medsam", action="store_true", dest="data.medsam", default=argparse.SUPPRESS,
                        help="Run MedSAM once per image and save ellipse-masked classifier inputs")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"),
                        dest="trainer.precision", default=argparse.SUPPRESS)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction,
                        dest="model.pretrained", default=argparse.SUPPRESS)
    parser.add_argument("--embeddings", action=argparse.BooleanOptionalAction,
                        dest="trainer.embeddings", default=argparse.SUPPRESS,
                        help="Generate 2D and 3D embedding plots from the first fold checkpoint of each model")
    freeze_options = parser.add_mutually_exclusive_group()
    freeze_options.add_argument("--freeze-backbone", nargs="*", choices=("true", "false"),
                                dest="model.freeze_backbone", default=argparse.SUPPRESS,
                                help="Freeze options to sweep; a bare flag means true")
    freeze_options.add_argument("--no-freeze-backbone", action="store_const", const=["false"],
                                dest="model.freeze_backbone", default=argparse.SUPPRESS)
    options = vars(parser.parse_args(argv))
    config_path = options.pop("config")
    overrides = {}
    for name, value in options.items():
        section, key = name.split(".")
        if name == "model.freeze_backbone":
            value = [option == "true" for option in (value or ["true"])]
        overrides.setdefault(section, {})[key] = value
    return load_config(config_path, overrides)


def run_backbone(config, frame, backbone, freeze_backbone):
    mode = MODE_NAMES[freeze_backbone].lower()
    output_dir = config.trainer.output_dir
    plots = output_dir / "plots" / "training" / backbone / mode
    for fold in range(config.trainer.folds):
        seed_everything(config.trainer.seed + fold)
        train_df = frame[frame["fold"] != fold]
        val_df = frame[frame["fold"] == fold]
        print(f"{backbone} ({mode}) | Fold {fold + 1}/{config.trainer.folds} | train: {len(train_df)} | val: {len(val_df)}")
        loaders = []
        for part, training in [(train_df, True), (val_df, False)]:
            loaders.append(DataLoader(
                CorneaDataset(part, make_transforms(config.transforms, training), config.data.labels),
                batch_size=config.trainer.batch_size, shuffle=training, num_workers=config.trainer.num_workers,
                pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(config.trainer.seed + fold),
            ))
        model = CorneaClassifier(
            num_labels=len(config.data.labels), freeze_backbone=freeze_backbone,
            pretrained=config.model.pretrained, backbone=backbone,
        )
        trainer_config = TrainerConfig(
            epochs=config.trainer.epochs, learning_rate=config.trainer.learning_rate,
            validation_interval=config.trainer.validation_interval,
            precision=config.trainer.precision,
            weight_decay=config.trainer.weight_decay, threshold=config.trainer.threshold,
            output_dir=output_dir / "folds" / backbone / mode / f"fold_{fold + 1}",
            backbone_dir=output_dir / "folds" / "backbones",
        )
        trainer = Trainer(model, trainer_config, *loaders)
        trainer.train(specificity_path=plots / f"fold_{fold + 1}.png")
        metrics, predictions = trainer.eval()
        identifiers = {"fold": fold + 1, "backbone": backbone, "freeze_backbone": freeze_backbone}
        del trainer, model, loaders
        yield metrics.assign(**identifiers), predictions.assign(**identifiers)


def create_sweep_directory(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"sweep_{timestamp}_", dir=output_root))


def main(argv=None):
    config = parse_config(argv)
    seed_everything(config.trainer.seed)
    frame = process_data(config.data.data_root, config.data.labels_csv, config.data.labels,
                         config.data.folder_labels, config.data.image_extensions)
    frame = frame.drop(columns=["preprocessed_path", *CORNEA_BOX_COLUMNS], errors="ignore")
    source = "raw"
    cache = config.data.preprocessed_images_csv
    if config.data.medsam:
        source = "medsam"
    elif config.data.cornea_boxes_csv is not None:
        source = "cornea_boxes"
        frame = attach_cornea_boxes(frame, config.data.cornea_boxes_csv, config.data.data_root)
    elif cache is not None and cache.is_file():
        source = "preprocessed"
        frame = attach_preprocessed_images(frame, cache, config.data.data_root)
    frame = assign_folds(frame, config.data.labels, config.trainer.folds, config.trainer.seed)
    config.trainer.output_dir = create_sweep_directory(config.trainer.output_dir)
    print(f"Sweep output: {config.trainer.output_dir.resolve()}")
    print(f"Image source: {source}")
    if source == "medsam":
        preprocessed = generate_preprocessed_images(frame, config.trainer.output_dir)
        frame = attach_preprocessed_images(frame, preprocessed, config.data.data_root)
    config.save(config.trainer.output_dir / "config.yml")
    manifest = frame.copy()
    manifest["image_path"] = manifest["image_path"].map(
        lambda p: str(Path(p).relative_to(config.data.data_root.resolve()))
        if Path(p).is_relative_to(config.data.data_root.resolve()) else p
    )
    if source == "medsam":
        manifest["preprocessed_path"] = manifest["preprocessed_path"].map(
            lambda path: str(Path(path).relative_to(config.trainer.output_dir.resolve()))
        )
    manifest.to_csv(config.trainer.output_dir / "manifest.csv", index=False)
    results = [
        result
        for backbone in config.model.backbones
        for freeze in config.model.freeze_backbone
        for result in run_backbone(config, frame, backbone, freeze)
    ]
    fold_metrics, fold_predictions = zip(*results)
    metrics = pd.concat(fold_metrics, ignore_index=True)
    predictions = pd.concat(fold_predictions, ignore_index=True)
    output = config.trainer.output_dir
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "out_of_fold_predictions.csv", index=False)
    summary = metrics.drop(columns=["fold", "positives"]).groupby(
        ["backbone", "freeze_backbone", "label"]
    ).agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    plot_sweep(metrics, predictions, config.data.labels, output / "plots", config.trainer.threshold)
    if config.trainer.embeddings:
        from cornea.embeddings import project_embeddings

        for backbone in config.model.backbones:
            for freeze in config.model.freeze_backbone:
                mode = MODE_NAMES[freeze].lower()
                project_embeddings(output, backbone, mode, plot_root=output / "plots" / "embeddings")
    print(f"Sweep complete: {output.resolve()}")
    return output


if __name__ == "__main__":
    main()
