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
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from cornea.config import DEFAULT_CONFIG, load_config
from cornea.dataset import CorneaDataset
from cornea.model import CorneaClassifier, available_backbones
from cornea.plots import plot_confusion_matrices, plot_experiment_comparison, plot_sweep_confusion_matrices
from cornea.trainer import Trainer, TrainerConfig


def guess_patient_group(path):
    prefix = re.split(
        r"\d|\b(?:fu|follow|od|os|kanan|kiri|nrm|copy)\b", path.stem.lower(), maxsplit=1
    )[0]
    return re.sub(r"[^a-z]+", " ", prefix).strip()


def process_data(data_root, labels_csv, labels, folder_labels, image_extensions):
    if labels_csv is not None:
        frame = pd.read_csv(labels_csv, dtype={"patient_id": "string"})
    else:
        rows = []
        for folder, targets in folder_labels.items():
            for path in sorted((data_root / folder).rglob("*")):
                if path.is_file() and path.suffix.lower() in image_extensions:
                    rows.append({
                        "image_path": str(path.relative_to(data_root)),
                        "patient_id": guess_patient_group(path),
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


def make_transforms(config, training=False):
    operations = [transforms.Resize((config.image_size, config.image_size))]
    if training:
        operations += [transforms.RandomHorizontalFlip(config.horizontal_flip_probability),
                       transforms.RandomRotation(config.rotation_degrees)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(config.mean, config.std)])


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
    parser = argparse.ArgumentParser(description="Run cross-validation for each configured backbone.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    # Keep CLI flags stable while routing overrides into their YAML sections.
    option_groups = [
        ("data", Path, ("data-root", "labels-csv")),
        ("trainer", Path, ("output-dir",)),
        ("data", int, ("folds", "batch-size", "image-size", "num-workers")),
        ("trainer", int, ("epochs", "seed", "validation-interval")),
        ("trainer", float, ("learning-rate", "weight-decay", "threshold")),
    ]
    for section, value_type, options in option_groups:
        for option in options:
            parser.add_argument(f"--{option}", type=value_type,
                                dest=f"{section}.{option.replace('-', '_')}",
                                default=argparse.SUPPRESS)
    parser.add_argument("--backbones", nargs="+", choices=available_backbones(),
                        dest="model.backbones", default=argparse.SUPPRESS)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"),
                        dest="trainer.precision", default=argparse.SUPPRESS)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction,
                        dest="model.pretrained", default=argparse.SUPPRESS)
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
    mode = "frozen" if freeze_backbone else "unfrozen"
    output_dir = config.trainer.output_dir / backbone / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = []
    fold_predictions = []
    for fold in range(config.data.folds):
        seed_everything(config.trainer.seed + fold)
        train_df = frame[frame["fold"] != fold]
        val_df = frame[frame["fold"] == fold]
        print(f"{backbone} ({mode}) | Fold {fold + 1}/{config.data.folds} | train: {len(train_df)} | val: {len(val_df)}")
        loaders = []
        for part, training in [(train_df, True), (val_df, False)]:
            loaders.append(DataLoader(
                CorneaDataset(part, make_transforms(config.data, training), config.data.labels),
                batch_size=config.data.batch_size, shuffle=training, num_workers=config.data.num_workers,
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
            output_dir=output_dir / f"fold_{fold + 1}",
            backbone_dir=config.trainer.output_dir / "backbones",
        )
        trainer = Trainer(model, trainer_config, *loaders)
        trainer.train()
        fold_metrics.append(trainer.eval().assign(
            fold=fold + 1, backbone=backbone, freeze_backbone=freeze_backbone,
        ))
        fold_predictions.append(pd.read_csv(trainer.output_dir / "predictions.csv").assign(fold=fold + 1))
        del trainer, model, loaders
    metrics = pd.concat(fold_metrics, ignore_index=True)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(output_dir / "summary.csv")
    predictions = pd.concat(fold_predictions, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    matrices = multilabel_confusion_matrix(
        predictions[config.data.labels],
        predictions[[f"{label}_prediction" for label in config.data.labels]],
    )
    plot_confusion_matrices(matrices, config.data.labels, output_dir)
    print(summary.round(4).to_string())
    return metrics


def summarize(metrics):
    return metrics.drop(columns=["fold", "positives"]).groupby(
        ["backbone", "freeze_backbone", "label"]
    ).agg(["mean", "std"])


def create_sweep_directory(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"sweep_{timestamp}_", dir=output_root))


def save_sweep_plots(metrics, config):
    output_dir = config.trainer.output_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_experiment_comparison(
        metrics, plots_dir / "experiment_comparison.png", threshold=config.trainer.threshold,
    )
    predictions = []
    for backbone in config.model.backbones:
        for freeze in config.model.freeze_backbone:
            mode = "frozen" if freeze else "unfrozen"
            path = output_dir / backbone / mode / "predictions.csv"
            predictions.append(pd.read_csv(path).assign(backbone=backbone, freeze_backbone=freeze))
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    plot_sweep_confusion_matrices(predictions, config.data.labels, plots_dir, config.trainer.threshold)


def main(argv=None):
    config = parse_config(argv)
    seed_everything(config.trainer.seed)
    frame = assign_folds(
        process_data(config.data.data_root, config.data.labels_csv, config.data.labels,
                     config.data.folder_labels, config.data.image_extensions),
        config.data.labels, config.data.folds, config.trainer.seed,
    )
    config.trainer.output_dir = create_sweep_directory(config.trainer.output_dir)
    print(f"Sweep output: {config.trainer.output_dir.resolve()}")
    config.save(config.trainer.output_dir / "config.yml")
    manifest = frame.copy()
    manifest["image_path"] = manifest["image_path"].map(
        lambda p: str(Path(p).relative_to(config.data.data_root.resolve()))
        if Path(p).is_relative_to(config.data.data_root.resolve()) else p
    )
    manifest.to_csv(config.trainer.output_dir / "manifest.csv", index=False)
    metrics = pd.concat(
        [run_backbone(config, frame, backbone, freeze_backbone)
         for backbone in config.model.backbones
         for freeze_backbone in config.model.freeze_backbone], ignore_index=True
    )
    metrics.to_csv(config.trainer.output_dir / "fold_metrics.csv", index=False)
    summarize(metrics).to_csv(config.trainer.output_dir / "summary.csv")
    save_sweep_plots(metrics, config)
    print(f"Sweep complete: {config.trainer.output_dir.resolve()}")
    return config.trainer.output_dir


if __name__ == "__main__":
    main()
