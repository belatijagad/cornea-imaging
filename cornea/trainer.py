from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from sklearn.metrics import (
    average_precision_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from cornea.checkpoints import CheckpointStore, load_checkpoint
from cornea.config import (
    CheckpointSelection,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    Precision,
    Probability,
)
from cornea.plots.metrics import plot_specificity

METRIC_COLUMNS = ["precision", "sensitivity", "specificity", "f1", "auroc", "average_precision"]


@dataclass(config=ConfigDict(extra="forbid"))
class TrainerConfig:
    epochs: PositiveInt = 50
    learning_rate: PositiveFloat = 1e-4
    output_dir: Path = Path("outputs")
    threshold: Probability = 0.5
    weight_decay: NonNegativeFloat = 1e-4
    validation_interval: PositiveInt = 5
    precision: Precision = "fp16"
    backbone_dir: Path | None = None
    checkpoint_selection: CheckpointSelection = "best"


def classification_metrics(targets, probabilities, labels, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    matrices = multilabel_confusion_matrix(targets, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    rows = []
    for i, label in enumerate(labels):
        tn, fp = matrices[i, 0]
        has_both = np.unique(targets[:, i]).size == 2
        rows.append({
            "label": label,
            "precision": precision[i],
            "sensitivity": recall[i] if support[i] else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "f1": f1[i],
            "positives": int(support[i]),
            "auroc": roc_auc_score(targets[:, i], probabilities[:, i]) if has_both else np.nan,
            "average_precision": average_precision_score(targets[:, i], probabilities[:, i])
            if has_both else np.nan,
        })
    return pd.DataFrame(rows), matrices


class Trainer:
    def __init__(self, model, config, train_dataloader, val_dataloader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = config.precision if self.device.type == "cuda" else "fp32"
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 precision is not supported by this CUDA device; use fp16")
        self.amp_enabled = self.precision != "fp32"
        self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.model = model.to(self.device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / f"{config.checkpoint_selection}.pt"
        self.checkpoints = CheckpointStore(self.model, self.checkpoint_path, config.backbone_dir)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")
        print(f"Device: {self.device} | Precision: {self.precision} (requested: {config.precision})")

    def _run_epoch(self, loader, training=False, description="Val"):
        self.model.train(training)
        total_loss, seen = 0.0, 0
        targets_list, probabilities_list = [], []
        progress = tqdm(loader, desc=description, leave=False)
        use_amp = self.amp_enabled and (training or self.config.checkpoint_selection == "best")
        for images, targets in progress:
            images, targets = images.to(self.device), targets.to(self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                    enabled=use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            seen += images.size(0)
            total_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{total_loss / seen:.4f}")
            if not training:
                targets_list.append(targets.cpu().numpy())
                probabilities_list.append(logits.detach().float().sigmoid().cpu().numpy())
        if training:
            return total_loss / seen
        return total_loss / seen, np.concatenate(targets_list), np.concatenate(probabilities_list)

    def train(self, *, specificity_path=None):
        best_loss = float("inf")
        history = []
        validation_metrics = []
        labels = list(self.val_dataloader.dataset.label_col)
        use_best = self.config.checkpoint_selection == "best"
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(
                self.train_dataloader, training=True, description=f"Train {epoch}/{self.config.epochs}"
            )
            val_loss = None
            save = False
            row = {"epoch": epoch, "train_loss": train_loss,
                   **{f"val_macro_{name}": None for name in METRIC_COLUMNS}}
            if epoch == self.config.epochs or (use_best and epoch % self.config.validation_interval == 0):
                val_loss, targets, probabilities = self._run_epoch(self.val_dataloader)
                metrics, _ = classification_metrics(targets, probabilities, labels, self.config.threshold)
                validation_metrics.append(metrics.assign(
                    epoch=epoch, val_loss=val_loss, threshold=self.config.threshold,
                ))
                row.update({f"val_macro_{name}": metrics[name].mean() for name in METRIC_COLUMNS})
                save = val_loss < best_loss if use_best else True
            row["val_loss"] = val_loss
            history.append(row)
            tqdm.write(
                f"Epoch {epoch:02d}/{self.config.epochs} | train loss: {train_loss:.4f}"
                + (f" | val loss: {val_loss:.4f}" if val_loss is not None else "")
                + (f" | specificity: {row['val_macro_specificity']:.4f}" if val_loss is not None else "")
                + (" | saved" if save else "")
            )
            if save:
                best_loss = val_loss
                self.checkpoints.save({
                    "epoch": epoch, "val_loss": val_loss,
                    "labels": labels,
                    "threshold": self.config.threshold,
                    "backbone": getattr(self.model, "backbone_name", None),
                    "freeze_backbone": getattr(self.model, "freeze_backbone", None),
                    "precision": self.precision,
                })
        pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
        validation_metrics = pd.concat(validation_metrics, ignore_index=True)
        validation_metrics.to_csv(self.output_dir / "validation_metrics.csv", index=False)
        plot_specificity(
            validation_metrics, specificity_path or self.output_dir / "specificity.png", self.config.threshold,
        )

    def eval(self):
        saved = load_checkpoint(self.model, self.checkpoint_path)
        val_loss, targets, probabilities = self._run_epoch(self.val_dataloader)
        labels = saved["labels"]
        metrics, _ = classification_metrics(targets, probabilities, labels, saved["threshold"])
        print(f"{self.config.checkpoint_selection.capitalize()} epoch: {saved['epoch']} | val loss: {val_loss:.4f}")
        print(metrics.round(4).to_string(index=False))
        predictions = self.val_dataloader.dataset.frame[["image_path", *labels]].copy()
        for i, label in enumerate(labels):
            predictions[f"{label}_probability"] = probabilities[:, i]
            predictions[f"{label}_prediction"] = (probabilities[:, i] >= saved["threshold"]).astype(int)
        return metrics, predictions
