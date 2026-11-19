from itertools import product
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cornea.config import DEFAULT_CONFIG, load_config
from cornea.model import CorneaClassifier, available_backbones, create_backbone
from cornea.trainer import Trainer, TrainerConfig, classification_metrics
from scripts.run import assign_folds, create_sweep_directory, main, parse_config


DEFAULTS = load_config()
LABELS = DEFAULTS.data.labels
FOLDER_LABELS = DEFAULTS.data.folder_labels


class TrainingTests(unittest.TestCase):
    def test_patient_folds(self):
        rows = [
            {"patient_id": f"patient_{patient}", **dict(zip(LABELS, targets))}
            for patient in range(10)
            for targets in FOLDER_LABELS.values()
        ]
        frame = assign_folds(pd.DataFrame(rows), LABELS)
        self.assertEqual(frame.groupby("patient_id")["fold"].nunique().max(), 1)
        self.assertEqual(set(frame["fold"]), set(range(5)))
        self.assertTrue(frame[LABELS].eq([1, 1, 0]).all(axis=1).any())

    def test_metrics(self):
        targets = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
        probabilities = np.array([[.9, .9, .1], [.1, .1, .9], [.9, .1, .1], [.1, .1, .1]])
        metrics, _ = classification_metrics(targets, probabilities, LABELS)
        np.testing.assert_allclose(metrics.sensitivity, [.5, 1, np.nan], equal_nan=True)
        np.testing.assert_allclose(metrics.specificity, [.5, 1, .75])

    def test_resnet_backbone(self):
        torch.set_num_threads(2)
        model = CorneaClassifier(pretrained=False, freeze_backbone=True).train()
        self.assertEqual(len(model.backbone.layer3), 6)
        self.assertFalse(model.backbone.training)
        logits = model(torch.randn(2, 3, 32, 32))
        nn.BCEWithLogitsLoss()(logits, torch.ones(2, 3)).backward()
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertIsNone(model.backbone.conv1.weight.grad)

    def test_training_and_evaluation(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in FOLDER_LABELS:
                (root / folder).mkdir()
                for patient in ["alpha", "bravo", "charlie", "delta"]:
                    Image.new("RGB", (40, 40), (80, 100, 120)).save(root / folder / f"{patient} 2024.jpg")
            output = root / "outputs"
            def tiny_model(**kwargs):
                backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 16))
                with patch("cornea.model.create_backbone", return_value=(backbone, 16)):
                    return CorneaClassifier(**kwargs)

            config = load_config()
            config.model.backbones = ["resnet18", "swin_t"]
            config.model.freeze_backbone = [False, True]
            config.trainer.threshold = 0.7
            config.trainer.weight_decay = 0.02
            config_path = root / "experiment.yml"
            config.save(config_path)
            with patch("scripts.run.CorneaClassifier", side_effect=tiny_model) as factory, patch("matplotlib.pyplot.show"):
                output = main(["--config", str(config_path), "--data-root", str(root), "--output-dir", str(output),
                      "--folds", "2", "--epochs", "1", "--batch-size", "4",
                      "--image-size", "32", "--no-pretrained"])
            self.assertTrue(output.name.startswith("sweep_"))
            self.assertEqual(output.parent, root / "outputs")
            for name in ("experiment_comparison.png", "experiment_comparison.svg",
                         "confusion_matrices_frozen.png", "confusion_matrices_unfrozen.png"):
                self.assertTrue((output / "plots" / name).is_file(), name)
                if name.endswith(".png"):
                    with Image.open(output / "plots" / name) as image:
                        image.verify()
            pooled = pd.read_csv(output / "out_of_fold_predictions.csv")
            self.assertEqual(len(pooled), 64)
            combinations = list(product(config.model.backbones, config.model.freeze_backbone))
            self.assertEqual(
                [(call.kwargs["backbone"], call.kwargs["freeze_backbone"])
                 for call in factory.call_args_list],
                [combination for combination in combinations for _ in range(2)],
            )
            for backbone, freeze in combinations:
                mode = "frozen" if freeze else "unfrozen"
                run_dir = output / "folds" / backbone / mode
                self.assertEqual({path.name for path in run_dir.iterdir()}, {"fold_1", "fold_2"})
                for fold in [1, 2]:
                    fold_dir = run_dir / f"fold_{fold}"
                    self.assertEqual({path.name for path in fold_dir.iterdir()},
                                     {"best.pt", "history.csv", "validation_metrics.csv"})
                    self.assertTrue((output / "plots" / "training" / backbone / mode / f"fold_{fold}.png").is_file())
                    saved = torch.load(fold_dir / "best.pt", weights_only=True)
                    self.assertEqual(saved["labels"], LABELS)
                    self.assertEqual(saved["backbone"], backbone)
                    self.assertEqual(saved["freeze_backbone"], freeze)
                    self.assertEqual(saved["threshold"], 0.7)
                    self.assertEqual("model_state_dict" in saved, not freeze)
                    if freeze:
                        self.assertTrue((fold_dir / saved["backbone_path"]).is_file())
                predictions = pooled[(pooled.backbone == backbone) & (pooled.freeze_backbone == freeze)]
                self.assertEqual(predictions["image_path"].nunique(), 16)
                self.assertEqual(len(predictions), 16)
            self.assertFalse(list((output / "plots").rglob("*.csv")))
            combined = pd.read_csv(output / "fold_metrics.csv")
            self.assertEqual(len(combined), 24)
            self.assertEqual(set(zip(combined.backbone, combined.freeze_backbone)), set(combinations))
            summary = pd.read_csv(output / "summary.csv")
            self.assertEqual(set(zip(summary.backbone, summary.freeze_backbone)), set(combinations))
            for stat in ("mean", "std"):
                expected = combined.groupby(["backbone", "freeze_backbone", "label"]).specificity.agg(stat)
                actual = summary.set_index(expected.index.names)[f"specificity_{stat}"]
                pd.testing.assert_series_equal(actual, expected, check_names=False)
            for fold in [1, 2]:
                paths = [group.image_path.tolist() for _, group in
                         pooled.loc[pooled.fold == fold].groupby(["backbone", "freeze_backbone"])]
                self.assertTrue(all(part == paths[0] for part in paths))
            saved_config = load_config(output / "config.yml")
            self.assertEqual(saved_config.model.backbones, config.model.backbones)
            self.assertEqual(saved_config.model.freeze_backbone, [False, True])
            self.assertEqual(saved_config.trainer.output_dir, output)

    def test_unique_sweep_directories(self):
        with tempfile.TemporaryDirectory() as directory, patch("scripts.run.datetime") as clock:
            clock.now.return_value.strftime.return_value = "20261110_120000"
            first = create_sweep_directory(Path(directory) / "outputs")
            marker = first / "existing.txt"
            marker.write_text("keep")
            second = create_sweep_directory(Path(directory) / "outputs")
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())
            self.assertEqual(marker.read_text(), "keep")

    def test_backbone_factory(self):
        torch.set_num_threads(2)
        for name in available_backbones():
            with self.subTest(backbone=name):
                model = CorneaClassifier(backbone=name, pretrained=False, freeze_backbone=True).train()
                self.assertFalse(model.backbone.training)
                parameter_count = sum(p.numel() for p in model.backbone.parameters())
                self.assertGreater(parameter_count, 10_000_000)
                self.assertLess(parameter_count, 50_000_000)
                image_size = 224 if name == "maxvit_t" else 64
                logits = model(torch.randn(2, 3, image_size, image_size))
                self.assertEqual(logits.shape, (2, 3))
                logits.sum().backward()
                self.assertIsNotNone(model.classifier.weight.grad)
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
                del model
        with self.assertRaisesRegex(ValueError, "Unknown backbone"):
            create_backbone("missing", pretrained=False)

    def test_parameter_budget_boundaries(self):
        for count in (10_000_000, 50_000_000):
            with self.subTest(parameters=count):
                backbone = nn.Module()
                backbone.features = nn.Parameter(torch.empty(count, device="meta"))
                backbone.fc = nn.Linear(1, 1, device="meta")
                with patch("cornea.model.models.get_model", return_value=backbone):
                    with self.assertRaisesRegex(ValueError, "strictly more than 10M and fewer than 50M"):
                        create_backbone("resnet18", pretrained=False)

    def test_config_and_overrides(self):
        config = parse_config(["--epochs", "2", "--backbones", "resnet18", "swin_s",
                               "--no-pretrained", "--freeze-backbone", "--validation-interval", "2",
                               "--precision", "bf16"])
        self.assertEqual(config.trainer.epochs, 2)
        self.assertEqual(config.trainer.validation_interval, 2)
        self.assertEqual(config.trainer.precision, "bf16")
        self.assertEqual(config.model.backbones, ["resnet18", "swin_s"])
        self.assertFalse(config.model.pretrained)
        self.assertEqual(config.model.freeze_backbone, [True])
        self.assertEqual(config.trainer.learning_rate, DEFAULTS.trainer.learning_rate)
        self.assertEqual(config.data, DEFAULTS.data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            config.save(path)
            self.assertEqual(set(yaml.safe_load(path.read_text())), {"model", "trainer", "data"})
            self.assertEqual(load_config(path), config)
        self.assertEqual(parse_config(["--freeze-backbone", "false", "true"]).model.freeze_backbone,
                         [False, True])
        self.assertEqual(parse_config(["--no-freeze-backbone"]).model.freeze_backbone, [False])
        self.assertEqual(load_config(overrides={"model": {"freeze_backbone": False}}).model.freeze_backbone,
                         [False])
        for overrides in (
            {"model": {"backbones": []}},
            {"model": {"freeze_backbone": []}},
            {"model": {"freeze_backbone": [True, True]}},
            {"model": {"freeze_backbone": ["false"]}},
            {"model": {"freeze_backbone": [0, 1]}},
            {"model": {"backbones": ["missing"]}},
            {"model": {"backbones": ["resnet18", "resnet18"]}},
            {"data": {"folds": 1}},
            {"trainer": {"validation_interval": 0}},
            {"trainer": {"validation_interval": -1}},
            {"trainer": {"validation_interval": 1.5}},
            {"trainer": {"validation_interval": True}},
            {"trainer": {"precision": "invalid"}},
            {"model": {"backbones": ["mobilenet_v3_small"]}},
            {"model": {"backbones": ["vit_b_16"]}},
            {"model": {"backbones": ["maxvit_t"]}, "data": {"image_size": 64}},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_config(DEFAULT_CONFIG, overrides)

    def test_precision_cpu_fallback(self):
        for precision in ("fp16", "bf16", "fp32"):
            with (self.subTest(precision=precision), tempfile.TemporaryDirectory() as directory,
                  patch("torch.cuda.is_available", return_value=False)):
                model = nn.Linear(4, 3)
                loader = DataLoader(TensorDataset(torch.randn(4, 4), torch.ones(4, 3)), batch_size=2)
                trainer = Trainer(model, TrainerConfig(output_dir=Path(directory), precision=precision), loader, loader)
                before = model.weight.detach().clone()
                loss = trainer._run_epoch(loader, training=True)
                self.assertTrue(np.isfinite(loss))
                self.assertFalse(torch.equal(before, model.weight))
                self.assertEqual(trainer.precision, "fp32")
                self.assertFalse(trainer.amp_enabled)
                self.assertFalse(trainer.scaler.is_enabled())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_cuda_mixed_precision_training(self):
        precisions = ["fp16"]
        if torch.cuda.is_bf16_supported():
            precisions.append("bf16")
        for precision in precisions:
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as directory:
                model = nn.Linear(4, 3)
                loader = DataLoader(TensorDataset(torch.randn(4, 4), torch.ones(4, 3)), batch_size=2)
                trainer = Trainer(model, TrainerConfig(output_dir=Path(directory), precision=precision), loader, loader)
                dtypes = []
                hook = model.register_forward_hook(lambda module, args, output: dtypes.append(output.dtype))
                try:
                    before = model.weight.detach().clone()
                    loss = trainer._run_epoch(loader, training=True)
                    val_loss, _, probabilities = trainer._run_epoch(loader)
                finally:
                    hook.remove()
                self.assertTrue(np.isfinite(loss) and np.isfinite(val_loss))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertFalse(torch.equal(before, model.weight))
                self.assertEqual(set(dtypes), {torch.float16 if precision == "fp16" else torch.bfloat16})
                self.assertEqual(model.weight.dtype, torch.float32)
                self.assertEqual(trainer.scaler.is_enabled(), precision == "fp16")

    def test_validation_schedule_and_checkpoint(self):
        for epochs, interval, expected in (
            (50, 5, list(range(5, 51, 5))),
            (12, 5, [5, 10, 12]),
            (3, 5, [3]),
            (3, 1, [1, 2, 3]),
        ):
            with self.subTest(epochs=epochs, interval=interval), tempfile.TemporaryDirectory() as directory:
                model = nn.Linear(1, 1)
                loader = SimpleNamespace(dataset=SimpleNamespace(label_col=["fungal", "bacterial"]))
                trainer = Trainer(model, TrainerConfig(
                    epochs=epochs, validation_interval=interval, output_dir=Path(directory),
                ), loader, loader)
                current_epoch = 0
                validated_epochs = []

                def run_epoch(loader, training=False, description="Val"):
                    nonlocal current_epoch
                    if training:
                        current_epoch += 1
                        with torch.no_grad():
                            model.weight.fill_(current_epoch)
                        return 0.5
                    validated_epochs.append(current_epoch)
                    # The second validation is best; subsequent checks get worse.
                    loss = 0.1 if len(validated_epochs) == 2 else 0.5
                    probabilities = (
                        [[0.9], [0.8], [0.2], [0.1]] if len(validated_epochs) == 1 else
                        [[0.9], [0.2], [0.1], [0.8]] if len(validated_epochs) == 2 else
                        [[0.9], [0.8], [0.7], [0.1]]
                    )
                    return (loss, np.repeat([[1], [0], [0], [1]], 2, axis=1),
                            np.repeat(probabilities, 2, axis=1))

                plots = []

                def capture_plot():
                    ax = plt.gcf().axes[0]
                    plots.append((ax.get_ylabel(), {
                        line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
                        for line in ax.get_lines()
                    }))

                with (patch.object(trainer, "_run_epoch", side_effect=run_epoch),
                      patch("cornea.trainer.tqdm.write"),
                      patch("matplotlib.pyplot.show", side_effect=capture_plot)):
                    trainer.train()
                self.assertEqual(current_epoch, epochs)
                self.assertEqual(validated_epochs, expected)
                history = pd.read_csv(Path(directory) / "history.csv")
                self.assertEqual(len(history), epochs)
                self.assertEqual(history.loc[history.val_loss.notna(), "epoch"].tolist(), expected)
                metrics = pd.read_csv(Path(directory) / "validation_metrics.csv")
                self.assertEqual(metrics.epoch.tolist(), [epoch for epoch in expected for _ in range(2)])
                expected_specificity = ([0.5, 1.0] + [0.0] * len(expected))[:len(expected)]
                np.testing.assert_allclose(metrics.specificity, np.repeat(expected_specificity, 2))
                for name in ("precision", "sensitivity", "specificity", "f1", "auroc", "average_precision"):
                    self.assertIn(name, metrics.columns)
                    np.testing.assert_allclose(
                        history.loc[history.val_loss.notna(), f"val_macro_{name}"],
                        metrics.groupby("epoch")[name].mean(),
                    )
                    self.assertTrue(history.loc[history.val_loss.isna(), f"val_macro_{name}"].isna().all())
                self.assertEqual(plots[0][0], "Specificity")
                self.assertEqual(plots[0][1]["fungal"][0], expected)
                np.testing.assert_allclose(plots[0][1]["fungal"][1], expected_specificity)
                np.testing.assert_allclose(plots[0][1]["Macro average"][1], expected_specificity)
                with Image.open(Path(directory) / "specificity.png") as image:
                    image.verify()
                saved = torch.load(trainer.checkpoint_path, weights_only=True)
                best_epoch = expected[1] if len(expected) > 1 else expected[0]
                self.assertEqual(saved["epoch"], best_epoch)
                self.assertEqual(saved["model_state_dict"]["weight"].item(), best_epoch)




if __name__ == "__main__":
    unittest.main()
