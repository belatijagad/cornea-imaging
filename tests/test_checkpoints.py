import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from cornea.checkpoints import CheckpointStore, load_checkpoint
from cornea.model import CorneaClassifier


def make_model(frozen=True):
    backbone = nn.Sequential(nn.Linear(4, 64), nn.BatchNorm1d(64), nn.ReLU())
    with patch("cornea.model.create_backbone", return_value=(backbone, 64)):
        return CorneaClassifier(pretrained=False, freeze_backbone=frozen)


class CheckpointTests(unittest.TestCase):
    def test_shared_backbone_and_relocated_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sweep"
            shared = root / "backbones"
            first = make_model().eval()
            second = make_model().eval()
            second.backbone.load_state_dict(first.backbone.state_dict())
            inputs = torch.randn(3, 4)
            expected = []
            for fold, model in enumerate((first, second), 1):
                path = root / "resnet50" / "frozen" / f"fold_{fold}" / "best.pt"
                store = CheckpointStore(model, path, shared)
                store.save({"epoch": 5, "freeze_backbone": True})
                saved = torch.load(path, weights_only=True)
                self.assertNotIn("model_state_dict", saved)
                self.assertEqual(set(saved["classifier_state_dict"]), {"weight", "bias"})
                expected.append(model(inputs).detach())
            self.assertEqual(len(list(shared.glob("*.pt"))), 1)
            moved = Path(directory) / "moved"
            shutil.move(root, moved)
            for fold, logits in enumerate(expected, 1):
                restored = make_model().eval()
                path = moved / "resnet50" / "frozen" / f"fold_{fold}" / "best.pt"
                saved = load_checkpoint(restored, path)
                self.assertEqual(saved["epoch"], 5)
                torch.testing.assert_close(restored(inputs), logits)

    def test_different_frozen_initializations_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold in (1, 2):
                CheckpointStore(make_model(), root / f"fold_{fold}" / "best.pt", root / "backbones").save({})
            self.assertEqual(len(list((root / "backbones").glob("*.pt"))), 2)

    def test_classifier_updates_and_full_checkpoint_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = make_model().eval()
            store = CheckpointStore(model, root / "best.pt", root / "backbones")
            store.save({"epoch": 5})
            with torch.no_grad():
                model.classifier.weight.add_(1)
            store.save({"epoch": 10})
            restored = make_model().eval()
            self.assertEqual(load_checkpoint(restored, root / "best.pt")["epoch"], 10)
            torch.testing.assert_close(restored.classifier.weight, model.classifier.weight)
            self.assertEqual(len(list((root / "backbones").glob("*.pt"))), 1)
            unfrozen = make_model(frozen=False).eval()
            CheckpointStore(unfrozen, root / "full.pt").save({"epoch": 15})
            full = torch.load(root / "full.pt", weights_only=True)
            self.assertIn("model_state_dict", full)
            self.assertNotIn("backbone_path", full)
            self.assertLess((root / "best.pt").stat().st_size, (root / "full.pt").stat().st_size)
            full.pop("checkpoint_type")
            torch.save(full, root / "legacy.pt")
            load_checkpoint(restored, root / "legacy.pt")
            inputs = torch.randn(2, 4)
            torch.testing.assert_close(restored(inputs), unfrozen(inputs))

    def test_missing_or_modified_shared_backbone_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = CheckpointStore(make_model(), root / "best.pt")
            store.save({})
            saved = torch.load(root / "best.pt", weights_only=True)
            backbone_path = root / saved["backbone_path"]
            state = torch.load(backbone_path, weights_only=True)
            state["1.running_mean"].add_(1)
            torch.save(state, backbone_path)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_checkpoint(make_model(), root / "best.pt")
            backbone_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Shared backbone is missing"):
                load_checkpoint(make_model(), root / "best.pt")

    def test_trainable_backbone_cannot_use_classifier_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = make_model()
            model.backbone.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "trainable backbone"):
                CheckpointStore(model, Path(directory) / "best.pt").save({})
