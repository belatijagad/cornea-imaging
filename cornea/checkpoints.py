import hashlib
import os
from pathlib import Path
import tempfile

import torch


def _state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class CheckpointStore:
    def __init__(self, model, path, backbone_dir=None):
        self.model = model
        self.path = Path(path)
        self.backbone_dir = Path(backbone_dir) if backbone_dir is not None else self.path.parent / "backbones"
        self._backbone_reference = None

    def save(self, metadata):
        saved = dict(metadata)
        if getattr(self.model, "freeze_backbone", False):
            if any(parameter.requires_grad for parameter in self.model.backbone.parameters()):
                raise ValueError("Cannot save a classifier-only checkpoint with trainable backbone weights")
            if self._backbone_reference is None:
                state = {name: tensor.detach().cpu() for name, tensor in self.model.backbone.state_dict().items()}
                digest = _state_digest(state)
                backbone_path = self.backbone_dir / f"{digest}.pt"
                if not backbone_path.exists():
                    _atomic_save(state, backbone_path)
                self._backbone_reference = {
                    "backbone_path": os.path.relpath(backbone_path, self.path.parent),
                    "backbone_sha256": digest,
                }
            saved.update(self._backbone_reference)
            saved["classifier_state_dict"] = self.model.classifier.state_dict()
            saved["checkpoint_type"] = "classifier"
        else:
            saved["model_state_dict"] = self.model.state_dict()
            saved["checkpoint_type"] = "full"
        _atomic_save(saved, self.path)


def fold_directory(run, backbone, mode, fold):
    return Path(run) / "folds" / backbone / mode / f"fold_{fold}"


def load_checkpoint(model, path):
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in saved:
        model.load_state_dict(saved["model_state_dict"])
    else:
        backbone_path = path.parent / saved["backbone_path"]
        if not backbone_path.is_file():
            raise FileNotFoundError(f"Shared backbone is missing: {backbone_path}")
        state = torch.load(backbone_path, map_location="cpu", weights_only=True)
        if _state_digest(state) != saved["backbone_sha256"]:
            raise ValueError(f"Shared backbone checksum mismatch: {backbone_path}")
        model.backbone.load_state_dict(state)
        model.classifier.load_state_dict(saved["classifier_state_dict"])
    return saved
