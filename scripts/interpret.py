import argparse
from pathlib import Path

import numpy as np
import torch

from cornea.checkpoints import checkpoint_path, load_checkpoint
from cornea.config import load_config
from cornea.dataset import CorneaDataset, load_manifest, make_transforms
from cornea.interpretability.explanation import explain_prediction
from cornea.model import CorneaClassifier


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare saliency methods for a held-out classifier prediction.")
    parser.add_argument("run", type=Path, help="Experiment output directory")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int, help="Zero-based image row in the run manifest")
    selection.add_argument("--image", type=Path, help="Original image path relative to data_root, or an absolute path")
    parser.add_argument("--backbone", help="Backbone to explain. Defaults to the first backbone in the run.")
    parser.add_argument("--mode", choices=("frozen", "unfrozen"), default="unfrozen")
    parser.add_argument("--label", help="Target class. Defaults to the class with the highest score.")
    parser.add_argument("--methods", nargs="+", choices=("gradcam", "ig", "shap"),
                        help="Defaults to all three for ResNet and IG plus SHAP for other backbones.")
    parser.add_argument("--steps", type=int, default=64, help="Integrated Gradients integration steps")
    parser.add_argument("--shap-grid", type=int, default=7, help="Number of SHAP patches along each image axis")
    parser.add_argument("--shap-samples", type=int, default=32, help="Number of sampled patch orders")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of model evaluations per batch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if min(args.steps, args.shap_grid, args.shap_samples, args.batch_size) < 1:
        parser.error("--steps, --shap-grid, --shap-samples, and --batch-size must be positive")

    config = load_config(args.run / "config.yml")
    frame = load_manifest(args.run, config)
    index = args.index
    if args.image is not None:
        path = str((config.data.data_root / args.image).resolve())
        matches = np.flatnonzero(frame["image_path"].eq(path))
        if len(matches) != 1:
            parser.error("--image must match exactly one row in the run manifest")
        index = int(matches[0])
    if not 0 <= index < len(frame):
        parser.error(f"--index must be between 0 and {len(frame) - 1}")

    backbone = args.backbone or config.model.backbones[0]
    resnet = backbone.startswith("resnet")
    methods = tuple(dict.fromkeys(args.methods)) if args.methods else (
        ("gradcam", "ig", "shap") if resnet else ("ig", "shap")
    )
    if "gradcam" in methods and not resnet:
        parser.error("Grad-CAM currently requires a ResNet backbone")
    if "shap" in methods and args.shap_grid > config.transforms.image_size:
        parser.error("--shap-grid cannot exceed the classifier image size")
    fold = int(frame.iloc[index]["fold"]) + 1
    checkpoint = checkpoint_path(args.run, backbone, args.mode, fold)
    if not checkpoint.is_file():
        parser.error(f"Checkpoint for held-out fold {fold} is missing: {checkpoint}")

    labels = config.data.labels
    if args.label is not None and args.label not in labels:
        parser.error(f"--label must be one of {', '.join(labels)}")
    model = CorneaClassifier(
        num_labels=len(labels), freeze_backbone=args.mode == "frozen", pretrained=False, backbone=backbone,
    )
    saved = load_checkpoint(model, checkpoint)
    if list(saved.get("labels", labels)) != labels:
        raise ValueError("Checkpoint labels do not match the experiment configuration")
    model.to(args.device).eval().requires_grad_(False)
    dataset = CorneaDataset(frame, make_transforms(config.transforms), labels)
    return explain_prediction(
        config, dataset, index, model, args.output_dir or args.run / "interpretability",
        label=args.label, methods=methods,
        steps=args.steps, shap_grid=args.shap_grid, shap_samples=args.shap_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
