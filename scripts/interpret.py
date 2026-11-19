import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn.functional import avg_pool2d

from cornea.checkpoints import load_checkpoint
from cornea.config import load_config
from cornea.dataset import CorneaDataset, load_manifest, make_transforms
from cornea.embeddings import checkpoint_path
from cornea.interpretability import grad_cam, integrated_gradients, patch_shap
from cornea.model import CorneaClassifier
from cornea.plots.style import STYLE, save_figure


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
    image, targets = dataset[index]
    image = image.to(args.device)
    mean = image.new_tensor(config.transforms.mean)[:, None, None]
    std = image.new_tensor(config.transforms.std)[:, None, None]
    baseline = (-mean / std).expand_as(image)
    with torch.no_grad():
        logits = model(torch.stack((image, baseline)))
        probabilities = logits[0].sigmoid()
        target = int(logits[0].argmax()) if args.label is None else labels.index(args.label)

    label = labels[target]
    probability = float(probabilities[target])
    logit_difference = float(logits[0, target] - logits[1, target])
    output = args.output_dir or args.run / "interpretability" / f"{index:04d}_{backbone}_{args.mode}_{label}"
    maps = {}
    method_details = {}
    if "gradcam" in methods:
        maps["gradcam"] = grad_cam(model, image, target, model.backbone.layer4[-1]).cpu()
        method_details["gradcam"] = {"layer": "layer4[-1]", "scale": "relative activation"}
    if "ig" in methods:
        maps["ig"] = integrated_gradients(
            model, image, baseline, target, steps=args.steps, batch_size=args.batch_size,
        ).sum(dim=0).cpu()
        method_details["ig"] = {
            "steps": args.steps, "integration": "gausslegendre",
            "completeness_residual": logit_difference - float(maps["ig"].sum()),
        }
    if "shap" in methods:
        maps["shap"] = patch_shap(
            model, image, baseline, target, grid=args.shap_grid, samples=args.shap_samples,
            batch_size=args.batch_size, seed=config.trainer.seed,
        ).cpu()
        method_details["shap"] = {
            "grid": [args.shap_grid, args.shap_grid],
            "permutations": args.shap_samples,
            "seed": config.trainer.seed,
            "completeness_residual": logit_difference - float(maps["shap"].sum()),
        }

    output.mkdir(parents=True, exist_ok=True)
    for method, attribution in maps.items():
        filename = "attribution.npy" if method == "ig" else f"{method}.npy"
        np.save(output / filename, attribution.numpy())
    (output / "prediction.json").write_text(json.dumps({
        "image_path": frame.iloc[index]["image_path"],
        "preprocessed_path": frame.iloc[index].get("preprocessed_path"),
        "checkpoint": str(checkpoint.resolve()),
        "fold": fold,
        "label": label,
        "probabilities": dict(zip(labels, probabilities.tolist(), strict=True)),
        "reference_labels": [name for name, value in zip(labels, targets.tolist(), strict=True) if value],
        "baseline": "black",
        "attribution_library": "captum",
        "methods": method_details,
        "logit_difference": logit_difference,
    }, indent=2))

    with Image.open(frame.iloc[index]["image_path"]) as original:
        original_rgb = ImageOps.exif_transpose(original).convert("RGB")
    model_rgb = (image * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    titles = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "shap": "Patch SHAP"}
    with plt.rc_context(STYLE):
        columns = 2 + len(methods)
        fig = plt.figure(figsize=(3.4 * columns, 4.6), layout="constrained")
        fig.suptitle(f"{label} · {probability:.1%} · {backbone} {args.mode} · fold {fold}")
        grid = fig.add_gridspec(2, columns, height_ratios=(1, 0.035))
        axes = [fig.add_subplot(grid[0, column]) for column in range(columns)]
        original_ax, input_ax, *overlays = axes
        original_ax.imshow(original_rgb)
        original_ax.set_title("Original")
        input_ax.imshow(model_rgb)
        input_ax.set_title("Classifier input")
        for ax in axes:
            ax.set_axis_off()
            ax.set_anchor("N")
        for column, (ax, method) in enumerate(zip(overlays, methods, strict=True), start=2):
            ax.imshow(model_rgb)
            ax.set_title(titles[method])
            attribution = maps[method]
            if method == "gradcam":
                display_map = attribution.numpy()
                heatmap = ax.imshow(display_map, cmap="magma", vmin=0, vmax=1, alpha=0.85 * display_map)
                scale_label = "Relative activation"
                ticks = [0, 0.5, 1]
            else:
                if method == "ig":
                    attribution = avg_pool2d(
                        attribution[None], kernel_size=5, stride=1, padding=2, count_include_pad=False,
                    )[0].masked_fill((image == baseline).all(dim=0).cpu(), 0)
                display_map = attribution.numpy()
                nonzero = np.abs(display_map[display_map != 0])
                limit = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
                opacity = 0.85 * np.clip(np.abs(display_map) / limit, 0, 1)
                heatmap = ax.imshow(display_map, cmap="coolwarm", vmin=-limit, vmax=limit, alpha=opacity)
                scale_label = "Logit contribution per pixel"
                ticks = [-limit, 0, limit]
            fig.colorbar(heatmap, cax=fig.add_subplot(grid[1, column]), orientation="horizontal",
                         label=scale_label, ticks=ticks, format="%.2g")
        save_figure(fig, output / "saliency.png")
    print(f"{label}: {probability:.1%}")
    print(output / "saliency.png")
    return output


if __name__ == "__main__":
    main()
