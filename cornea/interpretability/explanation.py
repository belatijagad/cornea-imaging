import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn.functional import avg_pool2d

from cornea.plots.style import INK, STYLE, TEAL, TERRACOTTA, save_figure

from .gradcam import grad_cam
from .integrated import integrated_gradients
from .shap import patch_shap


def explain_prediction(config, dataset, index, model, output_dir, *, label=None,
                       methods=("gradcam", "ig", "shap"), steps=64, shap_grid=7, shap_samples=32,
                       batch_size=4, outcome=None, recorded_probabilities=None, variant=None):
    row = dataset.frame.iloc[index]
    fold = int(row["fold"]) + 1
    labels = config.data.labels
    backbone = model.backbone_name
    mode = "frozen" if model.freeze_backbone else "unfrozen"
    image, targets = dataset[index]
    image = image.to(next(model.parameters()).device)
    mean = image.new_tensor(config.transforms.mean)[:, None, None]
    std = image.new_tensor(config.transforms.std)[:, None, None]
    baseline = (-mean / std).expand_as(image)
    probabilities = recorded_probabilities
    if probabilities is None:
        with torch.no_grad():
            probabilities = model(image.unsqueeze(0))[0].sigmoid().tolist()
    target = int(np.argmax(probabilities)) if label is None else labels.index(label)
    label = labels[target]
    maps = {}
    if "gradcam" in methods:
        maps["gradcam"] = grad_cam(model, image, target, model.backbone.layer4[-1]).cpu()
    if "ig" in methods:
        maps["ig"] = integrated_gradients(
            model, image, baseline, target, steps=steps, batch_size=batch_size,
        ).sum(dim=0).cpu()
    if "shap" in methods:
        maps["shap"] = patch_shap(
            model, image, baseline, target, grid=shap_grid, samples=shap_samples,
            batch_size=batch_size, seed=config.trainer.seed,
        ).cpu()
    case = {
        "image_path": row["image_path"],
        "input_type": "Masked image" if row.get("preprocessed_path") or dataset.cornea_boxes is not None else "Raw image",
        "fold": fold,
        "label": label,
        "probabilities": dict(zip(labels, probabilities, strict=True)),
        "predicted_labels": [name for name, value in zip(labels, probabilities, strict=True)
                             if value >= config.trainer.threshold],
        "outcome": outcome,
        "reference_labels": [name for name, value in zip(labels, targets.tolist(), strict=True) if value],
    }
    suffix = variant or f"{backbone}-{mode}"
    figure = output_dir / f"case-{index:04d}-{label.replace('_', '-')}-{suffix}.png"
    plot_saliency(
        image, {method: maps[method] for method in methods},
        config.transforms, figure, case, index=index, backbone=backbone, mode=mode,
    )
    print(figure)
    return figure


def plot_saliency(image, maps, transforms, output, case, *, index, backbone, mode):
    image = image.detach().cpu()
    mean = image.new_tensor(transforms.mean)[:, None, None]
    std = image.new_tensor(transforms.std)[:, None, None]
    baseline = -mean / std
    with Image.open(case["image_path"]) as original:
        original_rgb = ImageOps.exif_transpose(original).convert("RGB")
    model_rgb = (image * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    titles = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "shap": "Patch SHAP"}
    labels = {"viral_other": "Viral / Other"}
    labels.update({label: label.replace("_", " ").title() for label in case["probabilities"] if label not in labels})
    label = case["label"]
    reference = case["reference_labels"]
    predicted = case["predicted_labels"]
    outcome = case.get("outcome") or ("correct" if set(reference) == set(predicted) else "incorrect")
    input_type = case.get("input_type", "Masked image" if case.get("preprocessed_path") else "Raw image")
    model_name = backbone.replace("resnet", "ResNet-") if backbone.startswith("resnet") else backbone
    rows = []
    for heading, values in (("Reference", reference), ("Predicted", predicted)):
        if set(values) == {"fungal", "bacterial"}:
            value = "Mixed (fungal and bacterial)"
        else:
            value = " + ".join(labels[name] for name in values) or "No label"
        rows.append((heading, value))
    rows.append((f"{labels[label]} score", f"{case['probabilities'][label]:.1%}"))

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(9.6, 7.6))
        fig.suptitle(f"{labels[label]} attribution", x=0.06, y=0.96, ha="left", fontsize=19)
        grid = fig.add_gridspec(2, 3, left=0.06, right=0.96, top=0.88, bottom=0.13, wspace=0.20, hspace=0.34)
        original_ax, input_ax, summary_ax = [fig.add_subplot(grid[0, column]) for column in range(3)]
        original_ax.imshow(original_rgb)
        original_ax.set_title("Original photo", loc="left", fontsize=11)
        input_ax.imshow(model_rgb)
        input_ax.set_title("Model input", loc="left", fontsize=11)
        for ax in (original_ax, input_ax, summary_ax):
            ax.set_axis_off()
            ax.set_anchor("N")
        summary_ax.set_title(f"Case {index:04d}", loc="left", fontsize=11)
        summary_ax.text(0, 0.97, input_type, fontsize=10, va="top", color=INK)
        summary_ax.text(
            0, 0.85, outcome.capitalize(), fontsize=13, fontweight="bold", va="top",
            color=TEAL if outcome == "correct" else TERRACOTTA,
        )
        for (heading, value), y in zip(rows, (0.65, 0.45, 0.25), strict=True):
            summary_ax.text(0, y, heading, fontsize=8, va="top", color=INK)
            summary_ax.text(0, y - 0.075, value, fontsize=11, va="top", color=INK)
        summary_ax.text(0, 0.01, f"{model_name} ({mode}), fold {case['fold']}", fontsize=8, color=INK)

        for column, (method, attribution) in enumerate(maps.items()):
            ax = fig.add_subplot(grid[1, column])
            ax.set_axis_off()
            ax.set_anchor("N")
            ax.imshow(model_rgb)
            ax.set_title(titles[method], loc="left", fontsize=11)
            if method == "gradcam":
                display_map = attribution.numpy()
                if not display_map.any():
                    ax.text(0.5, -0.10, "No positive activation", ha="center", fontsize=8, transform=ax.transAxes)
                    continue
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
                scale_label = "Logit per pixel"
                ticks = [-limit, 0, limit]
            color_axis = ax.inset_axes([0.12, -0.11, 0.76, 0.035])
            colorbar = fig.colorbar(
                heatmap, cax=color_axis, orientation="horizontal", ticks=ticks, format="%.2g",
            )
            colorbar.set_label(scale_label, fontsize=8)
            colorbar.ax.tick_params(labelsize=8)
        save_figure(fig, output)
