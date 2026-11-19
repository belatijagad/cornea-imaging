import torch
from captum.attr import ShapleyValueSampling


@torch.no_grad()
def patch_shap(model, image, baseline, target, *, grid=7, samples=32, batch_size=4, seed=42):
    height, width = image.shape[-2:]
    if not 1 <= grid <= min(height, width) or samples < 1 or batch_size < 1:
        raise ValueError("SHAP grid must fit the image and samples and batch_size must be positive")

    rows = torch.arange(height, device=image.device) * grid // height
    cols = torch.arange(width, device=image.device) * grid // width
    patches = rows[:, None] * grid + cols[None, :]
    patch_sizes = torch.bincount(patches.flatten(), minlength=grid**2)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        attribution = ShapleyValueSampling(model).attribute(
            image.unsqueeze(0),
            baselines=baseline.unsqueeze(0),
            target=target,
            feature_mask=patches[None, None],
            n_samples=samples,
            perturbations_per_eval=batch_size,
        )
    return attribution[0, 0] / patch_sizes[patches]
