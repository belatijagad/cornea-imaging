import torch
from captum.attr import IntegratedGradients


@torch.enable_grad()
def integrated_gradients(model, image, baseline, target, *, steps=64, batch_size=4):
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")

    attribution = IntegratedGradients(model).attribute(
        image.detach().unsqueeze(0),
        baselines=baseline.detach().unsqueeze(0),
        target=target,
        n_steps=steps,
        internal_batch_size=batch_size,
        method="gausslegendre",
    )
    return attribution[0].detach()
