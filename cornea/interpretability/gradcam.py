import torch
from captum.attr import LayerAttribution, LayerGradCam


@torch.enable_grad()
def grad_cam(model, image, target, layer):
    inputs = image.detach().unsqueeze(0).requires_grad_()
    activations = LayerGradCam(model, layer).attribute(inputs, target=target, relu_attributions=True)
    heatmap = LayerAttribution.interpolate(activations.detach(), image.shape[-2:], interpolate_mode="bilinear")[0, 0]
    return heatmap / heatmap.amax().clamp_min(torch.finfo(heatmap.dtype).tiny)
