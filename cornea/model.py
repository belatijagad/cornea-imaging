import torch.nn as nn
from torchvision import models

BACKBONES = {
    "resnet18": "fc",
    "resnet34": "fc",
    "resnet50": "fc",
    "resnet101": "fc",
    "convnext_tiny": "classifier_last",
    "swin_t": "head",
    "swin_s": "head",
    "swin_v2_t": "head",
    "swin_v2_s": "head",
    "maxvit_t": "classifier_last",
}


def create_backbone(name, pretrained=True):
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {tuple(BACKBONES)}")
    weights = models.get_model_weights(name).DEFAULT if pretrained else None
    backbone = models.get_model(name, weights=weights)
    head_name = BACKBONES[name]
    if head_name == "classifier_last":
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
    else:
        in_features = getattr(backbone, head_name).in_features
        setattr(backbone, head_name, nn.Identity())
    parameter_count = sum(p.numel() for p in backbone.parameters())
    if not 10_000_000 < parameter_count < 50_000_000:
        raise ValueError(
            f"Backbone {name!r} has {parameter_count:,} parameters; "
            "must be strictly more than 10M and fewer than 50M"
        )
    return backbone, in_features


class CorneaClassifier(nn.Module):
    def __init__(self, num_labels=3, freeze_backbone=False, pretrained=True, backbone="resnet50"):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, in_features = create_backbone(backbone, pretrained)
        self.classifier = nn.Linear(in_features, num_labels)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
