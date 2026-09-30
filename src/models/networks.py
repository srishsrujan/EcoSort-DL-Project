from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class BaselineCNN(nn.Module):
    """Small convolutional baseline used as a deliberately simple reference model."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MobileNetV3SmallClassifier(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small fine-tuned for EcoSort classes."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        input_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(input_features, num_classes)
        if freeze_backbone:
            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.classifier.parameters():
                parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(kind: str, num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    kind = kind.lower().strip()
    if kind == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if kind == "transfer":
        return MobileNetV3SmallClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def load_checkpoint(model: nn.Module, checkpoint_path: str | None, device: str) -> dict:
    if not checkpoint_path:
        return {}
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["state_dict"] if "state_dict" in payload else payload)
    return payload
