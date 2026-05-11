from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    if num_classes < 2:
        raise ValueError("At least two classes are required for classification.")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(path: str, model: nn.Module, class_names: list[str]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
