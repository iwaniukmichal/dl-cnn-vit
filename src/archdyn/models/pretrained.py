from __future__ import annotations

import torch
from torch import nn

from archdyn.config import ModelConfig
from archdyn.models.custom_cnn import CustomCNN


TIMM_MODEL_NAMES = {
    "efficientnet_b3": "efficientnet_b3",
    "deit_tiny": "deit_tiny_patch16_224",
}


class TimmBackboneClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool, drop_path: float) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path,
        )
        self.feature_dim = getattr(self.backbone, "num_features")
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(model_config: ModelConfig) -> nn.Module:
    if model_config.name == "custom_cnn":
        return CustomCNN(num_classes=model_config.num_classes)
    if model_config.name in {"efficientnet_b3", "deit_tiny"}:
        return TimmBackboneClassifier(
            model_name=TIMM_MODEL_NAMES[model_config.name],
            num_classes=model_config.num_classes,
            pretrained=model_config.pretrained,
            drop_path=model_config.drop_path,
        )
    raise ValueError(f"Unsupported model.name: {model_config.name}")
