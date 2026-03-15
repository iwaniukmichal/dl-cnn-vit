from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        raise AttributeError("Backbone must implement forward_features")

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        support_embeddings = self.embed(support_images)
        query_embeddings = self.embed(query_images)
        prototypes = []
        num_classes = int(support_labels.max().item()) + 1
        for class_id in range(num_classes):
            class_mask = support_labels == class_id
            prototypes.append(support_embeddings[class_mask].mean(dim=0))
        prototype_tensor = torch.stack(prototypes)
        distances = torch.cdist(query_embeddings, prototype_tensor)
        return -distances


def prototypical_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)
