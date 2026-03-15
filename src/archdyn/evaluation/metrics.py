from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def distance_ratio(embeddings: np.ndarray, labels: np.ndarray) -> float:
    class_ids = np.unique(labels)
    centroids = []
    intra = []
    for class_id in class_ids:
        class_embeddings = embeddings[labels == class_id]
        centroid = class_embeddings.mean(axis=0)
        centroids.append(centroid)
        intra.append(np.linalg.norm(class_embeddings - centroid, axis=1).mean())
    centroids = np.stack(centroids)
    inter = []
    for left in range(len(centroids)):
        for right in range(left + 1, len(centroids)):
            inter.append(np.linalg.norm(centroids[left] - centroids[right]))
    return float(np.mean(inter) / max(np.mean(intra), 1e-8))


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, destination: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    pd.DataFrame(matrix).to_csv(destination, index=False)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())
