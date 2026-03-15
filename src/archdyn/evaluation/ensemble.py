from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from archdyn.evaluation.metrics import classification_metrics
from archdyn.paths import write_json


def predict_probabilities(model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def soft_voting(cnn_model, vit_model, dataloader, device: torch.device) -> tuple[dict[str, float], np.ndarray]:
    cnn_probs, labels = predict_probabilities(cnn_model, dataloader, device)
    vit_probs, _ = predict_probabilities(vit_model, dataloader, device)
    blended = (cnn_probs + vit_probs) / 2
    predictions = blended.argmax(axis=1)
    return classification_metrics(labels, predictions), predictions


def stacking(
    cnn_model,
    vit_model,
    meta_dataloader,
    eval_dataloader,
    device: torch.device,
    c_value: float,
    output_dir: Path,
) -> dict[str, float]:
    meta_features, meta_labels = _concatenated_embeddings(cnn_model, vit_model, meta_dataloader, device)
    eval_features, eval_labels = _concatenated_embeddings(cnn_model, vit_model, eval_dataloader, device)
    classifier = LogisticRegression(
        max_iter=500,
        C=c_value,
        solver="liblinear",
    )
    classifier.fit(meta_features, meta_labels)
    predictions = classifier.predict(eval_features)
    pd.DataFrame(classifier.coef_).to_csv(output_dir / "stacking_coefficients.csv", index=False)
    metrics = classification_metrics(eval_labels, predictions)
    write_json(metrics, output_dir / "stacking_metrics.json")
    return metrics


def _concatenated_embeddings(cnn_model, vit_model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    cnn_embeddings = []
    vit_embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            cnn_embeddings.append(cnn_model.forward_features(images).cpu().numpy())
            vit_embeddings.append(vit_model.forward_features(images).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate([np.concatenate(cnn_embeddings), np.concatenate(vit_embeddings)], axis=1), np.concatenate(labels)
