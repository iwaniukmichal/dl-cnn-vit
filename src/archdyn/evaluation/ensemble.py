from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from archdyn.evaluation.embeddings import extract_embeddings
from archdyn.evaluation.metrics import classification_metrics
from archdyn.paths import write_json


def predict_probabilities(
    model,
    dataloader,
    device: torch.device,
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    with torch.inference_mode():
        for batch_index, (images, batch_labels) in enumerate(dataloader, start=1):
            if progress_label is not None:
                _status(_batch_progress_message(progress_label, batch_index, total_batches))
            images = images.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def soft_voting(cnn_model, vit_model, cnn_dataloader, vit_dataloader, device: torch.device) -> tuple[dict[str, float], np.ndarray]:
    cnn_probs, labels = predict_probabilities(cnn_model, cnn_dataloader, device, progress_label="soft voting CNN")
    vit_probs, vit_labels = predict_probabilities(vit_model, vit_dataloader, device, progress_label="soft voting ViT")
    _assert_matching_labels(labels, vit_labels)
    blended = (cnn_probs + vit_probs) / 2
    predictions = blended.argmax(axis=1)
    return classification_metrics(labels, predictions), predictions


def stacking(
    cnn_model,
    vit_model,
    cnn_meta_dataloader,
    vit_meta_dataloader,
    cnn_eval_dataloader,
    vit_eval_dataloader,
    device: torch.device,
    c_value: float,
    output_dir: Path,
) -> dict[str, float]:
    meta_features, meta_labels = _concatenated_embeddings(
        cnn_model,
        vit_model,
        cnn_meta_dataloader,
        vit_meta_dataloader,
        device,
        progress_label="stacking meta",
    )
    eval_features, eval_labels = _concatenated_embeddings(
        cnn_model,
        vit_model,
        cnn_eval_dataloader,
        vit_eval_dataloader,
        device,
        progress_label="stacking eval",
    )
    classifier = LogisticRegression(
        max_iter=1000,
        C=c_value,
        solver="liblinear",
    )
    multiclass_classifier = OneVsRestClassifier(classifier)
    multiclass_classifier.fit(meta_features, meta_labels)
    predictions = multiclass_classifier.predict(eval_features)
    pd.DataFrame(_classifier_coefficients(multiclass_classifier)).to_csv(output_dir / "stacking_coefficients.csv", index=False)
    metrics = classification_metrics(eval_labels, predictions)
    write_json(metrics, output_dir / "stacking_metrics.json")
    return metrics


def protonet_logistic_regression(
    protonet_model,
    meta_dataloader,
    eval_dataloader,
    device: torch.device,
    c_value: float,
    output_dir: Path,
) -> dict[str, float]:
    meta_features, meta_labels = extract_embeddings(
        protonet_model,
        meta_dataloader,
        device,
        progress_label="protonet logreg meta",
    )
    eval_features, eval_labels = extract_embeddings(
        protonet_model,
        eval_dataloader,
        device,
        progress_label="protonet logreg eval",
    )
    classifier = LogisticRegression(
        max_iter=1000,
        C=c_value,
        solver="liblinear",
    )
    multiclass_classifier = OneVsRestClassifier(classifier)
    multiclass_classifier.fit(meta_features, meta_labels)
    predictions = multiclass_classifier.predict(eval_features)
    pd.DataFrame(_classifier_coefficients(multiclass_classifier)).to_csv(
        output_dir / "protonet_logreg_coefficients.csv",
        index=False,
    )
    pd.DataFrame({"label": eval_labels, "prediction": predictions}).to_csv(
        output_dir / "protonet_logreg_predictions.csv",
        index=False,
    )
    metrics = classification_metrics(eval_labels, predictions)
    write_json(metrics, output_dir / "protonet_logreg_metrics.json")
    return metrics


def _concatenated_embeddings(
    cnn_model,
    vit_model,
    cnn_dataloader,
    vit_dataloader,
    device: torch.device,
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cnn_model.eval()
    vit_model.eval()
    cnn_embeddings = []
    vit_embeddings = []
    labels = []
    cnn_total_batches = _dataloader_length(cnn_dataloader)
    vit_total_batches = _dataloader_length(vit_dataloader)
    total_batches = min(cnn_total_batches, vit_total_batches) if None not in (cnn_total_batches, vit_total_batches) else None
    with torch.inference_mode():
        for batch_index, ((cnn_images, cnn_labels), (vit_images, vit_labels)) in enumerate(
            zip(cnn_dataloader, vit_dataloader, strict=True),
            start=1,
        ):
            if progress_label is not None:
                _status(_batch_progress_message(progress_label, batch_index, total_batches))
            _assert_matching_labels(cnn_labels.numpy(), vit_labels.numpy())
            cnn_images = cnn_images.to(device, non_blocking=device.type == "cuda")
            vit_images = vit_images.to(device, non_blocking=device.type == "cuda")
            cnn_embeddings.append(cnn_model.forward_features(cnn_images).cpu().numpy())
            vit_embeddings.append(vit_model.forward_features(vit_images).cpu().numpy())
            labels.append(cnn_labels.numpy())
    return np.concatenate([np.concatenate(cnn_embeddings), np.concatenate(vit_embeddings)], axis=1), np.concatenate(labels)


def _assert_matching_labels(labels: np.ndarray, other_labels: np.ndarray) -> None:
    if not np.array_equal(labels, other_labels):
        raise ValueError("CNN and ViT dataloaders must yield samples in identical order")


def _dataloader_length(dataloader) -> int | None:
    return len(dataloader) if hasattr(dataloader, "__len__") else None


def _batch_progress_message(label: str, batch_index: int, total_batches: int | None) -> str:
    batch_suffix = f"/{total_batches}" if total_batches is not None else ""
    return f"{label}: batch {batch_index}{batch_suffix}"


def _status(message: str) -> None:
    print(f"[ensemble-eval] {message}", flush=True)


def _classifier_coefficients(classifier: OneVsRestClassifier) -> np.ndarray:
    return np.vstack([estimator.coef_.reshape(-1) for estimator in classifier.estimators_])
