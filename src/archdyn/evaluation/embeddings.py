from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score

from archdyn.evaluation.metrics import distance_ratio


def extract_embeddings(model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    dataset_size = len(dataloader.dataset) if hasattr(dataloader, "dataset") else "unknown"
    _status(f"Starting embedding extraction for {dataset_size} samples")
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            batch_embeddings = model.embed(images) if hasattr(model, "embed") else model.forward_features(images)
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.append(batch_labels.numpy())
    stacked_embeddings = np.concatenate(embeddings)
    stacked_labels = np.concatenate(labels)
    _status(
        f"Finished embedding extraction: samples={len(stacked_labels)} embedding_dim={stacked_embeddings.shape[1]}"
    )
    return stacked_embeddings, stacked_labels


def compute_embedding_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "silhouette_score": float(silhouette_score(embeddings, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(embeddings, labels)),
        "distance_ratio": float(distance_ratio(embeddings, labels)),
    }


def centroid_distance_matrix(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    class_ids = sorted(np.unique(labels).tolist())
    centroids = np.stack([embeddings[labels == class_id].mean(axis=0) for class_id in class_ids])
    distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    return distances, class_ids


def save_embedding_artifacts(
    embeddings: np.ndarray,
    labels: np.ndarray,
    run_dir: Path,
    max_tsne_samples: int,
    enable_tsne: bool,
    random_state: int,
    artifact_prefix: str | None = None,
) -> None:
    _status(f"Saving embeddings and metrics to {run_dir}")
    prefix = f"{artifact_prefix}_" if artifact_prefix else ""
    sklearn_embeddings = embeddings.astype(np.float64, copy=False)
    np.savez(run_dir / f"{prefix}embeddings.npz", embeddings=embeddings, labels=labels)
    metrics = compute_embedding_metrics(sklearn_embeddings, labels)
    pd.DataFrame([metrics]).to_csv(run_dir / f"{prefix}embedding_metrics.csv", index=False)

    _status("Running PCA projection")
    pca = PCA(n_components=2, random_state=random_state, svd_solver="full")
    pca_projection = pca.fit_transform(sklearn_embeddings)
    _scatter_plot(pca_projection, labels, run_dir / "plots" / f"{prefix}pca.png", "PCA")

    if enable_tsne:
        if len(sklearn_embeddings) > max_tsne_samples:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(len(sklearn_embeddings), size=max_tsne_samples, replace=False)
            tsne_embeddings = sklearn_embeddings[indices]
            tsne_labels = labels[indices]
        else:
            tsne_embeddings = sklearn_embeddings
            tsne_labels = labels
        _status(f"Running t-SNE projection on {len(tsne_embeddings)} samples")
        tsne = TSNE(n_components=2, random_state=random_state, init="pca")
        tsne_projection = tsne.fit_transform(tsne_embeddings)
        _scatter_plot(tsne_projection, tsne_labels, run_dir / "plots" / f"{prefix}tsne.png", "t-SNE")
    else:
        _status("Skipping t-SNE projection because analysis.enable_tsne=false")

    _status("Computing centroid distance matrix")
    distances, class_ids = centroid_distance_matrix(sklearn_embeddings, labels)
    pd.DataFrame(distances, index=class_ids, columns=class_ids).to_csv(run_dir / f"{prefix}centroid_distances.csv")
    _heatmap_plot(distances, class_ids, run_dir / "plots" / f"{prefix}centroid_heatmap.png")
    _status("Finished saving embedding artifacts")


def _scatter_plot(projection: np.ndarray, labels: np.ndarray, destination: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(projection[:, 0], projection[:, 1], c=labels, s=8, cmap="tab10", alpha=0.8)
    axis.set_title(f"{title} Projection of Image Embeddings")
    axis.set_xlabel(f"{title} Component 1")
    axis.set_ylabel(f"{title} Component 2")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Class ID")
    figure.text(
        0.5,
        0.01,
        "Each point is one image embedding; colors indicate the ground-truth class.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(destination)
    plt.close(figure)


def _heatmap_plot(matrix: np.ndarray, class_ids: list[int], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="viridis")
    axis.set_title("Centroid Distance Heatmap")
    axis.set_xlabel("Class ID")
    axis.set_ylabel("Class ID")
    axis.set_xticks(range(len(class_ids)), class_ids)
    axis.set_yticks(range(len(class_ids)), class_ids)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Euclidean Distance")
    figure.text(
        0.5,
        0.01,
        "Cells show pairwise distances between class centroids in embedding space.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(destination)
    plt.close(figure)


def _status(message: str) -> None:
    print(f"[embeddings] {message}", flush=True)
