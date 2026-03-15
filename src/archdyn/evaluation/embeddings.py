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
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            batch_embeddings = model.embed(images) if hasattr(model, "embed") else model.forward_features(images)
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


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
    random_state: int,
) -> None:
    np.savez(run_dir / "embeddings.npz", embeddings=embeddings, labels=labels)
    metrics = compute_embedding_metrics(embeddings, labels)
    pd.DataFrame([metrics]).to_csv(run_dir / "embedding_metrics.csv", index=False)

    pca = PCA(n_components=2, random_state=random_state)
    pca_projection = pca.fit_transform(embeddings)
    _scatter_plot(pca_projection, labels, run_dir / "plots" / "pca.png", "PCA")

    if len(embeddings) > max_tsne_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(embeddings), size=max_tsne_samples, replace=False)
        tsne_embeddings = embeddings[indices]
        tsne_labels = labels[indices]
    else:
        tsne_embeddings = embeddings
        tsne_labels = labels
    tsne = TSNE(n_components=2, random_state=random_state, init="pca")
    tsne_projection = tsne.fit_transform(tsne_embeddings)
    _scatter_plot(tsne_projection, tsne_labels, run_dir / "plots" / "tsne.png", "t-SNE")

    distances, class_ids = centroid_distance_matrix(embeddings, labels)
    pd.DataFrame(distances, index=class_ids, columns=class_ids).to_csv(run_dir / "centroid_distances.csv")
    _heatmap_plot(distances, class_ids, run_dir / "plots" / "centroid_heatmap.png")


def _scatter_plot(projection: np.ndarray, labels: np.ndarray, destination: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(projection[:, 0], projection[:, 1], c=labels, s=8, cmap="tab10")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def _heatmap_plot(matrix: np.ndarray, class_ids: list[int], destination: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="viridis")
    plt.xticks(range(len(class_ids)), class_ids)
    plt.yticks(range(len(class_ids)), class_ids)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()
