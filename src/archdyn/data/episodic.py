from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from archdyn.config import FewShotConfig


class EpisodeSampler:
    def __init__(self, dataset: Dataset, config: FewShotConfig, seed: int) -> None:
        self.dataset = dataset
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, (_, label) in enumerate(_dataset_samples(dataset)):
            self.class_to_indices[label].append(index)

    def sample_episode(self) -> dict[str, torch.Tensor]:
        classes = sorted(self.class_to_indices)
        chosen_classes = self.rng.choice(classes, size=self.config.n_way, replace=False)
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for episodic_label, class_id in enumerate(chosen_classes):
            candidates = self.class_to_indices[int(class_id)]
            count = self.config.k_shot + self.config.q_query
            chosen_indices = self.rng.choice(candidates, size=count, replace=False)
            for index in chosen_indices[: self.config.k_shot]:
                image, _ = self.dataset[index]
                support_images.append(image)
                support_labels.append(episodic_label)
            for index in chosen_indices[self.config.k_shot :]:
                image, _ = self.dataset[index]
                query_images.append(image)
                query_labels.append(episodic_label)

        return {
            "support_images": torch.stack(support_images),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_images": torch.stack(query_images),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
        }


def _dataset_samples(dataset: Dataset):
    if hasattr(dataset, "samples"):
        return dataset.samples
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_dataset = dataset.dataset
        if hasattr(base_dataset, "samples"):
            return [base_dataset.samples[index] for index in dataset.indices]
    raise AttributeError("Dataset must expose samples directly or via dataset/indices")
