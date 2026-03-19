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
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, (_, label) in enumerate(_dataset_samples(dataset)):
            self.class_to_indices[label].append(index)
        available_classes = len(self.class_to_indices)
        if self.config.n_way > available_classes:
            raise ValueError(
                f"fewshot.n_way={self.config.n_way} exceeds the number of classes available "
                f"in this split ({available_classes})"
            )
        required_samples = self.config.k_shot + self.config.q_query
        for class_id, indices in self.class_to_indices.items():
            if len(indices) < required_samples:
                raise ValueError(
                    f"Class {class_id} has {len(indices)} samples, but each episode needs "
                    f"{required_samples} per class (k_shot + q_query)"
                )

    def sample_episode(self, episode_index: int | None = None) -> dict[str, torch.Tensor]:
        rng = self.rng if episode_index is None else np.random.default_rng(self.seed + episode_index)
        classes = np.asarray(sorted(self.class_to_indices), dtype=np.int64)
        chosen_classes = rng.choice(classes, size=self.config.n_way, replace=False)
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for episodic_label, class_id in enumerate(chosen_classes):
            candidates = self.class_to_indices[int(class_id)]
            count = self.config.k_shot + self.config.q_query
            chosen_indices = rng.choice(candidates, size=count, replace=False)
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


class EpisodeDataset(Dataset):
    def __init__(self, sampler: EpisodeSampler, total_episodes: int) -> None:
        self.sampler = sampler
        self.total_episodes = total_episodes

    def __len__(self) -> int:
        return self.total_episodes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.sampler.sample_episode(index)


def _dataset_samples(dataset: Dataset):
    if hasattr(dataset, "samples"):
        return dataset.samples
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_dataset = dataset.dataset
        if hasattr(base_dataset, "samples"):
            return [base_dataset.samples[index] for index in dataset.indices]
    raise AttributeError("Dataset must expose samples directly or via dataset/indices")
