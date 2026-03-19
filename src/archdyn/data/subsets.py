from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from torch.utils.data import Dataset, Subset

from archdyn.config import RunConfig


def _manifest_path(config: RunConfig, split: str) -> Path:
    manifest_name = config.subset.manifest_name or f"{config.experiment_name}_{int(config.subset.fraction * 100)}.txt"
    base_path = Path(config.paths.subset_root) / manifest_name
    suffix = base_path.suffix
    if suffix:
        split_name = f"{base_path.stem}_{split}{suffix}"
    else:
        split_name = f"{base_path.name}_{split}"
    return base_path.with_name(split_name)


def load_or_create_manifest(config: RunConfig, dataset, split: str) -> list[str]:
    path = _manifest_path(config, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text(encoding="utf-8").splitlines()
    entries = create_class_balanced_manifest(config, dataset)
    path.write_text("\n".join(entries), encoding="utf-8")
    return entries


def create_class_balanced_manifest(config: RunConfig, dataset) -> list[str]:
    if config.seed is None:
        raise ValueError("seed must be set before creating a subset manifest")
    rng = np.random.default_rng(config.seed)
    grouped: dict[int, list[str]] = defaultdict(list)
    for sample_path, label in dataset.samples:
        grouped[label].append(sample_path)

    selected: list[str] = []
    for label in sorted(grouped):
        class_paths = sorted(grouped[label])
        take = max(1, int(len(class_paths) * config.subset.fraction))
        indices = rng.choice(len(class_paths), size=take, replace=False)
        selected.extend(class_paths[index] for index in sorted(indices))
    return sorted(selected)


def subset_from_manifest(dataset, manifest: Iterable[str]) -> Dataset:
    manifest_set = set(manifest)
    indices = [index for index, (sample_path, _) in enumerate(dataset.samples) if sample_path in manifest_set]
    return Subset(dataset, indices)


def resolve_manifest_path(subset_root: str | Path, manifest_name: str, split: str) -> Path:
    base_path = Path(manifest_name)
    if not base_path.is_absolute():
        base_path = Path(subset_root) / manifest_name
    if base_path.exists():
        return base_path
    suffix = base_path.suffix
    if suffix:
        split_name = f"{base_path.stem}_{split}{suffix}"
    else:
        split_name = f"{base_path.name}_{split}"
    return base_path.with_name(split_name)


def load_manifest_entries(path: str | Path) -> list[str]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Manifest not found: {resolved}")
    return resolved.read_text(encoding="utf-8").splitlines()


def dataset_samples(dataset: Dataset):
    if hasattr(dataset, "samples"):
        return dataset.samples
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_dataset = dataset.dataset
        if hasattr(base_dataset, "samples"):
            return [base_dataset.samples[index] for index in dataset.indices]
    raise AttributeError("Dataset must expose samples directly or via dataset/indices")


def sample_balanced_subset(dataset: Dataset, samples_per_class: int, seed: int) -> Dataset:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(dataset_samples(dataset)):
        grouped[int(label)].append(index)

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for label in sorted(grouped):
        class_indices = grouped[label]
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Class {label} only has {len(class_indices)} samples, but {samples_per_class} were requested"
            )
        chosen = rng.choice(class_indices, size=samples_per_class, replace=False)
        selected_indices.extend(int(index) for index in sorted(chosen))
    return Subset(dataset, sorted(selected_indices))
