from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from torch.utils.data import Dataset, Subset

from archdyn.config import RunConfig


def _manifest_path(config: RunConfig) -> Path:
    manifest_name = config.subset.manifest_name or f"{config.experiment_name}_{int(config.subset.fraction * 100)}.txt"
    return Path(config.paths.subset_root) / manifest_name


def load_or_create_manifest(config: RunConfig, dataset) -> list[str]:
    path = _manifest_path(config)
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
