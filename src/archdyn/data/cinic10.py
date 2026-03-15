from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from archdyn.config import RunConfig
from archdyn.data.subsets import load_or_create_manifest, subset_from_manifest


def split_path(config: RunConfig, split: str) -> Path:
    return Path(config.paths.data_root) / split


def build_dataset(config: RunConfig, split: str, transform) -> Dataset:
    dataset = ImageFolder(split_path(config, split), transform=transform)
    if split == config.dataset.train_split and config.subset.enabled:
        manifest = load_or_create_manifest(config, dataset)
        return subset_from_manifest(dataset, manifest)
    return dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device | None = None,
) -> DataLoader:
    resolved_workers = resolve_num_workers(num_workers)
    use_cuda = device is not None and device.type == "cuda"
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": resolved_workers,
        "pin_memory": use_cuda,
    }
    if resolved_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        **loader_kwargs,
    )


def resolve_num_workers(requested_workers: int) -> int:
    available_workers = _available_cpu_workers()
    if available_workers == 0:
        return requested_workers
    if requested_workers > available_workers:
        print(
            "[data] Reducing num_workers from "
            f"{requested_workers} to {available_workers} based on available CPU cores",
            flush=True,
        )
        return available_workers
    return requested_workers


def _available_cpu_workers() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 0


def build_supervised_loaders(config: RunConfig, train_transform, eval_transform, device: torch.device) -> dict[str, DataLoader]:
    train_dataset = build_dataset(config, config.dataset.train_split, train_transform)
    val_dataset = build_dataset(config, config.dataset.val_split, eval_transform)
    test_dataset = build_dataset(config, config.dataset.test_split, eval_transform)
    return {
        "train": build_dataloader(train_dataset, config.training.batch_size, config.training.num_workers, True, device),
        "val": build_dataloader(val_dataset, config.training.batch_size, config.training.num_workers, False, device),
        "test": build_dataloader(test_dataset, config.training.batch_size, config.training.num_workers, False, device),
    }
