from __future__ import annotations

from pathlib import Path

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


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_supervised_loaders(config: RunConfig, train_transform, eval_transform) -> dict[str, DataLoader]:
    train_dataset = build_dataset(config, config.dataset.train_split, train_transform)
    val_dataset = build_dataset(config, config.dataset.val_split, eval_transform)
    test_dataset = build_dataset(config, config.dataset.test_split, eval_transform)
    return {
        "train": build_dataloader(train_dataset, config.training.batch_size, config.training.num_workers, True),
        "val": build_dataloader(val_dataset, config.training.batch_size, config.training.num_workers, False),
        "test": build_dataloader(test_dataset, config.training.batch_size, config.training.num_workers, False),
    }
