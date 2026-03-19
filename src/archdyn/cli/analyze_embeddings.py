from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from archdyn.config import load_run_config, override_seed
from archdyn.data.cinic10 import build_dataset, build_dataloader
from archdyn.data.subsets import load_manifest_entries, resolve_manifest_path, sample_balanced_subset, subset_from_manifest
from archdyn.data.transforms import build_supervised_transforms
from archdyn.evaluation.embeddings import compute_embedding_metrics, extract_embeddings, save_embedding_artifacts
from archdyn.models.pretrained import build_model
from archdyn.models.prototypical import PrototypicalNetwork
from archdyn.paths import prepare_run_dir, write_config_snapshot, write_json
from archdyn.reproducibility import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    _status(f"Loading config: {args.config}")
    config = override_seed(load_run_config(args.config), args.seed)
    _, eval_transform = build_supervised_transforms(config)
    device = resolve_device(config.training.device)
    seed = config.seed
    _status(f"Seeding run with seed={seed}")
    seed_everything(seed, deterministic=config.training.deterministic)
    run_directory = prepare_run_dir(config, seed)
    _status(f"Preparing output directory: {run_directory}")
    write_config_snapshot(config, run_directory / "config.snapshot.yaml")
    split_keys = ["test"]
    if config.analysis.include_train_split:
        split_keys = ["train", "test"]

    _status(
        f"Building model and loading checkpoint from {Path(config.analysis.checkpoint_dir) / f'seed_{seed}' / 'checkpoint_best.pt'}"
    )
    model = _build_analysis_model(config, device)
    checkpoint_path = Path(config.analysis.checkpoint_dir) / f"seed_{seed}" / "checkpoint_best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    summary_metrics = {"seed": seed}
    for split_key in split_keys:
        resolved_split = _resolve_split_name(config, split_key)
        _status(f"Building dataset for split={resolved_split}")
        dataset = _build_analysis_dataset(config, resolved_split, eval_transform, seed, split_key)
        _status(f"Dataset ready for split={resolved_split} with {len(dataset)} samples")
        dataloader = build_dataloader(dataset, config.training.batch_size, config.training.num_workers, False, device)
        _status(f"Extracting embeddings for split={resolved_split} on device={device}")
        embeddings, labels = extract_embeddings(model, dataloader, device)
        _status(f"Saving embedding artifacts for split={resolved_split} to {run_directory}")
        save_embedding_artifacts(
            embeddings,
            labels,
            run_directory,
            config.analysis.max_tsne_samples,
            config.analysis.enable_tsne,
            config.analysis.tsne_random_state,
            artifact_prefix=split_key if config.analysis.include_train_split else None,
        )
        _status(f"Computing summary embedding metrics for split={resolved_split}")
        metrics = compute_embedding_metrics(embeddings, labels)
        if config.analysis.include_train_split:
            for metric_name, value in metrics.items():
                summary_metrics[f"{split_key}_{metric_name}"] = value
            summary_metrics[f"{split_key}_samples"] = int(len(labels))
        else:
            summary_metrics.update(metrics)
            summary_metrics[f"{split_key}_samples"] = int(len(labels))

    write_json(summary_metrics, run_directory / "test_metrics.json")
    _status("Embedding analysis complete")


def _status(message: str) -> None:
    print(f"[analysis] {message}", flush=True)


def _resolve_split_name(config, split_key: str) -> str:
    normalized = split_key.lower()
    if normalized == "train":
        return config.dataset.train_split
    if normalized in {"val", "valid"}:
        return config.dataset.val_split
    if normalized == "test":
        return config.dataset.test_split
    return split_key


def _build_analysis_dataset(config, split: str, transform, seed: int, split_key: str):
    dataset_config = copy.deepcopy(config)
    dataset_config.subset.enabled = False
    dataset = build_dataset(dataset_config, split, transform)
    if config.analysis.manifest_name:
        manifest_path = resolve_manifest_path(config.paths.subset_root, config.analysis.manifest_name, split)
        _status(f"Applying manifest for split={split_key}: {manifest_path}")
        dataset = subset_from_manifest(dataset, load_manifest_entries(manifest_path))
    if config.analysis.samples_per_class is not None:
        split_offset = 0 if split_key == "train" else 10_000
        _status(
            f"Sampling {config.analysis.samples_per_class} examples per class for split={split_key}"
        )
        dataset = sample_balanced_subset(
            dataset,
            samples_per_class=config.analysis.samples_per_class,
            seed=seed + split_offset,
        )
    return dataset


def _build_analysis_model(config, device):
    model_config = copy.deepcopy(config.model)
    model_config.pretrained = False
    if config.analysis.checkpoint_type == "fewshot":
        return PrototypicalNetwork(build_model(model_config)).to(device)
    if config.analysis.checkpoint_type == "supervised":
        return build_model(model_config).to(device)
    raise ValueError(f"Unsupported analysis.checkpoint_type: {config.analysis.checkpoint_type}")


if __name__ == "__main__":
    main()
