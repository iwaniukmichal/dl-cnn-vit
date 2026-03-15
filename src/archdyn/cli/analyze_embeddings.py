from __future__ import annotations

import argparse
from pathlib import Path

import torch

from archdyn.config import load_run_config, override_seed
from archdyn.data.cinic10 import build_dataset, build_dataloader
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
    config = override_seed(load_run_config(args.config), args.seed)
    _, eval_transform = build_supervised_transforms(config)
    device = resolve_device(config.training.device)
    seed = config.seed
    seed_everything(seed)
    run_directory = prepare_run_dir(config, seed)
    write_config_snapshot(config, run_directory / "config.snapshot.yaml")
    split = config.dataset.test_split if config.analysis.split == "test" else config.analysis.split
    dataset = build_dataset(_config_without_subset(config), split, eval_transform)
    dataloader = build_dataloader(dataset, config.training.batch_size, config.training.num_workers, False)
    model = PrototypicalNetwork(build_model(config.model)).to(device)
    checkpoint_path = Path(config.analysis.checkpoint_dir) / f"seed_{seed}" / "checkpoint_best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    embeddings, labels = extract_embeddings(model, dataloader, device)
    save_embedding_artifacts(
        embeddings,
        labels,
        run_directory,
        config.analysis.max_tsne_samples,
        config.analysis.tsne_random_state,
    )
    metrics = compute_embedding_metrics(embeddings, labels)
    metrics["seed"] = seed
    write_json(metrics, run_directory / "test_metrics.json")


def _config_without_subset(config):
    import copy

    clone = copy.deepcopy(config)
    clone.subset.enabled = False
    return clone


if __name__ == "__main__":
    main()
