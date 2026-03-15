from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch

from archdyn.config import load_run_config, override_seed
from archdyn.data.cinic10 import build_dataset, build_dataloader
from archdyn.data.transforms import build_supervised_transforms
from archdyn.evaluation.ensemble import soft_voting, stacking
from archdyn.models.pretrained import build_model
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
    seed_everything(seed, deterministic=config.training.deterministic)
    run_directory = prepare_run_dir(config, seed)
    write_config_snapshot(config, run_directory / "config.snapshot.yaml")

    eval_dataset = build_dataset(_without_subset(config), config.ensemble.eval_split, eval_transform)
    eval_loader = build_dataloader(eval_dataset, config.training.batch_size, config.training.num_workers, False, device)
    meta_dataset = build_dataset(_without_subset(config), config.ensemble.meta_split, eval_transform)
    meta_loader = build_dataloader(meta_dataset, config.training.batch_size, config.training.num_workers, False, device)

    cnn_config = copy.deepcopy(config)
    cnn_config.model.family = "pretrained_cnn"
    cnn_config.model.name = "efficientnet_b3"
    vit_config = copy.deepcopy(config)
    vit_config.model.family = "vit"
    vit_config.model.name = "deit_tiny"

    cnn_model = build_model(cnn_config.model).to(device)
    vit_model = build_model(vit_config.model).to(device)
    cnn_model.load_state_dict(torch.load(Path(config.ensemble.cnn_checkpoint_dir) / f"seed_{seed}" / "checkpoint_best.pt", map_location=device))
    vit_model.load_state_dict(torch.load(Path(config.ensemble.vit_checkpoint_dir) / f"seed_{seed}" / "checkpoint_best.pt", map_location=device))

    soft_metrics, soft_predictions = soft_voting(cnn_model, vit_model, eval_loader, device)
    write_json(soft_metrics, run_directory / "soft_voting_metrics.json")
    pd.DataFrame({"prediction": soft_predictions}).to_csv(run_directory / "soft_voting_predictions.csv", index=False)
    stacking_metrics = stacking(
        cnn_model,
        vit_model,
        meta_loader,
        eval_loader,
        device,
        config.ensemble.logistic_regression_c,
        run_directory,
    )
    summary = {
        "seed": seed,
        "soft_voting_accuracy": soft_metrics["accuracy"],
        "soft_voting_macro_f1": soft_metrics["macro_f1"],
        "stacking_accuracy": stacking_metrics["accuracy"],
        "stacking_macro_f1": stacking_metrics["macro_f1"],
    }
    write_json(summary, run_directory / "test_metrics.json")


def _without_subset(config):
    clone = copy.deepcopy(config)
    clone.subset.enabled = False
    return clone


if __name__ == "__main__":
    main()
