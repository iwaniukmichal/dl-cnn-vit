from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch

from archdyn.config import load_run_config, override_seed
from archdyn.data.cinic10 import build_dataset, build_dataloader
from archdyn.data.transforms import build_supervised_transforms
from archdyn.evaluation.ensemble import protonet_logistic_regression, soft_voting, stacking
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
    device = resolve_device(config.training.device)
    seed = config.seed
    seed_everything(seed, deterministic=config.training.deterministic)
    run_directory = prepare_run_dir(config, seed)
    write_config_snapshot(config, run_directory / "config.snapshot.yaml")

    summary = {"seed": seed}

    if config.ensemble.cnn_checkpoint_dir and config.ensemble.vit_checkpoint_dir:
        cnn_config = copy.deepcopy(config)
        cnn_config.model.family = "pretrained_cnn"
        cnn_config.model.name = "efficientnet_b3"
        cnn_config.dataset.input_size = config.ensemble.cnn_input_size
        vit_config = copy.deepcopy(config)
        vit_config.model.family = "vit"
        vit_config.model.name = "deit_tiny"
        vit_config.dataset.input_size = config.ensemble.vit_input_size

        _, cnn_eval_transform = build_supervised_transforms(cnn_config)
        _, vit_eval_transform = build_supervised_transforms(vit_config)

        cnn_eval_dataset = build_dataset(cnn_config, config.ensemble.eval_split, cnn_eval_transform)
        cnn_eval_loader = build_dataloader(
            cnn_eval_dataset,
            config.training.batch_size,
            config.training.num_workers,
            False,
            device,
        )
        vit_eval_dataset = build_dataset(vit_config, config.ensemble.eval_split, vit_eval_transform)
        vit_eval_loader = build_dataloader(
            vit_eval_dataset,
            config.training.batch_size,
            config.training.num_workers,
            False,
            device,
        )
        cnn_meta_dataset = build_dataset(cnn_config, config.ensemble.meta_split, cnn_eval_transform)
        cnn_meta_loader = build_dataloader(
            cnn_meta_dataset,
            config.training.batch_size,
            config.training.num_workers,
            False,
            device,
        )
        vit_meta_dataset = build_dataset(vit_config, config.ensemble.meta_split, vit_eval_transform)
        vit_meta_loader = build_dataloader(
            vit_meta_dataset,
            config.training.batch_size,
            config.training.num_workers,
            False,
            device,
        )

        cnn_model = build_model(cnn_config.model).to(device)
        vit_model = build_model(vit_config.model).to(device)
        cnn_model.load_state_dict(
            torch.load(Path(config.ensemble.cnn_checkpoint_dir) / f"seed_{seed}" / "checkpoint_best.pt", map_location=device)
        )
        vit_model.load_state_dict(
            torch.load(Path(config.ensemble.vit_checkpoint_dir) / f"seed_{seed}" / "checkpoint_best.pt", map_location=device)
        )

        soft_metrics, soft_predictions = soft_voting(cnn_model, vit_model, cnn_eval_loader, vit_eval_loader, device)
        write_json(soft_metrics, run_directory / "soft_voting_metrics.json")
        pd.DataFrame({"prediction": soft_predictions}).to_csv(run_directory / "soft_voting_predictions.csv", index=False)
        stacking_metrics = stacking(
            cnn_model,
            vit_model,
            cnn_meta_loader,
            vit_meta_loader,
            cnn_eval_loader,
            vit_eval_loader,
            device,
            config.ensemble.logistic_regression_c,
            run_directory,
        )
        summary.update(
            {
                "soft_voting_accuracy": soft_metrics["accuracy"],
                "soft_voting_macro_f1": soft_metrics["macro_f1"],
                "stacking_accuracy": stacking_metrics["accuracy"],
                "stacking_macro_f1": stacking_metrics["macro_f1"],
            }
        )

    if config.ensemble.protonet_checkpoint_dir:
        protonet_config = copy.deepcopy(config)
        _, protonet_eval_transform = build_supervised_transforms(protonet_config)
        protonet_meta_dataset = build_dataset(protonet_config, config.ensemble.meta_split, protonet_eval_transform)
        protonet_meta_loader = build_dataloader(
            protonet_meta_dataset,
            config.training.batch_size,
            config.training.num_workers,
            False,
            device,
        )
        protonet_eval_dataset = build_dataset(protonet_config, config.ensemble.eval_split, protonet_eval_transform)
        protonet_eval_loader = build_dataloader(
            protonet_eval_dataset,
            config.training.batch_size,
            config.training.num_workers,
            False,
            device,
        )
        protonet_model = PrototypicalNetwork(build_model(protonet_config.model)).to(device)
        protonet_model.load_state_dict(
            torch.load(
                Path(config.ensemble.protonet_checkpoint_dir) / f"seed_{seed}" / "checkpoint_best.pt",
                map_location=device,
            )
        )
        protonet_metrics = protonet_logistic_regression(
            protonet_model,
            protonet_meta_loader,
            protonet_eval_loader,
            device,
            config.ensemble.logistic_regression_c,
            run_directory,
        )
        summary.update(
            {
                "protonet_logreg_accuracy": protonet_metrics["accuracy"],
                "protonet_logreg_macro_f1": protonet_metrics["macro_f1"],
            }
        )

    if len(summary) == 1:
        raise ValueError("No ensemble experiment configured. Provide CNN+ViT checkpoints and/or ensemble.protonet_checkpoint_dir.")
    write_json(summary, run_directory / "test_metrics.json")
if __name__ == "__main__":
    main()
