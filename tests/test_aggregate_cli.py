from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from archdyn.cli import aggregate, search, train


def _base_config(mode: str, phase: str, experiment_name: str, data_root: Path, output_root: Path, subset_root: Path) -> dict:
    return {
        "mode": mode,
        "phase": phase,
        "experiment_name": experiment_name,
        "paths": {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "subset_root": str(subset_root),
        },
        "dataset": {
            "name": "cinic10",
            "train_split": "train",
            "val_split": "valid",
            "test_split": "test",
            "input_size": 32,
            "num_classes": 10,
        },
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "num_workers": 0,
            "device": "cpu",
        },
        "outputs": {
            "save_checkpoints": True,
            "save_predictions": True,
            "save_embeddings": True,
        },
    }


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def _run_seeded_cli(entrypoint, config_path: Path, monkeypatch, seed: int) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path), "--seed", str(seed)])
    entrypoint.main()


def test_aggregate_cli_aggregates_supervised_metrics(
    tiny_cinic10: Path,
    output_root: Path,
    manifest_root: Path,
    monkeypatch,
) -> None:
    config = _base_config("supervised", "phase1", "aggregate_supervised", tiny_cinic10, output_root, manifest_root)
    config["model"] = {
        "family": "custom_cnn",
        "name": "custom_cnn",
        "pretrained": False,
        "num_classes": 10,
        "drop_path": 0.0,
    }
    config["optimizer"] = {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001}
    config["scheduler"] = {"name": "none"}
    config["augmentation"] = {"name": "baseline"}
    config_path = _write_yaml(output_root.parent / "configs" / "aggregate_supervised.yaml", config)

    _run_seeded_cli(train, config_path, monkeypatch, seed=13)
    _run_seeded_cli(train, config_path, monkeypatch, seed=37)

    monkeypatch.setattr(sys, "argv", ["prog", "--output-root", str(output_root)])
    aggregate.main()

    aggregate_dir = output_root / "phase1" / "aggregate_supervised" / "aggregate"
    assert (aggregate_dir / "metrics_summary.csv").exists()
    assert (aggregate_dir / "metrics_mean_std.json").exists()
    with (aggregate_dir / "metrics_mean_std.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "accuracy" in payload
    assert "seed" in payload


def test_aggregate_cli_aggregates_search_results_across_seeds(
    tiny_cinic10: Path,
    output_root: Path,
    manifest_root: Path,
    monkeypatch,
) -> None:
    config = _base_config("search", "phase2", "aggregate_search", tiny_cinic10, output_root, manifest_root)
    config["subset"] = {
        "enabled": True,
        "fraction": 0.5,
        "class_balanced": True,
        "manifest_name": "aggregate_search_subset.txt",
    }
    config["model"] = {
        "family": "pretrained_cnn",
        "name": "efficientnet_b3",
        "pretrained": True,
        "num_classes": 10,
        "drop_path": 0.0,
    }
    config["optimizer"] = {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001}
    config["scheduler"] = {"name": "none"}
    config["augmentation"] = {"name": "baseline"}
    config["search"] = {
        "selection_metric": "val_accuracy",
        "space": {
            "lr": [0.001],
            "scheduler": ["none"],
            "drop_path": [0.0, 0.1],
            "weight_decay": [0.0001],
        },
    }
    config_path = _write_yaml(output_root.parent / "configs" / "aggregate_search.yaml", config)

    _run_seeded_cli(search, config_path, monkeypatch, seed=13)
    _run_seeded_cli(search, config_path, monkeypatch, seed=37)

    monkeypatch.setattr(sys, "argv", ["prog", "--output-root", str(output_root), "--phase", "phase2"])
    aggregate.main()

    aggregate_dir = output_root / "phase2" / "aggregate_search" / "aggregate"
    assert (aggregate_dir / "search_results_all_seeds.csv").exists()
    assert (aggregate_dir / "search_results_aggregated.csv").exists()
    assert (aggregate_dir / "best_search_result.json").exists()
    aggregated = pd.read_csv(aggregate_dir / "search_results_aggregated.csv")
    assert "config_id" in aggregated.columns
    assert "val_accuracy_mean" in aggregated.columns
