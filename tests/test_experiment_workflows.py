from __future__ import annotations

import sys
from pathlib import Path

import yaml

from archdyn.cli import analyze_embeddings, ensemble, fewshot, search, train
from archdyn.evaluation import embeddings as embedding_module


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


def _run_cli(entrypoint, config_path: Path, monkeypatch, seed: int = 13) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path), "--seed", str(seed)])
    entrypoint.main()


def test_supervised_train_cli_writes_expected_artifacts(
    tiny_cinic10: Path,
    output_root: Path,
    manifest_root: Path,
    monkeypatch,
) -> None:
    config = _base_config("supervised", "phase1", "custom_smoke", tiny_cinic10, output_root, manifest_root)
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
    config_path = _write_yaml(output_root.parent / "configs" / "custom_smoke.yaml", config)

    _run_cli(train, config_path, monkeypatch)

    run_dir = output_root / "phase1" / "custom_smoke" / "seed_13"
    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "train_history.csv").exists()
    assert (run_dir / "test_metrics.json").exists()


def test_search_cli_runs_reduced_grid_and_saves_results(
    tiny_cinic10: Path,
    output_root: Path,
    manifest_root: Path,
    monkeypatch,
) -> None:
    config = _base_config("search", "phase2", "efficientnet_search_smoke", tiny_cinic10, output_root, manifest_root)
    config["subset"] = {
        "enabled": True,
        "fraction": 0.5,
        "class_balanced": True,
        "manifest_name": "search_subset.txt",
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
    config_path = _write_yaml(output_root.parent / "configs" / "search_smoke.yaml", config)

    _run_cli(search, config_path, monkeypatch)

    search_run_dir = output_root / "phase2" / "efficientnet_search_smoke" / "seed_13"
    assert (manifest_root / "search_subset_train.txt").exists()
    assert (manifest_root / "search_subset_valid.txt").exists()
    assert (manifest_root / "search_subset_test.txt").exists()
    assert (search_run_dir / "search_results.csv").exists()
    assert (search_run_dir / "best_config.yaml").exists()


def test_fewshot_and_embedding_analysis_clis_run_end_to_end(
    tiny_cinic10: Path,
    output_root: Path,
    manifest_root: Path,
    monkeypatch,
) -> None:
    class DummyTSNE:
        def __init__(self, n_components: int, random_state: int, init: str) -> None:
            self.n_components = n_components

        def fit_transform(self, values):
            return values[:, : self.n_components]

    monkeypatch.setattr(embedding_module, "TSNE", DummyTSNE)

    fewshot_config = _base_config("fewshot", "phase4", "fewshot_smoke", tiny_cinic10, output_root, manifest_root)
    fewshot_config["subset"] = {
        "enabled": True,
        "fraction": 0.75,
        "class_balanced": True,
        "manifest_name": "fewshot_subset.txt",
    }
    fewshot_config["model"] = {
        "family": "pretrained_cnn",
        "name": "efficientnet_b3",
        "pretrained": True,
        "num_classes": 10,
        "drop_path": 0.0,
    }
    fewshot_config["optimizer"] = {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001}
    fewshot_config["augmentation"] = {"name": "baseline"}
    fewshot_config["fewshot"] = {
        "n_way": 10,
        "k_shot": 1,
        "q_query": 1,
        "train_episodes": 2,
        "val_episodes": 1,
        "test_episodes": 1,
    }
    fewshot_path = _write_yaml(output_root.parent / "configs" / "fewshot_smoke.yaml", fewshot_config)

    _run_cli(fewshot, fewshot_path, monkeypatch)

    fewshot_run_dir = output_root / "phase4" / "fewshot_smoke" / "seed_13"
    assert (fewshot_run_dir / "checkpoint_best.pt").exists()
    assert (fewshot_run_dir / "test_metrics.json").exists()
    assert (manifest_root / "fewshot_subset_train.txt").exists()
    assert (manifest_root / "fewshot_subset_valid.txt").exists()
    assert (manifest_root / "fewshot_subset_test.txt").exists()

    analysis_config = _base_config("embedding_analysis", "analysis", "embedding_smoke", tiny_cinic10, output_root, manifest_root)
    analysis_config["model"] = {
        "family": "pretrained_cnn",
        "name": "efficientnet_b3",
        "pretrained": True,
        "num_classes": 10,
        "drop_path": 0.0,
    }
    analysis_config["analysis"] = {
        "checkpoint_dir": str(output_root / "phase4" / "fewshot_smoke"),
        "split": "test",
        "max_tsne_samples": 40,
        "tsne_random_state": 42,
    }
    analysis_path = _write_yaml(output_root.parent / "configs" / "analysis_smoke.yaml", analysis_config)

    _run_cli(analyze_embeddings, analysis_path, monkeypatch)

    analysis_run_dir = output_root / "analysis" / "embedding_smoke" / "seed_13"
    assert (analysis_run_dir / "embeddings.npz").exists()
    assert (analysis_run_dir / "embedding_metrics.csv").exists()
    assert (analysis_run_dir / "plots" / "pca.png").exists()
    assert (analysis_run_dir / "plots" / "tsne.png").exists()
    assert (analysis_run_dir / "plots" / "centroid_heatmap.png").exists()


def test_ensemble_cli_uses_saved_supervised_checkpoints(
    tiny_cinic10: Path,
    output_root: Path,
    manifest_root: Path,
    monkeypatch,
) -> None:
    common = _base_config("supervised", "phase3", "ignored", tiny_cinic10, output_root, manifest_root)
    common["optimizer"] = {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001}
    common["scheduler"] = {"name": "none"}
    common["augmentation"] = {"name": "baseline"}

    cnn_config = dict(common)
    cnn_config["experiment_name"] = "efficientnet_supervised_for_ensemble"
    cnn_config["model"] = {
        "family": "pretrained_cnn",
        "name": "efficientnet_b3",
        "pretrained": True,
        "num_classes": 10,
        "drop_path": 0.0,
    }
    cnn_path = _write_yaml(output_root.parent / "configs" / "cnn_for_ensemble.yaml", cnn_config)
    _run_cli(train, cnn_path, monkeypatch)

    vit_config = dict(common)
    vit_config["experiment_name"] = "deit_supervised_for_ensemble"
    vit_config["model"] = {
        "family": "vit",
        "name": "deit_tiny",
        "pretrained": True,
        "num_classes": 10,
        "drop_path": 0.0,
    }
    vit_path = _write_yaml(output_root.parent / "configs" / "vit_for_ensemble.yaml", vit_config)
    _run_cli(train, vit_path, monkeypatch)

    ensemble_config = _base_config("ensemble", "ensembles", "ensemble_smoke", tiny_cinic10, output_root, manifest_root)
    ensemble_config["model"] = {
        "family": "pretrained_cnn",
        "name": "efficientnet_b3",
        "pretrained": True,
        "num_classes": 10,
        "drop_path": 0.0,
    }
    ensemble_config["ensemble"] = {
        "cnn_checkpoint_dir": str(output_root / "phase3" / "efficientnet_supervised_for_ensemble"),
        "vit_checkpoint_dir": str(output_root / "phase3" / "deit_supervised_for_ensemble"),
        "meta_split": "valid",
        "eval_split": "test",
        "logistic_regression_c": 1.0,
    }
    ensemble_path = _write_yaml(output_root.parent / "configs" / "ensemble_smoke.yaml", ensemble_config)

    _run_cli(ensemble, ensemble_path, monkeypatch)

    run_dir = output_root / "ensembles" / "ensemble_smoke" / "seed_13"
    assert (run_dir / "soft_voting_metrics.json").exists()
    assert (run_dir / "stacking_metrics.json").exists()
    assert (run_dir / "stacking_coefficients.csv").exists()
    assert (run_dir / "test_metrics.json").exists()
