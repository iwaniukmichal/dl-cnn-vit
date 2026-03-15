from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .config import RunConfig


def run_dir(config: RunConfig, seed: int) -> Path:
    return Path(config.paths.output_root) / config.phase / config.experiment_name / f"seed_{seed}"


def aggregate_dir(config: RunConfig) -> Path:
    return Path(config.paths.output_root) / config.phase / config.experiment_name / "aggregate"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_run_dir(config: RunConfig, seed: int) -> Path:
    path = ensure_dir(run_dir(config, seed))
    ensure_dir(path / "plots")
    return path


def prepare_aggregate_dir(config: RunConfig) -> Path:
    return ensure_dir(aggregate_dir(config))


def write_config_snapshot(config: RunConfig, destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)


def write_json(payload: dict[str, Any], destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
