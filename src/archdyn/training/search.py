from __future__ import annotations

import copy
import itertools
from dataclasses import asdict

import pandas as pd
import yaml

from archdyn.config import RunConfig
from archdyn.paths import prepare_run_dir, write_config_snapshot
from archdyn.training.supervised import run_supervised_experiment


def expand_search_space(space: dict[str, list]) -> list[dict]:
    keys = list(space)
    values = [space[key] for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def run_search(config: RunConfig) -> dict:
    rows = []
    best_summary = None
    best_config = None
    search_run_dir = prepare_run_dir(config, config.seed)
    _status(f"Preparing search output directory: {search_run_dir}")
    write_config_snapshot(config, search_run_dir / "config.snapshot.yaml")
    candidates = expand_search_space(config.search.space)
    _status(f"Expanded search space to {len(candidates)} candidate configurations")
    for index, override in enumerate(candidates, start=1):
        _status(f"Running candidate {index}/{len(candidates)} with override={override}")
        candidate = copy.deepcopy(config)
        candidate.mode = "supervised"
        candidate.experiment_name = f"{config.experiment_name}_cfg_{index:02d}"
        candidate.optimizer.lr = override["lr"]
        candidate.scheduler.name = override["scheduler"]
        candidate.model.drop_path = override["drop_path"]
        candidate.optimizer.weight_decay = override["weight_decay"]
        summary = run_supervised_experiment(candidate)
        row = {
            "config_id": candidate.experiment_name,
            **override,
            "val_accuracy": summary["best_val_accuracy"],
            "test_accuracy": summary["accuracy"],
        }
        rows.append(row)
        if best_summary is None or row["val_accuracy"] > best_summary["val_accuracy"]:
            best_summary = row
            best_config = candidate
            _status(
                f"New best candidate: {candidate.experiment_name} "
                f"val_accuracy={row['val_accuracy']:.4f} test_accuracy={row['test_accuracy']:.4f}"
            )
        print(
            f"[search] Candidate {index}/{len(candidates)} complete: "
            f"config_id={candidate.experiment_name} val_accuracy={row['val_accuracy']:.4f} "
            f"test_accuracy={row['test_accuracy']:.4f}",
            flush=True,
        )

    _status("Writing search summary files")
    pd.DataFrame(rows).to_csv(search_run_dir / "search_results.csv", index=False)
    if best_config is not None:
        with (search_run_dir / "best_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(best_config), handle, sort_keys=False)
        _status(f"Best configuration saved: {best_config.experiment_name}")
    return best_summary or {}


def _status(message: str) -> None:
    print(f"[search] {message}", flush=True)
