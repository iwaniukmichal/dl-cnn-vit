from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from archdyn.paths import prepare_aggregate_dir, write_json


def aggregate_seed_metrics(config, rows: list[dict]) -> dict:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    payload = summarize_numeric_frame(frame)
    aggregate_path = prepare_aggregate_dir(config)
    frame.to_csv(aggregate_path / "metrics_summary.csv", index=False)
    write_json(payload, aggregate_path / "metrics_mean_std.json")
    return payload


def summarize_numeric_frame(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    numeric = frame.select_dtypes(include=["number"])
    payload = {}
    for column in numeric.columns:
        payload[column] = {
            "mean": float(numeric[column].mean()),
            "std": float(numeric[column].std(ddof=0)),
        }
    return payload


def seed_dirs_for_experiment(experiment_dir: Path) -> list[Path]:
    return sorted(path for path in experiment_dir.iterdir() if path.is_dir() and path.name.startswith("seed_"))


def aggregate_experiment_metrics(experiment_dir: Path) -> dict[str, Any]:
    rows = []
    for seed_dir in seed_dirs_for_experiment(experiment_dir):
        metrics_path = seed_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(payload)
    if not rows:
        return {}

    frame = pd.DataFrame(rows)
    aggregate_dir = experiment_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(aggregate_dir / "metrics_summary.csv", index=False)
    payload = summarize_numeric_frame(frame)
    write_json(payload, aggregate_dir / "metrics_mean_std.json")
    return payload


def aggregate_search_results(experiment_dir: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in seed_dirs_for_experiment(experiment_dir):
        results_path = seed_dir / "search_results.csv"
        if not results_path.exists():
            continue
        frame = pd.read_csv(results_path)
        frame["seed"] = int(seed_dir.name.split("_", 1)[1])
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("config_id", dropna=False)

    metric_columns = [column for column in ("val_accuracy", "test_accuracy") if column in combined.columns]
    parameter_columns = [
        column
        for column in combined.columns
        if column not in {"config_id", "seed", *metric_columns}
    ]

    parameter_summary = grouped[parameter_columns].first() if parameter_columns else pd.DataFrame(index=grouped.size().index)
    metric_summary = grouped[metric_columns].agg(["mean", "std"]) if metric_columns else pd.DataFrame(index=grouped.size().index)
    metric_summary.columns = [f"{left}_{right}" for left, right in metric_summary.columns.to_flat_index()]
    summary = pd.concat([parameter_summary, metric_summary], axis=1).reset_index()

    aggregate_dir = experiment_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(aggregate_dir / "search_results_all_seeds.csv", index=False)
    summary.to_csv(aggregate_dir / "search_results_aggregated.csv", index=False)

    ranking_column = "val_accuracy_mean" if "val_accuracy_mean" in summary.columns else None
    if ranking_column is not None and not summary.empty:
        best_row = summary.sort_values(ranking_column, ascending=False).iloc[0].to_dict()
        write_json({key: _json_safe_value(value) for key, value in best_row.items()}, aggregate_dir / "best_search_result.json")
    return summary


def aggregate_output_tree(output_root: Path, phase: str | None = None, experiment: str | None = None) -> list[dict[str, Any]]:
    summaries = []
    phase_dirs = [output_root / phase] if phase else sorted(path for path in output_root.iterdir() if path.is_dir())
    for phase_dir in phase_dirs:
        if not phase_dir.exists():
            continue
        experiment_dirs = [phase_dir / experiment] if experiment else sorted(path for path in phase_dir.iterdir() if path.is_dir())
        for experiment_dir in experiment_dirs:
            if not experiment_dir.exists():
                continue
            metrics_summary = aggregate_experiment_metrics(experiment_dir)
            search_summary = aggregate_search_results(experiment_dir)
            if metrics_summary or not search_summary.empty:
                summaries.append(
                    {
                        "phase": phase_dir.name,
                        "experiment_name": experiment_dir.name,
                        "metrics_aggregated": bool(metrics_summary),
                        "search_aggregated": not search_summary.empty,
                    }
                )
    return summaries


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
