from __future__ import annotations

import argparse

from archdyn.config import load_run_config, override_seed
from archdyn.training.supervised import run_supervised_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(f"[train] Loading config: {args.config}", flush=True)
    config = override_seed(load_run_config(args.config), args.seed)
    print(
        f"[train] Starting supervised run: experiment={config.experiment_name} phase={config.phase} seed={config.seed}",
        flush=True,
    )
    run_supervised_experiment(config)
    print(f"[train] Finished supervised run: experiment={config.experiment_name} seed={config.seed}", flush=True)


if __name__ == "__main__":
    main()
