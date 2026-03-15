from __future__ import annotations

import argparse

from archdyn.config import load_run_config, override_seed
from archdyn.training.search import run_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(f"[search] Loading config: {args.config}", flush=True)
    config = override_seed(load_run_config(args.config), args.seed)
    print(
        f"[search] Starting search run: experiment={config.experiment_name} phase={config.phase} seed={config.seed}",
        flush=True,
    )
    run_search(config)
    print(f"[search] Finished search run: experiment={config.experiment_name} seed={config.seed}", flush=True)


if __name__ == "__main__":
    main()
