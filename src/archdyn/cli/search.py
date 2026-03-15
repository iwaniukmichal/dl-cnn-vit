from __future__ import annotations

import argparse

from archdyn.config import load_run_config, override_seed
from archdyn.training.search import run_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = override_seed(load_run_config(args.config), args.seed)
    run_search(config)


if __name__ == "__main__":
    main()
