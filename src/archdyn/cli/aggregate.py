from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from archdyn.evaluation.aggregate import aggregate_output_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--phase")
    parser.add_argument("--experiment")
    args = parser.parse_args()

    summaries = aggregate_output_tree(Path(args.output_root), phase=args.phase, experiment=args.experiment)
    if summaries:
        print(pd.DataFrame(summaries).to_string(index=False))
    else:
        print("No aggregatable experiments found.")


if __name__ == "__main__":
    main()
