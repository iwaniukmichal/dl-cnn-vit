from __future__ import annotations

import argparse

from archdyn.config import load_run_config, override_seed
from archdyn.training.fewshot import evaluate_fewshot_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "valid", "test"])
    parser.add_argument("--eval-n-way", type=int)
    parser.add_argument("--output-name")
    args = parser.parse_args()

    print(f"[fewshot-eval] Loading config: {args.config}", flush=True)
    config = override_seed(load_run_config(args.config), args.seed)
    effective_n_way = args.eval_n_way if args.eval_n_way is not None else config.dataset.num_classes
    print(
        "[fewshot-eval] Starting episodic evaluation: "
        f"experiment={config.experiment_name} phase={config.phase} seed={config.seed} "
        f"split={args.split} eval_n_way={effective_n_way}",
        flush=True,
    )
    metrics = evaluate_fewshot_experiment(
        config,
        checkpoint_path=args.checkpoint,
        eval_n_way=args.eval_n_way,
        split=args.split,
        output_filename=args.output_name,
    )
    print(
        "[fewshot-eval] Finished episodic evaluation: "
        f"experiment={config.experiment_name} seed={config.seed} accuracy={metrics['accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
