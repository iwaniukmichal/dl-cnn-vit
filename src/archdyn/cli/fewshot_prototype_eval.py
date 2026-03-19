from __future__ import annotations

import argparse

from archdyn.config import load_run_config, override_seed
from archdyn.training.fewshot import evaluate_protonet_with_fixed_prototypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--support-samples-per-class", type=int, default=64)
    parser.add_argument("--support-split", default="train", choices=["train", "val", "valid", "test"])
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "valid", "test"])
    parser.add_argument("--output-stem")
    args = parser.parse_args()

    print(f"[fewshot-prototype-eval] Loading config: {args.config}", flush=True)
    config = override_seed(load_run_config(args.config), args.seed)
    print(
        "[fewshot-prototype-eval] Starting fixed-prototype evaluation: "
        f"experiment={config.experiment_name} phase={config.phase} seed={config.seed} "
        f"support_split={args.support_split} support_samples_per_class={args.support_samples_per_class} "
        f"eval_split={args.eval_split}",
        flush=True,
    )
    metrics = evaluate_protonet_with_fixed_prototypes(
        config,
        checkpoint_path=args.checkpoint,
        support_samples_per_class=args.support_samples_per_class,
        support_split=args.support_split,
        eval_split=args.eval_split,
        output_stem=args.output_stem,
    )
    print(
        "[fewshot-prototype-eval] Finished fixed-prototype evaluation: "
        f"experiment={config.experiment_name} seed={config.seed} accuracy={metrics['accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
