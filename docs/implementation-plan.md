# Implementation Plan: CNN vs ViT Training Dynamics on CINIC-10

## Summary
Build a small Python research codebase for CINIC-10 that runs four experiment phases: supervised baselines, hyperparameter search, augmentation study, and few-shot learning with embedding analysis, plus a small ensemble extension. The required experiment matrix uses `custom_cnn`, `efficientnet_b3`, and `deit_tiny`. This plan is intended for `docs/plan/implementation-plan.md`.

## 1. System intent
The system exists to run a reproducible set of CINIC-10 experiments comparing a scratch-trained custom CNN, a pretrained CNN backbone (`efficientnet_b3`), and a Vision Transformer (`deit_tiny`) across optimization, augmentation, and few-shot settings, while saving enough artifacts to answer the project’s research questions with a small, explicit research codebase.

## 2. Assumptions and engineering philosophy
Assumptions:
- Use Python 3.11 with PyTorch-based training.
- `efficientnet_b3` and `deit_tiny` use pretrained weights and resized `224x224` inputs.
- `custom_cnn` trains from scratch and uses native `32x32` inputs.
- Phase 2 uses a class-balanced 20% subset of the training split.
- Phase 4 uses a class-balanced 10% reduced training subset.
- Few-shot defaults are `10-way, 5-shot, 15-query`.
- Best checkpoint per seed is selected by validation accuracy.
- Final reporting uses mean and standard deviation across seeds `[13, 37, 73]`.

Engineering philosophy:
- Keep the system explicit and small: YAML config -> build components -> run -> save outputs -> aggregate.
- Support CNN backbone selection in config because it is cheap to implement cleanly.
- Do not generalize beyond what is needed: the required experiment matrix includes only `custom_cnn`, `efficientnet_b3`, and `deit_tiny`.
- Share only stable pieces such as dataset loading, transforms, model builders, metrics, and artifact writing.
- Keep supervised and few-shot loops separate.
- Add only limited validation for missing files, invalid config fields, and contradictory settings.

## 3. Scope
### In scope
- CINIC-10 loading from official train/valid/test folders.
- Supervised training for `custom_cnn`, `efficientnet_b3`, and `deit_tiny`.
- Config-driven pretrained CNN selection, with `efficientnet_b3` required.
- Phase 2 hyperparameter search for `efficientnet_b3` and `deit_tiny`.
- Phase 3 augmentation comparison for `efficientnet_b3` and `deit_tiny`.
- Phase 4 prototypical-network few-shot experiments for `efficientnet_b3` and `deit_tiny`.
- Reduced-data supervised comparison in Phase 4.
- Embedding extraction, clustering metrics, PCA/t-SNE plots, and centroid heatmaps.
- Ensemble evaluation using the best pretrained CNN and ViT models.
- Per-seed outputs and seed aggregation.

### Out of scope
- Distributed training or cluster orchestration.
- Generic multi-dataset support.
- Heavy experiment-management frameworks.
- Arbitrary model registries or plugin architectures.
- Production inference or serving APIs.
- Automated report writing beyond saving CSV/JSON/plot artifacts.

## 4. Derived requirements from the spec
The codebase must support:
- Standard supervised training runs for:
  - `custom_cnn`
  - `efficientnet_b3`
  - `deit_tiny`
- A model config that can represent:
  - `model.family: custom_cnn | pretrained_cnn | vit`
  - `model.name: custom_cnn | efficientnet_b3 | deit_tiny`
- Deterministic reduced-data subset creation and reuse.
- Phase 2 grid search over four knobs for `efficientnet_b3` and `deit_tiny`:
  - `efficientnet_b3`: `lr in {3e-4, 1e-3}`, `scheduler in {none, cosine}`, `drop_path in {0.0, 0.1}`, `weight_decay in {1e-4, 5e-4}`
  - `deit_tiny`: `lr in {5e-5, 2e-4}`, `scheduler in {none, cosine}`, `drop_path in {0.0, 0.1}`, `weight_decay in {0.05, 0.1}`
- Augmentation strategies:
  - `baseline`
  - `standard`
  - `advanced`
  - `combined`
- Few-shot prototypical-network training using `efficientnet_b3` and `deit_tiny` as embedding backbones.
- Embedding analysis on trained few-shot checkpoints.
- Ensemble evaluation between the best pretrained CNN model and the best ViT model.

Important interfaces and types:
- Config dataclasses:
  - `DatasetConfig`
  - `ModelConfig`
  - `OptimizerConfig`
  - `AugmentationConfig`
  - `FewShotConfig`
  - `OutputConfig`
  - `RunConfig`
- CLI entrypoints:
  - `python -m archdyn.cli.train --config ...`
  - `python -m archdyn.cli.search --config ...`
  - `python -m archdyn.cli.fewshot --config ...`
  - `python -m archdyn.cli.analyze_embeddings --config ...`
  - `python -m archdyn.cli.ensemble --config ...`

## 5. Proposed tech stack
- PyTorch: training, checkpointing, dataloaders.
- torchvision: transforms and dataset utilities.
- timm: `efficientnet_b3`, `deit_tiny`, and DropPath-capable implementations.
- scikit-learn: logistic regression stacking, silhouette score, Davies-Bouldin index, PCA, t-SNE.
- PyYAML: readable YAML configs.
- matplotlib: plots and heatmaps.
- numpy: aggregation and distance calculations.
- pandas: simple result tables.

## 6. Repository structure
```text
README.md
pyproject.toml
.gitignore
docs/
  plan/
    implementation-plan.md
configs/
  phase1/
    custom_cnn_baseline.yaml
    efficientnet_b3_baseline.yaml
    deit_tiny_baseline.yaml
  phase2/
    efficientnet_b3_search.yaml
    deit_tiny_search.yaml
  phase3/
    efficientnet_b3_baseline.yaml
    efficientnet_b3_standard.yaml
    efficientnet_b3_advanced.yaml
    efficientnet_b3_combined.yaml
    deit_tiny_baseline.yaml
    deit_tiny_standard.yaml
    deit_tiny_advanced.yaml
    deit_tiny_combined.yaml
  phase4/
    protonet_efficientnet_b3_baseline.yaml
    protonet_efficientnet_b3_standard.yaml
    protonet_efficientnet_b3_advanced.yaml
    protonet_efficientnet_b3_combined.yaml
    protonet_deit_tiny_baseline.yaml
    protonet_deit_tiny_standard.yaml
    protonet_deit_tiny_advanced.yaml
    protonet_deit_tiny_combined.yaml
    reduced_supervised_efficientnet_b3.yaml
    reduced_supervised_deit_tiny.yaml
  analysis/
    embeddings_efficientnet_b3.yaml
    embeddings_deit_tiny.yaml
  ensembles/
    supervised_best_models.yaml
src/
  archdyn/
    config.py
    reproducibility.py
    paths.py
    data/
      cinic10.py
      transforms.py
      subsets.py
      episodic.py
    models/
      custom_cnn.py
      pretrained.py
      prototypical.py
    training/
      supervised.py
      search.py
      fewshot.py
    evaluation/
      metrics.py
      embeddings.py
      ensemble.py
      aggregate.py
    cli/
      train.py
      search.py
      fewshot.py
      analyze_embeddings.py
      ensemble.py
tests/
  test_config_loading.py
  test_subset_manifests.py
  test_episode_sampler.py
  test_metrics.py
data/
  .gitignore
outputs/
  .gitignore
```

## 7. Core modules and responsibilities
- `config.py`
  - Responsibility: load YAML into typed configs and validate obvious contradictions.
  - Why: keeps runners simple.
  - Should not do: run experiments.

- `reproducibility.py`
  - Responsibility: seed Python, NumPy, and Torch; set deterministic flags where reasonable.
  - Why: reproducibility is mandatory.
  - Should not do: output management.

- `paths.py`
  - Responsibility: create per-run directories and write config snapshots.
  - Why: keep artifacts consistent across phases.
  - Should not do: experiment logic.

- `data/cinic10.py`
  - Responsibility: dataset loading and standard dataloader creation.
  - Why: all phases depend on the same split conventions.
  - Should not do: episodic sampling.

- `data/transforms.py`
  - Responsibility: architecture-aware transforms and augmentation pipelines.
  - Why: `custom_cnn` and pretrained models need different input handling.
  - Should not do: model construction.

- `data/subsets.py`
  - Responsibility: deterministic, class-balanced subset manifests.
  - Why: fair comparisons in Phases 2 and 4.
  - Should not do: training.

- `data/episodic.py`
  - Responsibility: few-shot episode sampling.
  - Why: Phase 4 uses episodic learning, not minibatch classification.
  - Should not do: prototype computation.

- `models/custom_cnn.py`
  - Responsibility: define the scratch baseline CNN.
  - Why: it is a distinct architecture with separate preprocessing and training expectations.
  - Should not do: pretrained backbone selection.

- `models/pretrained.py`
  - Responsibility: build pretrained backbones from a small explicit list, initially `efficientnet_b3` and `deit_tiny`, expose classifier and embedding dimensions, and apply DropPath-capable settings where supported.
  - Why: keeps config-driven model choice straightforward.
  - Should not do: become a generic registry for arbitrary model names.

- `models/prototypical.py`
  - Responsibility: wrap a backbone as an embedding extractor and compute prototypes.
  - Why: Phase 4 requires a clean few-shot abstraction.
  - Should not do: standard supervised classification.

- `training/supervised.py`
  - Responsibility: standard train/val/test loop.
  - Why: used in Phases 1, 3, and the reduced-data comparison.
  - Should not do: grid expansion or episodic training.

- `training/search.py`
  - Responsibility: expand the fixed Phase 2 grid, launch per-seed runs, rank configs, save summaries.
  - Why: search bookkeeping should stay out of the trainer.
  - Should not do: act like a generic HPO engine.

- `training/fewshot.py`
  - Responsibility: episodic training and evaluation for prototypical networks.
  - Why: few-shot control flow is substantially different.
  - Should not do: full supervised training.

- `evaluation/metrics.py`
  - Responsibility: accuracy, macro-F1, confusion matrices, and embedding distance metrics.
  - Why: shared metrics logic across phases.
  - Should not do: plotting orchestration.

- `evaluation/embeddings.py`
  - Responsibility: embedding extraction, clustering metrics, PCA/t-SNE, centroid heatmaps.
  - Why: embedding-space analysis is a core contribution.
  - Should not do: training.

- `evaluation/ensemble.py`
  - Responsibility: soft voting and embedding-level stacking between the best pretrained CNN and ViT models.
  - Why: ensemble work is explicitly in scope.
  - Should not do: support arbitrary ensemble families.

- `evaluation/aggregate.py`
  - Responsibility: aggregate per-seed metrics into mean/std summaries.
  - Why: all reported results are over three seeds.
  - Should not do: model inference.

## 8. YAML configuration strategy
Organization:
- One YAML per runnable experiment or batch.
- Search jobs are one YAML per architecture.
- No YAML inheritance in v1.

Required fields:
- `mode`
- `experiment_name`
- `paths.data_root`, `paths.output_root`, `paths.subset_root`
- `dataset.*`
- `subset.*`
- `seeds`
- `model.family`
- `model.name`
- `model.pretrained`
- `model.num_classes`
- `model.drop_path`
- `optimizer.*`
- `scheduler.*`
- `training.*`
- `augmentation.*`
- `fewshot.*`
- `outputs.*`

Backbone selection policy:
- `custom_cnn` remains its own explicit model type.
- Pretrained backbone choice must be explicit in YAML.
- The planned experiment configs should use `efficientnet_b3` for the CNN branch.
- Supporting another pretrained CNN later is acceptable if it is added as a simple named branch in `models/pretrained.py`.

Readable example shape:
```yaml
mode: supervised
experiment_name: efficientnet_b3_standard
model:
  family: pretrained_cnn
  name: efficientnet_b3
  pretrained: true
  num_classes: 10
  drop_path: 0.1
augmentation:
  name: standard
```

## 9. Execution flows
- Standard supervised run
  - Load YAML, seed RNGs, resolve paths, build transforms and dataloaders, build model from `model.family` and `model.name`, train, select best checkpoint on validation accuracy, evaluate on test, write artifacts, aggregate across seeds.

- Hyperparameter search batch
  - Load the search YAML for `efficientnet_b3` or `deit_tiny`, expand the fixed 16-config grid, run all seeds on the 20% subset, save ranking summaries, emit `best_config.yaml`.

- Phase 3 augmentation run
  - Reuse the supervised trainer with the best Phase 2 hyperparameters and one of the four named augmentation strategies.

- Few-shot episodic run
  - Load reduced subset manifest, build episodic sampler, build a prototypical-network backbone using `efficientnet_b3` or `deit_tiny`, train over episodes, save checkpoints and episodic metrics.

- Embedding analysis job
  - Load trained few-shot checkpoints, extract embeddings on the test split, compute clustering metrics, save PCA/t-SNE plots and centroid-distance heatmaps, aggregate across seeds.

- Ensemble evaluation
  - Soft voting: average class probabilities from the best `efficientnet_b3` and `deit_tiny` supervised models.
  - Stacking: concatenate validation embeddings from the best pretrained CNN and ViT models, train logistic regression, evaluate on test.

## 10. Phase-by-phase implementation roadmap
- Milestone 1: Foundation
  - Deliverables: package skeleton, config loading, reproducibility utilities, path/output manager, CINIC-10 loader, subset manifests.
  - Dependencies: none.
  - Validation criteria: configs load, subset manifests are deterministic, dataloader smoke tests pass.

- Milestone 2: Phase 1 supervised baselines
  - Deliverables: `custom_cnn`, pretrained model builder, supervised trainer, baseline configs for `custom_cnn`, `efficientnet_b3`, `deit_tiny`.
  - Dependencies: Milestone 1.
  - Validation criteria: each model completes a smoke run and writes checkpoint plus metrics.

- Milestone 3: Phase 2 hyperparameter search
  - Deliverables: search runner, fixed grids for `efficientnet_b3` and `deit_tiny`, best-config export.
  - Dependencies: Milestone 2.
  - Validation criteria: dry-run ranking works and full jobs produce summary CSVs.

- Milestone 4: Phase 3 augmentation study
  - Deliverables: four named augmentation strategies and full-data configs for `efficientnet_b3` and `deit_tiny`.
  - Dependencies: Milestone 3.
  - Validation criteria: all architecture-strategy combinations produce comparable aggregated outputs.

- Milestone 5: Phase 4 few-shot learning
  - Deliverables: episode sampler, prototypical-network support for `efficientnet_b3` and `deit_tiny`, reduced-data supervised comparison configs.
  - Dependencies: Milestone 1 and Milestone 2.
  - Validation criteria: episodic smoke test passes and full runs save checkpoints plus metrics.

- Milestone 6: Embedding analysis and ensembles
  - Deliverables: embedding extraction, clustering metrics, PCA/t-SNE plots, heatmaps, soft voting, stacking.
  - Dependencies: Milestone 4 and Milestone 5.
  - Validation criteria: analysis and ensemble jobs reproduce all required outputs from saved checkpoints.

Test cases and scenarios:
- Config parsing rejects unknown `model.family` or unsupported `model.name`.
- Same subset seed and fraction generate identical manifests.
- Episode sampler returns correct support/query counts.
- Metric utilities behave correctly on tiny synthetic embeddings.
- End-to-end smoke runs work with `1 epoch`, `1 seed`, and tiny subsets.

## 11. Reuse vs deliberate duplication
Shared:
- Dataset loading, transforms, subset manifests, seeding, output writing, aggregation, pretrained model factory, metrics.

Separate by design:
- Supervised and few-shot training loops.
- Standard minibatch loaders and episodic samplers.
- Embedding analysis and ensemble evaluation.
- Search configs and single-run configs.

Reason:
- These flows differ enough that a unified abstraction would obscure the experiment logic.

## 12. Logging, outputs, and artifact organization
Per-run layout:
```text
outputs/<phase>/<experiment_name>/seed_<seed>/
  config.snapshot.yaml
  train_history.csv
  val_history.csv
  test_metrics.json
  checkpoint_best.pt
  confusion_matrix.csv
  predictions.csv
  embeddings.npz
  plots/
```

Aggregate layout:
```text
outputs/<phase>/<experiment_name>/aggregate/
  metrics_mean_std.json
  metrics_summary.csv
  comparison_plot.png
```

Additional outputs:
- Search jobs write `search_results.csv` and `best_config.yaml`.
- Embedding analysis writes metric JSON and all plots.
- Ensemble jobs write their own metrics JSON and summary CSV.

## 13. README requirements
The README should document:
- Project purpose and research questions.
- Supported experiment modes and CLI usage.
- CINIC-10 folder expectations under `data/`.
- Config philosophy: one YAML per runnable job, no inheritance.
- The meaning of `model.family` and `model.name`.
- Output directory conventions.
- Reproducibility notes: seeds, deterministic subsets, config snapshots.
- Limitations: no distributed training, no generic dataset support, no tracking platform integration.
- How the ensemble evaluation uses saved pretrained CNN and ViT checkpoints.

## 14. Deliberate non-features
- No generic model registry or plugin loader.
- No universal trainer covering both supervised and episodic modes.
- No automated HPO framework beyond the fixed grid search.
- No notebook-only analysis workflow.
- No experiment-tracking server.
- No arbitrary backbone support beyond a small hardcoded list.

## 15. Risks / open questions
- Exact epoch budgets are still unspecified and should remain configurable.
- `efficientnet_b3` is heavier than smaller CNNs, so batch size may need to be reduced if compute is tight.
- If `efficientnet_b3` backbone extraction for prototypical networks behaves awkwardly in practice, the implementation should expose penultimate pooled features directly rather than adding a broader abstraction.
- t-SNE may need to run on a stratified sample if full test-set visualization is too slow.

## Explicit assumptions and defaults chosen
- The CNN side of the planned experiments uses `efficientnet_b3`.
- Config-driven pretrained CNN selection is included.
- Only `efficientnet_b3` is required in the experiment plan.
- The repo should stay minimal and must not grow a registry or framework just to support that config field.
