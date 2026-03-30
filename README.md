# Architecture-Specific Training Dynamics

## Installation and quick start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Quick start

1. Put CINIC-10 under `data/cinic10/`.
2. Run one baseline experiment:

```bash
python -m archdyn.cli.train --config configs/phase1/efficientnet_b3_baseline.yaml --seed 13
```

3. Inspect outputs under:

```text
outputs/phase1/efficientnet_b3_baseline/seed_13/
```

4. Run local checks:

```bash
python3 -m compileall src tests
python3 -m pytest tests
```

## Repository structure

```text
configs/                Runnable YAML experiment definitions
configs/phase1/         Baseline supervised runs
configs/phase2/         Hyperparameter search jobs
configs/phase3/         Augmentation experiments
configs/phase4/         Few-shot and reduced-data supervised runs
configs/analysis/       Embedding analysis jobs
configs/ensembles/      Ensemble evaluation jobs
src/archdyn/            Implementation package
tests/                  Lightweight unit and smoke tests
data/                   Local dataset root and subset manifests
outputs/                Experiment artifacts and summaries
docs/                   Project and implementation plan documents
```

## Core modules

### Core infrastructure

- `src/archdyn/config.py`
  - Loads YAML configs into typed dataclasses.
  - Validates modes, model names, and core config constraints.

- `src/archdyn/reproducibility.py`
  - Sets the single runtime seed for a run.
  - Resolves the compute device.

- `src/archdyn/paths.py`
  - Creates output directories.
  - Writes config snapshots and JSON artifacts.

### Data

- `src/archdyn/data/cinic10.py`
  - Loads CINIC-10 using `ImageFolder`.
  - Builds train, validation, and test dataloaders.

- `src/archdyn/data/transforms.py`
  - Builds architecture-aware transforms.
  - Applies augmentation strategy selection.

- `src/archdyn/data/subsets.py`
  - Creates deterministic, class-balanced subset manifests.
  - Reuses stored manifests across runs.

- `src/archdyn/data/episodic.py`
  - Samples few-shot episodes with support and query sets.

### Models

- `src/archdyn/models/custom_cnn.py`
  - Defines the custom CNN baseline.

- `src/archdyn/models/pretrained.py`
  - Builds `efficientnet_b3` and `deit_tiny`.
  - Exposes `forward_features()` for embedding extraction.

- `src/archdyn/models/prototypical.py`
  - Wraps a backbone for prototypical-network training and inference.

### Training and evaluation

- `src/archdyn/training/supervised.py`
  - Standard supervised training, validation, testing, checkpointing, and metric export.

- `src/archdyn/training/search.py`
  - Fixed-grid hyperparameter search orchestration.

- `src/archdyn/training/fewshot.py`
  - Episodic few-shot training and evaluation.

- `src/archdyn/evaluation/metrics.py`
  - Classification metrics and embedding distance metrics.

- `src/archdyn/evaluation/aggregate.py`
  - Aggregation helper kept for future multi-run summarization.

- `src/archdyn/evaluation/embeddings.py`
  - Embedding extraction, clustering metrics, PCA, t-SNE, and centroid heatmaps.

- `src/archdyn/evaluation/ensemble.py`
  - Soft voting and embedding-level stacking.

### CLI entrypoints

- `src/archdyn/cli/train.py`
  - Supervised runs.

- `src/archdyn/cli/search.py`
  - Hyperparameter search runs.

- `src/archdyn/cli/fewshot.py`
  - Few-shot prototypical-network runs.

- `src/archdyn/cli/analyze_embeddings.py`
  - Embedding analysis jobs.

- `src/archdyn/cli/ensemble.py`
  - Ensemble evaluation jobs.

- `src/archdyn/cli/aggregate.py`
  - Aggregates finished seed runs from the filesystem.

## Full workflow and pipeline

### Pipeline overview

The system is executed as a sequence of CLI jobs. Each job:

1. loads one YAML config
2. builds the required datasets, models, and transforms
3. runs the corresponding training or evaluation loop
4. writes artifacts under the seed-specific output directory

There is no hidden state beyond:

- the dataset under `data/`
- subset manifests under `data/manifests/`
- prior checkpoints needed by analysis and ensemble jobs

## End-to-end experiments

This is the intended run order for reproducing the planned project.
Use the project-plan seed set `13`, `37`, and `73`: every training, search, few-shot, analysis, and ensemble command below should be rerun once per seed before you compare results or report mean/std values.

### 1. Prepare data

Put CINIC-10 under `data/cinic10/`.

### 2. Run Phase 1 baselines

On full data

```bash
python -m archdyn.cli.train --config configs/phase1/custom_cnn_baseline.yaml --seed 13
python -m archdyn.cli.train --config configs/phase1/efficientnet_b3_baseline.yaml --seed 13
python -m archdyn.cli.train --config configs/phase1/deit_tiny_baseline.yaml --seed 13
```

Repeat the same three commands with `--seed 37` and `--seed 73`, then aggregate Phase 1 results:

```bash
python -m archdyn.cli.aggregate --output-root outputs --phase phase1
```

### 3. Run Phase 2 hyperparameter search

On 5% of data

```bash
python -m archdyn.cli.search --config configs/phase2/efficientnet_b3_search.yaml --seed 13
python -m archdyn.cli.search --config configs/phase2/deit_tiny_search.yaml --seed 13
```

Repeat both search commands with `--seed 37` and `--seed 73`, then aggregate each search experiment before choosing downstream hyperparameters:

```bash
python -m archdyn.cli.aggregate --output-root outputs --phase phase2 --experiment efficientnet_b3_search
python -m archdyn.cli.aggregate --output-root outputs --phase phase2 --experiment deit_tiny_search
```

Before moving to Phase 3 and Phase 4:

1. Use `outputs/phase2/<experiment>/aggregate/search_results_aggregated.csv` and `best_search_result.json` as the source of truth for the winning Phase 2 settings across all three seeds.
2. Do not choose the Phase 2 winner from a single `seed_<seed>/best_config.yaml` if you are following the project plan.
3. Manually copy the winning `lr`, `weight_decay`, `scheduler`, and `drop_path` into:
   - `configs/phase3/*.yaml` for the matching backbone
   - `configs/phase4/protonet_*.yaml` for the matching backbone
   - `configs/phase4/(not)_reduced_supervised_*.yaml` for the matching backbone

### 4. Run Phase 3 augmentation experiments

On 5% of data

```bash
python -m archdyn.cli.train --config configs/phase3/efficientnet_b3_baseline.yaml --seed 13
python -m archdyn.cli.train --config configs/phase3/efficientnet_b3_standard.yaml --seed 13
python -m archdyn.cli.train --config configs/phase3/efficientnet_b3_advanced.yaml --seed 13
python -m archdyn.cli.train --config configs/phase3/efficientnet_b3_combined.yaml --seed 13

python -m archdyn.cli.train --config configs/phase3/deit_tiny_baseline.yaml --seed 13
python -m archdyn.cli.train --config configs/phase3/deit_tiny_standard.yaml --seed 13
python -m archdyn.cli.train --config configs/phase3/deit_tiny_advanced.yaml --seed 13
python -m archdyn.cli.train --config configs/phase3/deit_tiny_combined.yaml --seed 13
```

Repeat the full Phase 3 matrix with `--seed 37` and `--seed 73`, then aggregate Phase 3 before picking the best augmentation per backbone:

```bash
python -m archdyn.cli.aggregate --output-root outputs --phase phase3
```


Before moving to the few-shot, (not)_reduced-data supervised, analysis, and ensemble steps:

1. Compare `outputs/phase3/<experiment>/aggregate/metrics_mean_std.json` across experiments and pick the best augmentation strategy per backbone from the aggregated multi-seed results.
2. Keep the full augmentation matrix for few-shot:
   - `configs/phase4/protonet_<backbone>_baseline.yaml`
   - `configs/phase4/protonet_<backbone>_standard.yaml`
   - `configs/phase4/protonet_<backbone>_advanced.yaml`
   - `configs/phase4/protonet_<backbone>_combined.yaml`
3. Manually propagate only the best Phase 2 hyperparameters into all matching Phase 4 configs for that backbone:
   - `optimizer.lr`
   - `optimizer.weight_decay`
   - `scheduler`
   - `model.drop_path`
4. Use the winning Phase 3 augmentation when preparing (not)reduced supervised comparisons, and when choosing which checkpoint lineage should be treated as the main downstream reference.


### 5. Run Phase 4 few-shot experiments

```bash
python -m archdyn.cli.fewshot --config configs/phase4/protonet_efficientnet_b3_baseline.yaml --seed 13
python -m archdyn.cli.fewshot --config configs/phase4/protonet_efficientnet_b3_standard.yaml --seed 13
python -m archdyn.cli.fewshot --config configs/phase4/protonet_efficientnet_b3_advanced.yaml --seed 13
python -m archdyn.cli.fewshot --config configs/phase4/protonet_efficientnet_b3_combined.yaml --seed 13

python -m archdyn.cli.fewshot --config configs/phase4/protonet_deit_tiny_baseline.yaml --seed 13
python -m archdyn.cli.fewshot --config configs/phase4/protonet_deit_tiny_standard.yaml --seed 13
python -m archdyn.cli.fewshot --config configs/phase4/protonet_deit_tiny_advanced.yaml --seed 13
python -m archdyn.cli.fewshot --config configs/phase4/protonet_deit_tiny_combined.yaml --seed 13
```

The default Phase 4 Protonet configs train in `5-way` episodes. That is normal for meta-learning, but it is easier than closed-set 10-class classification. For a fairer comparison to the standard supervised models, rerun evaluation on the test split with `--eval-n-way 10` after each training run:

```bash
python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_efficientnet_b3_baseline.yaml --seed 13 --eval-n-way 10
python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_efficientnet_b3_standard.yaml --seed 13 --eval-n-way 10
python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_efficientnet_b3_advanced.yaml --seed 13 --eval-n-way 10
python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_efficientnet_b3_combined.yaml --seed 13 --eval-n-way 10

python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_deit_tiny_baseline.yaml --seed 13 --eval-n-way 10
python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_deit_tiny_standard.yaml --seed 13 --eval-n-way 10
python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_deit_tiny_advanced.yaml --seed 13 --eval-n-way 10
python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_deit_tiny_combined.yaml --seed 13 --eval-n-way 10
```

Each fair-evaluation run writes a separate artifact into the training run directory, for example `episodic_eval_test_nway_10.json`, so the original `test_metrics.json` from the training-time `5-way` evaluation is preserved.

You can also evaluate each saved Protonet checkpoint as a fixed-prototype classifier by sampling `64` training images per class, averaging their embeddings into one prototype per class, and classifying the configured test split against those prototypes:

```bash
python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_efficientnet_b3_baseline.yaml --seed 13 --support-samples-per-class 64
python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_efficientnet_b3_standard.yaml --seed 13 --support-samples-per-class 64
python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_efficientnet_b3_advanced.yaml --seed 13 --support-samples-per-class 64
python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_efficientnet_b3_combined.yaml --seed 13 --support-samples-per-class 64

python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_deit_tiny_baseline.yaml --seed 13 --support-samples-per-class 64
python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_deit_tiny_standard.yaml --seed 13 --support-samples-per-class 64
python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_deit_tiny_advanced.yaml --seed 13 --support-samples-per-class 64
python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_deit_tiny_combined.yaml --seed 13 --support-samples-per-class 64
```

Each run writes a separate artifact such as `prototype_eval_train64_test.json`, plus matching predictions and confusion matrix files, into the same Phase 4 run directory.

Update downstream checkpoint references when needed:
   - set `analysis.checkpoint_dir` in `configs/analysis/embeddings_*.yaml` to `outputs/phase4/<best_fewshot_experiment>`
   - set `ensemble.protonet_checkpoint_dir` in `configs/ensembles/protonet_efficientnet_b3_logreg.yaml` or `configs/ensembles/protonet_deit_tiny_logreg.yaml` to the selected Phase 4 Protonet experiment directory for that backbone

### 6. Run reduced-data and notreduced supervised comparisons

```bash
python -m archdyn.cli.train --config configs/phase4/reduced_supervised_efficientnet_b3.yaml --seed 13
python -m archdyn.cli.train --config configs/phase4/reduced_supervised_deit_tiny.yaml --seed 13

python -m archdyn.cli.train --config configs/phase4/not_reduced_supervised_efficientnet_b3.yaml --seed 13
python -m archdyn.cli.train --config configs/phase4/not_reduced_supervised_deit_tiny.yaml --seed 13
```

Repeat Phase 4 few-shot and reduced supervised runs with `--seed 37` and `--seed 73`, then aggregate Phase 4 before comparing few-shot against reduced supervised baselines:

```bash
python -m archdyn.cli.aggregate --output-root outputs --phase phase4
```

Set `ensemble.cnn_checkpoint_dir` and `ensemble.vit_checkpoint_dir` in `configs/ensembles/supervised_best_models.yaml` to the selected Phase 4 experiment directories under `outputs/phase4/` - not reduced experiments with best augmentation (phase3) and hyperparameters (phase2)


### 7. Run embedding analysis

```bash
python -m archdyn.cli.analyze_embeddings --config configs/analysis/embeddings_efficientnet_b3.yaml --seed 13
python -m archdyn.cli.analyze_embeddings --config configs/analysis/embeddings_deit_tiny.yaml --seed 13
python -m archdyn.cli.analyze_embeddings --config configs/analysis/embeddings_efficientnet_b3_supervised.yaml --seed 13
python -m archdyn.cli.analyze_embeddings --config configs/analysis/embeddings_deit_tiny_supervised.yaml --seed 13
```

The Protonet analysis configs read selected Phase 4 few-shot checkpoints. The supervised analysis configs read `outputs/phase4/not_reduced_supervised_efficientnet_b3` and `outputs/phase4/not_reduced_supervised_deit_tiny`, so you can compare Protonet representations against standard supervised models on the same sampled train/test subsets.

Repeat all analysis jobs with `--seed 37` and `--seed 73`, then aggregate the `analysis` phase to obtain mean/std summaries for the embedding metrics:

```bash
python -m archdyn.cli.aggregate --output-root outputs --phase analysis
```

### 8. Run ensemble evaluation


```bash
python -m archdyn.cli.ensemble --config configs/ensembles/supervised_best_models.yaml --seed 13
python -m archdyn.cli.ensemble --config configs/ensembles/protonet_efficientnet_b3_logreg.yaml --seed 13
python -m archdyn.cli.ensemble --config configs/ensembles/protonet_deit_tiny_logreg.yaml --seed 13
```

Repeat the ensemble run with `--seed 37` and `--seed 73`, then aggregate the ensemble phase:

```bash
python -m archdyn.cli.aggregate --output-root outputs --phase ensembles
```

Note:

- The ensemble config supports separate `ensemble.cnn_input_size` and `ensemble.vit_input_size`, so EfficientNet and DeiT can be evaluated at different resolutions.
- The Protonet linear-probe configs use `ensemble.meta_split: train` and `ensemble.eval_split: test`, training logistic regression on frozen Protonet embeddings from the selected checkpoint directory.

### 9. Aggregate results across seeds

After running the same experiment for multiple seeds:

```bash
python -m archdyn.cli.aggregate --output-root outputs
```

To aggregate only one phase or one experiment:

```bash
python -m archdyn.cli.aggregate --output-root outputs --phase phase3
python -m archdyn.cli.aggregate --output-root outputs --phase phase3 --experiment efficientnet_b3_combined
```
