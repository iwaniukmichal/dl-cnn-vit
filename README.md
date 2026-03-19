# Architecture-Specific Training Dynamics

## Repository description and purpose

This repository is a minimal research codebase for running CINIC-10 experiments that compare:

- `custom_cnn`
- `efficientnet_b3`
- `deit_tiny`

The system is built to support the project plan end to end:

- supervised baseline training
- fixed-grid hyperparameter search
- augmentation studies
- few-shot prototypical-network training
- embedding-space analysis
- ensemble evaluation


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

## How to use the system

### Main entrypoint concept

There is no single orchestration script. The system is operated through CLI entrypoints, each of which takes one YAML config and one required runtime seed:

```bash
python -m archdyn.cli.<command> --config <path-to-yaml> --seed <int>
```

Each invocation runs exactly one seed. To compare multiple seeds, rerun the same command multiple times with different `--seed` values.
Training and few-shot runs show a simple progress bar for epochs and batches or episodes.


### CLI to config mapping

#### `python -m archdyn.cli.train`

Use for standard supervised experiments.

Config sources:
- `configs/phase1/*.yaml`
- `configs/phase3/*.yaml`
- `configs/phase4/reduced_supervised_*.yaml`

Examples:

```bash
python -m archdyn.cli.train --config configs/phase1/custom_cnn_baseline.yaml --seed 37
python -m archdyn.cli.train --config configs/phase3/efficientnet_b3_combined.yaml --seed 37
python -m archdyn.cli.train --config configs/phase4/reduced_supervised_deit_tiny.yaml --seed 37
python -m archdyn.cli.train --config configs/phase1/efficientnet_b3_baseline.yaml --seed 37
```

#### `python -m archdyn.cli.search`

Use for fixed-grid hyperparameter search.

Config sources:
- `configs/phase2/*.yaml`

Examples:

```bash
python -m archdyn.cli.search --config configs/phase2/efficientnet_b3_search.yaml --seed 13
python -m archdyn.cli.search --config configs/phase2/deit_tiny_search.yaml --seed 13
```

#### `python -m archdyn.cli.fewshot`

Use for prototypical-network few-shot training.

Config sources:
- `configs/phase4/protonet_*.yaml`

Examples:

```bash
python -m archdyn.cli.fewshot --config configs/phase4/protonet_efficientnet_b3_standard.yaml --seed 13
python -m archdyn.cli.fewshot --config configs/phase4/protonet_deit_tiny_combined.yaml --seed 13
```

#### `python -m archdyn.cli.fewshot_eval`

Use after few-shot training to rerun episodic evaluation from a saved checkpoint, for example with `--eval-n-way 10` so Protonet is tested on all CINIC-10 classes.

Config sources:
- `configs/phase4/protonet_*.yaml`

Examples:

```bash
python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_efficientnet_b3_standard.yaml --seed 13 --eval-n-way 10
python -m archdyn.cli.fewshot_eval --config configs/phase4/protonet_deit_tiny_combined.yaml --seed 13 --eval-n-way 10
```

#### `python -m archdyn.cli.fewshot_prototype_eval`

Use after few-shot training to build one fixed prototype per class from the training split, for example from `64` randomly sampled training examples per class, and then classify the test split against those prototypes.

Config sources:
- `configs/phase4/protonet_*.yaml`

Examples:

```bash
python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_efficientnet_b3_standard.yaml --seed 13 --support-samples-per-class 64
python -m archdyn.cli.fewshot_prototype_eval --config configs/phase4/protonet_deit_tiny_combined.yaml --seed 13 --support-samples-per-class 64
```

#### `python -m archdyn.cli.analyze_embeddings`

Use after few-shot or supervised training to analyze saved checkpoints.

Config sources:
- `configs/analysis/*.yaml`

Examples:

```bash
python -m archdyn.cli.analyze_embeddings --config configs/analysis/embeddings_efficientnet_b3.yaml --seed 13
python -m archdyn.cli.analyze_embeddings --config configs/analysis/embeddings_deit_tiny.yaml --seed 13
python -m archdyn.cli.analyze_embeddings --config configs/analysis/embeddings_efficientnet_b3_supervised.yaml --seed 13
```

#### `python -m archdyn.cli.ensemble`

Use after supervised or few-shot training to evaluate ensemble-style downstream experiments from saved checkpoints.

Config sources:
- `configs/ensembles/*.yaml`

Examples:

```bash
python -m archdyn.cli.ensemble --config configs/ensembles/supervised_best_models.yaml --seed 13
python -m archdyn.cli.ensemble --config configs/ensembles/protonet_efficientnet_b3_logreg.yaml --seed 13
```

#### `python -m archdyn.cli.aggregate`

Use after multiple seeds have already been run to compute mean and standard deviation summaries from finished `seed_*` directories.

Examples:

```bash
python -m archdyn.cli.aggregate --output-root outputs
python -m archdyn.cli.aggregate --output-root outputs --phase phase3
python -m archdyn.cli.aggregate --output-root outputs --phase phase3 --experiment efficientnet_b3_combined
```

### Configs

Each runnable YAML represents one experiment or one batch job.

Important config fields:

- `mode`
  - `supervised`, `search`, `fewshot`, `embedding_analysis`, or `ensemble`

- `phase`
  - Controls output grouping such as `phase1`, `phase2`, `phase3`, `phase4`, `analysis`, or `ensembles`

- `experiment_name`
  - Run name under `outputs/<phase>/`

- `seed` is not stored in YAML
  - it is supplied at runtime through `--seed`

- `paths.data_root`
  - Root of CINIC-10

- `paths.output_root`
  - Root output directory

- `paths.subset_root`
  - Where subset manifests are stored

- `model.family`
  - `custom_cnn`, `pretrained_cnn`, or `vit`

- `model.name`
  - `custom_cnn`, `efficientnet_b3`, or `deit_tiny`

- `subset.*`
  - Reduced-data controls for Phase 2, Phase 3, and Phase 4

- `dataset.input_size`
  - Input resolution used by the selected model config

- `optimizer.*`, `scheduler.*`, `model.drop_path`
  - Core hyperparameters that should be propagated from Phase 2 into later experiment configs

- `augmentation.name`
  - `baseline`, `standard`, `advanced`, or `combined`
  - In few-shot, `advanced` and `combined` now also apply CutMix to query images during training

- `fewshot.*`
  - Episode shape and episode counts
  - `training.batch_size` is not listed in few-shot YAMLs because episodes are controlled through `fewshot.*`

- `analysis.checkpoint_dir`
  - Source directory for embedding-analysis checkpoints
- `analysis.checkpoint_type`
  - `fewshot` for Protonet checkpoints or `supervised` for standard classifier checkpoints

- `ensemble.cnn_checkpoint_dir` and `ensemble.vit_checkpoint_dir`
  - Source directories for ensemble checkpoints
- `ensemble.cnn_input_size` and `ensemble.vit_input_size`
  - Allow the ensemble job to evaluate CNN and ViT checkpoints with different input resolutions

### Data formats and input expectations

#### Dataset format

The code expects CINIC-10 in standard `ImageFolder` layout:

```text
data/cinic10/
  train/
    airplane/
    automobile/
    ...
  valid/
    airplane/
    automobile/
    ...
  test/
    airplane/
    automobile/
    ...
```

#### Input expectations by mode

- `supervised`
  - Reads dataset splits from `data_root`
  - May also read a subset manifest if `subset.enabled: true`

- `search`
  - Reads the full training split
  - Creates or reuses a deterministic reduced subset manifest
  - Expands a fixed config search space from YAML

- `fewshot`
  - Reads a reduced training subset and samples episodic support/query batches
  - Uses the same optimizer, scheduler, weight decay, drop path, and augmentation config fields as later supervised phases

- `embedding_analysis`
  - Reads saved few-shot or supervised checkpoints from `analysis.checkpoint_dir`
  - Runs embedding extraction on the configured dataset split

- `ensemble`
  - Reads saved supervised checkpoints from the configured CNN and ViT checkpoint directories

## Full workflow and pipeline

### Pipeline overview

The system is executed as a sequence of CLI jobs. Each job:

1. loads one YAML config
2. builds the required datasets, models, and transforms
3. runs the corresponding training or evaluation loop
4. shows a simple progress bar during training or episode loops
5. writes artifacts under the seed-specific output directory

There is no hidden state beyond:

- the dataset under `data/`
- subset manifests under `data/manifests/`
- prior checkpoints needed by analysis and ensemble jobs

### What is trained and from which inputs

#### Phase 1

Trains:
- `custom_cnn`
- `efficientnet_b3`
- `deit_tiny`

Inputs:
- full CINIC-10 training split
- validation split for model selection
- test split for final evaluation
- phase-specific YAML config

#### Phase 2

Trains:
- `efficientnet_b3`
- `deit_tiny`

Inputs:
- deterministic reduced training subset
- validation split
- search grid from YAML

#### Phase 3

Trains:
- `efficientnet_b3`
- `deit_tiny`

Inputs:
- deterministic reduced training subset
- configured augmentation strategy
- tuned optimization settings encoded in YAML

#### Phase 4 few-shot

Trains:
- prototypical networks using `efficientnet_b3` or `deit_tiny` backbones

Inputs:
- deterministic reduced training subset
- episodic few-shot settings from YAML
- validation and test splits for episodic evaluation

Note:
- Phase 4 training configs are `5-way` by default. This is valid meta-learning training, but for comparison against standard 10-class supervised models you should run the separate `fewshot_eval` step with `--eval-n-way 10`.
- You can also run `fewshot_prototype_eval` to convert a saved Protonet checkpoint into a standard closed-set classifier by averaging `64` randomly sampled train examples per class into fixed prototypes and evaluating the test split against them.

#### Phase 4 reduced supervised comparison

Trains:
- standard supervised `efficientnet_b3`
- standard supervised `deit_tiny`

Inputs:
- same reduced subset policy used for Phase 4

#### Embedding analysis

Consumes:
- saved few-shot checkpoints

Produces:
- raw embeddings
- clustering metrics
- PCA and t-SNE plots
- centroid-distance heatmaps

#### Ensemble evaluation

Consumes:
- saved supervised checkpoints for `efficientnet_b3`
- saved supervised checkpoints for `deit_tiny`

Produces:
- soft-voting metrics
- stacking metrics
- stacking coefficients

### Output directory conventions

Outputs are written to:

```text
outputs/<phase>/<experiment_name>/seed_<seed>/
```

Typical files:

- `config.snapshot.yaml`
- `train_history.csv`
- `val_history.csv`
- `test_metrics.json`
- `checkpoint_best.pt`
- `confusion_matrix.csv`
- `predictions.csv`
- `embeddings.npz`
- `plots/`
- `episodic_eval_*.json`
- `prototype_eval_*.json`

Additional mode-specific outputs:

- search jobs:
  - `search_results.csv`
  - `best_config.yaml`
  - both written to the search run directory for the current seed

- aggregate jobs:
  - `aggregate/metrics_summary.csv`
  - `aggregate/metrics_mean_std.json`
  - and, for search experiments, `aggregate/search_results_all_seeds.csv`, `aggregate/search_results_aggregated.csv`, and `aggregate/best_search_result.json`

- embedding analysis jobs:
  - `embedding_metrics.csv`
  - `centroid_distances.csv`
  - PCA, t-SNE, and heatmap images

- ensemble jobs:
  - `soft_voting_metrics.json`
  - `stacking_metrics.json`
  - `stacking_coefficients.csv`
  - `protonet_logreg_metrics.json`
  - `protonet_logreg_coefficients.csv`

## End-to-end experiments for the project plan

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

## Assumptions, limitations, and intended usage boundaries

### Assumptions

- CINIC-10 is present locally in `ImageFolder` layout.
- The user runs one seed per CLI invocation, then repeats commands for the planned seed set `13`, `37`, and `73`.
- The user runs experiments through YAML configs rather than through notebooks.
- The current experiment matrix is limited to `custom_cnn`, `efficientnet_b3`, and `deit_tiny`.
- Phase 2, Phase 3, and Phase 4 reduced-data runs use deterministic class-balanced subsets.

### Limitations

- No distributed training
- No generic support for arbitrary datasets
- No experiment-tracking service integration
- No automatic dataset download
- No automatic orchestration of the entire project from one command
- No automatic propagation of Phase 2 best configs into later YAML files

### Intended usage boundaries

This repository is intended for a small-team or student research workflow where:

- experiments are run explicitly from the command line
- outputs are inspected from the filesystem
- configs are edited manually and versioned as plain YAML
- extending the system means adding a small number of new modules or configs, not introducing a framework
