from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    data_root: str = "data/cinic10"
    output_root: str = "outputs"
    subset_root: str = "data/manifests"


@dataclass(slots=True)
class DatasetConfig:
    name: str = "cinic10"
    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"
    input_size: int = 224
    num_classes: int = 10
    normalization: str = "imagenet"


@dataclass(slots=True)
class SubsetConfig:
    enabled: bool = False
    fraction: float = 1.0
    class_balanced: bool = True
    manifest_name: str | None = None


@dataclass(slots=True)
class ModelConfig:
    family: str = "pretrained_cnn"
    name: str = "efficientnet_b3"
    pretrained: bool = True
    num_classes: int = 10
    drop_path: float = 0.0


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4


@dataclass(slots=True)
class SchedulerConfig:
    name: str = "none"
    t_max: int | None = None


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    num_workers: int = 4
    device: str = "auto"
    deterministic: bool = False
    mixed_precision: bool = True
    log_every: int = 20


@dataclass(slots=True)
class AugmentationConfig:
    name: str = "baseline"
    cutmix_alpha: float = 1.0


@dataclass(slots=True)
class FewShotConfig:
    n_way: int = 10
    k_shot: int = 5
    q_query: int = 15
    train_episodes: int = 200
    val_episodes: int = 100
    test_episodes: int = 200


@dataclass(slots=True)
class OutputConfig:
    save_checkpoints: bool = True
    save_predictions: bool = True
    save_embeddings: bool = True


@dataclass(slots=True)
class AnalysisConfig:
    checkpoint_dir: str = ""
    split: str = "test"
    max_tsne_samples: int = 2000
    tsne_random_state: int = 42


@dataclass(slots=True)
class EnsembleConfig:
    cnn_checkpoint_dir: str = ""
    vit_checkpoint_dir: str = ""
    meta_split: str = "valid"
    eval_split: str = "test"
    logistic_regression_c: float = 1.0


@dataclass(slots=True)
class SearchConfig:
    space: dict[str, list[Any]] = field(default_factory=dict)
    selection_metric: str = "val_accuracy"


@dataclass(slots=True)
class RunConfig:
    mode: str
    phase: str
    experiment_name: str
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    seed: int | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig)
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_dataclass(cls: type[Any], payload: dict[str, Any] | None) -> Any:
    payload = payload or {}
    return cls(**payload)


def load_run_config(path: str | Path) -> RunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = RunConfig(
        mode=raw["mode"],
        phase=raw.get("phase", raw["mode"]),
        experiment_name=raw["experiment_name"],
        paths=_build_dataclass(PathsConfig, raw.get("paths")),
        dataset=_build_dataclass(DatasetConfig, raw.get("dataset")),
        subset=_build_dataclass(SubsetConfig, raw.get("subset")),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        optimizer=_build_dataclass(OptimizerConfig, raw.get("optimizer")),
        scheduler=_build_dataclass(SchedulerConfig, raw.get("scheduler")),
        training=_build_dataclass(TrainingConfig, raw.get("training")),
        augmentation=_build_dataclass(
            AugmentationConfig, raw.get("augmentation")),
        fewshot=_build_dataclass(FewShotConfig, raw.get("fewshot")),
        outputs=_build_dataclass(OutputConfig, raw.get("outputs")),
        analysis=_build_dataclass(AnalysisConfig, raw.get("analysis")),
        ensemble=_build_dataclass(EnsembleConfig, raw.get("ensemble")),
        search=_build_dataclass(SearchConfig, raw.get("search")),
    )
    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    valid_modes = {"supervised", "search",
                   "fewshot", "embedding_analysis", "ensemble"}
    if config.mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {config.mode}")

    valid_families = {"custom_cnn", "pretrained_cnn", "vit"}
    if config.model.family not in valid_families:
        raise ValueError(f"Unsupported model.family: {config.model.family}")

    valid_names = {"custom_cnn", "efficientnet_b3", "deit_tiny"}
    if config.model.name not in valid_names:
        raise ValueError(f"Unsupported model.name: {config.model.name}")

    if config.model.family == "custom_cnn" and config.model.name != "custom_cnn":
        raise ValueError("custom_cnn family must use model.name=custom_cnn")

    if config.model.family == "vit" and config.model.name != "deit_tiny":
        raise ValueError("vit family must use model.name=deit_tiny")

    if config.model.family == "pretrained_cnn" and config.model.name == "deit_tiny":
        raise ValueError("pretrained_cnn family cannot use deit_tiny")

    if not 0 < config.subset.fraction <= 1:
        raise ValueError("subset.fraction must be in (0, 1]")

    if config.fewshot.n_way <= 0 or config.fewshot.k_shot <= 0 or config.fewshot.q_query <= 0:
        raise ValueError("fewshot fields must be positive")

    if config.mode == "search" and not config.search.space:
        raise ValueError("search.space is required for search mode")


def dump_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    raise TypeError("Expected dataclass instance")


def override_seed(config: RunConfig, seed: int | None) -> RunConfig:
    if seed is None:
        raise ValueError("--seed is required")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    config.seed = seed
    return config
