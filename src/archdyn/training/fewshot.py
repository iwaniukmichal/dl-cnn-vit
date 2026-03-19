from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from archdyn.config import RunConfig
from archdyn.data.cinic10 import build_dataloader, build_dataset, resolve_num_workers
from archdyn.data.episodic import EpisodeDataset, EpisodeSampler
from archdyn.data.transforms import build_supervised_transforms
from archdyn.evaluation.metrics import accuracy_from_logits, classification_metrics, save_confusion_matrix
from archdyn.models.pretrained import build_model
from archdyn.models.prototypical import PrototypicalNetwork, prototypical_loss
from archdyn.paths import prepare_run_dir, write_config_snapshot, write_json
from archdyn.reproducibility import resolve_device, seed_everything
from archdyn.training.supervised import apply_cutmix, build_scheduler


def run_fewshot_experiment(config: RunConfig) -> dict:
    _status("Building transforms")
    train_transform, eval_transform = build_supervised_transforms(config)
    seed = config.seed
    _status(f"Seeding run with seed={seed}")
    seed_everything(seed, deterministic=config.training.deterministic)
    device = resolve_device(config.training.device)
    _status(f"Using device={device}")
    amp_enabled = device.type == "cuda" and config.training.mixed_precision
    run_directory = prepare_run_dir(config, seed)
    _status(f"Preparing output directory: {run_directory}")
    write_config_snapshot(config, run_directory / "config.snapshot.yaml")

    _status("Building datasets")
    train_dataset = build_dataset(config, config.dataset.train_split, train_transform)
    val_dataset = build_dataset(config, config.dataset.val_split, eval_transform)
    test_dataset = build_dataset(config, config.dataset.test_split, eval_transform)
    _status(
        f"Datasets ready: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"episodes(train/val/test)={config.fewshot.train_episodes}/{config.fewshot.val_episodes}/{config.fewshot.test_episodes}"
    )

    _status("Building episode samplers")
    train_sampler = EpisodeSampler(train_dataset, config.fewshot, seed)
    val_sampler = EpisodeSampler(val_dataset, config.fewshot, seed + 1)
    test_sampler = EpisodeSampler(test_dataset, config.fewshot, seed + 2)
    _status("Building episode dataloaders")
    train_loader = build_episode_dataloader(
        train_sampler,
        config.fewshot.train_episodes * config.training.epochs,
        config.training.num_workers,
        device,
    )
    val_loader = build_episode_dataloader(
        val_sampler,
        config.fewshot.val_episodes * config.training.epochs,
        config.training.num_workers,
        device,
    )
    test_loader = build_episode_dataloader(
        test_sampler,
        config.fewshot.test_episodes,
        config.training.num_workers,
        device,
    )
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)
    test_iterator = iter(test_loader)

    _status(f"Building backbone and prototypical network: name={config.model.name}")
    model = build_protonet_model(config, device)
    _status(f"Building optimizer: {config.optimizer.name}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    scheduler = build_scheduler(config, optimizer)
    scaler = _build_grad_scaler(device, amp_enabled)

    best_state = None
    best_val_accuracy = -1.0
    train_rows = []
    val_rows = []
    _status(f"Starting few-shot training loop for {config.training.epochs} epochs")
    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_accuracy = run_episode_epoch(
            model,
            train_iterator,
            config.fewshot.train_episodes,
            optimizer,
            device,
            scaler,
            amp_enabled,
            config,
            train=True,
            epoch=epoch,
            total_epochs=config.training.epochs,
            seed=seed,
        )
        val_loss, val_accuracy = run_episode_epoch(
            model,
            val_iterator,
            config.fewshot.val_episodes,
            optimizer,
            device,
            scaler,
            amp_enabled,
            config,
            train=False,
            epoch=epoch,
            total_epochs=config.training.epochs,
            seed=seed,
        )
        train_rows.append({"epoch": epoch, "loss": train_loss, "accuracy": train_accuracy})
        val_rows.append({"epoch": epoch, "loss": val_loss, "accuracy": val_accuracy})
        if scheduler is not None:
            scheduler.step()
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            _status(f"New best checkpoint at epoch={epoch} val_accuracy={val_accuracy:.4f}")
        print(
            f"[fewshot] Epoch {epoch}/{config.training.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        if config.outputs.save_checkpoints:
            _status("Saving best checkpoint")
            torch.save(best_state, run_directory / "checkpoint_best.pt")

    _status("Evaluating best model on test episodes")
    test_loss, test_accuracy = run_episode_epoch(
        model,
        test_iterator,
        config.fewshot.test_episodes,
        optimizer,
        device,
        scaler,
        amp_enabled,
        config,
        train=False,
        epoch=config.training.epochs,
        total_epochs=config.training.epochs,
        seed=seed,
    )
    _status("Writing training history and final metrics")
    pd.DataFrame(train_rows).to_csv(run_directory / "train_history.csv", index=False)
    pd.DataFrame(val_rows).to_csv(run_directory / "val_history.csv", index=False)
    metrics = {
        "seed": seed,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "accuracy": test_accuracy,
    }
    write_json(metrics, run_directory / "test_metrics.json")
    _status(
        f"Run complete: accuracy={test_accuracy:.4f} test_loss={test_loss:.4f} "
        f"best_val_accuracy={best_val_accuracy:.4f}"
    )
    return metrics


def evaluate_fewshot_experiment(
    config: RunConfig,
    *,
    checkpoint_path: str | Path | None = None,
    eval_n_way: int | None = None,
    split: str = "test",
    output_filename: str | None = None,
) -> dict:
    if config.seed is None:
        raise ValueError("config.seed must be set before evaluation")

    _status("Building evaluation transforms")
    _, eval_transform = build_supervised_transforms(config)
    seed = config.seed
    _status(f"Seeding evaluation with seed={seed}")
    seed_everything(seed, deterministic=config.training.deterministic)
    device = resolve_device(config.training.device)
    _status(f"Using device={device}")
    amp_enabled = device.type == "cuda" and config.training.mixed_precision
    run_directory = prepare_run_dir(config, seed)
    resolved_split = _resolve_split_name(config, split)
    _status(f"Building {resolved_split} dataset for episodic evaluation")
    eval_dataset = build_dataset(config, resolved_split, eval_transform)

    resolved_n_way = eval_n_way if eval_n_way is not None else config.dataset.num_classes
    if resolved_n_way <= 0:
        raise ValueError("eval_n_way must be positive")
    eval_fewshot = replace(config.fewshot, n_way=resolved_n_way)
    num_episodes = _episodes_for_split(config, split)
    sampler_seed = seed + _split_seed_offset(split)
    _status(
        f"Sampling {num_episodes} evaluation episodes on split={resolved_split} "
        f"with n_way={eval_fewshot.n_way}"
    )
    eval_sampler = EpisodeSampler(eval_dataset, eval_fewshot, sampler_seed)
    eval_loader = build_episode_dataloader(
        eval_sampler,
        num_episodes,
        config.training.num_workers,
        device,
    )

    _status(f"Building backbone and prototypical network: name={config.model.name}")
    model = build_protonet_model(config, device)
    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else run_directory / "checkpoint_best.pt"
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")
    _status(f"Loading checkpoint: {resolved_checkpoint}")
    state_dict = torch.load(resolved_checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    loss, accuracy = run_episode_epoch(
        model,
        iter(eval_loader),
        num_episodes,
        optimizer=None,
        device=device,
        scaler=None,
        amp_enabled=amp_enabled,
        config=config,
        train=False,
        epoch=1,
        total_epochs=1,
        seed=seed,
    )
    metrics = {
        "seed": seed,
        "split": resolved_split,
        "train_n_way": config.fewshot.n_way,
        "eval_n_way": eval_fewshot.n_way,
        "k_shot": eval_fewshot.k_shot,
        "q_query": eval_fewshot.q_query,
        "episodes": num_episodes,
        "checkpoint_path": str(resolved_checkpoint),
        "loss": loss,
        "accuracy": accuracy,
    }
    metrics_name = output_filename or f"episodic_eval_{resolved_split}_nway_{eval_fewshot.n_way}.json"
    write_json(metrics, run_directory / metrics_name)
    _status(
        f"Evaluation complete: split={resolved_split} eval_n_way={eval_fewshot.n_way} "
        f"accuracy={accuracy:.4f} loss={loss:.4f}"
    )
    return metrics


def evaluate_protonet_with_fixed_prototypes(
    config: RunConfig,
    *,
    checkpoint_path: str | Path | None = None,
    support_samples_per_class: int = 64,
    support_split: str = "train",
    eval_split: str = "test",
    output_stem: str | None = None,
) -> dict:
    if config.seed is None:
        raise ValueError("config.seed must be set before evaluation")
    if support_samples_per_class <= 0:
        raise ValueError("support_samples_per_class must be positive")

    _status("Building fixed-prototype evaluation transforms")
    _, eval_transform = build_supervised_transforms(config)
    seed = config.seed
    _status(f"Seeding fixed-prototype evaluation with seed={seed}")
    seed_everything(seed, deterministic=config.training.deterministic)
    device = resolve_device(config.training.device)
    _status(f"Using device={device}")
    run_directory = prepare_run_dir(config, seed)

    support_split_name = _resolve_split_name(config, support_split)
    eval_split_name = _resolve_split_name(config, eval_split)
    support_dataset = build_dataset(config, support_split_name, eval_transform)
    eval_dataset = build_dataset(config, eval_split_name, eval_transform)
    support_indices, class_ids = _sample_fixed_support_indices(
        support_dataset,
        support_samples_per_class=support_samples_per_class,
        seed=seed,
    )
    support_subset = Subset(support_dataset, support_indices)
    support_loader = build_dataloader(
        support_subset,
        config.training.batch_size,
        config.training.num_workers,
        False,
        device,
    )
    eval_loader = build_dataloader(
        eval_dataset,
        config.training.batch_size,
        config.training.num_workers,
        False,
        device,
    )

    _status(f"Building backbone and prototypical network: name={config.model.name}")
    model = build_protonet_model(config, device)
    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else run_directory / "checkpoint_best.pt"
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")
    _status(f"Loading checkpoint: {resolved_checkpoint}")
    state_dict = torch.load(resolved_checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    prototype_tensor, prototype_labels = _build_fixed_prototypes(model, support_loader, device)
    if len(prototype_labels) != config.dataset.num_classes:
        _status(
            f"Fixed-prototype evaluation uses {len(prototype_labels)} prototype classes; "
            f"dataset.num_classes={config.dataset.num_classes}"
        )
    predictions, labels, average_loss = _predict_with_fixed_prototypes(
        model,
        eval_loader,
        prototype_tensor,
        prototype_labels,
        device,
    )

    metrics = classification_metrics(labels, predictions)
    metrics.update(
        {
            "seed": seed,
            "support_split": support_split_name,
            "eval_split": eval_split_name,
            "support_samples_per_class": support_samples_per_class,
            "prototype_classes": int(len(class_ids)),
            "checkpoint_path": str(resolved_checkpoint),
            "loss": average_loss,
        }
    )
    stem = output_stem or f"prototype_eval_{support_split_name}{support_samples_per_class}_{eval_split_name}"
    write_json(metrics, run_directory / f"{stem}.json")
    save_confusion_matrix(labels, predictions, run_directory / f"{stem}_confusion_matrix.csv")
    pd.DataFrame({"label": labels, "prediction": predictions}).to_csv(
        run_directory / f"{stem}_predictions.csv",
        index=False,
    )
    _status(
        f"Fixed-prototype evaluation complete: support={support_split_name} "
        f"samples_per_class={support_samples_per_class} eval={eval_split_name} "
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )
    return metrics


def run_episode_epoch(
    model,
    episode_iterator,
    num_episodes: int,
    optimizer,
    device: torch.device,
    scaler,
    amp_enabled: bool,
    config: RunConfig,
    train: bool,
    epoch: int,
    total_epochs: int,
    seed: int,
) -> tuple[float, float]:
    losses = []
    accuracies = []
    if train and optimizer is None:
        raise ValueError("optimizer is required when train=True")
    if train and scaler is None:
        raise ValueError("scaler is required when train=True")
    if train:
        model.train()
    else:
        model.eval()
    for _ in range(num_episodes):
        episode = next(episode_iterator)
        support_images = episode["support_images"].to(device, non_blocking=device.type == "cuda")
        support_labels = episode["support_labels"].to(device, non_blocking=device.type == "cuda")
        query_images = episode["query_images"].to(device, non_blocking=device.type == "cuda")
        query_labels = episode["query_labels"].to(device, non_blocking=device.type == "cuda")
        if train:
            optimizer.zero_grad(set_to_none=True)
        query_inputs = query_images
        query_labels_a = query_labels
        query_labels_b = query_labels
        lam = 1.0
        if train and config.augmentation.name in {"advanced", "combined"}:
            query_inputs, query_labels_a, query_labels_b, lam = apply_cutmix(
                query_images.clone(),
                query_labels,
                alpha=config.augmentation.cutmix_alpha,
            )
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(support_images, support_labels, query_inputs)
                if lam < 1.0:
                    loss = lam * prototypical_loss(logits, query_labels_a) + (1 - lam) * prototypical_loss(
                        logits,
                        query_labels_b,
                    )
                else:
                    loss = prototypical_loss(logits, query_labels)
            if train:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        losses.append(float(loss.item()))
        accuracies.append(_episode_accuracy(logits.detach(), query_labels_a, query_labels_b, lam))
    return float(np.mean(losses)), float(np.mean(accuracies))


def _build_grad_scaler(device: torch.device, amp_enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")


def build_protonet_model(config: RunConfig, device: torch.device) -> PrototypicalNetwork:
    backbone = build_model(config.model)
    return PrototypicalNetwork(backbone).to(device)


def build_episode_dataloader(
    sampler: EpisodeSampler,
    total_episodes: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    resolved_workers = resolve_num_workers(num_workers)
    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "dataset": EpisodeDataset(sampler, total_episodes),
        "batch_size": None,
        "shuffle": False,
        "num_workers": resolved_workers,
        "pin_memory": use_cuda,
    }
    if resolved_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def _episode_accuracy(
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> float:
    if lam >= 1.0:
        return accuracy_from_logits(logits, labels_a)
    predictions = logits.argmax(dim=1)
    accuracy_a = (predictions == labels_a).float().mean().item()
    accuracy_b = (predictions == labels_b).float().mean().item()
    return float(lam * accuracy_a + (1 - lam) * accuracy_b)


def _sample_fixed_support_indices(
    dataset: Dataset,
    *,
    support_samples_per_class: int,
    seed: int,
) -> tuple[list[int], np.ndarray]:
    class_to_indices: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(_dataset_samples(dataset)):
        class_to_indices.setdefault(int(label), []).append(index)
    class_ids = np.asarray(sorted(class_to_indices), dtype=np.int64)
    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []
    for class_id in class_ids:
        candidates = class_to_indices[int(class_id)]
        if len(candidates) < support_samples_per_class:
            raise ValueError(
                f"Class {class_id} has {len(candidates)} samples, but fixed-prototype evaluation "
                f"needs {support_samples_per_class}"
            )
        chosen = rng.choice(candidates, size=support_samples_per_class, replace=False)
        sampled_indices.extend(int(index) for index in chosen)
    return sampled_indices, class_ids


def _build_fixed_prototypes(
    model: PrototypicalNetwork,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    embedding_groups: dict[int, list[np.ndarray]] = {}
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=device.type == "cuda")
            embeddings = model.embed(images).cpu().numpy()
            for embedding, label in zip(embeddings, labels.numpy(), strict=True):
                embedding_groups.setdefault(int(label), []).append(embedding)
    class_ids = np.asarray(sorted(embedding_groups), dtype=np.int64)
    prototypes = np.stack(
        [
            np.mean(np.stack(embedding_groups[int(class_id)], axis=0), axis=0)
            for class_id in class_ids
        ],
        axis=0,
    )
    return torch.tensor(prototypes, dtype=torch.float32, device=device), class_ids


def _predict_with_fixed_prototypes(
    model: PrototypicalNetwork,
    dataloader: DataLoader,
    prototypes: torch.Tensor,
    prototype_labels: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    label_to_position = {int(label): index for index, label in enumerate(prototype_labels.tolist())}
    losses = []
    all_predictions = []
    all_labels = []
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=device.type == "cuda")
            embeddings = model.embed(images)
            logits = -torch.cdist(embeddings, prototypes)
            target_positions = torch.tensor(
                [label_to_position[int(label)] for label in labels.tolist()],
                dtype=torch.long,
                device=device,
            )
            losses.append(float(prototypical_loss(logits, target_positions).item()))
            prediction_positions = logits.argmax(dim=1).cpu().numpy()
            predictions = prototype_labels[prediction_positions]
            all_predictions.append(predictions)
            all_labels.append(labels.numpy())
    return np.concatenate(all_predictions), np.concatenate(all_labels), float(np.mean(losses))


def _episodes_for_split(config: RunConfig, split: str) -> int:
    normalized = split.lower()
    if normalized == "train":
        return config.fewshot.train_episodes
    if normalized in {"val", "valid"}:
        return config.fewshot.val_episodes
    if normalized == "test":
        return config.fewshot.test_episodes
    raise ValueError(f"Unsupported split: {split}")


def _resolve_split_name(config: RunConfig, split: str) -> str:
    normalized = split.lower()
    if normalized == "train":
        return config.dataset.train_split
    if normalized in {"val", "valid"}:
        return config.dataset.val_split
    if normalized == "test":
        return config.dataset.test_split
    raise ValueError(f"Unsupported split: {split}")


def _split_seed_offset(split: str) -> int:
    normalized = split.lower()
    if normalized == "train":
        return 0
    if normalized in {"val", "valid"}:
        return 1
    if normalized == "test":
        return 2
    raise ValueError(f"Unsupported split: {split}")


def _dataset_samples(dataset: Dataset):
    if hasattr(dataset, "samples"):
        return dataset.samples
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_dataset = dataset.dataset
        if hasattr(base_dataset, "samples"):
            return [base_dataset.samples[index] for index in dataset.indices]
    raise AttributeError("Dataset must expose samples directly or via dataset/indices")


def _status(message: str) -> None:
    print(f"[fewshot] {message}", flush=True)
