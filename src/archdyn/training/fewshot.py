from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from archdyn.config import RunConfig
from archdyn.data.cinic10 import build_dataset, resolve_num_workers
from archdyn.data.episodic import EpisodeDataset, EpisodeSampler
from archdyn.data.transforms import build_supervised_transforms
from archdyn.evaluation.metrics import accuracy_from_logits
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
    backbone = build_model(config.model)
    model = PrototypicalNetwork(backbone).to(device)
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


def _status(message: str) -> None:
    print(f"[fewshot] {message}", flush=True)
