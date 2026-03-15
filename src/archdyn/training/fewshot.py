from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from archdyn.config import RunConfig
from archdyn.data.cinic10 import build_dataset
from archdyn.data.episodic import EpisodeSampler
from archdyn.data.transforms import build_supervised_transforms
from archdyn.evaluation.metrics import accuracy_from_logits
from archdyn.models.pretrained import build_model
from archdyn.models.prototypical import PrototypicalNetwork, prototypical_loss
from archdyn.paths import prepare_run_dir, write_config_snapshot, write_json
from archdyn.progress import progress
from archdyn.reproducibility import resolve_device, seed_everything


def run_fewshot_experiment(config: RunConfig) -> dict:
    train_transform, eval_transform = build_supervised_transforms(config)
    seed = config.seed
    seed_everything(seed)
    device = resolve_device(config.training.device)
    run_directory = prepare_run_dir(config, seed)
    write_config_snapshot(config, run_directory / "config.snapshot.yaml")

    train_dataset = build_dataset(config, config.dataset.train_split, train_transform)
    val_dataset = build_dataset(_without_subset(config), config.dataset.val_split, eval_transform)
    test_dataset = build_dataset(_without_subset(config), config.dataset.test_split, eval_transform)

    train_sampler = EpisodeSampler(train_dataset, config.fewshot, seed)
    val_sampler = EpisodeSampler(val_dataset, config.fewshot, seed + 1)
    test_sampler = EpisodeSampler(test_dataset, config.fewshot, seed + 2)

    backbone = build_model(config.model)
    model = PrototypicalNetwork(backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    best_state = None
    best_val_accuracy = -1.0
    train_rows = []
    val_rows = []
    epoch_iterator = progress(
        range(1, config.training.epochs + 1),
        desc=f"{config.experiment_name} seed={seed}",
        leave=False,
    )
    for epoch in epoch_iterator:
        train_loss, train_accuracy = run_episode_epoch(
            model,
            train_sampler,
            config.fewshot.train_episodes,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            total_epochs=config.training.epochs,
            seed=seed,
        )
        val_loss, val_accuracy = run_episode_epoch(
            model,
            val_sampler,
            config.fewshot.val_episodes,
            optimizer,
            device,
            train=False,
            epoch=epoch,
            total_epochs=config.training.epochs,
            seed=seed,
        )
        train_rows.append({"epoch": epoch, "loss": train_loss, "accuracy": train_accuracy})
        val_rows.append({"epoch": epoch, "loss": val_loss, "accuracy": val_accuracy})
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
        if hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_accuracy:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if config.outputs.save_checkpoints:
            torch.save(best_state, run_directory / "checkpoint_best.pt")

    test_loss, test_accuracy = run_episode_epoch(
        model,
        test_sampler,
        config.fewshot.test_episodes,
        optimizer,
        device,
        train=False,
        epoch=config.training.epochs,
        total_epochs=config.training.epochs,
        seed=seed,
    )
    pd.DataFrame(train_rows).to_csv(run_directory / "train_history.csv", index=False)
    pd.DataFrame(val_rows).to_csv(run_directory / "val_history.csv", index=False)
    metrics = {
        "seed": seed,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "accuracy": test_accuracy,
    }
    write_json(metrics, run_directory / "test_metrics.json")
    return metrics


def run_episode_epoch(model, sampler: EpisodeSampler, num_episodes: int, optimizer, device: torch.device, train: bool, epoch: int, total_epochs: int, seed: int) -> tuple[float, float]:
    losses = []
    accuracies = []
    if train:
        model.train()
    else:
        model.eval()
    episode_iterator = progress(
        range(num_episodes),
        desc=f"{'train' if train else 'eval'} epoch={epoch}/{total_epochs} seed={seed}",
        leave=False,
    )
    for _ in episode_iterator:
        episode = sampler.sample_episode()
        support_images = episode["support_images"].to(device)
        support_labels = episode["support_labels"].to(device)
        query_images = episode["query_images"].to(device)
        query_labels = episode["query_labels"].to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(support_images, support_labels, query_images)
            loss = prototypical_loss(logits, query_labels)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        accuracies.append(accuracy_from_logits(logits.detach(), query_labels))
        if hasattr(episode_iterator, "set_postfix"):
            episode_iterator.set_postfix(loss=f"{loss.item():.4f}")
    return float(np.mean(losses)), float(np.mean(accuracies))


def _without_subset(config: RunConfig) -> RunConfig:
    clone = copy.deepcopy(config)
    clone.subset.enabled = False
    return clone
