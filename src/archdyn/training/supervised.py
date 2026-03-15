from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from archdyn.config import RunConfig
from archdyn.data.cinic10 import build_supervised_loaders
from archdyn.data.transforms import build_supervised_transforms
from archdyn.evaluation.metrics import accuracy_from_logits, classification_metrics, save_confusion_matrix
from archdyn.models.pretrained import build_model
from archdyn.paths import prepare_run_dir, write_config_snapshot, write_json
from archdyn.progress import progress
from archdyn.reproducibility import resolve_device, seed_everything


def run_supervised_experiment(config: RunConfig) -> dict:
    _status("Building transforms")
    train_transform, eval_transform = build_supervised_transforms(config)
    seed = config.seed
    _status(f"Seeding run with seed={seed}")
    seed_everything(seed, deterministic=config.training.deterministic)
    device = resolve_device(config.training.device)
    _status(f"Using device={device}")
    amp_enabled = device.type == "cuda" and config.training.mixed_precision
    _status(f"CUDA mixed precision={'enabled' if amp_enabled else 'disabled'}")
    run_directory = prepare_run_dir(config, seed)
    _status(f"Preparing output directory: {run_directory}")
    write_config_snapshot(config, run_directory / "config.snapshot.yaml")

    _status("Building dataloaders")
    loaders = build_supervised_loaders(config, train_transform, eval_transform, device)
    _status(
        "Dataloaders ready: "
        f"train_steps={len(loaders['train'])} val_steps={len(loaders['val'])} test_steps={len(loaders['test'])}"
    )
    _status(f"Building model: family={config.model.family} name={config.model.name}")
    model = build_model(config.model).to(device)
    _status(
        f"Building optimizer and scheduler: optimizer={config.optimizer.name} "
        f"scheduler={config.scheduler.name}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    scheduler = build_scheduler(config, optimizer)
    criterion = nn.CrossEntropyLoss()
    scaler = _build_grad_scaler(device, amp_enabled)

    best_state = None
    best_val_accuracy = -1.0
    train_history = []
    val_history = []
    _status(f"Starting training loop for {config.training.epochs} epochs")
    epoch_iterator = progress(
        range(1, config.training.epochs + 1),
        desc=f"{config.experiment_name} seed={seed}",
        leave=False,
    )
    for epoch in epoch_iterator:
        train_loss, train_accuracy = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            config,
            scaler,
            amp_enabled,
            epoch,
            seed,
        )
        val_loss, val_accuracy, _, _ = evaluate_classifier(model, loaders["val"], criterion, device, amp_enabled)
        train_history.append({"epoch": epoch, "loss": train_loss, "accuracy": train_accuracy})
        val_history.append({"epoch": epoch, "loss": val_loss, "accuracy": val_accuracy})
        if scheduler is not None:
            scheduler.step()
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            _status(f"New best checkpoint at epoch={epoch} val_accuracy={val_accuracy:.4f}")
        if hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_accuracy:.3f}")
        print(
            f"[supervised] Epoch {epoch}/{config.training.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        if config.outputs.save_checkpoints:
            _status("Saving best checkpoint")
            torch.save(best_state, run_directory / "checkpoint_best.pt")

    _status("Writing training history")
    pd.DataFrame(train_history).to_csv(run_directory / "train_history.csv", index=False)
    pd.DataFrame(val_history).to_csv(run_directory / "val_history.csv", index=False)

    _status("Evaluating best model on test split")
    _, _, test_labels, test_predictions = evaluate_classifier(model, loaders["test"], criterion, device, amp_enabled)
    metrics = classification_metrics(test_labels, test_predictions)
    metrics["seed"] = seed
    metrics["best_val_accuracy"] = best_val_accuracy

    _status("Writing final metrics and artifacts")
    save_confusion_matrix(test_labels, test_predictions, run_directory / "confusion_matrix.csv")
    write_json(metrics, run_directory / "test_metrics.json")
    if config.outputs.save_predictions:
        pd.DataFrame({"label": test_labels, "prediction": test_predictions}).to_csv(
            run_directory / "predictions.csv",
            index=False,
        )
    _status(
        f"Run complete: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"best_val_accuracy={best_val_accuracy:.4f}"
    )
    return metrics


def build_scheduler(config: RunConfig, optimizer):
    if config.scheduler.name == "cosine":
        t_max = config.scheduler.t_max or config.training.epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    return None


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
    config: RunConfig,
    scaler,
    amp_enabled: bool,
    epoch: int,
    seed: int,
) -> tuple[float, float]:
    model.train()
    losses = []
    accuracies = []
    batch_iterator = progress(
        dataloader,
        desc=f"train epoch={epoch}/{config.training.epochs} seed={seed}",
        leave=False,
    )
    use_async_transfer = device.type == "cuda"
    for images, labels in batch_iterator:
        images = images.to(device, non_blocking=use_async_transfer)
        labels = labels.to(device, non_blocking=use_async_transfer)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if config.augmentation.name in {"advanced", "combined"}:
                images, labels_a, labels_b, lam = apply_cutmix(images, labels, alpha=config.augmentation.cutmix_alpha)
                logits = model(images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        accuracies.append(accuracy_from_logits(logits.detach(), labels))
        if hasattr(batch_iterator, "set_postfix"):
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")
    return float(np.mean(losses)), float(np.mean(accuracies))


def evaluate_classifier(
    model,
    dataloader,
    criterion,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_labels = []
    all_predictions = []
    use_async_transfer = device.type == "cuda"
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=use_async_transfer)
            labels = labels.to(device, non_blocking=use_async_transfer)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    metrics = classification_metrics(labels, predictions)
    return float(np.mean(losses)), metrics["accuracy"], labels, predictions


def apply_cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0), device=images.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    return images, labels, labels[indices], float(lam)


def rand_bbox(size, lam: float) -> tuple[int, int, int, int]:
    width = size[2]
    height = size[3]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)
    return int(bbx1), int(bby1), int(bbx2), int(bby2)


def _build_grad_scaler(device: torch.device, amp_enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")


def _status(message: str) -> None:
    print(f"[supervised] {message}", flush=True)
