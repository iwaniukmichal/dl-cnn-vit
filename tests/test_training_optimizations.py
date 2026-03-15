import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import archdyn.data.cinic10 as cinic10
from archdyn.config import RunConfig
from archdyn.training.supervised import train_one_epoch


class CountingClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.forward_calls = 0
        self.classifier = nn.Linear(3 * 4 * 4, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.classifier(inputs.flatten(start_dim=1))


class DisabledScaler:
    def is_enabled(self) -> bool:
        return False


def test_resolve_num_workers_clamps_to_available_cpu(monkeypatch) -> None:
    monkeypatch.setattr(cinic10, "_available_cpu_workers", lambda: 2)
    assert cinic10.resolve_num_workers(4) == 2


def test_cutmix_training_uses_single_forward_per_batch() -> None:
    images = torch.randn(8, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    model = CountingClassifier(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    config = RunConfig(mode="supervised", phase="phase1", experiment_name="cutmix_smoke")
    config.training.epochs = 1
    config.augmentation.name = "advanced"
    config.augmentation.cutmix_alpha = 1.0

    train_one_epoch(
        model,
        dataloader,
        optimizer,
        criterion,
        torch.device("cpu"),
        config,
        DisabledScaler(),
        False,
        epoch=1,
        seed=13,
    )

    assert model.forward_calls == len(dataloader)
