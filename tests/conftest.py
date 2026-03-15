from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

matplotlib.use("Agg")


try:
    import torchvision  # noqa: F401
except ModuleNotFoundError:
    torchvision_module = ModuleType("torchvision")
    datasets_module = ModuleType("torchvision.datasets")
    transforms_module = ModuleType("torchvision.transforms")

    class ImageFolder:
        def __init__(self, root: str | Path, transform=None) -> None:
            self.root = Path(root)
            self.transform = transform
            self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
            self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
            self.samples = []
            for class_name in self.classes:
                class_dir = self.root / class_name
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file():
                        self.samples.append((str(image_path), self.class_to_idx[class_name]))

        def __getitem__(self, index: int):
            image_path, label = self.samples[index]
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, label

        def __len__(self) -> int:
            return len(self.samples)

    class Compose:
        def __init__(self, transforms_list):
            self.transforms = transforms_list

        def __call__(self, image):
            for transform in self.transforms:
                image = transform(image)
            return image

    class Resize:
        def __init__(self, size):
            if isinstance(size, tuple):
                self.size = size
            else:
                self.size = (size, size)

        def __call__(self, image):
            return image.resize(self.size)

    class RandomCrop:
        def __init__(self, size, padding: int = 0):
            self.size = size
            self.padding = padding

        def __call__(self, image):
            if self.padding:
                array = np.array(image)
                array = np.pad(array, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode="edge")
                image = Image.fromarray(array.astype(np.uint8))
            return image.crop((0, 0, self.size, self.size))

    class RandomHorizontalFlip:
        def __call__(self, image):
            return image.transpose(Image.FLIP_LEFT_RIGHT)

    class ColorJitter:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def __call__(self, image):
            return image

    class RandomResizedCrop:
        def __init__(self, size) -> None:
            self.size = size

        def __call__(self, image):
            return image.resize((self.size, self.size))

    class ToTensor:
        def __call__(self, image):
            array = np.array(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1)

    class Normalize:
        def __init__(self, mean, std) -> None:
            self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        def __call__(self, tensor):
            return (tensor - self.mean) / self.std

    datasets_module.ImageFolder = ImageFolder
    transforms_module.Compose = Compose
    transforms_module.Resize = Resize
    transforms_module.RandomCrop = RandomCrop
    transforms_module.RandomHorizontalFlip = RandomHorizontalFlip
    transforms_module.ColorJitter = ColorJitter
    transforms_module.RandomResizedCrop = RandomResizedCrop
    transforms_module.ToTensor = ToTensor
    transforms_module.Normalize = Normalize

    torchvision_module.datasets = datasets_module
    torchvision_module.transforms = transforms_module
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.datasets"] = datasets_module
    sys.modules["torchvision.transforms"] = transforms_module


class FakeTimmBackbone(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_features = 16
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(8, self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


@pytest.fixture(autouse=True)
def fake_timm(monkeypatch):
    module = ModuleType("timm")

    def create_model(model_name: str, pretrained: bool, num_classes: int, drop_path_rate: float):
        return FakeTimmBackbone(model_name)

    module.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", module)
    yield


@pytest.fixture
def tiny_cinic10(tmp_path: Path) -> Path:
    root = tmp_path / "cinic10"
    classes = [f"class_{index}" for index in range(10)]
    splits = ("train", "valid", "test")
    for split in splits:
        for class_index, class_name in enumerate(classes):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(4):
                base_value = (class_index * 20 + image_index * 3) % 255
                array = np.full((32, 32, 3), base_value, dtype=np.uint8)
                array[:, :, (class_index + image_index) % 3] = min(255, base_value + 25)
                Image.fromarray(array).save(class_dir / f"sample_{image_index}.png")
    return root


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    path = tmp_path / "outputs"
    path.mkdir()
    return path


@pytest.fixture
def manifest_root(tmp_path: Path) -> Path:
    path = tmp_path / "manifests"
    path.mkdir()
    return path


def write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path
