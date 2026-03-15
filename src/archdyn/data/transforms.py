from __future__ import annotations

from torchvision import transforms

from archdyn.config import RunConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def input_size_for_model(config: RunConfig) -> int:
    if config.model.name == "custom_cnn":
        return 32
    return config.dataset.input_size


def _normalization():
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def build_supervised_transforms(config: RunConfig) -> tuple[transforms.Compose, transforms.Compose]:
    size = input_size_for_model(config)
    train_ops = []
    eval_ops = []

    if size != 32:
        eval_ops.extend([transforms.Resize((size, size))])
    if config.augmentation.name in {"standard", "combined"}:
        if size == 32:
            train_ops.extend(
                [
                    transforms.RandomCrop(size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                ]
            )
        else:
            train_ops.extend(
                [
                    transforms.RandomResizedCrop(size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                ]
            )
    else:
        if size != 32:
            train_ops.append(transforms.Resize((size, size)))

    if size == 32:
        eval_ops.append(transforms.Resize((size, size)))

    common_tail = [transforms.ToTensor(), _normalization()]
    return transforms.Compose(train_ops + common_tail), transforms.Compose(eval_ops + common_tail)
