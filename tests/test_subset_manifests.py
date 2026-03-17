from pathlib import Path

from archdyn.config import PathsConfig, RunConfig, SubsetConfig
from archdyn.data.cinic10 import build_dataset
from archdyn.data.subsets import create_class_balanced_manifest


class DummyDataset:
    def __init__(self) -> None:
        self.samples = []
        for label in range(2):
            for index in range(10):
                self.samples.append((f"/tmp/class_{label}_{index}.png", label))


def test_create_class_balanced_manifest() -> None:
    config = RunConfig(
        mode="supervised",
        phase="phase1",
        experiment_name="subset_test",
        seed=13,
        subset=SubsetConfig(enabled=True, fraction=0.2, class_balanced=True, manifest_name="dummy.txt"),
    )
    manifest = create_class_balanced_manifest(config, DummyDataset())
    assert len(manifest) == 4
    assert len([path for path in manifest if "class_0" in path]) == 2
    assert len([path for path in manifest if "class_1" in path]) == 2


def test_build_dataset_applies_subset_to_all_splits(
    tiny_cinic10: Path,
    manifest_root: Path,
) -> None:
    config = RunConfig(
        mode="supervised",
        phase="phase1",
        experiment_name="subset_smoke",
        seed=13,
        paths=PathsConfig(data_root=str(tiny_cinic10), subset_root=str(manifest_root)),
        subset=SubsetConfig(enabled=True, fraction=0.5, class_balanced=True, manifest_name="subset_smoke.txt"),
    )

    train_dataset = build_dataset(config, "train", transform=None)
    val_dataset = build_dataset(config, "valid", transform=None)
    test_dataset = build_dataset(config, "test", transform=None)

    assert len(train_dataset) == 20
    assert len(val_dataset) == 20
    assert len(test_dataset) == 20
    assert (manifest_root / "subset_smoke_train.txt").exists()
    assert (manifest_root / "subset_smoke_valid.txt").exists()
    assert (manifest_root / "subset_smoke_test.txt").exists()
