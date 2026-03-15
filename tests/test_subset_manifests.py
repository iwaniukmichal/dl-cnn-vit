from pathlib import Path

from archdyn.config import RunConfig, SubsetConfig
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
