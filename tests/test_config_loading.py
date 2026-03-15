from pathlib import Path

import pytest

from archdyn.config import load_run_config


def test_load_run_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mode: supervised
phase: phase1
experiment_name: smoke
model:
  family: pretrained_cnn
  name: efficientnet_b3
  pretrained: true
""".strip(),
        encoding="utf-8",
    )
    config = load_run_config(config_path)
    assert config.experiment_name == "smoke"
    assert config.model.name == "efficientnet_b3"
    assert config.seed is None


def test_invalid_model_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mode: supervised
phase: phase1
experiment_name: bad
model:
  family: pretrained_cnn
  name: made_up_model
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_run_config(config_path)
