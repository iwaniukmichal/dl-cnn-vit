from archdyn.config import ModelConfig
from archdyn.models.pretrained import TimmBackboneClassifier, build_model


def test_deit_tiny_resolves_to_supported_timm_name() -> None:
    model = build_model(
        ModelConfig(
            family="vit",
            name="deit_tiny",
            pretrained=True,
            num_classes=10,
            drop_path=0.0,
        )
    )

    assert isinstance(model, TimmBackboneClassifier)
    assert model.backbone.model_name == "deit_tiny_patch16_224"
