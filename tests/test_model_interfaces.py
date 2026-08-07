"""Tests for shared PlagNetPlus model infrastructure."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from plagnetplus.models import (
    RepresentationModel,
    build_backbone,
    normalise_backbone_name,
)


@pytest.mark.parametrize(
    ("provided_name", "expected_name"),
    [
        ("resnet18", "resnet18"),
        ("resnet-18", "resnet18"),
        ("resnet_18", "resnet18"),
        ("rn18", "resnet18"),
        ("resnet50", "resnet50"),
        ("resnet-50", "resnet50"),
        ("rn50", "resnet50"),
    ],
)
def test_backbone_name_aliases(
    provided_name: str,
    expected_name: str,
) -> None:
    """Common user-facing aliases should map to one canonical name."""

    assert normalise_backbone_name(provided_name) == expected_name


@pytest.mark.parametrize(
    ("name", "expected_dimension"),
    [
        ("resnet18", 512),
        ("resnet50", 2048),
    ],
)
def test_backbone_feature_dimensions(
    name: str,
    expected_dimension: int,
) -> None:
    """Each supported backbone should return its expected feature width."""

    built = build_backbone(name)

    # A small synthetic image batch is sufficient to verify that removal
    # of the classification head exposes the intended representation.
    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    built.model.eval()

    with torch.no_grad():
        features = built.model(images)

    assert built.name == name
    assert built.initialization == "scratch"
    assert built.feature_dimension == expected_dimension
    assert features.shape == (
        2,
        expected_dimension,
    )
    assert torch.isfinite(features).all()


def test_unknown_backbone_is_rejected() -> None:
    """Architectures outside the controlled comparison should be rejected."""

    with pytest.raises(
        ValueError,
        match="Unsupported backbone",
    ):
        build_backbone("resnet101")


def test_pretrained_initialization_is_rejected() -> None:
    """The shared builder should not silently introduce pretrained weights."""

    with pytest.raises(
        ValueError,
        match="only initialization='scratch'",
    ):
        build_backbone(
            "resnet18",
            initialization="pretrained",  # type: ignore[arg-type]
        )


def test_representation_model_requires_common_methods() -> None:
    """Concrete models must provide both embedding and training operations."""

    class IncompleteModel(RepresentationModel):
        def encode(self, images: Tensor) -> Tensor:
            return images

    with pytest.raises(TypeError):
        IncompleteModel()
