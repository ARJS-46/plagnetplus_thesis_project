import pytest
import torch

from plagnetplus.models import (
    SimCLRModel,
    build_backbone,
    nt_xent_loss,
)


@pytest.mark.parametrize(
    ("name", "expected_dimension"),
    [
        ("resnet18", 512),
        ("resnet-50", 2048),
    ],
)
def test_backbone_feature_dimensions(
    name,
    expected_dimension,
):
    built = build_backbone(name)

    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    built.model.eval()

    with torch.no_grad():
        features = built.model(
            images
        )

    assert built.feature_dimension == (
        expected_dimension
    )

    assert features.shape == (
        2,
        expected_dimension,
    )

    assert torch.isfinite(
        features
    ).all()


def test_simclr_resnet18_forward_and_backward():
    torch.manual_seed(42)

    model = SimCLRModel(
        backbone_name="resnet18",
        projection_hidden_dimension=128,
        projection_dimension=32,
        temperature=0.2,
    )

    views = (
        torch.randn(
            4,
            3,
            64,
            64,
        ),
        torch.randn(
            4,
            3,
            64,
            64,
        ),
    )

    first_features, first_projection = (
        model(views[0])
    )

    assert first_features.shape == (
        4,
        512,
    )

    assert first_projection.shape == (
        4,
        32,
    )

    loss, accuracy = (
        model.contrastive_objective(
            views
        )
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(accuracy)

    assert 0.0 <= float(
        accuracy
    ) <= 1.0

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients

    assert all(
        torch.isfinite(
            gradient
        ).all()
        for gradient in gradients
    )


def test_simclr_resnet50_metadata():
    model = SimCLRModel(
        backbone_name="resnet50",
        projection_dimension=128,
        temperature=0.2,
    )

    metadata = (
        model.model_metadata()
    )

    assert metadata["method"] == (
        "simclr"
    )

    assert metadata["backbone"] == (
        "resnet50"
    )

    assert metadata[
        "feature_dimension"
    ] == 2048

    assert metadata[
        "projection_hidden_dimension"
    ] == 2048

    assert metadata[
        "projection_dimension"
    ] == 128


def test_nt_xent_rejects_invalid_temperature():
    projections = torch.randn(
        4,
        32,
    )

    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        nt_xent_loss(
            projections,
            projections,
            temperature=0.0,
        )


def test_simclr_requires_exactly_two_views():
    model = SimCLRModel(
        backbone_name="resnet18"
    )

    with pytest.raises(
        ValueError,
        match="exactly two",
    ):
        model.training_loss(
            (
                torch.randn(
                    2,
                    3,
                    64,
                    64,
                ),
            )
        )


def test_unknown_backbone_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported backbone",
    ):
        build_backbone(
            "resnet101"
        )
