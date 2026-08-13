import pytest
import torch

from plagnetplus.models import (
    SimCLRModel,
    build_backbone,
    nt_xent_loss,
)




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
