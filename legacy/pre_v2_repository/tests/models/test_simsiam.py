"""Tests for the native PlagNetPlus SimSiam implementation."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from plagnetplus.models.simsiam import (
    SimSiamModel,
    SimSiamPredictionHead,
    SimSiamProjectionHead,
    negative_cosine_similarity,
    symmetric_negative_cosine_loss,
)




def test_symmetric_loss_detaches_both_targets() -> None:
    generator = torch.Generator().manual_seed(7)

    first_prediction = torch.randn(
        4,
        8,
        generator=generator,
        requires_grad=True,
    )

    second_prediction = torch.randn(
        4,
        8,
        generator=generator,
        requires_grad=True,
    )

    first_projection = torch.randn(
        4,
        8,
        generator=generator,
        requires_grad=True,
    )

    second_projection = torch.randn(
        4,
        8,
        generator=generator,
        requires_grad=True,
    )

    loss = symmetric_negative_cosine_loss(
        first_prediction,
        second_projection,
        second_prediction,
        first_projection,
    )

    loss.backward()

    assert first_prediction.grad is not None
    assert second_prediction.grad is not None

    assert torch.isfinite(
        first_prediction.grad
    ).all()

    assert torch.isfinite(
        second_prediction.grad
    ).all()

    assert first_projection.grad is None
    assert second_projection.grad is None






def test_model_shapes_and_objective_backward() -> None:
    torch.manual_seed(11)

    model = SimSiamModel(
        backbone_name="resnet18",
        initialization="scratch",
        projection_hidden_dimension=2048,
        projection_dimension=2048,
        prediction_hidden_dimension=512,
        prediction_dimension=2048,
        final_projection_batch_norm_affine=False,
    )

    model.train()

    first_view = torch.randn(
        4,
        3,
        64,
        64,
    )

    second_view = torch.randn(
        4,
        3,
        64,
        64,
    )

    features, projections, predictions = (
        model(first_view)
    )

    assert features.shape == (
        4,
        512,
    )

    assert projections.shape == (
        4,
        2048,
    )

    assert predictions.shape == (
        4,
        2048,
    )

    loss, diagnostics = model.simsiam_objective(
        (
            first_view,
            second_view,
        )
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    assert set(diagnostics) == {
        "mean_cosine_similarity",
        "projection_standard_deviation",
        "collapse_level",
        "backbone_feature_norm",
        "projection_norm",
        "prediction_norm",
    }

    for value in diagnostics.values():
        assert value.ndim == 0
        assert torch.isfinite(value)

    assert (
        0.0
        <= diagnostics[
            "collapse_level"
        ].item()
        <= 1.0
    )

    loss.backward()

    backbone_gradients = [
        parameter.grad
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    projector_gradients = [
        parameter.grad
        for parameter in model.projector.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    predictor_gradients = [
        parameter.grad
        for parameter in model.predictor.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    assert backbone_gradients
    assert projector_gradients
    assert predictor_gradients

    assert all(
        torch.isfinite(gradient).all()
        for gradient in (
            backbone_gradients
            + projector_gradients
            + predictor_gradients
        )
    )
