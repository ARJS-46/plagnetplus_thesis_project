"""Tests for the native SimSiam training engine."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional

from plagnetplus.models.simsiam import (
    symmetric_negative_cosine_loss,
)
from plagnetplus.training.simsiam_engine import (
    build_simsiam_optimizer,
    evaluate_one_epoch,
    learning_rate_for_epoch,
    load_checkpoint,
    optimizer_learning_rates,
    save_checkpoint_atomic,
    train_one_epoch,
    write_history_csv_atomic,
)


class TinySimSiam(nn.Module):
    """Small SimSiam-compatible model for fast engine tests."""

    def __init__(self) -> None:
        super().__init__()

        self.backbone = nn.Linear(
            4,
            6,
        )

        self.projector = nn.Linear(
            6,
            8,
        )

        self.predictor = nn.Linear(
            8,
            8,
        )

    def simsiam_objective(
        self,
        views: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        first_features = self.backbone(
            views[0]
        )

        second_features = self.backbone(
            views[1]
        )

        first_projection = self.projector(
            first_features
        )

        second_projection = self.projector(
            second_features
        )

        first_prediction = self.predictor(
            first_projection
        )

        second_prediction = self.predictor(
            second_projection
        )

        loss = symmetric_negative_cosine_loss(
            first_prediction,
            second_projection,
            second_prediction,
            first_projection,
        )

        normalised = functional.normalize(
            first_projection.detach(),
            dim=1,
        )

        projection_std = (
            normalised.std(
                dim=0,
            ).mean()
        )

        diagnostics = {
            "mean_cosine_similarity": (
                -loss.detach()
            ),
            "projection_standard_deviation": (
                projection_std
            ),
            "collapse_level": torch.clamp(
                1.0
                - projection_std
                * (8**0.5),
                min=0.0,
                max=1.0,
            ),
            "backbone_feature_norm": (
                first_features.detach()
                .norm(dim=1)
                .mean()
            ),
            "projection_norm": (
                first_projection.detach()
                .norm(dim=1)
                .mean()
            ),
            "prediction_norm": (
                first_prediction.detach()
                .norm(dim=1)
                .mean()
            ),
        }

        return loss, diagnostics


def loader_batches() -> list[
    dict[str, object]
]:
    generator = (
        torch.Generator()
        .manual_seed(42)
    )

    return [
        {
            "image": (
                torch.randn(
                    4,
                    4,
                    generator=generator,
                ),
                torch.randn(
                    4,
                    4,
                    generator=generator,
                ),
            )
        },
        {
            "image": (
                torch.randn(
                    3,
                    4,
                    generator=generator,
                ),
                torch.randn(
                    3,
                    4,
                    generator=generator,
                ),
            )
        },
    ]






def test_train_and_validation_epochs() -> None:
    torch.manual_seed(5)

    model = TinySimSiam()

    optimizer = build_simsiam_optimizer(
        model=model,
        learning_rate=0.05,
        predictor_learning_rate=0.05,
        momentum=0.9,
        weight_decay=0.0001,
    )

    train_metrics = train_one_epoch(
        model=model,
        loader=loader_batches(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=1,
        backbone_projector_learning_rate=0.02,
        predictor_learning_rate=0.05,
        use_amp=False,
        scaler=None,
        gradient_clip_norm=5.0,
    )

    assert train_metrics.split == "train"
    assert train_metrics.images == 7
    assert train_metrics.batches == 2
    assert train_metrics.loss == pytest.approx(
        train_metrics.loss
    )

    assert (
        train_metrics
        .backbone_projector_learning_rate
        == pytest.approx(0.02)
    )

    assert (
        train_metrics.predictor_learning_rate
        == pytest.approx(0.05)
    )

    validation_metrics = evaluate_one_epoch(
        model=model,
        loader=loader_batches(),
        device=torch.device("cpu"),
        epoch=1,
        backbone_projector_learning_rate=0.02,
        predictor_learning_rate=0.05,
        use_amp=False,
    )

    assert (
        validation_metrics.split
        == "validation"
    )

    assert validation_metrics.images == 7
    assert validation_metrics.batches == 2

    for value in (
        train_metrics.loss,
        train_metrics.collapse_level,
        validation_metrics.loss,
        validation_metrics.collapse_level,
    ):
        assert value == pytest.approx(value)
