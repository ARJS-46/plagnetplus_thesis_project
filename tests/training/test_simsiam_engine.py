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


def test_learning_rate_schedule() -> None:
    assert learning_rate_for_epoch(
        epoch_index=0,
        total_epochs=100,
        warmup_epochs=10,
        base_learning_rate=0.05,
        minimum_learning_rate=0.0,
    ) == pytest.approx(0.005)

    assert learning_rate_for_epoch(
        epoch_index=9,
        total_epochs=100,
        warmup_epochs=10,
        base_learning_rate=0.05,
        minimum_learning_rate=0.0,
    ) == pytest.approx(0.05)

    assert learning_rate_for_epoch(
        epoch_index=10,
        total_epochs=100,
        warmup_epochs=10,
        base_learning_rate=0.05,
        minimum_learning_rate=0.0,
    ) == pytest.approx(0.05)

    assert learning_rate_for_epoch(
        epoch_index=99,
        total_epochs=100,
        warmup_epochs=10,
        base_learning_rate=0.05,
        minimum_learning_rate=0.0,
    ) == pytest.approx(0.0)


def test_optimizer_has_exact_parameter_groups() -> None:
    model = TinySimSiam()

    optimizer = build_simsiam_optimizer(
        model=model,
        learning_rate=0.05,
        predictor_learning_rate=0.04,
        momentum=0.9,
        weight_decay=0.0001,
    )

    assert {
        group["group_name"]
        for group in optimizer.param_groups
    } == {
        "backbone_projector",
        "predictor",
    }

    assert optimizer_learning_rates(
        optimizer
    ) == pytest.approx(
        (
            0.05,
            0.04,
        )
    )


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


def test_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    torch.manual_seed(9)

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
        backbone_projector_learning_rate=0.05,
        predictor_learning_rate=0.05,
        use_amp=False,
        scaler=None,
        gradient_clip_norm=5.0,
    )

    expected_state = {
        key: value.detach().clone()
        for key, value
        in model.state_dict().items()
    }

    checkpoint = (
        tmp_path
        / "simsiam_epoch_001.pt"
    )

    generator = (
        torch.Generator()
        .manual_seed(123)
    )

    save_checkpoint_atomic(
        checkpoint,
        completed_epoch=1,
        model=model,
        optimizer=optimizer,
        scaler=None,
        history=[
            train_metrics.to_dict()
        ],
        config={
            "epochs": 100,
        },
        benchmark_id="benchmark",
        manifest_fingerprint_sha256=(
            "fingerprint"
        ),
        train_manifest_sha256=(
            "train-sha"
        ),
        train_generator=generator,
    )

    assert checkpoint.is_file()
    assert not checkpoint.with_suffix(
        ".pt.partial"
    ).exists()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    loaded = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        expected_benchmark_id="benchmark",
        expected_manifest_fingerprint_sha256=(
            "fingerprint"
        ),
        scaler=None,
        train_generator=generator,
        restore_rng=False,
    )

    assert loaded["completed_epoch"] == 1

    for key, value in model.state_dict().items():
        assert torch.equal(
            value,
            expected_state[key],
        )


def test_checkpoint_rejects_wrong_fingerprint(
    tmp_path: Path,
) -> None:
    model = TinySimSiam()

    optimizer = build_simsiam_optimizer(
        model=model,
        learning_rate=0.05,
        predictor_learning_rate=0.05,
        momentum=0.9,
        weight_decay=0.0001,
    )

    checkpoint = (
        tmp_path
        / "checkpoint.pt"
    )

    save_checkpoint_atomic(
        checkpoint,
        completed_epoch=1,
        model=model,
        optimizer=optimizer,
        scaler=None,
        history=[],
        config={},
        benchmark_id="benchmark",
        manifest_fingerprint_sha256=(
            "correct"
        ),
        train_manifest_sha256="train",
    )

    with pytest.raises(
        ValueError,
        match="fingerprint",
    ):
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            expected_benchmark_id="benchmark",
            expected_manifest_fingerprint_sha256=(
                "wrong"
            ),
            restore_rng=False,
        )


def test_history_csv(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.csv"

    row = {
        "epoch": 1,
        "split": "train",
        "loss": -0.2,
        "mean_cosine_similarity": 0.2,
        "projection_standard_deviation": 0.02,
        "collapse_level": 0.1,
        "backbone_feature_norm": 3.0,
        "projection_norm": 4.0,
        "prediction_norm": 2.0,
        "backbone_projector_learning_rate": 0.05,
        "predictor_learning_rate": 0.05,
        "batches": 2,
        "images": 7,
        "seconds": 1.2,
    }

    write_history_csv_atomic(
        path,
        [row],
    )

    with path.open(
        encoding="utf-8"
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    assert len(rows) == 1
    assert rows[0]["split"] == "train"
    assert rows[0]["epoch"] == "1"


def test_invalid_view_batch_is_rejected() -> None:
    model = TinySimSiam()

    optimizer = build_simsiam_optimizer(
        model=model,
        learning_rate=0.05,
        predictor_learning_rate=0.05,
        momentum=0.9,
        weight_decay=0.0001,
    )

    with pytest.raises(
        TypeError,
        match="two augmented views",
    ):
        train_one_epoch(
            model=model,
            loader=[
                {
                    "image": torch.randn(
                        4,
                        4,
                    )
                }
            ],
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=1,
            backbone_projector_learning_rate=0.05,
            predictor_learning_rate=0.05,
            use_amp=False,
            scaler=None,
            gradient_clip_norm=5.0,
        )
