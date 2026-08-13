from __future__ import annotations

import torch
import pytest

from plagnetplus.models import (
    DeepClusterModel,
    RepresentationModel,
)


def test_deepcluster_shapes_and_metadata() -> None:
    model = DeepClusterModel(
        backbone_name="resnet18",
        cluster_count=7,
    )

    images = torch.randn(
        3,
        3,
        64,
        64,
    )

    features, logits = model(images)

    assert isinstance(
        model,
        RepresentationModel,
    )

    assert features.shape == (
        3,
        512,
    )

    assert logits.shape == (
        3,
        7,
    )

    assert model.model_metadata() == {
        "method": "deepcluster",
        "backbone": "resnet18",
        "initialization": "scratch",
        "feature_dimension": 512,
        "cluster_count": 7,
        "classifier_reinitialised_each_cycle": True,
    }


def test_deepcluster_pseudo_label_loss() -> None:
    model = DeepClusterModel(
        backbone_name="resnet18",
        cluster_count=4,
    )

    images = torch.randn(
        4,
        3,
        64,
        64,
    )

    labels = torch.tensor(
        [0, 1, 2, 3]
    )

    loss = model.training_loss(
        (
            images,
            labels,
        )
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_classifier_reset_is_seeded() -> None:
    model = DeepClusterModel(
        backbone_name="resnet18",
        cluster_count=5,
    )

    original_classifier = (
        model.classifier
    )

    model.reset_classifier(
        seed=123
    )

    first_weights = (
        model.classifier.weight
        .detach()
        .clone()
    )

    assert (
        model.classifier
        is not original_classifier
    )

    model.reset_classifier(
        seed=123
    )

    second_weights = (
        model.classifier.weight
        .detach()
        .clone()
    )

    assert torch.equal(
        first_weights,
        second_weights,
    )

    assert torch.count_nonzero(
        model.classifier.bias
    ) == 0


def test_deepcluster_rejects_invalid_labels() -> None:
    model = DeepClusterModel(
        backbone_name="resnet18",
        cluster_count=3,
    )

    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    with pytest.raises(
        ValueError,
        match="exceed",
    ):
        model.pseudo_label_loss(
            images,
            torch.tensor(
                [0, 3]
            ),
        )
