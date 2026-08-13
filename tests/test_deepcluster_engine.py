from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch.optim import SGD
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from plagnetplus.models import (
    DeepClusterModel,
)
from plagnetplus.training.deepcluster_engine import (
    PseudoLabelDataset,
    assignment_stability,
    build_balanced_sampler,
    cluster_statistics,
    cycle_learning_rate,
    extract_backbone_features,
    fit_kmeans,
    inverse_cluster_frequency_weights,
    preprocess_features,
    train_pseudo_label_epoch,
    validate_cluster_health,
)


Tensor = torch.Tensor


class MetadataDataset(Dataset):
    def __init__(
        self,
        row_count: int,
    ) -> None:
        self.row_count = row_count

    def __len__(self) -> int:
        return self.row_count

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        return {
            "image": torch.full(
                (3, 32, 32),
                fill_value=(
                    index / 10.0
                ),
                dtype=torch.float32,
            ),
            "metadata": {
                "crystal_id": (
                    f"crystal_{index}"
                ),
                "manifest_order": index,
            },
        }


class ImageDataset(Dataset):
    def __init__(
        self,
        row_count: int,
    ) -> None:
        generator = torch.Generator()
        generator.manual_seed(42)

        self.images = torch.randn(
            row_count,
            3,
            32,
            32,
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        return {
            "image": self.images[index]
        }


def test_cycle_learning_rate() -> None:
    assert cycle_learning_rate(
        cycle=1,
        total_cycles=10,
        warmup_cycles=2,
        learning_rate=0.03,
    ) == pytest.approx(0.015)

    assert cycle_learning_rate(
        cycle=2,
        total_cycles=10,
        warmup_cycles=2,
        learning_rate=0.03,
    ) == pytest.approx(0.03)

    assert cycle_learning_rate(
        cycle=10,
        total_cycles=10,
        warmup_cycles=2,
        learning_rate=0.03,
    ) == pytest.approx(0.0)


def test_extract_backbone_features_alignment() -> None:
    dataset = MetadataDataset(6)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    model = DeepClusterModel(
        backbone_name="resnet18",
        cluster_count=3,
    )

    result = extract_backbone_features(
        model=model,
        loader=loader,
        expected_crystal_ids=[
            f"crystal_{index}"
            for index in range(6)
        ],
        expected_manifest_orders=list(
            range(6)
        ),
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert result.features.shape == (
        6,
        512,
    )

    assert np.isfinite(
        result.features
    ).all()

    assert result.crystal_ids.tolist() == [
        f"crystal_{index}"
        for index in range(6)
    ]


def test_preprocessing_and_kmeans_are_deterministic() -> None:
    rng = np.random.default_rng(42)

    centres = np.asarray(
        [
            [-5.0] * 16,
            [0.0] * 16,
            [5.0] * 16,
        ],
        dtype=np.float32,
    )

    features = np.vstack(
        [
            rng.normal(
                centre,
                0.5,
                size=(40, 16),
            )
            for centre in centres
        ]
    ).astype(np.float32)

    first_preprocessing = preprocess_features(
        features,
        pca_components=8,
        whiten=True,
        solver="randomized",
        seed=42,
    )

    second_preprocessing = preprocess_features(
        features,
        pca_components=8,
        whiten=True,
        solver="randomized",
        seed=42,
    )

    assert np.allclose(
        first_preprocessing.features,
        second_preprocessing.features,
    )

    first = fit_kmeans(
        first_preprocessing.features,
        cluster_count=3,
        seed=42,
    )

    second = fit_kmeans(
        second_preprocessing.features,
        cluster_count=3,
        seed=42,
    )

    assert np.array_equal(
        first.assignments,
        second.assignments,
    )

    assert (
        first.assignment_sha256
        == second.assignment_sha256
    )

    assert (
        first.statistics.occupied_clusters
        == 3
    )


def test_cluster_health_rejects_collapse() -> None:
    statistics = cluster_statistics(
        np.asarray(
            [0] * 98
            + [1]
            + [2],
            dtype=np.int64,
        ),
        cluster_count=3,
    )

    with pytest.raises(
        RuntimeError,
        match="collapse check failed",
    ):
        validate_cluster_health(
            statistics,
            required_occupied_cluster_fraction=1.0,
            minimum_normalised_cluster_entropy=0.75,
            maximum_largest_cluster_fraction=0.20,
        )


def test_inverse_frequency_weights_equalise_mass() -> None:
    assignments = np.asarray(
        [0, 0, 0, 1],
        dtype=np.int64,
    )

    weights = (
        inverse_cluster_frequency_weights(
            assignments,
            cluster_count=2,
        )
    )

    cluster_zero_mass = float(
        weights[
            assignments == 0
        ].sum()
    )

    cluster_one_mass = float(
        weights[
            assignments == 1
        ].sum()
    )

    assert cluster_zero_mass == pytest.approx(
        cluster_one_mass
    )

    sampler = build_balanced_sampler(
        assignments,
        cluster_count=2,
        samples_per_epoch=20,
        seed=42,
    )

    assert sampler.num_samples == 20
    assert sampler.replacement is True


def test_pseudo_label_dataset_preserves_image() -> None:
    base = ImageDataset(4)

    wrapped = PseudoLabelDataset(
        base,
        np.asarray(
            [3, 2, 1, 0],
            dtype=np.int64,
        ),
    )

    item = wrapped[1]

    assert torch.equal(
        item["image"],
        base[1]["image"],
    )

    assert item["pseudo_label"] == 2
    assert item["dataset_index"] == 1


def test_assignment_stability() -> None:
    previous = np.asarray(
        [0, 0, 1, 1]
    )

    permuted = np.asarray(
        [1, 1, 0, 0]
    )

    metrics = assignment_stability(
        permuted,
        previous,
    )

    assert metrics[
        "adjusted_rand_index"
    ] == pytest.approx(1.0)

    assert metrics[
        "normalised_mutual_information"
    ] == pytest.approx(1.0)


def test_train_pseudo_label_epoch() -> None:
    base = ImageDataset(8)

    dataset = PseudoLabelDataset(
        base,
        np.asarray(
            [0, 1, 2, 0, 1, 2, 0, 1],
            dtype=np.int64,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    model = DeepClusterModel(
        backbone_name="resnet18",
        cluster_count=3,
    )

    optimizer = SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
    )

    metrics = train_pseudo_label_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        use_amp=False,
        gradient_clip_norm=5.0,
    )

    assert metrics.images == 8
    assert metrics.batches == 2
    assert np.isfinite(metrics.loss)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.learning_rate == pytest.approx(
        0.01
    )
