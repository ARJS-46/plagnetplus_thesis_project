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
