"""Reusable feature clustering and pseudo-label training utilities."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import normalize
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from plagnetplus.models.deepcluster import DeepClusterModel


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Deterministic backbone feature extraction output."""

    features: np.ndarray
    crystal_ids: np.ndarray
    manifest_orders: np.ndarray
    seconds: float

    @property
    def row_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class FeaturePreprocessingResult:
    """PCA-whitened and row-normalised feature output."""

    features: np.ndarray
    pca: PCA
    retained_variance: float
    seconds: float


@dataclass(frozen=True)
class ClusterStatistics:
    """Cluster occupancy and balance diagnostics."""

    cluster_count: int
    occupied_clusters: int
    empty_clusters: int
    minimum_cluster_size: int
    median_cluster_size: float
    mean_cluster_size: float
    maximum_cluster_size: int
    largest_cluster_fraction: float
    normalised_cluster_entropy: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ClusteringResult:
    """One deterministic PCA/KMeans clustering result."""

    assignments: np.ndarray
    centres: np.ndarray
    inertia: float
    iterations: int
    statistics: ClusterStatistics
    assignment_sha256: str
    centres_sha256: str
    seconds: float


@dataclass(frozen=True)
class PseudoLabelEpochMetrics:
    """Metrics from one pseudo-label optimisation epoch."""

    loss: float
    accuracy: float
    batches: int
    images: int
    seconds: float
    learning_rate: float


class PseudoLabelDataset(Dataset):
    """Attach one manifest-aligned pseudo-label to every item."""

    def __init__(
        self,
        dataset: Dataset,
        assignments: np.ndarray,
    ) -> None:
        self.dataset = dataset

        labels = np.asarray(
            assignments
        )

        if labels.ndim != 1:
            raise ValueError(
                "assignments must be one-dimensional."
            )

        if len(labels) != len(dataset):
            raise ValueError(
                "Assignment count does not match dataset length: "
                f"{len(labels)} != {len(dataset)}."
            )

        if not np.issubdtype(
            labels.dtype,
            np.integer,
        ):
            raise TypeError(
                "assignments must contain integers."
            )

        if np.any(labels < 0):
            raise ValueError(
                "assignments contain negative values."
            )

        self.assignments = labels.astype(
            np.int64,
            copy=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        item = self.dataset[index]

        if not isinstance(
            item,
            Mapping,
        ):
            raise TypeError(
                "The wrapped dataset must return a mapping."
            )

        if "image" not in item:
            raise KeyError(
                "The wrapped dataset item lacks 'image'."
            )

        output = dict(item)

        output["pseudo_label"] = int(
            self.assignments[index]
        )

        output["dataset_index"] = int(
            index
        )

        return output


def _metadata_values(
    value: object,
) -> list[object]:
    if isinstance(
        value,
        Tensor,
    ):
        result = value.detach().cpu().tolist()

        if isinstance(result, list):
            return result

        return [result]

    if isinstance(
        value,
        np.ndarray,
    ):
        result = value.tolist()

        if isinstance(result, list):
            return result

        return [result]

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return list(value)

    return [value]


def array_sha256(
    values: np.ndarray,
) -> str:
    """Hash array identity, dtype, shape and contiguous bytes."""
    array = np.ascontiguousarray(
        values
    )

    digest = hashlib.sha256()

    digest.update(
        str(array.dtype).encode(
            "utf-8"
        )
    )

    digest.update(
        repr(array.shape).encode(
            "utf-8"
        )
    )

    digest.update(
        array.tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def cycle_learning_rate(
    *,
    cycle: int,
    total_cycles: int,
    warmup_cycles: int,
    learning_rate: float,
    minimum_learning_rate: float = 0.0,
) -> float:
    """Return one-based warm-up and cosine learning rate."""
    if not 1 <= cycle <= total_cycles:
        raise ValueError(
            "cycle must lie between one and total_cycles."
        )

    if total_cycles < 1:
        raise ValueError(
            "total_cycles must be positive."
        )

    if not 0 <= warmup_cycles <= total_cycles:
        raise ValueError(
            "warmup_cycles must lie between zero and total_cycles."
        )

    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be positive."
        )

    if not (
        0.0
        <= minimum_learning_rate
        <= learning_rate
    ):
        raise ValueError(
            "minimum_learning_rate is invalid."
        )

    if (
        warmup_cycles > 0
        and cycle <= warmup_cycles
    ):
        return float(
            learning_rate
            * cycle
            / warmup_cycles
        )

    cosine_cycles = (
        total_cycles
        - warmup_cycles
    )

    if cosine_cycles <= 0:
        return float(
            learning_rate
        )

    progress = (
        cycle
        - warmup_cycles
    ) / cosine_cycles

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )

    return float(
        minimum_learning_rate
        + (
            learning_rate
            - minimum_learning_rate
        )
        * cosine
    )


def set_optimizer_learning_rate(
    optimizer: Optimizer,
    learning_rate: float,
) -> None:
    """Set all optimiser parameter-group learning rates."""
    if learning_rate < 0.0:
        raise ValueError(
            "learning_rate cannot be negative."
        )

    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def extract_backbone_features(
    *,
    model: DeepClusterModel,
    loader: DataLoader,
    expected_crystal_ids: Sequence[str],
    expected_manifest_orders: Sequence[int],
    device: torch.device,
    use_amp: bool,
) -> FeatureExtractionResult:
    """Extract deterministic, exactly aligned backbone features."""
    expected_ids = np.asarray(
        expected_crystal_ids
    ).astype(str)

    expected_orders = np.asarray(
        expected_manifest_orders,
        dtype=np.int64,
    )

    if expected_ids.ndim != 1:
        raise ValueError(
            "expected_crystal_ids must be one-dimensional."
        )

    if expected_orders.ndim != 1:
        raise ValueError(
            "expected_manifest_orders must be one-dimensional."
        )

    if len(expected_ids) != len(
        expected_orders
    ):
        raise ValueError(
            "Expected ID and order counts differ."
        )

    model.eval()

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    feature_blocks: list[
        np.ndarray
    ] = []

    crystal_ids: list[str] = []
    manifest_orders: list[int] = []

    started = time.perf_counter()

    # Feature extraction is a frozen inference pass; gradients are unnecessary
    # until pseudo-label training begins.
    with torch.inference_mode():
        for batch in loader:
            if "metadata" not in batch:
                raise KeyError(
                    "Feature-extraction batches require metadata."
                )

            images = batch["image"].to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                features = model.encode(
                    images
                )

            if features.ndim != 2:
                raise RuntimeError(
                    "Model returned non-matrix features."
                )

            if not torch.isfinite(
                features
            ).all():
                raise RuntimeError(
                    "Feature extraction produced non-finite values."
                )

            feature_blocks.append(
                features.detach()
                .to(
                    dtype=torch.float32
                )
                .cpu()
                .numpy()
            )

            metadata = batch[
                "metadata"
            ]

            crystal_ids.extend(
                str(value)
                for value in _metadata_values(
                    metadata[
                        "crystal_id"
                    ]
                )
            )

            manifest_orders.extend(
                int(value)
                for value in _metadata_values(
                    metadata[
                        "manifest_order"
                    ]
                )
            )

    if not feature_blocks:
        raise RuntimeError(
            "Feature loader produced no batches."
        )

    feature_array = np.concatenate(
        feature_blocks,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    extracted_ids = np.asarray(
        crystal_ids
    ).astype(str)

    extracted_orders = np.asarray(
        manifest_orders,
        dtype=np.int64,
    )

    if feature_array.shape[0] != len(
        expected_ids
    ):
        raise RuntimeError(
            "Feature row count does not match expected rows."
        )

    if feature_array.shape[1] != (
        model.feature_dimension
    ):
        raise RuntimeError(
            "Unexpected backbone feature dimension."
        )

    if not np.array_equal(
        extracted_ids,
        expected_ids,
    ):
        raise RuntimeError(
            "Extracted crystal IDs are misaligned."
        )

    if not np.array_equal(
        extracted_orders,
        expected_orders,
    ):
        raise RuntimeError(
            "Extracted manifest orders are misaligned."
        )

    if not np.isfinite(
        feature_array
    ).all():
        raise RuntimeError(
            "Feature output contains non-finite values."
        )

    return FeatureExtractionResult(
        features=feature_array,
        crystal_ids=extracted_ids,
        manifest_orders=extracted_orders,
        seconds=(
            time.perf_counter()
            - started
        ),
    )


def preprocess_features(
    features: np.ndarray,
    *,
    pca_components: int,
    whiten: bool,
    solver: str,
    seed: int,
) -> FeaturePreprocessingResult:
    """Mean-centre, fit PCA, optionally whiten and L2-normalise."""
    values = np.asarray(
        features
    )

    if values.ndim != 2:
        raise ValueError(
            "features must be two-dimensional."
        )

    if not np.issubdtype(
        values.dtype,
        np.number,
    ):
        raise TypeError(
            "features must have a numeric dtype."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "features contain non-finite values."
        )

    maximum_components = min(
        values.shape
    )

    if not (
        1
        <= pca_components
        <= maximum_components
    ):
        raise ValueError(
            "pca_components exceeds the available matrix rank."
        )

    if solver not in {
        "randomized",
        "full",
    }:
        raise ValueError(
            "Unsupported PCA solver."
        )

    started = time.perf_counter()

    # Reduce and whiten backbone features before k-means so high-variance
    # dimensions do not dominate the pseudo-label assignments.
    pca = PCA(
        n_components=pca_components,
        whiten=whiten,
        svd_solver=solver,
        random_state=(
            seed
            if solver == "randomized"
            else None
        ),
    )

    transformed = pca.fit_transform(
        values.astype(
            np.float32,
            copy=False,
        )
    )

    processed = normalize(
        transformed,
        norm="l2",
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    if processed.shape != (
        values.shape[0],
        pca_components,
    ):
        raise RuntimeError(
            "Unexpected PCA output shape."
        )

    if not np.isfinite(
        processed
    ).all():
        raise RuntimeError(
            "Feature preprocessing produced non-finite values."
        )

    norms = np.linalg.norm(
        processed,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError(
            "Processed feature rows are not L2-normalised."
        )

    return FeaturePreprocessingResult(
        features=processed,
        pca=pca,
        retained_variance=float(
            pca.explained_variance_ratio_.sum()
        ),
        seconds=(
            time.perf_counter()
            - started
        ),
    )


def cluster_statistics(
    assignments: np.ndarray,
    *,
    cluster_count: int,
) -> ClusterStatistics:
    """Calculate cluster occupancy, size and entropy diagnostics."""
    labels = np.asarray(
        assignments
    )

    if labels.ndim != 1:
        raise ValueError(
            "assignments must be one-dimensional."
        )

    if not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise TypeError(
            "assignments must contain integers."
        )

    if len(labels) < 1:
        raise ValueError(
            "assignments cannot be empty."
        )

    if cluster_count < 2:
        raise ValueError(
            "cluster_count must be at least two."
        )

    if np.any(labels < 0):
        raise ValueError(
            "assignments contain negative labels."
        )

    if np.any(
        labels >= cluster_count
    ):
        raise ValueError(
            "assignments exceed cluster_count."
        )

    # Track cluster occupancy and entropy explicitly so collapsed pseudo-label
    # partitions can be rejected before they drive another training cycle.
    counts = np.bincount(
        labels,
        minlength=cluster_count,
    )

    occupied = counts[
        counts > 0
    ]

    probabilities = (
        occupied.astype(
            np.float64
        )
        / occupied.sum()
    )

    entropy = float(
        -np.sum(
            probabilities
            * np.log(probabilities)
        )
    )

    normalised_entropy = float(
        entropy
        / np.log(cluster_count)
    )

    return ClusterStatistics(
        cluster_count=int(
            cluster_count
        ),
        occupied_clusters=int(
            len(occupied)
        ),
        empty_clusters=int(
            cluster_count
            - len(occupied)
        ),
        minimum_cluster_size=int(
            occupied.min()
        ),
        median_cluster_size=float(
            np.median(occupied)
        ),
        mean_cluster_size=float(
            occupied.mean()
        ),
        maximum_cluster_size=int(
            occupied.max()
        ),
        largest_cluster_fraction=float(
            occupied.max()
            / len(labels)
        ),
        normalised_cluster_entropy=(
            normalised_entropy
        ),
    )


def validate_cluster_health(
    statistics: ClusterStatistics,
    *,
    required_occupied_cluster_fraction: float,
    minimum_normalised_cluster_entropy: float,
    maximum_largest_cluster_fraction: float,
) -> None:
    """Raise when a clustering violates frozen collapse thresholds."""
    if not (
        0.0
        < required_occupied_cluster_fraction
        <= 1.0
    ):
        raise ValueError(
            "required_occupied_cluster_fraction is invalid."
        )

    required_occupied = math.ceil(
        statistics.cluster_count
        * required_occupied_cluster_fraction
    )

    failures: list[str] = []

    if (
        statistics.occupied_clusters
        < required_occupied
    ):
        failures.append(
            "insufficient occupied clusters"
        )

    if (
        statistics.normalised_cluster_entropy
        < minimum_normalised_cluster_entropy
    ):
        failures.append(
            "cluster entropy below threshold"
        )

    if (
        statistics.largest_cluster_fraction
        > maximum_largest_cluster_fraction
    ):
        failures.append(
            "largest cluster fraction above threshold"
        )

    if failures:
        raise RuntimeError(
            "DeepCluster collapse check failed: "
            + "; ".join(failures)
            + "."
        )


def fit_kmeans(
    features: np.ndarray,
    *,
    cluster_count: int,
    seed: int,
    initialization: str = "k-means++",
    n_init: int = 1,
    maximum_iterations: int = 100,
    tolerance: float = 1e-4,
    algorithm: str = "lloyd",
) -> ClusteringResult:
    """Fit full scikit-learn KMeans to processed features."""
    values = np.asarray(
        features,
        dtype=np.float32,
    )

    if values.ndim != 2:
        raise ValueError(
            "features must be two-dimensional."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "features contain non-finite values."
        )

    if not (
        2
        <= cluster_count
        <= values.shape[0]
    ):
        raise ValueError(
            "cluster_count is incompatible with row count."
        )

    started = time.perf_counter()

    estimator = KMeans(
        n_clusters=cluster_count,
        init=initialization,
        n_init=n_init,
        max_iter=maximum_iterations,
        tol=tolerance,
        random_state=seed,
        algorithm=algorithm,
    )

    assignments = estimator.fit_predict(
        values
    ).astype(
        np.int32,
        copy=False,
    )

    centres = np.asarray(
        estimator.cluster_centers_,
        dtype=np.float32,
    )

    statistics = cluster_statistics(
        assignments,
        cluster_count=cluster_count,
    )

    return ClusteringResult(
        assignments=assignments,
        centres=centres,
        inertia=float(
            estimator.inertia_
        ),
        iterations=int(
            estimator.n_iter_
        ),
        statistics=statistics,
        assignment_sha256=(
            array_sha256(
                assignments
            )
        ),
        centres_sha256=(
            array_sha256(
                centres
            )
        ),
        seconds=(
            time.perf_counter()
            - started
        ),
    )


def assignment_stability(
    current_assignments: np.ndarray,
    previous_assignments: np.ndarray,
) -> dict[str, float]:
    """Compare current and preceding pseudo-label partitions."""
    current = np.asarray(
        current_assignments
    )

    previous = np.asarray(
        previous_assignments
    )

    if current.ndim != 1:
        raise ValueError(
            "current_assignments must be one-dimensional."
        )

    if previous.ndim != 1:
        raise ValueError(
            "previous_assignments must be one-dimensional."
        )

    if current.shape != previous.shape:
        raise ValueError(
            "Assignment vectors have different shapes."
        )

    return {
        "adjusted_rand_index": float(
            adjusted_rand_score(
                previous,
                current,
            )
        ),
        "normalised_mutual_information": float(
            normalized_mutual_info_score(
                previous,
                current,
            )
        ),
    }


def inverse_cluster_frequency_weights(
    assignments: np.ndarray,
    *,
    cluster_count: int,
) -> np.ndarray:
    """Return row weights giving every occupied cluster equal mass."""
    labels = np.asarray(
        assignments
    )

    statistics = cluster_statistics(
        labels,
        cluster_count=cluster_count,
    )

    if statistics.empty_clusters > 0:
        raise ValueError(
            "Cannot balance assignments containing empty clusters."
        )

    counts = np.bincount(
        labels,
        minlength=cluster_count,
    ).astype(
        np.float64
    )

    # Inverse cluster-frequency weighting reduces the influence of very large
    # pseudo-label clusters during the following training epoch.
    weights = (
        1.0
        / counts[labels]
    )

    weights *= (
        len(weights)
        / weights.sum()
    )

    if not np.isfinite(
        weights
    ).all():
        raise RuntimeError(
            "Sampling weights contain non-finite values."
        )

    if np.any(weights <= 0.0):
        raise RuntimeError(
            "Sampling weights must be positive."
        )

    return weights


def build_balanced_sampler(
    assignments: np.ndarray,
    *,
    cluster_count: int,
    samples_per_epoch: int,
    seed: int,
) -> WeightedRandomSampler:
    """Build a deterministic inverse-frequency replacement sampler."""
    if samples_per_epoch < 1:
        raise ValueError(
            "samples_per_epoch must be positive."
        )

    weights = inverse_cluster_frequency_weights(
        assignments,
        cluster_count=cluster_count,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(
            weights,
            dtype=torch.double,
        ),
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )


def train_pseudo_label_epoch(
    *,
    model: DeepClusterModel,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: float | None,
    scaler: Any | None = None,
) -> PseudoLabelEpochMetrics:
    """Train one epoch against the current pseudo-label assignments."""
    model.train()

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    if (
        amp_enabled
        and scaler is None
    ):
        raise ValueError(
            "CUDA AMP requires a gradient scaler."
        )

    started = time.perf_counter()

    total_loss = 0.0
    correct = 0
    image_count = 0
    batch_count = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        pseudo_labels = torch.as_tensor(
            batch["pseudo_label"],
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            features, logits = model(
                images
            )

            if not torch.isfinite(
                features
            ).all():
                raise RuntimeError(
                    "Training produced non-finite features."
                )

            if not torch.isfinite(
                logits
            ).all():
                raise RuntimeError(
                    "Training produced non-finite logits."
                )

            loss = nn.functional.cross_entropy(
                logits,
                pseudo_labels,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training produced a non-finite loss."
            )

        if amp_enabled:
            scaler.scale(loss).backward()

            if gradient_clip_norm is not None:
                scaler.unscale_(
                    optimizer
                )

                gradient_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip_norm,
                    )
                )

                if not torch.isfinite(
                    gradient_norm
                ):
                    raise RuntimeError(
                        "Gradient norm is non-finite."
                    )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:
            loss.backward()

            if gradient_clip_norm is not None:
                gradient_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip_norm,
                    )
                )

                if not torch.isfinite(
                    gradient_norm
                ):
                    raise RuntimeError(
                        "Gradient norm is non-finite."
                    )

            optimizer.step()

        batch_size = int(
            images.shape[0]
        )

        total_loss += (
            float(loss.detach().item())
            * batch_size
        )

        correct += int(
            (
                logits.detach().argmax(
                    dim=1
                )
                == pseudo_labels
            ).sum().item()
        )

        image_count += batch_size
        batch_count += 1

    if image_count < 1:
        raise RuntimeError(
            "Pseudo-label loader produced no images."
        )

    learning_rates = {
        float(group["lr"])
        for group in optimizer.param_groups
    }

    if len(learning_rates) != 1:
        raise RuntimeError(
            "Optimiser parameter groups have different learning rates."
        )

    return PseudoLabelEpochMetrics(
        loss=float(
            total_loss
            / image_count
        ),
        accuracy=float(
            correct
            / image_count
        ),
        batches=int(
            batch_count
        ),
        images=int(
            image_count
        ),
        seconds=float(
            time.perf_counter()
            - started
        ),
        learning_rate=(
            learning_rates.pop()
        ),
    )
