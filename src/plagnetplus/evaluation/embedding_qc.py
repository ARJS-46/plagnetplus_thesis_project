"""Representation-health diagnostics for embedding matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from plagnetplus.evaluation.embedding_io import (
    EmbeddingBundle,
)


def _validated_features(
    features: np.ndarray,
) -> np.ndarray:
    array = np.asarray(features)

    if array.ndim != 2:
        raise ValueError(
            "features must be two-dimensional; "
            f"received shape {array.shape}."
        )

    if array.shape[0] < 2:
        raise ValueError(
            "At least two embedding rows are required."
        )

    if array.shape[1] < 1:
        raise ValueError(
            "At least one embedding dimension is required."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise TypeError(
            "features must have a numeric dtype."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "features contain NaN or infinite values."
        )

    return array


def dimension_statistics(
    features: np.ndarray,
    *,
    near_zero_variance_threshold: float = 1e-12,
) -> pd.DataFrame:
    """Calculate summary statistics for every dimension."""
    array = _validated_features(
        features
    ).astype(
        np.float64,
        copy=False,
    )

    means = array.mean(axis=0)
    variances = array.var(axis=0)
    standard_deviations = np.sqrt(
        variances
    )

    return pd.DataFrame(
        {
            "dimension": np.arange(
                array.shape[1],
                dtype=np.int64,
            ),
            "mean": means,
            "standard_deviation": (
                standard_deviations
            ),
            "variance": variances,
            "minimum": array.min(axis=0),
            "maximum": array.max(axis=0),
            "zero_variance": (
                variances == 0.0
            ),
            "near_zero_variance": (
                variances
                <= near_zero_variance_threshold
            ),
        }
    )


def entropy_effective_rank(
    features: np.ndarray,
) -> tuple[float, float]:
    """Return absolute and dimension-normalised effective rank."""
    array = _validated_features(
        features
    ).astype(
        np.float64,
        copy=False,
    )

    centred = (
        array
        - array.mean(
            axis=0,
            keepdims=True,
        )
    )

    # Effective rank is derived from the embedding covariance spectrum and is
    # used as a compact diagnostic for representational compression or collapse.
    covariance = (
        centred.T
        @ centred
    ) / float(
        array.shape[0] - 1
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    eigenvalues = np.clip(
        eigenvalues,
        a_min=0.0,
        a_max=None,
    )

    total = float(
        eigenvalues.sum()
    )

    if total <= 0.0:
        return 0.0, 0.0

    probabilities = (
        eigenvalues
        / total
    )

    positive = probabilities[
        probabilities > 0.0
    ]

    entropy = -float(
        np.sum(
            positive
            * np.log(positive)
        )
    )

    effective_rank = float(
        np.exp(entropy)
    )

    normalised_rank = (
        effective_rank
        / float(array.shape[1])
    )

    return (
        effective_rank,
        normalised_rank,
    )


def sampled_mean_pairwise_cosine_similarity(
    features: np.ndarray,
    *,
    pair_count: int = 100_000,
    seed: int = 42,
    chunk_size: int = 10_000,
) -> float:
    """Estimate mean cosine similarity for unrelated row pairs."""
    array = _validated_features(
        features
    ).astype(
        np.float64,
        copy=False,
    )

    if pair_count < 1:
        raise ValueError(
            "pair_count must be positive."
        )

    if chunk_size < 1:
        raise ValueError(
            "chunk_size must be positive."
        )

    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    normalised = np.divide(
        array,
        norms,
        out=np.zeros_like(array),
        where=norms > 0.0,
    )

    row_count = array.shape[0]
    rng = np.random.default_rng(seed)

    similarity_sum = 0.0
    completed_pairs = 0

    while completed_pairs < pair_count:
        current_count = min(
            chunk_size,
            pair_count - completed_pairs,
        )

        left_indices = rng.integers(
            0,
            row_count,
            size=current_count,
        )

        right_indices = rng.integers(
            0,
            row_count - 1,
            size=current_count,
        )

        right_indices += (
            right_indices
            >= left_indices
        )

        similarities = np.einsum(
            "ij,ij->i",
            normalised[left_indices],
            normalised[right_indices],
        )

        similarity_sum += float(
            similarities.sum()
        )

        completed_pairs += current_count

    return (
        similarity_sum
        / float(pair_count)
    )


def embedding_qc_summary(
    bundle: EmbeddingBundle,
    *,
    near_zero_variance_threshold: float = 1e-12,
    pair_count: int = 100_000,
    pair_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate the standard representation-health report."""
    features = _validated_features(
        bundle.features
    )

    dimension_table = dimension_statistics(
        features,
        near_zero_variance_threshold=(
            near_zero_variance_threshold
        ),
    )

    effective_rank, normalised_rank = (
        entropy_effective_rank(features)
    )

    norms = np.linalg.norm(
        features.astype(
            np.float64,
            copy=False,
        ),
        axis=1,
    )

    # Exact duplicate rows are tracked separately from low variance because
    # both can signal degenerate embedding behaviour.
    unique_row_count = int(
        np.unique(
            np.ascontiguousarray(features),
            axis=0,
        ).shape[0]
    )

    duplicate_row_count = (
        bundle.row_count
        - unique_row_count
    )

    standard_deviations = (
        dimension_table[
            "standard_deviation"
        ].to_numpy()
    )

    summary: dict[str, Any] = {
        "row_count": bundle.row_count,
        "feature_dimension": (
            bundle.feature_dimension
        ),
        "feature_dtype": str(
            features.dtype
        ),
        "all_finite": bool(
            np.isfinite(features).all()
        ),
        "zero_variance_dimensions": int(
            dimension_table[
                "zero_variance"
            ].sum()
        ),
        "near_zero_variance_dimensions": int(
            dimension_table[
                "near_zero_variance"
            ].sum()
        ),
        "near_zero_variance_threshold": float(
            near_zero_variance_threshold
        ),
        "minimum_dimension_standard_deviation": float(
            standard_deviations.min()
        ),
        "median_dimension_standard_deviation": float(
            np.median(
                standard_deviations
            )
        ),
        "maximum_dimension_standard_deviation": float(
            standard_deviations.max()
        ),
        "effective_rank": float(
            effective_rank
        ),
        "normalised_effective_rank": float(
            normalised_rank
        ),
        "mean_embedding_norm": float(
            norms.mean()
        ),
        "embedding_norm_standard_deviation": float(
            norms.std()
        ),
        "minimum_embedding_norm": float(
            norms.min()
        ),
        "maximum_embedding_norm": float(
            norms.max()
        ),
        "zero_norm_rows": int(
            np.count_nonzero(
                norms == 0.0
            )
        ),
        "unique_embedding_rows": (
            unique_row_count
        ),
        "exact_duplicate_embedding_rows": (
            duplicate_row_count
        ),
        "exact_duplicate_embedding_rate": float(
            duplicate_row_count
            / bundle.row_count
        ),
        "sampled_mean_pairwise_cosine_similarity": float(
            sampled_mean_pairwise_cosine_similarity(
                features,
                pair_count=pair_count,
                seed=pair_seed,
            )
        ),
        "pairwise_cosine_pair_count": int(
            pair_count
        ),
        "pairwise_cosine_seed": int(
            pair_seed
        ),
    }

    return summary, dimension_table
