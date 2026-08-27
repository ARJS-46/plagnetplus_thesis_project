"""Deterministic PCA and dimensionality summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PCAResult:
    """Deterministic PCA outputs for reusable analysis."""

    coordinates: np.ndarray
    components: np.ndarray
    mean: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    input_dimension: int
    retained_components: int


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
            "At least two rows are required for PCA."
        )

    if array.shape[1] < 1:
        raise ValueError(
            "At least one feature dimension is required."
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
            "features contain non-finite values."
        )

    return array.astype(
        np.float64,
        copy=False,
    )


def _canonicalise_component_signs(
    components: np.ndarray,
) -> np.ndarray:
    # Eigenvector signs are mathematically arbitrary, so fix them deterministically
    # to make saved PCA coordinates reproducible between runs.
    canonical = components.copy()

    for index in range(
        canonical.shape[0]
    ):
        component = canonical[index]

        anchor = int(
            np.argmax(
                np.abs(component)
            )
        )

        if component[anchor] < 0.0:
            canonical[index] *= -1.0

    return canonical


def fit_deterministic_pca(
    features: np.ndarray,
    *,
    retained_components: int = 50,
) -> PCAResult:
    """Fit covariance PCA with deterministic component signs."""
    array = _validated_features(
        features
    )

    maximum_components = min(
        array.shape[0] - 1,
        array.shape[1],
    )

    if retained_components < 1:
        raise ValueError(
            "retained_components must be positive."
        )

    if retained_components > maximum_components:
        raise ValueError(
            "retained_components exceeds the available "
            f"PCA rank: {retained_components} > "
            f"{maximum_components}."
        )

    mean = array.mean(
        axis=0
    )

    centred = (
        array
        - mean
    )

    # Fit PCA explicitly from the covariance matrix so ordering and sign handling
    # are controlled rather than delegated to version-dependent defaults.
    covariance = (
        centred.T
        @ centred
    ) / float(
        array.shape[0] - 1
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    descending = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = np.clip(
        eigenvalues[descending],
        a_min=0.0,
        a_max=None,
    )

    eigenvectors = eigenvectors[
        :,
        descending,
    ]

    total_variance = float(
        eigenvalues.sum()
    )

    if total_variance <= 0.0:
        raise ValueError(
            "PCA cannot be fitted because total variance "
            "is zero."
        )

    explained_variance_ratio = (
        eigenvalues
        / total_variance
    )

    cumulative = np.cumsum(
        explained_variance_ratio
    )

    components = (
        eigenvectors[
            :,
            :retained_components,
        ]
        .T
    )

    components = (
        _canonicalise_component_signs(
            components
        )
    )

    # Project onto only the requested leading components after the deterministic
    # eigenvalue ordering and sign convention have been applied.
    coordinates = (
        centred
        @ components.T
    )

    return PCAResult(
        coordinates=coordinates.astype(
            np.float32
        ),
        components=components.astype(
            np.float32
        ),
        mean=mean.astype(
            np.float32
        ),
        eigenvalues=eigenvalues.astype(
            np.float64
        ),
        explained_variance_ratio=(
            explained_variance_ratio.astype(
                np.float64
            )
        ),
        cumulative_explained_variance=(
            cumulative.astype(
                np.float64
            )
        ),
        input_dimension=int(
            array.shape[1]
        ),
        retained_components=int(
            retained_components
        ),
    )


def components_for_variance_threshold(
    cumulative_explained_variance: np.ndarray,
    threshold: float,
) -> int:
    """Return the component count reaching a variance threshold."""
    cumulative = np.asarray(
        cumulative_explained_variance,
        dtype=np.float64,
    )

    if cumulative.ndim != 1:
        raise ValueError(
            "cumulative_explained_variance must be "
            "one-dimensional."
        )

    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "threshold must lie in (0, 1]."
        )

    index = int(
        np.searchsorted(
            cumulative,
            threshold,
            side="left",
        )
    )

    return min(
        index + 1,
        len(cumulative),
    )


def l2_normalise_rows(
    values: np.ndarray,
) -> np.ndarray:
    """L2-normalise rows while safely preserving zero rows."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            "values must be two-dimensional."
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

    return normalised.astype(
        np.float32
    )


def summarise_pca(
    result: PCAResult,
) -> dict[str, Any]:
    """Create the standard PCA summary."""
    thresholds = [
        0.50,
        0.80,
        0.90,
        0.95,
        0.99,
    ]

    component_counts = {
        f"components_for_{int(threshold * 100)}_percent_variance": (
            components_for_variance_threshold(
                result.cumulative_explained_variance,
                threshold,
            )
        )
        for threshold in thresholds
    }

    retained_variance = float(
        result.cumulative_explained_variance[
            result.retained_components - 1
        ]
    )

    return {
        "input_dimension": (
            result.input_dimension
        ),
        "retained_components": (
            result.retained_components
        ),
        "variance_retained_by_selected_components": (
            retained_variance
        ),
        "pc1_explained_variance_ratio": float(
            result.explained_variance_ratio[0]
        ),
        "pc2_explained_variance_ratio": float(
            result.explained_variance_ratio[1]
        ),
        "pc3_explained_variance_ratio": float(
            result.explained_variance_ratio[2]
        ),
        **component_counts,
    }
