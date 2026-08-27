"""Shared clustering separation and stability metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
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

    if array.shape[0] < 3:
        raise ValueError(
            "At least three rows are required."
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
            "features contain NaN or infinite values."
        )

    return array.astype(
        np.float32,
        copy=False,
    )


def normalised_cluster_entropy(
    labels: np.ndarray,
    *,
    number_of_clusters: int,
) -> float:
    """Measure cluster-size balance on a zero-to-one scale."""
    values = np.asarray(labels)

    if values.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if number_of_clusters < 2:
        raise ValueError(
            "number_of_clusters must be at least two."
        )

    if values.size < 1:
        raise ValueError(
            "labels must not be empty."
        )

    if values.min() < 0:
        raise ValueError(
            "Cluster labels must be non-negative."
        )

    counts = np.bincount(
        values.astype(
            np.int64,
            copy=False,
        ),
        minlength=number_of_clusters,
    )

    probabilities = (
        counts[counts > 0]
        / float(values.size)
    )

    entropy = -float(
        np.sum(
            probabilities
            * np.log(probabilities)
        )
    )

    maximum_entropy = float(
        np.log(number_of_clusters)
    )

    return entropy / maximum_entropy


def evaluate_kmeans_run(
    features: np.ndarray,
    *,
    number_of_clusters: int,
    seed: int,
    silhouette_indices: np.ndarray,
    maximum_iterations: int = 300,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one k-means run and calculate standard metrics."""
    array = _validated_features(features)

    if not 2 <= number_of_clusters < array.shape[0]:
        raise ValueError(
            "number_of_clusters must lie between two "
            "and the number of rows minus one."
        )

    indices = np.asarray(
        silhouette_indices,
        dtype=np.int64,
    )

    if indices.ndim != 1:
        raise ValueError(
            "silhouette_indices must be one-dimensional."
        )

    if indices.size < number_of_clusters + 1:
        raise ValueError(
            "The silhouette sample is too small for "
            "the requested cluster count."
        )

    if indices.min() < 0 or indices.max() >= array.shape[0]:
        raise ValueError(
            "silhouette_indices contain out-of-range rows."
        )

    if np.unique(indices).size != indices.size:
        raise ValueError(
            "silhouette_indices must be unique."
        )

    # Use a fixed seed and one explicit initialisation per run; stability is
    # measured by repeating this function with the predefined seed set.
    model = KMeans(
        n_clusters=number_of_clusters,
        init="k-means++",
        n_init=1,
        max_iter=maximum_iterations,
        tol=tolerance,
        random_state=seed,
        algorithm="lloyd",
    )

    labels = model.fit_predict(array).astype(
        np.int32,
        copy=False,
    )

    centres = np.asarray(
        model.cluster_centers_,
        dtype=np.float32,
    )

    counts = np.bincount(
        labels,
        minlength=number_of_clusters,
    )

    occupied_clusters = int(
        np.count_nonzero(counts)
    )

    sampled_labels = labels[indices]

    sampled_cluster_count = int(
        np.unique(sampled_labels).size
    )

    if sampled_cluster_count < 2:
        raise ValueError(
            "The silhouette sample contains fewer than "
            "two occupied clusters."
        )

    # Silhouette is calculated on the fixed sampled rows rather than all
    # validation crystals to keep the common comparison computationally manageable.
    sampled_features = array[indices]

    silhouette = float(
        silhouette_score(
            sampled_features,
            sampled_labels,
            metric="euclidean",
        )
    )

    metrics: dict[str, Any] = {
        "k": int(number_of_clusters),
        "seed": int(seed),
        "rows": int(array.shape[0]),
        "dimensions": int(array.shape[1]),
        "silhouette_sample_rows": int(
            indices.size
        ),
        "silhouette": silhouette,
        "davies_bouldin": float(
            davies_bouldin_score(
                array,
                labels,
            )
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(
                array,
                labels,
            )
        ),
        "inertia": float(
            model.inertia_
        ),
        "inertia_per_row": float(
            model.inertia_
            / array.shape[0]
        ),
        "iterations": int(
            model.n_iter_
        ),
        "occupied_clusters": (
            occupied_clusters
        ),
        "minimum_cluster_size": int(
            counts.min()
        ),
        "maximum_cluster_size": int(
            counts.max()
        ),
        "minimum_cluster_fraction": float(
            counts.min()
            / array.shape[0]
        ),
        "maximum_cluster_fraction": float(
            counts.max()
            / array.shape[0]
        ),
        "normalised_cluster_entropy": float(
            normalised_cluster_entropy(
                labels,
                number_of_clusters=(
                    number_of_clusters
                ),
            )
        ),
    }

    return labels, centres, metrics


def pairwise_clustering_stability(
    label_matrix: np.ndarray,
    *,
    seeds: list[int] | tuple[int, ...],
    number_of_clusters: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare every pair of clustering seeds using ARI and NMI."""
    matrix = np.asarray(label_matrix)

    if matrix.ndim != 2:
        raise ValueError(
            "label_matrix must have shape "
            "(runs, observations)."
        )

    if matrix.shape[0] != len(seeds):
        raise ValueError(
            "Number of label rows does not match seeds."
        )

    if matrix.shape[0] < 2:
        raise ValueError(
            "At least two clustering runs are required."
        )

    rows: list[dict[str, Any]] = []

    # Compare every pair of seeded clusterings so ARI/NMI summarise how
    # reproducible the partition is rather than relying on one k-means fit.
    for left_index, right_index in combinations(
        range(matrix.shape[0]),
        2,
    ):
        left_labels = matrix[left_index]
        right_labels = matrix[right_index]

        rows.append(
            {
                "k": int(number_of_clusters),
                "seed_left": int(
                    seeds[left_index]
                ),
                "seed_right": int(
                    seeds[right_index]
                ),
                "adjusted_rand_index": float(
                    adjusted_rand_score(
                        left_labels,
                        right_labels,
                    )
                ),
                "normalised_mutual_information": float(
                    normalized_mutual_info_score(
                        left_labels,
                        right_labels,
                    )
                ),
            }
        )

    table = pd.DataFrame(rows)

    ari = table[
        "adjusted_rand_index"
    ].to_numpy(
        dtype=np.float64
    )

    nmi = table[
        "normalised_mutual_information"
    ].to_numpy(
        dtype=np.float64
    )

    summary: dict[str, Any] = {
        "k": int(number_of_clusters),
        "seed_count": int(
            len(seeds)
        ),
        "pair_count": int(
            len(table)
        ),
        "mean_pairwise_ari": float(
            ari.mean()
        ),
        "median_pairwise_ari": float(
            np.median(ari)
        ),
        "minimum_pairwise_ari": float(
            ari.min()
        ),
        "maximum_pairwise_ari": float(
            ari.max()
        ),
        "mean_pairwise_nmi": float(
            nmi.mean()
        ),
        "median_pairwise_nmi": float(
            np.median(nmi)
        ),
        "minimum_pairwise_nmi": float(
            nmi.min()
        ),
        "maximum_pairwise_nmi": float(
            nmi.max()
        ),
    }

    return table, summary
