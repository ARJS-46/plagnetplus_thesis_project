"""Exact cosine retrieval and neighbourhood diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def prepare_cosine_features(
    features: np.ndarray,
    *,
    centre: bool = False,
) -> np.ndarray:
    """Return finite row-wise L2-normalised float32 features."""
    array = np.asarray(features)

    if array.ndim != 2:
        raise ValueError(
            "features must be two-dimensional; "
            f"received {array.shape}."
        )

    if array.shape[0] < 2:
        raise ValueError(
            "At least two feature rows are required."
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

    values = array.astype(
        np.float64,
        copy=False,
    )

    if centre:
        values = (
            values
            - values.mean(
                axis=0,
                keepdims=True,
            )
        )

    norms = np.linalg.norm(
        values,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0.0):
        raise ValueError(
            "Cosine preparation produced zero-norm rows."
        )

    normalised = (
        values
        / norms
    )

    return normalised.astype(
        np.float32
    )


def exact_cosine_neighbours(
    normalised_features: np.ndarray,
    *,
    neighbour_count: int,
    exclusion_codes: np.ndarray | None = None,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate exact top-k cosine neighbours in chunks."""
    features = np.asarray(
        normalised_features,
        dtype=np.float32,
    )

    if features.ndim != 2:
        raise ValueError(
            "normalised_features must be two-dimensional."
        )

    row_count = features.shape[0]

    if not 1 <= neighbour_count < row_count:
        raise ValueError(
            "neighbour_count must lie between one and "
            "the number of rows minus one."
        )

    if chunk_size < 1:
        raise ValueError(
            "chunk_size must be positive."
        )

    codes: np.ndarray | None = None

    if exclusion_codes is not None:
        codes = np.asarray(
            exclusion_codes
        )

        if codes.ndim != 1:
            raise ValueError(
                "exclusion_codes must be one-dimensional."
            )

        if len(codes) != row_count:
            raise ValueError(
                "exclusion_codes length does not match features."
            )

        _, counts = np.unique(
            codes,
            return_counts=True,
        )

        maximum_group_size = int(
            counts.max()
        )

        if (
            row_count
            - maximum_group_size
            < neighbour_count
        ):
            raise ValueError(
                "At least one exclusion group leaves too few "
                "eligible neighbours."
            )

    reference = torch.from_numpy(
        np.ascontiguousarray(features)
    )

    all_indices = np.empty(
        (
            row_count,
            neighbour_count,
        ),
        dtype=np.int32,
    )

    all_similarities = np.empty(
        (
            row_count,
            neighbour_count,
        ),
        dtype=np.float32,
    )

    reference_codes = (
        torch.from_numpy(
            np.ascontiguousarray(codes)
        )
        if codes is not None
        else None
    )

    # Compute exact cosine neighbours in chunks to avoid materialising the full
    # N x N similarity matrix while preserving exact top-k retrieval.
    with torch.inference_mode():
        for start in range(
            0,
            row_count,
            chunk_size,
        ):
            stop = min(
                start + chunk_size,
                row_count,
            )

            query = reference[
                start:stop
            ]

            similarities = (
                query
                @ reference.T
            )

            local_rows = torch.arange(
                stop - start,
            )

            global_rows = torch.arange(
                start,
                stop,
            )

            # Exclude each query from retrieving itself before applying any optional
            # physical-sample exclusion.
            similarities[
                local_rows,
                global_rows,
            ] = -torch.inf

            if reference_codes is not None:
                query_codes = (
                    reference_codes[
                        start:stop
                    ]
                )

                exclusion_mask = (
                    query_codes[:, None]
                    == reference_codes[None, :]
                )

                similarities.masked_fill_(
                    exclusion_mask,
                    -torch.inf,
                )

            values, indices = torch.topk(
                similarities,
                k=neighbour_count,
                dim=1,
                largest=True,
                sorted=True,
            )

            if not torch.isfinite(values).all():
                raise RuntimeError(
                    "Neighbour search returned non-finite "
                    "selected similarities."
                )

            all_indices[
                start:stop
            ] = indices.numpy().astype(
                np.int32,
                copy=False,
            )

            all_similarities[
                start:stop
            ] = values.numpy().astype(
                np.float32,
                copy=False,
            )

    return (
        all_indices,
        all_similarities,
    )


def categorical_precision_at_k(
    neighbour_indices: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    query_mask: np.ndarray | None = None,
) -> float:
    """Measure mean same-label precision among the first k neighbours."""
    neighbours = np.asarray(
        neighbour_indices
    )

    values = np.asarray(labels)

    if neighbours.ndim != 2:
        raise ValueError(
            "neighbour_indices must be two-dimensional."
        )

    if values.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if len(values) != neighbours.shape[0]:
        raise ValueError(
            "labels length does not match neighbour rows."
        )

    if not 1 <= k <= neighbours.shape[1]:
        raise ValueError(
            "k exceeds the available neighbours."
        )

    selected_queries = np.ones(
        len(values),
        dtype=bool,
    )

    if query_mask is not None:
        selected_queries = np.asarray(
            query_mask,
            dtype=bool,
        )

        if selected_queries.shape != (
            len(values),
        ):
            raise ValueError(
                "query_mask has the wrong shape."
            )

    if not selected_queries.any():
        raise ValueError(
            "query_mask selects no rows."
        )

    selected_neighbours = neighbours[
        selected_queries,
        :k,
    ]

    query_labels = values[
        selected_queries
    ]

    matches = (
        values[selected_neighbours]
        == query_labels[:, None]
    )

    return float(
        matches.mean()
    )


def categorical_enrichment_at_k(
    neighbour_indices: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    query_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare observed same-label retrieval with random expectation."""
    neighbours = np.asarray(
        neighbour_indices
    )

    values = np.asarray(labels)

    if neighbours.ndim != 2:
        raise ValueError(
            "neighbour_indices must be two-dimensional."
        )

    if values.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if len(values) != neighbours.shape[0]:
        raise ValueError(
            "labels length does not match neighbour rows."
        )

    if not 1 <= k <= neighbours.shape[1]:
        raise ValueError(
            "k exceeds the available neighbours."
        )

    selected_queries = np.ones(
        len(values),
        dtype=bool,
    )

    if query_mask is not None:
        selected_queries = np.asarray(
            query_mask,
            dtype=bool,
        )

    if not selected_queries.any():
        raise ValueError(
            "query_mask selects no rows."
        )

    _, inverse, counts = np.unique(
        values,
        return_inverse=True,
        return_counts=True,
    )

    query_indices = np.flatnonzero(
        selected_queries
    )

    selected_neighbours = neighbours[
        query_indices,
        :k,
    ]

    query_labels = values[
        query_indices
    ]

    observed_matches = int(
        (
            values[selected_neighbours]
            == query_labels[:, None]
        ).sum()
    )

    # Compare observed same-category retrieval with the frequency expected by
    # chance, giving the enrichment statistic used for source-image diagnostics.
    expected_matches = float(
        np.sum(
            k
            * (
                counts[
                    inverse[query_indices]
                ]
                - 1
            )
            / float(
                len(values) - 1
            )
        )
    )

    total_slots = int(
        len(query_indices)
        * k
    )

    observed_precision = (
        observed_matches
        / total_slots
    )

    expected_precision = (
        expected_matches
        / total_slots
    )

    enrichment = (
        observed_matches
        / expected_matches
        if expected_matches > 0.0
        else float("nan")
    )

    return {
        "query_rows": int(
            len(query_indices)
        ),
        "k": int(k),
        "observed_matches": (
            observed_matches
        ),
        "expected_matches": float(
            expected_matches
        ),
        "observed_precision": float(
            observed_precision
        ),
        "expected_precision": float(
            expected_precision
        ),
        "enrichment": float(
            enrichment
        ),
    }
