"""Loading and validating exported representation embeddings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingBundle:
    """Backbone embeddings and their traceability identifiers."""

    features: np.ndarray
    crystal_ids: np.ndarray
    manifest_orders: np.ndarray
    source_path: Path

    @property
    def row_count(self) -> int:
        """Return the number of embedded crystals."""
        return int(self.features.shape[0])

    @property
    def feature_dimension(self) -> int:
        """Return the dimensionality of each embedding."""
        return int(self.features.shape[1])


def sha256_file(path: str | Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    resolved = Path(path)

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    digest = hashlib.sha256()

    with resolved.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_embedding_bundle(
    path: str | Path,
    *,
    expected_feature_dimension: int | None = None,
) -> EmbeddingBundle:
    """Load the standard PlagNetPlus NPZ embedding export."""
    resolved = Path(path)

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    # Embedding archives carry identifiers as well as features so numerical
    # arrays can be checked against the exact manifest rows that produced them.
    required_keys = {
        "features",
        "crystal_ids",
        "manifest_orders",
    }

    with np.load(
        resolved,
        allow_pickle=False,
    ) as payload:
        missing_keys = required_keys - set(payload.files)

        if missing_keys:
            raise ValueError(
                "Embedding archive is missing keys: "
                f"{sorted(missing_keys)}"
            )

        features = np.asarray(
            payload["features"]
        ).copy()

        crystal_ids = np.asarray(
            payload["crystal_ids"]
        ).copy()

        manifest_orders = np.asarray(
            payload["manifest_orders"]
        ).copy()

    if features.ndim != 2:
        raise ValueError(
            "features must be a two-dimensional matrix; "
            f"received shape {features.shape}."
        )

    if crystal_ids.ndim != 1:
        raise ValueError(
            "crystal_ids must be one-dimensional; "
            f"received shape {crystal_ids.shape}."
        )

    if manifest_orders.ndim != 1:
        raise ValueError(
            "manifest_orders must be one-dimensional; "
            f"received shape {manifest_orders.shape}."
        )

    row_count = features.shape[0]

    if crystal_ids.shape[0] != row_count:
        raise ValueError(
            "crystal_ids length does not match features: "
            f"{crystal_ids.shape[0]} != {row_count}."
        )

    if manifest_orders.shape[0] != row_count:
        raise ValueError(
            "manifest_orders length does not match features: "
            f"{manifest_orders.shape[0]} != {row_count}."
        )

    if not np.issubdtype(
        features.dtype,
        np.number,
    ):
        raise TypeError(
            "features must have a numeric dtype; "
            f"received {features.dtype}."
        )

    if expected_feature_dimension is not None:
        if features.shape[1] != expected_feature_dimension:
            raise ValueError(
                "Unexpected feature dimension: "
                f"{features.shape[1]} != "
                f"{expected_feature_dimension}."
            )

    try:
        manifest_orders = manifest_orders.astype(
            np.int64,
            copy=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "manifest_orders could not be converted "
            "to integer values."
        ) from exc

    crystal_ids = crystal_ids.astype(
        str,
        copy=False,
    )

    features.setflags(write=False)
    crystal_ids.setflags(write=False)
    manifest_orders.setflags(write=False)

    return EmbeddingBundle(
        features=features,
        crystal_ids=crystal_ids,
        manifest_orders=manifest_orders,
        source_path=resolved,
    )


def load_embedding_manifest(
    path: str | Path,
    *,
    id_column: str = "crystal_id",
    order_column: str = "manifest_order",
) -> pd.DataFrame:
    """Load a manifest required for embedding alignment."""
    resolved = Path(path)

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    manifest = pd.read_csv(
        resolved,
        low_memory=False,
    )

    required_columns = {
        id_column,
        order_column,
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return manifest


def validate_embedding_manifest_alignment(
    bundle: EmbeddingBundle,
    manifest: pd.DataFrame,
    *,
    id_column: str = "crystal_id",
    order_column: str = "manifest_order",
) -> dict[str, Any]:
    """Require exact row-by-row alignment with a manifest."""
    if len(manifest) != bundle.row_count:
        raise ValueError(
            "Manifest and embedding row counts differ: "
            f"{len(manifest)} != {bundle.row_count}."
        )

    # Compare both crystal_id and manifest_order; matching only one key could
    # hide a reordered or partially mismatched embedding export.
    manifest_ids = (
        manifest[id_column]
        .astype(str)
        .to_numpy()
    )

    manifest_orders = pd.to_numeric(
        manifest[order_column],
        errors="raise",
    ).to_numpy(
        dtype=np.int64
    )

    embedding_id_duplicates = int(
        pd.Series(bundle.crystal_ids)
        .duplicated()
        .sum()
    )

    embedding_order_duplicates = int(
        pd.Series(bundle.manifest_orders)
        .duplicated()
        .sum()
    )

    manifest_id_duplicates = int(
        manifest[id_column]
        .astype(str)
        .duplicated()
        .sum()
    )

    manifest_order_duplicates = int(
        manifest[order_column]
        .duplicated()
        .sum()
    )

    if embedding_id_duplicates:
        raise ValueError(
            "Embedding crystal_ids contain "
            f"{embedding_id_duplicates} duplicates."
        )

    if embedding_order_duplicates:
        raise ValueError(
            "Embedding manifest_orders contain "
            f"{embedding_order_duplicates} duplicates."
        )

    if manifest_id_duplicates:
        raise ValueError(
            "Manifest crystal IDs contain "
            f"{manifest_id_duplicates} duplicates."
        )

    if manifest_order_duplicates:
        raise ValueError(
            "Manifest orders contain "
            f"{manifest_order_duplicates} duplicates."
        )

    ids_aligned = np.array_equal(
        bundle.crystal_ids,
        manifest_ids,
    )

    orders_aligned = np.array_equal(
        bundle.manifest_orders,
        manifest_orders,
    )

    if not ids_aligned:
        mismatch_indices = np.flatnonzero(
            bundle.crystal_ids
            != manifest_ids
        )

        first_index = int(
            mismatch_indices[0]
        )

        raise ValueError(
            "Embedding crystal IDs are not aligned. "
            f"First mismatch at row {first_index}: "
            f"{bundle.crystal_ids[first_index]!r} != "
            f"{manifest_ids[first_index]!r}."
        )

    if not orders_aligned:
        mismatch_indices = np.flatnonzero(
            bundle.manifest_orders
            != manifest_orders
        )

        first_index = int(
            mismatch_indices[0]
        )

        raise ValueError(
            "Embedding manifest orders are not aligned. "
            f"First mismatch at row {first_index}: "
            f"{bundle.manifest_orders[first_index]} != "
            f"{manifest_orders[first_index]}."
        )

    return {
        "row_count": bundle.row_count,
        "feature_dimension": (
            bundle.feature_dimension
        ),
        "ids_aligned": ids_aligned,
        "manifest_orders_aligned": (
            orders_aligned
        ),
        "embedding_id_duplicates": (
            embedding_id_duplicates
        ),
        "embedding_order_duplicates": (
            embedding_order_duplicates
        ),
        "manifest_id_duplicates": (
            manifest_id_duplicates
        ),
        "manifest_order_duplicates": (
            manifest_order_duplicates
        ),
    }
