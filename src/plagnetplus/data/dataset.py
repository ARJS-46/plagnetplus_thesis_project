"""Manifest-backed crystal-image datasets.

The dataset is locality-agnostic. Aegina, Methana, FCT, Romania, or any future
source can be used provided every row follows the common manifest contract and
``relative_path`` is relative to the configured image root.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image

from plagnetplus.data.manifest import read_manifest, validate_manifest

_REQUIRED_METADATA = (
    "crystal_id",
    "locality",
    "sample_id",
    "scan_id",
    "relative_path",
    "split",
)

VerifyMode = Literal["all", "sample", "none"]


def _normalise_filter(values: str | Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def _safe_resolve(root: Path, relative_path: str) -> Path:
    relative = Path(str(relative_path).replace("\\", "/"))

    if relative.is_absolute():
        raise ValueError(f"Manifest path must be relative: {relative_path!r}")
    if ".." in relative.parts:
        raise ValueError(f"Parent traversal is not allowed: {relative_path!r}")

    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"Resolved image path escapes the configured root: {relative_path!r}"
        )
    return resolved


def locality_sampling_weights(
    manifest: pd.DataFrame,
    *,
    locality_column: str = "locality",
    power: float = 1.0,
    normalise_mean: bool = True,
) -> np.ndarray:
    """Return inverse-frequency per-row locality weights.

    ``power=1`` gives each locality equal total weight. ``power=0`` gives
    uniform crystal sampling. Intermediate values partially rebalance sources.
    """
    if locality_column not in manifest.columns:
        raise ValueError(f"Missing locality column: {locality_column}")
    if power < 0:
        raise ValueError("power must be non-negative.")
    if manifest.empty:
        raise ValueError("Cannot compute weights for an empty manifest.")

    labels = manifest[locality_column].astype(str)
    counts = labels.value_counts()

    weights = labels.map(
        lambda label: 1.0 / (float(counts[label]) ** power)
    ).to_numpy(dtype=np.float64, copy=True)

    if normalise_mean:
        weights /= weights.mean()

    return weights


class CrystalDataset(Sequence[dict[str, Any]]):
    """Map-style dataset backed by the common PlagNetPlus manifest.

    The class deliberately does not subclass a PyTorch class. PyTorch
    ``DataLoader`` accepts any object implementing ``__len__`` and
    ``__getitem__``, while this design also allows lightweight data inspection
    without installing PyTorch.

    Parameters
    ----------
    manifest:
        CSV/Parquet path or an already-loaded DataFrame.
    image_root:
        Directory against which ``relative_path`` values are resolved.
    transform:
        Optional callable applied to a fresh RGB PIL image. For SSL this may
        return one tensor, two augmented views, or a model-specific structure.
    localities, samples, splits:
        Optional manifest filters.
    sort_by:
        Stable deterministic ordering column. Set to ``None`` to preserve the
        incoming order.
    verify_files:
        ``"all"`` checks every image, ``"sample"`` checks a reproducible
        subset, and ``"none"`` skips existence checks.
    verify_sample_size:
        Number of rows checked by ``verify_files="sample"``.
    verify_seed:
        Reproducible file-check sampling seed.
    include_metadata:
        Include manifest identifiers and selected extra fields in each item.
    metadata_columns:
        Additional columns to include alongside the required identifiers.
    """
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        image_root: str | Path,
        *,
        transform: Callable[[Image.Image], Any] | None = None,
        localities: str | Iterable[str] | None = None,
        samples: str | Iterable[str] | None = None,
        splits: str | Iterable[str] | None = None,
        sort_by: str | None = "relative_path",
        verify_files: VerifyMode = "all",
        verify_sample_size: int = 256,
        verify_seed: int = 42,
        include_metadata: bool = True,
        metadata_columns: Iterable[str] | None = None,
    ) -> None:
        if isinstance(manifest, pd.DataFrame):
            frame = manifest.copy()
            self.manifest_source: Path | None = None
        else:
            self.manifest_source = Path(manifest).expanduser().resolve()
            frame = read_manifest(self.manifest_source)

        errors = validate_manifest(frame)
        if errors:
            raise ValueError("Manifest validation failed: " + "; ".join(errors))

        for column in _REQUIRED_METADATA:
            if column not in frame.columns:
                raise ValueError(f"Required manifest column is missing: {column}")

        frame = frame.copy()
        frame["_source_manifest_row"] = frame.index

        filters = {
            "locality": _normalise_filter(localities),
            "sample_id": _normalise_filter(samples),
            "split": _normalise_filter(splits),
        }
        for column, allowed in filters.items():
            if allowed is not None:
                frame = frame.loc[
                    frame[column].astype(str).isin(allowed)
                ].copy()

        if frame.empty:
            raise ValueError("Dataset filters removed every manifest row.")

        if sort_by is not None:
            if sort_by not in frame.columns:
                raise ValueError(f"sort_by column does not exist: {sort_by}")
            frame = frame.sort_values(sort_by, kind="stable")

        frame = frame.reset_index(drop=True)

        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.is_dir():
            raise NotADirectoryError(self.image_root)

        frame["_absolute_path"] = frame["relative_path"].map(
            lambda value: _safe_resolve(self.image_root, str(value))
        )

        self.transform = transform
        self.include_metadata = include_metadata

        requested_metadata = list(metadata_columns or [])
        unknown = set(requested_metadata) - set(frame.columns)
        if unknown:
            raise ValueError(
                f"Unknown metadata columns: {sorted(unknown)}"
            )

        self.metadata_columns = list(dict.fromkeys(
            [*_REQUIRED_METADATA, *requested_metadata]
        ))
        self.manifest = frame

        self._verify_files(
            mode=verify_files,
            sample_size=verify_sample_size,
            seed=verify_seed,
        )

    def _verify_files(
        self,
        *,
        mode: VerifyMode,
        sample_size: int,
        seed: int,
    ) -> None:
        if mode not in {"all", "sample", "none"}:
            raise ValueError(
                "verify_files must be one of: 'all', 'sample', 'none'."
            )
        if mode == "none":
            return

        if mode == "all":
            indices = np.arange(len(self.manifest))
        else:
            if sample_size <= 0:
                raise ValueError("verify_sample_size must be positive.")
            rng = np.random.default_rng(seed)
            count = min(sample_size, len(self.manifest))
            indices = np.sort(
                rng.choice(len(self.manifest), size=count, replace=False)
            )

        missing = [
            self.manifest.iloc[int(index)]["_absolute_path"]
            for index in indices
            if not self.manifest.iloc[int(index)]["_absolute_path"].is_file()
        ]
        if missing:
            examples = "\n".join(str(path) for path in missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} checked image paths are missing. "
                f"Examples:\n{examples}"
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        path: Path = row["_absolute_path"]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image_value = self.transform(image) if self.transform else image.copy()

        item: dict[str, Any] = {
            "image": image_value,
            "index": int(index),
        }

        if self.include_metadata:
            item["metadata"] = {
                column: row[column]
                for column in self.metadata_columns
            }
            item["metadata"]["absolute_path"] = str(path)
            item["metadata"]["source_manifest_row"] = int(
                row["_source_manifest_row"]
            )

        return item

    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(self.manifest),
            "image_root": str(self.image_root),
            "manifest_source": (
                str(self.manifest_source)
                if self.manifest_source is not None
                else None
            ),
            "locality_counts": (
                self.manifest["locality"]
                .astype(str)
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "sample_counts": (
                self.manifest["sample_id"]
                .astype(str)
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "scan_groups": int(self.manifest["scan_id"].nunique()),
            "split_counts": (
                self.manifest["split"]
                .astype(str)
                .value_counts()
                .sort_index()
                .to_dict()
            ),
        }

    def group_labels(self, unit: Literal["locality", "sample_id", "scan_id"]) -> np.ndarray:
        """Return group labels for leakage-safe splitting or evaluation."""
        if unit not in {"locality", "sample_id", "scan_id"}:
            raise ValueError(
                "Grouping unit must be locality, sample_id, or scan_id."
            )
        return self.manifest[unit].astype(str).to_numpy(copy=True)

    def sampling_weights(
        self,
        *,
        power: float = 1.0,
    ) -> np.ndarray:
        """Return inverse-frequency locality weights aligned to dataset rows."""
        return locality_sampling_weights(
            self.manifest,
            power=power,
        )
