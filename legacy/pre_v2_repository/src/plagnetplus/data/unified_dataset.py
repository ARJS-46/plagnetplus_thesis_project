"""Multi-root manifest-backed datasets for unified PlagNetPlus benchmarks."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image

from plagnetplus.data.dataset import locality_sampling_weights


VerifyMode = Literal["all", "sample", "none"]

_REQUIRED_COLUMNS = (
    "manifest_order",
    "crystal_id",
    "locality",
    "sample_id",
    "physical_sample_id",
    "acquisition_id",
    "scan_id",
    "split_group_id",
    "benchmark_id",
    "benchmark_split",
    "image_root_key",
    "image_relative_path",
    "image_sha256",
)

_DEFAULT_METADATA_COLUMNS = (
    "manifest_order",
    "crystal_id",
    "locality",
    "sample_id",
    "physical_sample_id",
    "acquisition_id",
    "scan_id",
    "split_group_id",
    "benchmark_id",
    "benchmark_split",
    "image_root_key",
    "image_relative_path",
    "image_locator",
    "image_sha256",
    "image_width_px",
    "image_height_px",
    "source_dataset",
)


def _read_unified_manifest(
    path: Path,
) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            low_memory=False,
        )

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        "Manifest must be CSV or Parquet."
    )


def _normalise_filter(
    values: str | Iterable[str] | None,
) -> set[str] | None:
    if values is None:
        return None

    if isinstance(values, str):
        return {values}

    return {str(value) for value in values}


def _safe_resolve(
    root: Path,
    relative_path: str,
) -> Path:
    relative = Path(
        str(relative_path).replace("\\", "/")
    )

    if relative.is_absolute():
        raise ValueError(
            f"Manifest image path must be relative: {relative_path!r}"
        )

    if ".." in relative.parts:
        raise ValueError(
            f"Parent traversal is not allowed: {relative_path!r}"
        )

    resolved = (root / relative).resolve()

    if not resolved.is_relative_to(root):
        raise ValueError(
            "Resolved image path escapes its configured root: "
            f"{relative_path!r}"
        )

    return resolved


def load_image_root_map(
    root_map: str | Path | Mapping[str, str],
    *,
    data_root: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve the image-root keys used by unified manifests.

    Resolution priority:

    1. Explicit ``data_root`` with the portable root templates.
    2. The environment variable named in the root-map JSON.
    3. ``current_resolved_roots`` recorded when the dataset was built.
    """

    if isinstance(root_map, Mapping):
        payload = dict(root_map)
        root_map_source: Path | None = None
    else:
        root_map_source = Path(
            root_map
        ).expanduser().resolve()

        with root_map_source.open(
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

    environment_variable = str(
        payload.get(
            "environment_variable",
            "PNP_DATA_ROOT",
        )
    )

    portable_roots = payload.get(
        "portable_roots",
        {},
    )

    current_roots = payload.get(
        "current_resolved_roots",
        {},
    )

    selected_data_root: Path | None

    if data_root is not None:
        selected_data_root = Path(
            data_root
        ).expanduser().resolve()
    else:
        environment_value = os.environ.get(
            environment_variable
        )

        selected_data_root = (
            Path(
                environment_value
            ).expanduser().resolve()
            if environment_value
            else None
        )

    resolved_roots: dict[str, Path] = {}

    if selected_data_root is not None:
        placeholder = (
            "${"
            + environment_variable
            + "}"
        )

        for key, template in portable_roots.items():
            expanded = str(template).replace(
                placeholder,
                str(selected_data_root),
            )

            resolved_roots[str(key)] = Path(
                expanded
            ).expanduser().resolve()

    else:
        resolved_roots = {
            str(key): Path(
                value
            ).expanduser().resolve()
            for key, value in current_roots.items()
        }

    if not resolved_roots:
        source_note = (
            str(root_map_source)
            if root_map_source is not None
            else "provided mapping"
        )

        raise ValueError(
            "No image roots could be resolved from "
            f"{source_note}."
        )

    missing_roots = {
        key: path
        for key, path in resolved_roots.items()
        if not path.is_dir()
    }

    if missing_roots:
        details = "\n".join(
            f"{key}: {path}"
            for key, path in sorted(
                missing_roots.items()
            )
        )

        raise NotADirectoryError(
            "Configured image roots do not exist:\n"
            + details
        )

    return resolved_roots


class UnifiedCrystalDataset(
    Sequence[dict[str, Any]]
):
    """Dataset for frozen multi-root PlagNetPlus benchmark manifests."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        image_root_map: str | Path | Mapping[str, str],
        *,
        data_root: str | Path | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        localities: str | Iterable[str] | None = None,
        samples: str | Iterable[str] | None = None,
        splits: str | Iterable[str] | None = None,
        sort_by: str | None = "manifest_order",
        verify_files: VerifyMode = "sample",
        verify_sample_size: int = 256,
        verify_seed: int = 42,
        include_metadata: bool = True,
        metadata_columns: Iterable[str] | None = None,
    ) -> None:
        if isinstance(manifest, pd.DataFrame):
            frame = manifest.copy()
            self.manifest_source: Path | None = None
        else:
            self.manifest_source = Path(
                manifest
            ).expanduser().resolve()

            frame = _read_unified_manifest(
                self.manifest_source
            )

        missing_columns = [
            column
            for column in _REQUIRED_COLUMNS
            if column not in frame.columns
        ]

        if missing_columns:
            raise ValueError(
                "Unified manifest is missing required columns: "
                f"{missing_columns}"
            )

        if frame["crystal_id"].duplicated().any():
            raise ValueError(
                "crystal_id values are not unique."
            )

        if "image_locator" in frame.columns:
            if frame["image_locator"].duplicated().any():
                raise ValueError(
                    "image_locator values are not unique."
                )

        frame = frame.copy()
        frame["_source_manifest_row"] = frame.index

        filters = {
            "locality": _normalise_filter(
                localities
            ),
            "sample_id": _normalise_filter(
                samples
            ),
            "benchmark_split": _normalise_filter(
                splits
            ),
        }

        for column, allowed in filters.items():
            if allowed is not None:
                frame = frame.loc[
                    frame[column]
                    .astype(str)
                    .isin(allowed)
                ].copy()

        if frame.empty:
            raise ValueError(
                "Dataset filters removed every manifest row."
            )

        if sort_by is not None:
            if sort_by not in frame.columns:
                raise ValueError(
                    f"sort_by column does not exist: {sort_by}"
                )

            frame = frame.sort_values(
                sort_by,
                kind="stable",
            )

        frame = frame.reset_index(
            drop=True
        )

        self.image_roots = load_image_root_map(
            image_root_map,
            data_root=data_root,
        )

        unknown_root_keys = sorted(
            set(
                frame["image_root_key"]
                .astype(str)
            )
            - set(self.image_roots)
        )

        if unknown_root_keys:
            raise ValueError(
                "Manifest contains unknown image_root_key values: "
                f"{unknown_root_keys}"
            )

        frame["_absolute_path"] = [
            _safe_resolve(
                self.image_roots[
                    str(root_key)
                ],
                str(relative_path),
            )
            for root_key, relative_path in zip(
                frame["image_root_key"],
                frame["image_relative_path"],
                strict=True,
            )
        ]

        self.transform = transform
        self.include_metadata = (
            include_metadata
        )

        requested_metadata = list(
            metadata_columns or []
        )

        default_metadata = [
            column
            for column
            in _DEFAULT_METADATA_COLUMNS
            if column in frame.columns
        ]

        unknown_metadata = (
            set(requested_metadata)
            - set(frame.columns)
        )

        if unknown_metadata:
            raise ValueError(
                "Unknown metadata columns: "
                f"{sorted(unknown_metadata)}"
            )

        self.metadata_columns = list(
            dict.fromkeys(
                [
                    *default_metadata,
                    *requested_metadata,
                ]
            )
        )

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
        if mode not in {
            "all",
            "sample",
            "none",
        }:
            raise ValueError(
                "verify_files must be one of: "
                "'all', 'sample', 'none'."
            )

        if mode == "none":
            return

        if mode == "all":
            indices = np.arange(
                len(self.manifest)
            )
        else:
            if sample_size <= 0:
                raise ValueError(
                    "verify_sample_size must be positive."
                )

            rng = np.random.default_rng(
                seed
            )

            count = min(
                sample_size,
                len(self.manifest),
            )

            indices = np.sort(
                rng.choice(
                    len(self.manifest),
                    size=count,
                    replace=False,
                )
            )

        missing = [
            self.manifest.iloc[
                int(index)
            ]["_absolute_path"]
            for index in indices
            if not self.manifest.iloc[
                int(index)
            ]["_absolute_path"].is_file()
        ]

        if missing:
            examples = "\n".join(
                str(path)
                for path in missing[:10]
            )

            raise FileNotFoundError(
                f"{len(missing)} checked image paths are missing. "
                f"Examples:\n{examples}"
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        path: Path = row[
            "_absolute_path"
        ]

        with Image.open(path) as image:
            image = image.convert("RGB")

            image_value = (
                self.transform(image)
                if self.transform
                else image.copy()
            )

        item: dict[str, Any] = {
            "image": image_value,
            "index": int(index),
        }

        if self.include_metadata:
            item["metadata"] = {
                column: row[column]
                for column
                in self.metadata_columns
            }

            item["metadata"][
                "absolute_path"
            ] = str(path)

            item["metadata"][
                "source_manifest_row"
            ] = int(
                row[
                    "_source_manifest_row"
                ]
            )

        return item

    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(
                self.manifest
            ),
            "manifest_source": (
                str(
                    self.manifest_source
                )
                if self.manifest_source
                is not None
                else None
            ),
            "image_roots": {
                key: str(path)
                for key, path
                in sorted(
                    self.image_roots.items()
                )
            },
            "root_key_counts": (
                self.manifest[
                    "image_root_key"
                ]
                .astype(str)
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "locality_counts": (
                self.manifest[
                    "locality"
                ]
                .astype(str)
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "split_counts": (
                self.manifest[
                    "benchmark_split"
                ]
                .astype(str)
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "split_group_count": int(
                self.manifest[
                    "split_group_id"
                ].nunique()
            ),
        }

    def group_labels(
        self,
        unit: Literal[
            "locality",
            "sample_id",
            "physical_sample_id",
            "scan_id",
            "split_group_id",
        ],
    ) -> np.ndarray:
        allowed = {
            "locality",
            "sample_id",
            "physical_sample_id",
            "scan_id",
            "split_group_id",
        }

        if unit not in allowed:
            raise ValueError(
                "Unsupported grouping unit: "
                f"{unit}"
            )

        return (
            self.manifest[unit]
            .astype(str)
            .to_numpy(copy=True)
        )

    def sampling_weights(
        self,
        *,
        power: float = 1.0,
    ) -> np.ndarray:
        return locality_sampling_weights(
            self.manifest,
            power=power,
        )
