"""Configuration loading and validation for DeepCluster training."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from plagnetplus.models.backbones import (
    normalise_backbone_name,
)


@dataclass(frozen=True)
class DeepClusterTrainingConfig:
    """Validated configuration for one DeepCluster run."""

    run_name: str
    benchmark_id: str
    sampling: str

    backbone_name: str
    initialization: str

    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    image_root_map: Path
    transform_config: Path
    output_root: Path

    expected_train_rows: int
    expected_validation_rows: int
    expected_test_rows: int

    cycles: int
    cluster_count: int
    pca_components: int

    epochs_per_cycle: int = 1

    pca_whiten: bool = True
    pca_solver: str = "randomized"

    kmeans_algorithm: str = "lloyd"
    kmeans_initialization: str = "k-means++"
    kmeans_n_init: int = 1
    kmeans_maximum_iterations: int = 100
    kmeans_tolerance: float = 1e-4

    batch_size: int = 256
    evaluation_batch_size: int = 256
    num_workers: int = 8

    learning_rate: float = 0.03
    minimum_learning_rate: float = 0.0
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_cycles: int = 5
    gradient_clip_norm: float | None = 5.0

    samples_per_epoch: int | None = None
    inverse_cluster_frequency_sampling: bool = True
    classifier_reset_each_cycle: bool = True

    checkpoint_every: int = 5

    seed: int = 42
    device: str = "auto"
    use_amp: bool = True

    verify_files: str = "sample"
    verify_sample_size: int = 2048

    required_occupied_cluster_fraction: float = 1.0
    minimum_normalised_cluster_entropy: float = 0.75
    maximum_largest_cluster_fraction: float = 0.20
    minimum_cluster_size_warning: int = 10

    limit_train_items: int | None = None

    def __post_init__(self) -> None:
        canonical_backbone = (
            normalise_backbone_name(
                self.backbone_name
            )
        )

        object.__setattr__(
            self,
            "backbone_name",
            canonical_backbone,
        )

        if not self.run_name.strip():
            raise ValueError(
                "run_name cannot be empty."
            )

        if not self.benchmark_id.strip():
            raise ValueError(
                "benchmark_id cannot be empty."
            )

        if (
            self.sampling
            != "natural_frequency"
        ):
            raise ValueError(
                "sampling must be "
                "'natural_frequency'."
            )

        if self.initialization != "scratch":
            raise ValueError(
                "Only scratch initialisation is "
                "supported."
            )

        input_paths = {
            "train_manifest": (
                self.train_manifest
            ),
            "validation_manifest": (
                self.validation_manifest
            ),
            "test_manifest": (
                self.test_manifest
            ),
            "image_root_map": (
                self.image_root_map
            ),
            "transform_config": (
                self.transform_config
            ),
        }

        for name, path in input_paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"{name}: {path}"
                )

        for name, value in {
            "expected_train_rows": (
                self.expected_train_rows
            ),
            "expected_validation_rows": (
                self.expected_validation_rows
            ),
            "expected_test_rows": (
                self.expected_test_rows
            ),
        }.items():
            if value < 1:
                raise ValueError(
                    f"{name} must be positive."
                )

        if self.cycles < 1:
            raise ValueError(
                "cycles must be positive."
            )

        if self.epochs_per_cycle < 1:
            raise ValueError(
                "epochs_per_cycle must be positive."
            )

        available_train_rows = (
            self.expected_train_rows
            if self.limit_train_items is None
            else min(
                self.expected_train_rows,
                self.limit_train_items,
            )
        )

        if available_train_rows < 2:
            raise ValueError(
                "At least two training rows are "
                "required."
            )

        if not (
            2
            <= self.cluster_count
            <= available_train_rows
        ):
            raise ValueError(
                "cluster_count must lie between "
                "two and the available train rows."
            )

        feature_dimension = (
            512
            if self.backbone_name
            == "resnet18"
            else 2048
        )

        if not (
            1
            <= self.pca_components
            <= feature_dimension
        ):
            raise ValueError(
                "pca_components exceeds the "
                "backbone feature dimension."
            )

        if (
            self.pca_components
            >= available_train_rows
        ):
            raise ValueError(
                "pca_components must be smaller "
                "than the available train rows."
            )

        if self.pca_solver not in {
            "randomized",
            "full",
        }:
            raise ValueError(
                "Unsupported pca_solver."
            )

        if self.kmeans_algorithm != "lloyd":
            raise ValueError(
                "kmeans_algorithm must be 'lloyd'."
            )

        if (
            self.kmeans_initialization
            != "k-means++"
        ):
            raise ValueError(
                "kmeans_initialization must be "
                "'k-means++'."
            )

        if self.kmeans_n_init < 1:
            raise ValueError(
                "kmeans_n_init must be positive."
            )

        if (
            self.kmeans_maximum_iterations
            < 1
        ):
            raise ValueError(
                "kmeans_maximum_iterations must "
                "be positive."
            )

        if self.kmeans_tolerance < 0.0:
            raise ValueError(
                "kmeans_tolerance cannot be "
                "negative."
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.evaluation_batch_size < 1:
            raise ValueError(
                "evaluation_batch_size must be "
                "positive."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if not (
            0.0
            <= self.minimum_learning_rate
            <= self.learning_rate
        ):
            raise ValueError(
                "minimum_learning_rate must lie "
                "between zero and learning_rate."
            )

        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(
                "momentum must lie in [0, 1)."
            )

        if self.weight_decay < 0.0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if not (
            0
            <= self.warmup_cycles
            <= self.cycles
        ):
            raise ValueError(
                "warmup_cycles must lie between "
                "zero and cycles."
            )

        if (
            self.gradient_clip_norm
            is not None
            and self.gradient_clip_norm <= 0.0
        ):
            raise ValueError(
                "gradient_clip_norm must be "
                "positive when provided."
            )

        if (
            self.samples_per_epoch is not None
            and self.samples_per_epoch < 1
        ):
            raise ValueError(
                "samples_per_epoch must be "
                "positive when provided."
            )

        if not self.classifier_reset_each_cycle:
            raise ValueError(
                "The frozen design requires the "
                "classifier to reset each cycle."
            )

        if self.checkpoint_every < 1:
            raise ValueError(
                "checkpoint_every must be positive."
            )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative."
            )

        if self.device not in {
            "auto",
            "cpu",
            "cuda",
        }:
            raise ValueError(
                "device must be auto, cpu or cuda."
            )

        if self.verify_files not in {
            "none",
            "sample",
            "all",
        }:
            raise ValueError(
                "verify_files must be none, "
                "sample or all."
            )

        if self.verify_sample_size < 1:
            raise ValueError(
                "verify_sample_size must be "
                "positive."
            )

        if not (
            0.0
            < self.required_occupied_cluster_fraction
            <= 1.0
        ):
            raise ValueError(
                "Invalid occupied-cluster "
                "fraction."
            )

        if not (
            0.0
            <= self.minimum_normalised_cluster_entropy
            <= 1.0
        ):
            raise ValueError(
                "Invalid cluster-entropy "
                "threshold."
            )

        if not (
            0.0
            < self.maximum_largest_cluster_fraction
            <= 1.0
        ):
            raise ValueError(
                "Invalid largest-cluster "
                "threshold."
            )

        if self.minimum_cluster_size_warning < 1:
            raise ValueError(
                "minimum_cluster_size_warning "
                "must be positive."
            )

        if (
            self.limit_train_items is not None
            and self.limit_train_items < 1
        ):
            raise ValueError(
                "limit_train_items must be "
                "positive when provided."
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        payload = asdict(self)

        for key in [
            "train_manifest",
            "validation_manifest",
            "test_manifest",
            "image_root_map",
            "transform_config",
            "output_root",
        ]:
            payload[key] = str(payload[key])

        return payload


def _expand_path(
    value: object,
) -> Path:
    expanded = os.path.expandvars(
        os.path.expanduser(
            str(value)
        )
    )

    return Path(expanded)


def _optional_integer(
    value: object,
) -> int | None:
    if value is None:
        return None

    return int(value)


def _optional_float(
    value: object,
) -> float | None:
    if value is None:
        return None

    return float(value)


def config_from_mapping(
    mapping: Mapping[str, Any],
) -> DeepClusterTrainingConfig:
    """Build a validated configuration from a flat mapping."""
    required = {
        "run_name",
        "benchmark_id",
        "sampling",
        "backbone_name",
        "initialization",
        "train_manifest",
        "validation_manifest",
        "test_manifest",
        "image_root_map",
        "transform_config",
        "output_root",
        "expected_train_rows",
        "expected_validation_rows",
        "expected_test_rows",
        "cycles",
        "cluster_count",
        "pca_components",
    }

    missing = required - set(mapping)

    if missing:
        raise KeyError(
            "Missing DeepCluster configuration "
            f"keys: {sorted(missing)}"
        )

    return DeepClusterTrainingConfig(
        run_name=str(
            mapping["run_name"]
        ),
        benchmark_id=str(
            mapping["benchmark_id"]
        ),
        sampling=str(
            mapping["sampling"]
        ),
        backbone_name=str(
            mapping["backbone_name"]
        ),
        initialization=str(
            mapping["initialization"]
        ),
        train_manifest=_expand_path(
            mapping["train_manifest"]
        ),
        validation_manifest=_expand_path(
            mapping["validation_manifest"]
        ),
        test_manifest=_expand_path(
            mapping["test_manifest"]
        ),
        image_root_map=_expand_path(
            mapping["image_root_map"]
        ),
        transform_config=_expand_path(
            mapping["transform_config"]
        ),
        output_root=_expand_path(
            mapping["output_root"]
        ),
        expected_train_rows=int(
            mapping["expected_train_rows"]
        ),
        expected_validation_rows=int(
            mapping[
                "expected_validation_rows"
            ]
        ),
        expected_test_rows=int(
            mapping["expected_test_rows"]
        ),
        cycles=int(mapping["cycles"]),
        cluster_count=int(
            mapping["cluster_count"]
        ),
        pca_components=int(
            mapping["pca_components"]
        ),
        epochs_per_cycle=int(
            mapping.get(
                "epochs_per_cycle",
                1,
            )
        ),
        pca_whiten=bool(
            mapping.get(
                "pca_whiten",
                True,
            )
        ),
        pca_solver=str(
            mapping.get(
                "pca_solver",
                "randomized",
            )
        ),
        kmeans_algorithm=str(
            mapping.get(
                "kmeans_algorithm",
                "lloyd",
            )
        ),
        kmeans_initialization=str(
            mapping.get(
                "kmeans_initialization",
                "k-means++",
            )
        ),
        kmeans_n_init=int(
            mapping.get(
                "kmeans_n_init",
                1,
            )
        ),
        kmeans_maximum_iterations=int(
            mapping.get(
                "kmeans_maximum_iterations",
                100,
            )
        ),
        kmeans_tolerance=float(
            mapping.get(
                "kmeans_tolerance",
                1e-4,
            )
        ),
        batch_size=int(
            mapping.get(
                "batch_size",
                256,
            )
        ),
        evaluation_batch_size=int(
            mapping.get(
                "evaluation_batch_size",
                256,
            )
        ),
        num_workers=int(
            mapping.get(
                "num_workers",
                8,
            )
        ),
        learning_rate=float(
            mapping.get(
                "learning_rate",
                0.03,
            )
        ),
        minimum_learning_rate=float(
            mapping.get(
                "minimum_learning_rate",
                0.0,
            )
        ),
        momentum=float(
            mapping.get(
                "momentum",
                0.9,
            )
        ),
        weight_decay=float(
            mapping.get(
                "weight_decay",
                1e-4,
            )
        ),
        warmup_cycles=int(
            mapping.get(
                "warmup_cycles",
                5,
            )
        ),
        gradient_clip_norm=(
            _optional_float(
                mapping.get(
                    "gradient_clip_norm",
                    5.0,
                )
            )
        ),
        samples_per_epoch=(
            _optional_integer(
                mapping.get(
                    "samples_per_epoch"
                )
            )
        ),
        inverse_cluster_frequency_sampling=bool(
            mapping.get(
                "inverse_cluster_frequency_sampling",
                True,
            )
        ),
        classifier_reset_each_cycle=bool(
            mapping.get(
                "classifier_reset_each_cycle",
                True,
            )
        ),
        checkpoint_every=int(
            mapping.get(
                "checkpoint_every",
                5,
            )
        ),
        seed=int(
            mapping.get(
                "seed",
                42,
            )
        ),
        device=str(
            mapping.get(
                "device",
                "auto",
            )
        ),
        use_amp=bool(
            mapping.get(
                "use_amp",
                True,
            )
        ),
        verify_files=str(
            mapping.get(
                "verify_files",
                "sample",
            )
        ),
        verify_sample_size=int(
            mapping.get(
                "verify_sample_size",
                2048,
            )
        ),
        required_occupied_cluster_fraction=float(
            mapping.get(
                "required_occupied_cluster_fraction",
                1.0,
            )
        ),
        minimum_normalised_cluster_entropy=float(
            mapping.get(
                "minimum_normalised_cluster_entropy",
                0.75,
            )
        ),
        maximum_largest_cluster_fraction=float(
            mapping.get(
                "maximum_largest_cluster_fraction",
                0.20,
            )
        ),
        minimum_cluster_size_warning=int(
            mapping.get(
                "minimum_cluster_size_warning",
                10,
            )
        ),
        limit_train_items=(
            _optional_integer(
                mapping.get(
                    "limit_train_items"
                )
            )
        ),
    )


def load_deepcluster_training_config(
    path: str | Path,
) -> DeepClusterTrainingConfig:
    """Read JSON or YAML and return a validated configuration."""
    selected = Path(path)

    if not selected.is_file():
        raise FileNotFoundError(selected)

    suffix = selected.suffix.lower()

    with selected.open(
        "r",
        encoding="utf-8",
    ) as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {
            ".yaml",
            ".yml",
        }:
            payload = yaml.safe_load(
                handle
            )
        else:
            raise ValueError(
                "Configuration must be JSON "
                "or YAML."
            )

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            "Configuration root must be a mapping."
        )

    return config_from_mapping(
        payload
    )
