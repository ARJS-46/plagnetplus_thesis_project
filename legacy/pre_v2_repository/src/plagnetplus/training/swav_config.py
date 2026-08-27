"""Validated configuration for native PlagNetPlus SwAV training."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from plagnetplus.models.backbones import (
    normalise_backbone_name,
)


@dataclass(frozen=True)
class SwAVTrainingConfig:
    """Frozen two-global-crop SwAV baseline configuration."""

    run_name: str
    output_root: Path

    method: str
    backbone_name: str
    initialization: str
    projection_hidden_dimension: int
    projection_dimension: int
    prototype_count: int
    temperature: float
    sinkhorn_epsilon: float
    sinkhorn_iterations: int
    global_crops: int
    local_crops: int
    queue_length: int

    benchmark_id: str
    sampling: str
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    image_root_map: Path
    transform_config: Path
    expected_manifest_fingerprint_sha256: str
    expected_train_rows: int
    expected_validation_rows: int
    expected_test_rows: int
    limit_train_items: int | None
    limit_validation_items: int | None
    verify_files: str
    verify_sample_size: int

    epochs: int
    batch_size: int
    evaluation_batch_size: int
    optimizer_name: str
    learning_rate: float
    minimum_learning_rate: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    gradient_clip_norm: float | None
    prototype_freeze_epochs: int

    num_workers: int
    checkpoint_every: int
    validation_every: int
    seed: int
    use_amp: bool
    export_train_embeddings: bool
    export_validation_embeddings: bool

    track_assignment_entropy: bool
    track_prototype_occupancy: bool
    track_assignment_agreement: bool
    track_backbone_feature_norm: bool
    track_projection_norm: bool
    track_prototype_weight_norm: bool
    require_finite_loss: bool
    require_finite_gradients: bool

    def __post_init__(
        self,
    ) -> None:
        if not self.run_name.strip():
            raise ValueError(
                "run_name cannot be empty."
            )

        if self.method.strip().lower() != "swav":
            raise ValueError(
                "method must be 'swav'."
            )

        normalise_backbone_name(
            self.backbone_name
        )

        if self.initialization != "scratch":
            raise ValueError(
                "The controlled baseline requires "
                "initialization='scratch'."
            )

        dimensions = {
            "projection_hidden_dimension":
            self.projection_hidden_dimension,
            "projection_dimension":
            self.projection_dimension,
        }

        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        if self.prototype_count < 2:
            raise ValueError(
                "prototype_count must be at least two."
            )

        if self.temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        if self.sinkhorn_epsilon <= 0:
            raise ValueError(
                "sinkhorn_epsilon must be positive."
            )

        if self.sinkhorn_iterations < 1:
            raise ValueError(
                "sinkhorn_iterations must be at least one."
            )

        if self.global_crops != 2:
            raise ValueError(
                "The initial controlled SwAV baseline "
                "requires exactly two global crops."
            )

        if self.local_crops != 0:
            raise ValueError(
                "Local crops are not enabled in the "
                "initial SwAV baseline."
            )

        if self.queue_length != 0:
            raise ValueError(
                "The initial SwAV baseline does not "
                "enable an assignment queue."
            )

        if not self.benchmark_id.strip():
            raise ValueError(
                "benchmark_id cannot be empty."
            )

        if self.sampling != "natural_frequency":
            raise ValueError(
                "The controlled comparison requires "
                "sampling='natural_frequency'."
            )

        fingerprint = (
            self
            .expected_manifest_fingerprint_sha256
            .strip()
            .lower()
        )

        if re.fullmatch(
            r"[0-9a-f]{64}",
            fingerprint,
        ) is None:
            raise ValueError(
                "expected_manifest_fingerprint_sha256 "
                "must be a 64-character lowercase "
                "SHA-256 digest."
            )

        row_counts = {
            "expected_train_rows":
            self.expected_train_rows,
            "expected_validation_rows":
            self.expected_validation_rows,
            "expected_test_rows":
            self.expected_test_rows,
        }

        for name, value in row_counts.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        limits = {
            "limit_train_items": (
                self.limit_train_items,
                self.expected_train_rows,
            ),
            "limit_validation_items": (
                self.limit_validation_items,
                self.expected_validation_rows,
            ),
        }

        for name, (
            value,
            expected,
        ) in limits.items():
            if value is None:
                continue

            if value <= 0:
                raise ValueError(
                    f"{name} must be positive "
                    "when provided."
                )

            if value > expected:
                raise ValueError(
                    f"{name} cannot exceed the "
                    "corresponding expected row count."
                )

        if self.verify_files not in {
            "none",
            "sample",
            "all",
        }:
            raise ValueError(
                "verify_files must be one of "
                "'none', 'sample', or 'all'."
            )

        if self.verify_sample_size <= 0:
            raise ValueError(
                "verify_sample_size must be positive."
            )

        integer_optimization_values = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "evaluation_batch_size":
            self.evaluation_batch_size,
        }

        for (
            name,
            value,
        ) in integer_optimization_values.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        if self.optimizer_name.lower() != "sgd":
            raise ValueError(
                "The initial SwAV baseline requires "
                "optimizer='sgd'."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.minimum_learning_rate < 0:
            raise ValueError(
                "minimum_learning_rate cannot "
                "be negative."
            )

        if (
            self.minimum_learning_rate
            > self.learning_rate
        ):
            raise ValueError(
                "minimum_learning_rate cannot exceed "
                "learning_rate."
            )

        if not 0 <= self.momentum < 1:
            raise ValueError(
                "momentum must be in [0, 1)."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError(
                "warmup_epochs must be in "
                "[0, epochs)."
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive "
                "when provided."
            )

        if not (
            0
            <= self.prototype_freeze_epochs
            < self.epochs
        ):
            raise ValueError(
                "prototype_freeze_epochs must be in "
                "[0, epochs)."
            )

        runtime_integers = {
            "num_workers": self.num_workers,
            "checkpoint_every":
            self.checkpoint_every,
            "validation_every":
            self.validation_every,
        }

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        for name in (
            "checkpoint_every",
            "validation_every",
        ):
            if runtime_integers[name] <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative."
            )

        if not self.export_train_embeddings:
            raise ValueError(
                "The baseline must export train "
                "backbone embeddings."
            )

        if not self.export_validation_embeddings:
            raise ValueError(
                "The baseline must export validation "
                "backbone embeddings."
            )

        required_monitoring = {
            "track_assignment_entropy":
            self.track_assignment_entropy,
            "track_prototype_occupancy":
            self.track_prototype_occupancy,
            "track_assignment_agreement":
            self.track_assignment_agreement,
            "track_backbone_feature_norm":
            self.track_backbone_feature_norm,
            "track_projection_norm":
            self.track_projection_norm,
            "track_prototype_weight_norm":
            self.track_prototype_weight_norm,
            "require_finite_loss":
            self.require_finite_loss,
            "require_finite_gradients":
            self.require_finite_gradients,
        }

        disabled = [
            name
            for name, enabled
            in required_monitoring.items()
            if not enabled
        ]

        if disabled:
            raise ValueError(
                "The controlled baseline requires all "
                "monitoring checks. Disabled: "
                + ", ".join(disabled)
                + "."
            )

    def validate_input_paths(
        self,
    ) -> None:
        """Require all manifests and metadata inputs to exist."""

        required_paths = {
            "train_manifest":
            self.train_manifest,
            "validation_manifest":
            self.validation_manifest,
            "test_manifest":
            self.test_manifest,
            "image_root_map":
            self.image_root_map,
            "transform_config":
            self.transform_config,
        }

        for name, path in required_paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"{name} does not exist: {path}"
                )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return a JSON-compatible structured configuration."""

        return {
            "run": {
                "name": self.run_name,
                "output_root": str(
                    self.output_root
                ),
            },
            "model": {
                "method": self.method,
                "backbone": self.backbone_name,
                "initialization":
                self.initialization,
                "projection_hidden_dimension":
                self.projection_hidden_dimension,
                "projection_dimension":
                self.projection_dimension,
                "prototype_count":
                self.prototype_count,
                "temperature": self.temperature,
                "sinkhorn_epsilon":
                self.sinkhorn_epsilon,
                "sinkhorn_iterations":
                self.sinkhorn_iterations,
                "global_crops":
                self.global_crops,
                "local_crops":
                self.local_crops,
                "queue_length":
                self.queue_length,
            },
            "data": {
                "benchmark_id":
                self.benchmark_id,
                "sampling": self.sampling,
                "train_manifest": str(
                    self.train_manifest
                ),
                "validation_manifest": str(
                    self.validation_manifest
                ),
                "test_manifest": str(
                    self.test_manifest
                ),
                "image_root_map": str(
                    self.image_root_map
                ),
                "transform_config": str(
                    self.transform_config
                ),
                "expected_manifest_fingerprint_sha256":
                self
                .expected_manifest_fingerprint_sha256,
                "expected_train_rows":
                self.expected_train_rows,
                "expected_validation_rows":
                self.expected_validation_rows,
                "expected_test_rows":
                self.expected_test_rows,
                "limit_train_items":
                self.limit_train_items,
                "limit_validation_items":
                self.limit_validation_items,
                "verify_files":
                self.verify_files,
                "verify_sample_size":
                self.verify_sample_size,
            },
            "optimization": {
                "epochs": self.epochs,
                "batch_size":
                self.batch_size,
                "evaluation_batch_size":
                self.evaluation_batch_size,
                "optimizer":
                self.optimizer_name,
                "learning_rate":
                self.learning_rate,
                "minimum_learning_rate":
                self.minimum_learning_rate,
                "momentum": self.momentum,
                "weight_decay":
                self.weight_decay,
                "warmup_epochs":
                self.warmup_epochs,
                "gradient_clip_norm":
                self.gradient_clip_norm,
                "prototype_freeze_epochs":
                self.prototype_freeze_epochs,
            },
            "runtime": {
                "num_workers":
                self.num_workers,
                "checkpoint_every":
                self.checkpoint_every,
                "validation_every":
                self.validation_every,
                "seed": self.seed,
                "use_amp": self.use_amp,
                "export_train_embeddings":
                self.export_train_embeddings,
                "export_validation_embeddings":
                self
                .export_validation_embeddings,
            },
            "monitoring": {
                "track_assignment_entropy":
                self.track_assignment_entropy,
                "track_prototype_occupancy":
                self.track_prototype_occupancy,
                "track_assignment_agreement":
                self.track_assignment_agreement,
                "track_backbone_feature_norm":
                self.track_backbone_feature_norm,
                "track_projection_norm":
                self.track_projection_norm,
                "track_prototype_weight_norm":
                self.track_prototype_weight_norm,
                "require_finite_loss":
                self.require_finite_loss,
                "require_finite_gradients":
                self.require_finite_gradients,
            },
        }


def _section(
    mapping: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = mapping.get(name)

    if not isinstance(value, Mapping):
        raise ValueError(
            f"Configuration section {name!r} "
            "must be a mapping."
        )

    return value


def _expand_path(
    value: object,
    *,
    source_directory: Path,
) -> Path:
    expanded = os.path.expandvars(
        os.path.expanduser(
            str(value)
        )
    )

    path = Path(expanded)

    if not path.is_absolute():
        path = source_directory / path

    return path.resolve(
        strict=False
    )


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


def _boolean(
    value: object,
    *,
    name: str,
) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be a boolean."
        )

    return value


def config_from_mapping(
    mapping: Mapping[str, Any],
    *,
    source_directory: Path,
) -> SwAVTrainingConfig:
    """Construct a validated SwAV configuration."""

    run = _section(
        mapping,
        "run",
    )

    model = _section(
        mapping,
        "model",
    )

    data = _section(
        mapping,
        "data",
    )

    optimization = _section(
        mapping,
        "optimization",
    )

    runtime = _section(
        mapping,
        "runtime",
    )

    monitoring = _section(
        mapping,
        "monitoring",
    )

    return SwAVTrainingConfig(
        run_name=str(
            run["name"]
        ),
        output_root=_expand_path(
            run["output_root"],
            source_directory=source_directory,
        ),
        method=str(
            model["method"]
        ),
        backbone_name=str(
            model["backbone"]
        ),
        initialization=str(
            model["initialization"]
        ),
        projection_hidden_dimension=int(
            model[
                "projection_hidden_dimension"
            ]
        ),
        projection_dimension=int(
            model["projection_dimension"]
        ),
        prototype_count=int(
            model["prototype_count"]
        ),
        temperature=float(
            model["temperature"]
        ),
        sinkhorn_epsilon=float(
            model["sinkhorn_epsilon"]
        ),
        sinkhorn_iterations=int(
            model["sinkhorn_iterations"]
        ),
        global_crops=int(
            model["global_crops"]
        ),
        local_crops=int(
            model["local_crops"]
        ),
        queue_length=int(
            model["queue_length"]
        ),
        benchmark_id=str(
            data["benchmark_id"]
        ),
        sampling=str(
            data["sampling"]
        ),
        train_manifest=_expand_path(
            data["train_manifest"],
            source_directory=source_directory,
        ),
        validation_manifest=_expand_path(
            data["validation_manifest"],
            source_directory=source_directory,
        ),
        test_manifest=_expand_path(
            data["test_manifest"],
            source_directory=source_directory,
        ),
        image_root_map=_expand_path(
            data["image_root_map"],
            source_directory=source_directory,
        ),
        transform_config=_expand_path(
            data["transform_config"],
            source_directory=source_directory,
        ),
        expected_manifest_fingerprint_sha256=str(
            data[
                "expected_manifest_fingerprint_sha256"
            ]
        ),
        expected_train_rows=int(
            data["expected_train_rows"]
        ),
        expected_validation_rows=int(
            data[
                "expected_validation_rows"
            ]
        ),
        expected_test_rows=int(
            data["expected_test_rows"]
        ),
        limit_train_items=_optional_integer(
            data.get(
                "limit_train_items"
            )
        ),
        limit_validation_items=_optional_integer(
            data.get(
                "limit_validation_items"
            )
        ),
        verify_files=str(
            data["verify_files"]
        ),
        verify_sample_size=int(
            data["verify_sample_size"]
        ),
        epochs=int(
            optimization["epochs"]
        ),
        batch_size=int(
            optimization["batch_size"]
        ),
        evaluation_batch_size=int(
            optimization[
                "evaluation_batch_size"
            ]
        ),
        optimizer_name=str(
            optimization["optimizer"]
        ),
        learning_rate=float(
            optimization["learning_rate"]
        ),
        minimum_learning_rate=float(
            optimization[
                "minimum_learning_rate"
            ]
        ),
        momentum=float(
            optimization["momentum"]
        ),
        weight_decay=float(
            optimization["weight_decay"]
        ),
        warmup_epochs=int(
            optimization["warmup_epochs"]
        ),
        gradient_clip_norm=_optional_float(
            optimization.get(
                "gradient_clip_norm"
            )
        ),
        prototype_freeze_epochs=int(
            optimization[
                "prototype_freeze_epochs"
            ]
        ),
        num_workers=int(
            runtime["num_workers"]
        ),
        checkpoint_every=int(
            runtime["checkpoint_every"]
        ),
        validation_every=int(
            runtime["validation_every"]
        ),
        seed=int(
            runtime["seed"]
        ),
        use_amp=_boolean(
            runtime["use_amp"],
            name="runtime.use_amp",
        ),
        export_train_embeddings=_boolean(
            runtime[
                "export_train_embeddings"
            ],
            name=(
                "runtime."
                "export_train_embeddings"
            ),
        ),
        export_validation_embeddings=_boolean(
            runtime[
                "export_validation_embeddings"
            ],
            name=(
                "runtime."
                "export_validation_embeddings"
            ),
        ),
        track_assignment_entropy=_boolean(
            monitoring[
                "track_assignment_entropy"
            ],
            name=(
                "monitoring."
                "track_assignment_entropy"
            ),
        ),
        track_prototype_occupancy=_boolean(
            monitoring[
                "track_prototype_occupancy"
            ],
            name=(
                "monitoring."
                "track_prototype_occupancy"
            ),
        ),
        track_assignment_agreement=_boolean(
            monitoring[
                "track_assignment_agreement"
            ],
            name=(
                "monitoring."
                "track_assignment_agreement"
            ),
        ),
        track_backbone_feature_norm=_boolean(
            monitoring[
                "track_backbone_feature_norm"
            ],
            name=(
                "monitoring."
                "track_backbone_feature_norm"
            ),
        ),
        track_projection_norm=_boolean(
            monitoring[
                "track_projection_norm"
            ],
            name=(
                "monitoring."
                "track_projection_norm"
            ),
        ),
        track_prototype_weight_norm=_boolean(
            monitoring[
                "track_prototype_weight_norm"
            ],
            name=(
                "monitoring."
                "track_prototype_weight_norm"
            ),
        ),
        require_finite_loss=_boolean(
            monitoring[
                "require_finite_loss"
            ],
            name=(
                "monitoring."
                "require_finite_loss"
            ),
        ),
        require_finite_gradients=_boolean(
            monitoring[
                "require_finite_gradients"
            ],
            name=(
                "monitoring."
                "require_finite_gradients"
            ),
        ),
    )


def load_swav_training_config(
    path: str | Path,
) -> SwAVTrainingConfig:
    """Load a SwAV configuration from JSON."""

    import json

    source = Path(path).resolve()

    with source.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError(
            "The configuration root must "
            "be a mapping."
        )

    return config_from_mapping(
        payload,
        source_directory=source.parent,
    )
