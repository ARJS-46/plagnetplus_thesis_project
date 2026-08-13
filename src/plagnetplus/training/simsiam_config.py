"""Validated configuration loading for SimSiam experiments."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class SimSiamTrainingConfig:
    """Validated configuration for one SimSiam experiment."""

    run_name: str
    output_root: Path

    method: str
    backbone_name: str
    initialization: str
    projection_layers: int
    projection_hidden_dimension: int
    projection_dimension: int
    prediction_layers: int
    prediction_hidden_dimension: int
    prediction_dimension: int
    final_projection_batch_norm_affine: bool
    explicit_stop_gradient: bool

    benchmark_id: str
    sampling: str
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    image_root_map: Path
    transform_config: Path
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
    predictor_learning_rate_mode: str
    predictor_learning_rate: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    gradient_clip_norm: float | None

    num_workers: int
    checkpoint_every: int
    validation_every: int
    seed: int
    use_amp: bool
    export_train_embeddings: bool
    export_validation_embeddings: bool

    track_projection_standard_deviation: bool
    track_collapse_level: bool
    track_backbone_feature_norm: bool
    track_projection_norm: bool
    require_finite_loss: bool
    require_finite_gradients: bool

    def __post_init__(self) -> None:
        """Validate scalar values and the frozen baseline contract."""

        for name, value in (
            ("run_name", self.run_name),
            ("method", self.method),
            ("backbone_name", self.backbone_name),
            ("initialization", self.initialization),
            ("benchmark_id", self.benchmark_id),
            ("sampling", self.sampling),
            ("optimizer_name", self.optimizer_name),
            (
                "predictor_learning_rate_mode",
                self.predictor_learning_rate_mode,
            ),
        ):
            if not str(value).strip():
                raise ValueError(
                    f"{name} must be non-empty."
                )

        if self.method.lower() != "simsiam":
            raise ValueError(
                "SimSiam configuration method must be 'simsiam'."
            )

        if self.sampling != "natural_frequency":
            raise ValueError(
                "The frozen baseline requires natural_frequency sampling."
            )

        if self.projection_layers != 3:
            raise ValueError(
                "The baseline projector must contain three layers."
            )

        if self.prediction_layers != 2:
            raise ValueError(
                "The baseline predictor must contain two layers."
            )

        for name, value in (
            (
                "projection_hidden_dimension",
                self.projection_hidden_dimension,
            ),
            (
                "projection_dimension",
                self.projection_dimension,
            ),
            (
                "prediction_hidden_dimension",
                self.prediction_hidden_dimension,
            ),
            (
                "prediction_dimension",
                self.prediction_dimension,
            ),
        ):
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        if (
            self.prediction_dimension
            != self.projection_dimension
        ):
            raise ValueError(
                "Prediction and projection dimensions must match."
            )

        if self.final_projection_batch_norm_affine:
            raise ValueError(
                "The baseline final projector BatchNorm must be non-affine."
            )

        if not self.explicit_stop_gradient:
            raise ValueError(
                "The baseline requires explicit stop-gradient targets."
            )

        for name, value in (
            ("expected_train_rows", self.expected_train_rows),
            (
                "expected_validation_rows",
                self.expected_validation_rows,
            ),
            ("expected_test_rows", self.expected_test_rows),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            (
                "evaluation_batch_size",
                self.evaluation_batch_size,
            ),
            ("verify_sample_size", self.verify_sample_size),
            ("checkpoint_every", self.checkpoint_every),
            ("validation_every", self.validation_every),
        ):
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        for name, limit, expected in (
            (
                "limit_train_items",
                self.limit_train_items,
                self.expected_train_rows,
            ),
            (
                "limit_validation_items",
                self.limit_validation_items,
                self.expected_validation_rows,
            ),
        ):
            if limit is not None and not (
                0 < limit <= expected
            ):
                raise ValueError(
                    f"{name} must be positive and no greater "
                    "than its expected split size."
                )

        if self.verify_files not in {
            "none",
            "sample",
            "all",
        }:
            raise ValueError(
                "verify_files must be 'none', 'sample', or 'all'."
            )

        if self.optimizer_name.lower() != "sgd":
            raise ValueError(
                "The frozen baseline optimiser must be SGD."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if not (
            0
            <= self.minimum_learning_rate
            <= self.learning_rate
        ):
            raise ValueError(
                "minimum_learning_rate must lie between "
                "zero and learning_rate."
            )

        if (
            self.predictor_learning_rate_mode
            != "fixed"
        ):
            raise ValueError(
                "The baseline predictor learning-rate mode "
                "must be fixed."
            )

        if self.predictor_learning_rate <= 0:
            raise ValueError(
                "predictor_learning_rate must be positive."
            )

        if not 0 <= self.momentum < 1:
            raise ValueError(
                "momentum must lie in [0, 1)."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError(
                "warmup_epochs must lie between zero and epochs."
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive when set."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        if not self.export_train_embeddings:
            raise ValueError(
                "The baseline must export train embeddings."
            )

        if not self.export_validation_embeddings:
            raise ValueError(
                "The baseline must export validation embeddings."
            )

        required_monitoring = {
            "track_projection_standard_deviation": (
                self.track_projection_standard_deviation
            ),
            "track_collapse_level": (
                self.track_collapse_level
            ),
            "track_backbone_feature_norm": (
                self.track_backbone_feature_norm
            ),
            "track_projection_norm": (
                self.track_projection_norm
            ),
            "require_finite_loss": (
                self.require_finite_loss
            ),
            "require_finite_gradients": (
                self.require_finite_gradients
            ),
        }

        disabled = sorted(
            name
            for name, enabled
            in required_monitoring.items()
            if not enabled
        )

        if disabled:
            raise ValueError(
                "Required SimSiam monitoring cannot be disabled: "
                + ", ".join(disabled)
            )

    def validate_input_paths(self) -> None:
        """Require all frozen benchmark inputs to exist as files."""

        required_paths = {
            "train_manifest": self.train_manifest,
            "validation_manifest": self.validation_manifest,
            "test_manifest": self.test_manifest,
            "image_root_map": self.image_root_map,
            "transform_config": self.transform_config,
        }

        missing = [
            f"{name}={path}"
            for name, path in required_paths.items()
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "Required SimSiam inputs are missing: "
                + "; ".join(missing)
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration."""

        payload = asdict(self)

        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)

        return payload


def _section(
    mapping: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = mapping.get(name)

    if not isinstance(value, Mapping):
        raise ValueError(
            f"Missing or invalid configuration section: {name!r}."
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

    return path.resolve()


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
            f"{name} must be a JSON/YAML boolean."
        )

    return value


def config_from_mapping(
    mapping: Mapping[str, Any],
    *,
    source_directory: Path,
) -> SimSiamTrainingConfig:
    """Construct a validated SimSiam configuration."""

    run = _section(mapping, "run")
    model = _section(mapping, "model")
    data = _section(mapping, "data")
    optimization = _section(
        mapping,
        "optimization",
    )
    runtime = _section(mapping, "runtime")
    monitoring = _section(
        mapping,
        "monitoring",
    )

    return SimSiamTrainingConfig(
        run_name=str(run["name"]),
        output_root=_expand_path(
            run["output_root"],
            source_directory=source_directory,
        ),
        method=str(model["method"]),
        backbone_name=str(model["backbone"]),
        initialization=str(
            model["initialization"]
        ),
        projection_layers=int(
            model["projection_layers"]
        ),
        projection_hidden_dimension=int(
            model["projection_hidden_dimension"]
        ),
        projection_dimension=int(
            model["projection_dimension"]
        ),
        prediction_layers=int(
            model["prediction_layers"]
        ),
        prediction_hidden_dimension=int(
            model["prediction_hidden_dimension"]
        ),
        prediction_dimension=int(
            model["prediction_dimension"]
        ),
        final_projection_batch_norm_affine=_boolean(
            model[
                "final_projection_batch_norm_affine"
            ],
            name=(
                "final_projection_batch_norm_affine"
            ),
        ),
        explicit_stop_gradient=_boolean(
            model["explicit_stop_gradient"],
            name="explicit_stop_gradient",
        ),
        benchmark_id=str(
            data["benchmark_id"]
        ),
        sampling=str(
            data.get(
                "sampling",
                "natural_frequency",
            )
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
        expected_train_rows=int(
            data["expected_train_rows"]
        ),
        expected_validation_rows=int(
            data["expected_validation_rows"]
        ),
        expected_test_rows=int(
            data["expected_test_rows"]
        ),
        limit_train_items=_optional_integer(
            data.get("limit_train_items")
        ),
        limit_validation_items=_optional_integer(
            data.get(
                "limit_validation_items"
            )
        ),
        verify_files=str(
            data.get(
                "verify_files",
                "sample",
            )
        ),
        verify_sample_size=int(
            data.get(
                "verify_sample_size",
                2048,
            )
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
        predictor_learning_rate_mode=str(
            optimization[
                "predictor_learning_rate_mode"
            ]
        ),
        predictor_learning_rate=float(
            optimization[
                "predictor_learning_rate"
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
        num_workers=int(
            runtime["num_workers"]
        ),
        checkpoint_every=int(
            runtime["checkpoint_every"]
        ),
        validation_every=int(
            runtime["validation_every"]
        ),
        seed=int(runtime["seed"]),
        use_amp=_boolean(
            runtime["use_amp"],
            name="use_amp",
        ),
        export_train_embeddings=_boolean(
            runtime[
                "export_train_embeddings"
            ],
            name="export_train_embeddings",
        ),
        export_validation_embeddings=_boolean(
            runtime[
                "export_validation_embeddings"
            ],
            name="export_validation_embeddings",
        ),
        track_projection_standard_deviation=(
            _boolean(
                monitoring[
                    "track_projection_standard_deviation"
                ],
                name=(
                    "track_projection_standard_deviation"
                ),
            )
        ),
        track_collapse_level=_boolean(
            monitoring[
                "track_collapse_level"
            ],
            name="track_collapse_level",
        ),
        track_backbone_feature_norm=_boolean(
            monitoring[
                "track_backbone_feature_norm"
            ],
            name=(
                "track_backbone_feature_norm"
            ),
        ),
        track_projection_norm=_boolean(
            monitoring[
                "track_projection_norm"
            ],
            name="track_projection_norm",
        ),
        require_finite_loss=_boolean(
            monitoring[
                "require_finite_loss"
            ],
            name="require_finite_loss",
        ),
        require_finite_gradients=_boolean(
            monitoring[
                "require_finite_gradients"
            ],
            name="require_finite_gradients",
        ),
    )


def load_simsiam_training_config(
    path: str | Path,
) -> SimSiamTrainingConfig:
    """Load a SimSiam configuration from JSON or YAML."""

    config_path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    text = config_path.read_text(
        encoding="utf-8",
    )

    suffix = config_path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {
        ".yaml",
        ".yml",
    }:
        payload = yaml.safe_load(text)
    else:
        raise ValueError(
            "Configuration file must use "
            ".json, .yaml, or .yml."
        )

    if not isinstance(payload, Mapping):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    return config_from_mapping(
        payload,
        source_directory=config_path.parent,
    )


__all__ = [
    "SimSiamTrainingConfig",
    "config_from_mapping",
    "load_simsiam_training_config",
]
