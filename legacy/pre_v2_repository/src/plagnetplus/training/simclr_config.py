"""Configuration loading and validation for SimCLR training."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from plagnetplus.models.backbones import normalise_backbone_name


@dataclass(frozen=True)
class SimCLRTrainingConfig:
    """Validated configuration for one SimCLR experiment."""

    run_name: str
    benchmark_id: str
    sampling: str

    backbone_name: str
    initialization: str
    projection_hidden_dimension: int | None
    projection_dimension: int
    temperature: float

    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    image_root_map: Path
    transform_config: Path
    output_root: Path

    expected_train_rows: int
    expected_validation_rows: int
    expected_test_rows: int

    epochs: int
    batch_size: int
    evaluation_batch_size: int
    num_workers: int

    learning_rate: float
    minimum_learning_rate: float
    weight_decay: float
    warmup_epochs: int
    gradient_clip_norm: float | None

    checkpoint_every: int
    validation_every: int
    seed: int
    use_amp: bool

    verify_files: str
    verify_sample_size: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backbone_name",
            normalise_backbone_name(
                self.backbone_name
            ),
        )

        if not self.run_name.strip():
            raise ValueError(
                "run_name cannot be empty."
            )

        if not self.benchmark_id.strip():
            raise ValueError(
                "benchmark_id cannot be empty."
            )

        if self.sampling != "natural_frequency":
            raise ValueError(
                "The initial benchmark requires "
                "sampling='natural_frequency'."
            )

        if self.initialization != "scratch":
            raise ValueError(
                "The controlled comparison currently requires "
                "initialization='scratch'."
            )

        if (
            self.projection_hidden_dimension is not None
            and self.projection_hidden_dimension <= 0
        ):
            raise ValueError(
                "projection_hidden_dimension must be "
                "positive or null."
            )

        if self.projection_dimension <= 0:
            raise ValueError(
                "projection_dimension must be positive."
            )

        if self.temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        if self.batch_size < 2:
            raise ValueError(
                "batch_size must be at least two."
            )

        if self.evaluation_batch_size <= 0:
            raise ValueError(
                "evaluation_batch_size must be positive."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.minimum_learning_rate < 0:
            raise ValueError(
                "minimum_learning_rate cannot be negative."
            )

        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError(
                "minimum_learning_rate cannot exceed "
                "learning_rate."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError(
                "warmup_epochs must be between zero "
                "and epochs."
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive "
                "or null."
            )

        if self.checkpoint_every <= 0:
            raise ValueError(
                "checkpoint_every must be positive."
            )

        if self.validation_every <= 0:
            raise ValueError(
                "validation_every must be positive."
            )

        expected_counts = (
            self.expected_train_rows,
            self.expected_validation_rows,
            self.expected_test_rows,
        )

        if any(
            count <= 0
            for count in expected_counts
        ):
            raise ValueError(
                "Expected split row counts must be positive."
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""

        values = asdict(self)

        path_fields = (
            "train_manifest",
            "validation_manifest",
            "test_manifest",
            "image_root_map",
            "transform_config",
            "output_root",
        )

        for field in path_fields:
            values[field] = str(
                values[field]
            )

        return values

    def validate_input_paths(self) -> None:
        """Check that all required input files exist."""

        required = {
            "train_manifest": self.train_manifest,
            "validation_manifest": self.validation_manifest,
            "test_manifest": self.test_manifest,
            "image_root_map": self.image_root_map,
            "transform_config": self.transform_config,
        }

        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "Required configuration inputs are missing:\n"
                + "\n".join(missing)
            )


def _section(
    mapping: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = mapping.get(name)

    if not isinstance(
        value,
        Mapping,
    ):
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


def config_from_mapping(
    mapping: Mapping[str, Any],
    *,
    source_directory: Path,
) -> SimCLRTrainingConfig:
    """Construct a configuration from nested JSON or YAML data."""

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

    return SimCLRTrainingConfig(
        run_name=str(
            run["name"]
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
        backbone_name=str(
            model["backbone"]
        ),
        initialization=str(
            model.get(
                "initialization",
                "scratch",
            )
        ),
        projection_hidden_dimension=_optional_integer(
            model.get(
                "projection_hidden_dimension"
            )
        ),
        projection_dimension=int(
            model.get(
                "projection_dimension",
                128,
            )
        ),
        temperature=float(
            model.get(
                "temperature",
                0.2,
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
        output_root=_expand_path(
            run["output_root"],
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
        epochs=int(
            optimization["epochs"]
        ),
        batch_size=int(
            optimization["batch_size"]
        ),
        evaluation_batch_size=int(
            optimization.get(
                "evaluation_batch_size",
                optimization["batch_size"],
            )
        ),
        num_workers=int(
            runtime.get(
                "num_workers",
                8,
            )
        ),
        learning_rate=float(
            optimization["learning_rate"]
        ),
        minimum_learning_rate=float(
            optimization.get(
                "minimum_learning_rate",
                0.0,
            )
        ),
        weight_decay=float(
            optimization.get(
                "weight_decay",
                1.0e-4,
            )
        ),
        warmup_epochs=int(
            optimization.get(
                "warmup_epochs",
                0,
            )
        ),
        gradient_clip_norm=_optional_float(
            optimization.get(
                "gradient_clip_norm"
            )
        ),
        checkpoint_every=int(
            runtime.get(
                "checkpoint_every",
                1,
            )
        ),
        validation_every=int(
            runtime.get(
                "validation_every",
                1,
            )
        ),
        seed=int(
            runtime.get(
                "seed",
                42,
            )
        ),
        use_amp=bool(
            runtime.get(
                "use_amp",
                True,
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
                1024,
            )
        ),
    )


def load_simclr_training_config(
    path: str | Path,
) -> SimCLRTrainingConfig:
    """Load a SimCLR configuration from JSON or YAML."""

    config_path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    text = config_path.read_text(
        encoding="utf-8"
    )

    suffix = config_path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(
            text
        )
    elif suffix in {
        ".yaml",
        ".yml",
    }:
        payload = yaml.safe_load(
            text
        )
    else:
        raise ValueError(
            "Configuration file must use "
            ".json, .yaml, or .yml."
        )

    if not isinstance(
        payload,
        Mapping,
    ):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    return config_from_mapping(
        payload,
        source_directory=config_path.parent,
    )
