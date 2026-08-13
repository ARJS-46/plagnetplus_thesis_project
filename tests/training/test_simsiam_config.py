"""Tests for the validated SimSiam configuration loader."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from plagnetplus.training.simsiam_config import (
    SimSiamTrainingConfig,
    config_from_mapping,
)


def base_mapping(
    root: Path,
) -> dict[str, object]:
    """Return a valid compact SimSiam configuration mapping."""

    return {
        "run": {
            "name": "simsiam_test",
            "output_root": str(
                root / "runs"
            ),
        },
        "model": {
            "method": "simsiam",
            "backbone": "resnet18",
            "initialization": "scratch",
            "projection_layers": 3,
            "projection_hidden_dimension": 2048,
            "projection_dimension": 2048,
            "prediction_layers": 2,
            "prediction_hidden_dimension": 512,
            "prediction_dimension": 2048,
            "final_projection_batch_norm_affine": False,
            "explicit_stop_gradient": True,
        },
        "data": {
            "benchmark_id": "benchmark",
            "sampling": "natural_frequency",
            "train_manifest": str(
                root / "train.csv"
            ),
            "validation_manifest": str(
                root / "validation.csv"
            ),
            "test_manifest": str(
                root / "test.csv"
            ),
            "image_root_map": str(
                root / "image_root_map.json"
            ),
            "transform_config": str(
                root / "transform.json"
            ),
            "expected_train_rows": 100,
            "expected_validation_rows": 20,
            "expected_test_rows": 30,
            "limit_train_items": None,
            "limit_validation_items": None,
            "verify_files": "sample",
            "verify_sample_size": 10,
        },
        "optimization": {
            "epochs": 100,
            "batch_size": 256,
            "evaluation_batch_size": 256,
            "optimizer": "sgd",
            "learning_rate": 0.05,
            "minimum_learning_rate": 0.0,
            "predictor_learning_rate_mode": "fixed",
            "predictor_learning_rate": 0.05,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "warmup_epochs": 10,
            "gradient_clip_norm": 5.0,
        },
        "runtime": {
            "num_workers": 8,
            "checkpoint_every": 10,
            "validation_every": 1,
            "seed": 42,
            "use_amp": True,
            "export_train_embeddings": True,
            "export_validation_embeddings": True,
        },
        "monitoring": {
            "track_projection_standard_deviation": True,
            "track_collapse_level": True,
            "track_backbone_feature_norm": True,
            "track_projection_norm": True,
            "require_finite_loss": True,
            "require_finite_gradients": True,
        },
    }


def create_input_files(
    root: Path,
) -> None:
    """Create the files checked by validate_input_paths()."""

    for name in (
        "train.csv",
        "validation.csv",
        "test.csv",
        "image_root_map.json",
        "transform.json",
    ):
        (root / name).write_text(
            "test\n",
            encoding="utf-8",
        )


def test_valid_mapping_loads(
    tmp_path: Path,
) -> None:
    config = config_from_mapping(
        base_mapping(tmp_path),
        source_directory=tmp_path,
    )

    assert isinstance(
        config,
        SimSiamTrainingConfig,
    )

    assert config.method == "simsiam"
    assert config.projection_dimension == 2048
    assert config.prediction_dimension == 2048
    assert config.seed == 42
    assert config.export_train_embeddings
    assert config.export_validation_embeddings


def test_validate_input_paths(
    tmp_path: Path,
) -> None:
    create_input_files(tmp_path)

    config = config_from_mapping(
        base_mapping(tmp_path),
        source_directory=tmp_path,
    )

    config.validate_input_paths()


def test_validate_input_paths_rejects_missing_file(
    tmp_path: Path,
) -> None:
    create_input_files(tmp_path)
    (tmp_path / "validation.csv").unlink()

    config = config_from_mapping(
        base_mapping(tmp_path),
        source_directory=tmp_path,
    )

    with pytest.raises(
        FileNotFoundError,
        match="validation_manifest",
    ):
        config.validate_input_paths()


def test_prediction_dimension_must_match_projection(
    tmp_path: Path,
) -> None:
    mapping = base_mapping(tmp_path)
    model = mapping["model"]
    assert isinstance(model, dict)

    model["prediction_dimension"] = 512

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        config_from_mapping(
            mapping,
            source_directory=tmp_path,
        )


def test_stop_gradient_cannot_be_disabled(
    tmp_path: Path,
) -> None:
    mapping = base_mapping(tmp_path)
    model = mapping["model"]
    assert isinstance(model, dict)

    model["explicit_stop_gradient"] = False

    with pytest.raises(
        ValueError,
        match="stop-gradient",
    ):
        config_from_mapping(
            mapping,
            source_directory=tmp_path,
        )


def test_monitoring_cannot_be_disabled(
    tmp_path: Path,
) -> None:
    mapping = base_mapping(tmp_path)
    monitoring = mapping["monitoring"]
    assert isinstance(monitoring, dict)

    monitoring[
        "track_collapse_level"
    ] = False

    with pytest.raises(
        ValueError,
        match="monitoring",
    ):
        config_from_mapping(
            mapping,
            source_directory=tmp_path,
        )


def test_warmup_cannot_exceed_epochs(
    tmp_path: Path,
) -> None:
    mapping = base_mapping(tmp_path)
    optimization = mapping[
        "optimization"
    ]
    assert isinstance(
        optimization,
        dict,
    )

    optimization["warmup_epochs"] = 101

    with pytest.raises(
        ValueError,
        match="warmup_epochs",
    ):
        config_from_mapping(
            mapping,
            source_directory=tmp_path,
        )


def test_limit_cannot_exceed_expected_rows(
    tmp_path: Path,
) -> None:
    mapping = base_mapping(tmp_path)
    data = mapping["data"]
    assert isinstance(data, dict)

    data["limit_train_items"] = 101

    with pytest.raises(
        ValueError,
        match="limit_train_items",
    ):
        config_from_mapping(
            mapping,
            source_directory=tmp_path,
        )


def test_boolean_strings_are_rejected(
    tmp_path: Path,
) -> None:
    mapping = base_mapping(tmp_path)
    runtime = mapping["runtime"]
    assert isinstance(runtime, dict)

    runtime["use_amp"] = "false"

    with pytest.raises(
        ValueError,
        match="boolean",
    ):
        config_from_mapping(
            mapping,
            source_directory=tmp_path,
        )


def test_to_dict_converts_paths(
    tmp_path: Path,
) -> None:
    config = config_from_mapping(
        base_mapping(tmp_path),
        source_directory=tmp_path,
    )

    payload = config.to_dict()

    assert isinstance(
        payload["output_root"],
        str,
    )

    assert isinstance(
        payload["train_manifest"],
        str,
    )


def test_mapping_is_not_mutated(
    tmp_path: Path,
) -> None:
    mapping = base_mapping(tmp_path)
    original = deepcopy(mapping)

    config_from_mapping(
        mapping,
        source_directory=tmp_path,
    )

    assert mapping == original
