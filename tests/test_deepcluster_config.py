from __future__ import annotations

import json
from pathlib import Path

import pytest

from plagnetplus.training.deepcluster_config import (
    config_from_mapping,
    load_deepcluster_training_config,
)


def valid_mapping(
    tmp_path: Path,
) -> dict[str, object]:
    paths = {}

    for name in [
        "train.csv",
        "validation.csv",
        "test.csv",
        "roots.json",
        "transforms.json",
    ]:
        path = tmp_path / name
        path.write_text(
            "{}\n",
            encoding="utf-8",
        )
        paths[name] = path

    return {
        "run_name": "deepcluster_smoke",
        "benchmark_id": (
            "pooled_broad_p80_10_10_v1"
        ),
        "sampling": "natural_frequency",
        "backbone_name": "resnet18",
        "initialization": "scratch",
        "train_manifest": str(
            paths["train.csv"]
        ),
        "validation_manifest": str(
            paths["validation.csv"]
        ),
        "test_manifest": str(
            paths["test.csv"]
        ),
        "image_root_map": str(
            paths["roots.json"]
        ),
        "transform_config": str(
            paths["transforms.json"]
        ),
        "output_root": str(
            tmp_path / "runs"
        ),
        "expected_train_rows": 100,
        "expected_validation_rows": 20,
        "expected_test_rows": 20,
        "cycles": 2,
        "cluster_count": 10,
        "pca_components": 8,
        "warmup_cycles": 1,
        "limit_train_items": 50,
    }


def test_deepcluster_config_from_mapping(
    tmp_path: Path,
) -> None:
    config = config_from_mapping(
        valid_mapping(tmp_path)
    )

    assert config.run_name == (
        "deepcluster_smoke"
    )

    assert config.backbone_name == (
        "resnet18"
    )

    assert config.cluster_count == 10
    assert config.pca_components == 8
    assert config.cycles == 2
    assert config.limit_train_items == 50

    payload = config.to_dict()

    assert isinstance(
        payload["train_manifest"],
        str,
    )


def test_load_deepcluster_json_config(
    tmp_path: Path,
) -> None:
    mapping = valid_mapping(
        tmp_path
    )

    path = tmp_path / "config.json"

    path.write_text(
        json.dumps(mapping),
        encoding="utf-8",
    )

    config = (
        load_deepcluster_training_config(
            path
        )
    )

    assert config.cluster_count == 10
    assert config.cycles == 2


def test_deepcluster_config_rejects_too_many_clusters(
    tmp_path: Path,
) -> None:
    mapping = valid_mapping(
        tmp_path
    )

    mapping["cluster_count"] = 51

    with pytest.raises(
        ValueError,
        match="cluster_count",
    ):
        config_from_mapping(
            mapping
        )


def test_deepcluster_config_rejects_missing_key(
    tmp_path: Path,
) -> None:
    mapping = valid_mapping(
        tmp_path
    )

    del mapping["benchmark_id"]

    with pytest.raises(
        KeyError,
        match="benchmark_id",
    ):
        config_from_mapping(
            mapping
        )
