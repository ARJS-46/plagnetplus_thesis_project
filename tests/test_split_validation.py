import pandas as pd
import pytest

from plagnetplus.data.split_validation import (
    assert_valid_benchmark_manifest,
    validate_benchmark_manifest,
)


def make_valid_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crystal_id": [
                "a1",
                "a2",
                "b1",
                "c1",
            ],
            "manifest_order": [
                0,
                1,
                2,
                3,
            ],
            "physical_sample_id": [
                "sample_a",
                "sample_a",
                "sample_b",
                "sample_c",
            ],
            "split_group_id": [
                "group_a",
                "group_a",
                "group_b",
                "group_c",
            ],
            "benchmark_split": [
                "train",
                "train",
                "validation",
                "test",
            ],
        }
    )


def test_valid_benchmark_manifest():
    frame = make_valid_manifest()

    assert (
        validate_benchmark_manifest(
            frame
        )
        == []
    )


def test_split_group_cannot_cross_partitions():
    frame = make_valid_manifest()

    frame.loc[
        frame["crystal_id"] == "a2",
        "benchmark_split",
    ] = "validation"

    errors = validate_benchmark_manifest(
        frame
    )

    assert any(
        "split_group_id leakage detected"
        in error
        for error in errors
    )


def test_physical_sample_cannot_use_multiple_split_groups():
    frame = make_valid_manifest()

    frame.loc[
        frame["crystal_id"] == "a2",
        "split_group_id",
    ] = "different_group"

    errors = validate_benchmark_manifest(
        frame
    )

    assert any(
        "physical_sample_id maps to multiple"
        in error
        for error in errors
    )


def test_invalid_split_name_is_rejected():
    frame = make_valid_manifest()

    frame.loc[
        0,
        "benchmark_split",
    ] = "development"

    errors = validate_benchmark_manifest(
        frame
    )

    assert any(
        "invalid benchmark_split values"
        in error
        for error in errors
    )


def test_duplicate_crystal_id_is_rejected():
    frame = make_valid_manifest()

    frame.loc[
        1,
        "crystal_id",
    ] = "a1"

    errors = validate_benchmark_manifest(
        frame
    )

    assert (
        "crystal_id contains duplicates"
        in errors
    )


def test_duplicate_manifest_order_is_rejected():
    frame = make_valid_manifest()

    frame.loc[
        1,
        "manifest_order",
    ] = 0

    errors = validate_benchmark_manifest(
        frame
    )

    assert (
        "manifest_order contains duplicates"
        in errors
    )


def test_missing_required_column_is_rejected():
    frame = (
        make_valid_manifest()
        .drop(
            columns=[
                "split_group_id"
            ]
        )
    )

    errors = validate_benchmark_manifest(
        frame
    )

    assert errors == [
        "missing required columns: "
        "split_group_id"
    ]


def test_missing_group_value_is_rejected():
    frame = make_valid_manifest()

    frame.loc[
        0,
        "split_group_id",
    ] = None

    errors = validate_benchmark_manifest(
        frame
    )

    assert (
        "split_group_id contains missing values"
        in errors
    )


def test_assert_valid_manifest_passes():
    frame = make_valid_manifest()

    assert_valid_benchmark_manifest(
        frame
    )


def test_assert_valid_manifest_raises():
    frame = make_valid_manifest()

    frame.loc[
        frame["crystal_id"] == "a2",
        "benchmark_split",
    ] = "test"

    with pytest.raises(
        ValueError,
        match="split_group_id leakage detected",
    ):
        assert_valid_benchmark_manifest(
            frame
        )
