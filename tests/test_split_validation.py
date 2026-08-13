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
