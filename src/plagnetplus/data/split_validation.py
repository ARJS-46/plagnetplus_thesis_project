from __future__ import annotations

import pandas as pd


ALLOWED_BENCHMARK_SPLITS = frozenset(
    {
        "train",
        "validation",
        "test",
    }
)


def validate_benchmark_manifest(
    frame: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []

    required_columns = {
        "crystal_id",
        "manifest_order",
        "physical_sample_id",
        "split_group_id",
        "benchmark_split",
    }

    missing_columns = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        errors.append(
            "missing required columns: "
            + ", ".join(missing_columns)
        )
        return errors

    if frame.empty:
        errors.append(
            "manifest is empty"
        )
        return errors

    if frame["crystal_id"].isna().any():
        errors.append(
            "crystal_id contains missing values"
        )

    if frame["crystal_id"].duplicated().any():
        errors.append(
            "crystal_id contains duplicates"
        )

    if frame["manifest_order"].isna().any():
        errors.append(
            "manifest_order contains missing values"
        )

    if frame["manifest_order"].duplicated().any():
        errors.append(
            "manifest_order contains duplicates"
        )

    for column in (
        "physical_sample_id",
        "split_group_id",
        "benchmark_split",
    ):
        if frame[column].isna().any():
            errors.append(
                f"{column} contains missing values"
            )

    observed_splits = set(
        frame["benchmark_split"]
        .dropna()
        .astype(str)
        .unique()
    )

    invalid_splits = sorted(
        observed_splits
        - ALLOWED_BENCHMARK_SPLITS
    )

    if invalid_splits:
        errors.append(
            "invalid benchmark_split values: "
            + ", ".join(invalid_splits)
        )

    group_split_counts = (
        frame
        .dropna(
            subset=[
                "split_group_id",
                "benchmark_split",
            ]
        )
        .groupby(
            "split_group_id",
            observed=True,
        )["benchmark_split"]
        .nunique()
    )

    leaking_groups = sorted(
        group_split_counts[
            group_split_counts > 1
        ].index.astype(str)
    )

    if leaking_groups:
        errors.append(
            "split_group_id leakage detected: "
            + ", ".join(leaking_groups)
        )

    physical_group_counts = (
        frame
        .dropna(
            subset=[
                "physical_sample_id",
                "split_group_id",
            ]
        )
        .groupby(
            "physical_sample_id",
            observed=True,
        )["split_group_id"]
        .nunique()
    )

    inconsistent_physical_samples = sorted(
        physical_group_counts[
            physical_group_counts > 1
        ].index.astype(str)
    )

    if inconsistent_physical_samples:
        errors.append(
            "physical_sample_id maps to multiple "
            "split_group_id values: "
            + ", ".join(
                inconsistent_physical_samples
            )
        )

    return errors


def assert_valid_benchmark_manifest(
    frame: pd.DataFrame,
) -> None:
    errors = validate_benchmark_manifest(
        frame
    )

    if errors:
        raise ValueError(
            "; ".join(errors)
        )
