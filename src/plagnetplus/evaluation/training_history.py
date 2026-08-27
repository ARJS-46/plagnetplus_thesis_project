"""Training-history validation and summary calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "epoch",
    "split",
    "loss",
    "positive_pair_accuracy",
    "learning_rate",
    "batches",
    "examples",
    "seconds",
}

_EXPECTED_SPLITS = {
    "train",
    "validation",
}

_COLUMN_ALIASES = {
    "batches": (
        "batches",
        "batch_count",
        "num_batches",
        "n_batches",
    ),
    "examples": (
        "examples",
        "example_count",
        "num_examples",
        "n_examples",
        "examples_seen",
        "sample_count",
        "num_samples",
        "samples",
        "images",
    ),
}


def _normalise_column_aliases(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Convert supported history-column aliases to canonical names."""
    rename: dict[str, str] = {}

    for canonical, candidates in _COLUMN_ALIASES.items():
        present = [
            candidate
            for candidate in candidates
            if candidate in history.columns
        ]

        if len(present) > 1:
            raise ValueError(
                "Training history contains multiple aliases "
                f"for {canonical!r}: {present}."
            )

        if not present:
            continue

        source = present[0]

        if source != canonical:
            rename[source] = canonical

    return history.rename(
        columns=rename
    )


def load_training_history(
    path: str | Path,
    *,
    expected_epochs: int | None = None,
) -> pd.DataFrame:
    """Load and strictly validate a PlagNetPlus history CSV."""
    resolved = Path(path)

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    history = pd.read_csv(resolved)

    history = _normalise_column_aliases(
        history
    )

    missing_columns = (
        _REQUIRED_COLUMNS
        - set(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training history is missing columns: "
            f"{sorted(missing_columns)}"
        )

    numeric_columns = [
        "epoch",
        "loss",
        "positive_pair_accuracy",
        "learning_rate",
        "batches",
        "examples",
        "seconds",
    ]

    for column in numeric_columns:
        history[column] = pd.to_numeric(
            history[column],
            errors="raise",
        )

    history["epoch"] = history[
        "epoch"
    ].astype(np.int64)

    history["split"] = history[
        "split"
    ].astype(str)

    unknown_splits = (
        set(history["split"].unique())
        - _EXPECTED_SPLITS
    )

    if unknown_splits:
        raise ValueError(
            "Unexpected training-history splits: "
            f"{sorted(unknown_splits)}"
        )

    duplicate_count = int(
        history.duplicated(
            subset=[
                "epoch",
                "split",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "Training history contains "
            f"{duplicate_count} duplicate epoch/split rows."
        )

    if not np.isfinite(
        history[numeric_columns].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "Training history contains non-finite values."
        )

    if (history["epoch"] < 1).any():
        raise ValueError(
            "Epoch numbers must be positive."
        )

    if (history["seconds"] <= 0.0).any():
        raise ValueError(
            "Every history row must have positive runtime."
        )

    split_counts = (
        history.groupby(
            "epoch",
            observed=True,
        )["split"]
        .nunique()
    )

    incomplete_epochs = split_counts[
        split_counts
        != len(_EXPECTED_SPLITS)
    ]

    if not incomplete_epochs.empty:
        raise ValueError(
            "Some epochs do not contain exactly one "
            "train and one validation row: "
            f"{incomplete_epochs.index.tolist()}"
        )

    observed_epochs = sorted(
        history["epoch"].unique().tolist()
    )

    if expected_epochs is not None:
        expected = list(
            range(
                1,
                expected_epochs + 1,
            )
        )

        if observed_epochs != expected:
            raise ValueError(
                "Observed epochs do not match the expected "
                f"range 1..{expected_epochs}."
            )

    return history.sort_values(
        [
            "epoch",
            "split",
        ],
        kind="stable",
    ).reset_index(drop=True)


def summarise_training_history(
    history: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate method-internal training diagnostics."""
    train = (
        history.loc[
            history["split"] == "train"
        ]
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    validation = (
        history.loc[
            history["split"] == "validation"
        ]
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    if train.empty or validation.empty:
        raise ValueError(
            "Both train and validation rows are required."
        )

    if not np.array_equal(
        train["epoch"].to_numpy(),
        validation["epoch"].to_numpy(),
    ):
        raise ValueError(
            "Train and validation epochs are not aligned."
        )

    best_index = int(
        validation["loss"].idxmin()
    )

    best_row = validation.loc[
        best_index
    ]

    final_train = train.iloc[-1]
    final_validation = validation.iloc[-1]

    def final_window_mean(
        values: pd.Series,
        size: int,
    ) -> float:
        return float(
            values.tail(
                min(
                    size,
                    len(values),
                )
            ).mean()
        )

    summary: dict[str, Any] = {
        "completed_epochs": int(
            train["epoch"].max()
        ),
        "history_rows": int(
            len(history)
        ),
        "best_validation_epoch": int(
            best_row["epoch"]
        ),
        "best_validation_loss": float(
            best_row["loss"]
        ),
        "final_train_loss": float(
            final_train["loss"]
        ),
        "final_validation_loss": float(
            final_validation["loss"]
        ),
        "final_train_validation_gap": float(
            final_validation["loss"]
            - final_train["loss"]
        ),
        "first_validation_loss": float(
            validation.iloc[0]["loss"]
        ),
        "validation_loss_change_first_to_final": float(
            final_validation["loss"]
            - validation.iloc[0]["loss"]
        ),
        "mean_validation_loss_final_10_epochs": (
            final_window_mean(
                validation["loss"],
                10,
            )
        ),
        "mean_validation_loss_final_20_epochs": (
            final_window_mean(
                validation["loss"],
                20,
            )
        ),
        "validation_loss_standard_deviation_final_10_epochs": float(
            validation["loss"]
            .tail(
                min(
                    10,
                    len(validation),
                )
            )
            .std(ddof=0)
        ),
        "final_train_positive_pair_accuracy": float(
            final_train[
                "positive_pair_accuracy"
            ]
        ),
        "final_validation_positive_pair_accuracy": float(
            final_validation[
                "positive_pair_accuracy"
            ]
        ),
        "initial_learning_rate": float(
            train.iloc[0][
                "learning_rate"
            ]
        ),
        "final_learning_rate": float(
            final_train[
                "learning_rate"
            ]
        ),
        "mean_train_epoch_seconds": float(
            train["seconds"].mean()
        ),
        "mean_validation_epoch_seconds": float(
            validation["seconds"].mean()
        ),
        "total_recorded_train_seconds": float(
            train["seconds"].sum()
        ),
        "total_recorded_validation_seconds": float(
            validation["seconds"].sum()
        ),
        "total_recorded_epoch_seconds": float(
            history["seconds"].sum()
        ),
        "train_examples_per_epoch": sorted(
            {
                int(value)
                for value in train[
                    "examples"
                ].unique()
            }
        ),
        "validation_examples_per_epoch": sorted(
            {
                int(value)
                for value in validation[
                    "examples"
                ].unique()
            }
        ),
    }

    return summary
