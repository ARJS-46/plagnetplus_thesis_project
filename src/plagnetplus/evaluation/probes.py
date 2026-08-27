"""Linear probes for frozen representation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LocalityProbeResult:
    """Outputs from one train-to-validation locality probe."""

    pipeline: Pipeline
    summary: dict[str, Any]
    per_class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    predictions: pd.DataFrame


def _validated_features(
    features: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(features)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional; "
            f"received shape {array.shape}."
        )

    if array.shape[0] < 1:
        raise ValueError(
            f"{name} contains no rows."
        )

    if array.shape[1] < 1:
        raise ValueError(
            f"{name} contains no feature dimensions."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise TypeError(
            f"{name} must have a numeric dtype."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return array.astype(
        np.float32,
        copy=False,
    )


def _validated_labels(
    labels: np.ndarray,
    *,
    expected_rows: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(labels).astype(str)

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(values) != expected_rows:
        raise ValueError(
            f"{name} length does not match features: "
            f"{len(values)} != {expected_rows}."
        )

    if np.any(values == ""):
        raise ValueError(
            f"{name} contains empty labels."
        )

    return values


def fit_locality_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    regularisation_c: float = 1.0,
    class_weight: str | None = "balanced",
    maximum_iterations: int = 500,
    tolerance: float = 1e-5,
    seed: int = 42,
) -> LocalityProbeResult:
    """Fit a frozen-feature locality probe and evaluate validation."""
    train_x = _validated_features(
        train_features,
        name="train_features",
    )

    validation_x = _validated_features(
        validation_features,
        name="validation_features",
    )

    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError(
            "Train and validation feature dimensions differ: "
            f"{train_x.shape[1]} != {validation_x.shape[1]}."
        )

    train_y = _validated_labels(
        train_labels,
        expected_rows=train_x.shape[0],
        name="train_labels",
    )

    validation_y = _validated_labels(
        validation_labels,
        expected_rows=validation_x.shape[0],
        name="validation_labels",
    )

    training_classes = np.asarray(
        sorted(
            np.unique(train_y).tolist()
        ),
        dtype=str,
    )

    validation_classes = np.asarray(
        sorted(
            np.unique(validation_y).tolist()
        ),
        dtype=str,
    )

    if len(training_classes) < 2:
        raise ValueError(
            "At least two training classes are required."
        )

    unseen_validation_classes = (
        set(validation_classes)
        - set(training_classes)
    )

    if unseen_validation_classes:
        raise ValueError(
            "Validation contains classes absent from training: "
            f"{sorted(unseen_validation_classes)}"
        )

    if regularisation_c <= 0.0:
        raise ValueError(
            "regularisation_c must be positive."
        )

    if maximum_iterations < 1:
        raise ValueError(
            "maximum_iterations must be positive."
        )

    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be positive."
        )

    # Use class-balanced logistic regression so the probe is not dominated by
    # the largest geographic location in the training split.
    classifier = LogisticRegression(
        C=regularisation_c,
        l1_ratio=0.0,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=maximum_iterations,
        tol=tolerance,
        random_state=seed,
    )

    pipeline = Pipeline(
        [
            (
                "standard_scaler",
                StandardScaler(),
            ),
            (
                "logistic_regression",
                classifier,
            ),
        ]
    )

    # Standardisation is fitted on training embeddings only; validation remains
    # a held-out diagnostic of how strongly location is encoded.
    pipeline.fit(
        train_x,
        train_y,
    )

    predicted = pipeline.predict(
        validation_x
    ).astype(str)

    probabilities = pipeline.predict_proba(
        validation_x
    )

    fitted_classifier = pipeline.named_steps[
        "logistic_regression"
    ]

    model_classes = np.asarray(
        fitted_classifier.classes_
    ).astype(str)

    iterations = int(
        np.max(
            fitted_classifier.n_iter_
        )
    )

    converged = (
        iterations
        < maximum_iterations
    )

    macro_f1 = float(
        f1_score(
            validation_y,
            predicted,
            labels=validation_classes,
            average="macro",
            zero_division=0,
        )
    )

    accuracy = float(
        accuracy_score(
            validation_y,
            predicted,
        )
    )

    validation_log_loss = float(
        log_loss(
            validation_y,
            probabilities,
            labels=model_classes,
        )
    )

    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            validation_y,
            predicted,
            labels=validation_classes,
            zero_division=0,
        )
    )

    balanced_accuracy = float(
        np.mean(recall)
    )

    per_class = pd.DataFrame(
        {
            "locality": validation_classes,
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
            "support": support.astype(
                np.int64
            ),
        }
    )

    all_class_confusion = confusion_matrix(
        validation_y,
        predicted,
        labels=model_classes,
    )

    confusion = pd.DataFrame(
        all_class_confusion,
        index=[
            f"true::{label}"
            for label in model_classes
        ],
        columns=[
            f"predicted::{label}"
            for label in model_classes
        ],
    )

    prediction_table = pd.DataFrame(
        {
            "true_locality": validation_y,
            "predicted_locality": predicted,
            "prediction_correct": (
                validation_y
                == predicted
            ),
            "maximum_probability": (
                probabilities.max(axis=1)
            ),
        }
    )

    for class_index, locality in enumerate(
        model_classes
    ):
        prediction_table[
            f"probability::{locality}"
        ] = probabilities[
            :,
            class_index,
        ]

    validation_class_mask = np.isin(
        predicted,
        validation_classes,
    )

    train_counts = pd.Series(
        train_y
    ).value_counts()

    validation_counts = pd.Series(
        validation_y
    ).value_counts()

    # Record a majority-class reference alongside macro-F1 so probe performance
    # can be interpreted against the imbalanced location distribution.
    train_majority_class = str(
        train_counts.index[0]
    )

    train_majority_predictions = np.repeat(
        train_majority_class,
        len(validation_y),
    ).astype(str)

    summary: dict[str, Any] = {
        "train_rows": int(
            train_x.shape[0]
        ),
        "validation_rows": int(
            validation_x.shape[0]
        ),
        "feature_dimension": int(
            train_x.shape[1]
        ),
        "training_classes": (
            training_classes.tolist()
        ),
        "validation_classes": (
            validation_classes.tolist()
        ),
        "model_classes": (
            model_classes.tolist()
        ),
        "train_class_counts": {
            str(label): int(count)
            for label, count
            in train_counts.sort_index().items()
        },
        "validation_class_counts": {
            str(label): int(count)
            for label, count
            in validation_counts.sort_index().items()
        },
        "configuration": {
            "feature_preprocessing": (
                "standard_scaler_fit_on_train"
            ),
            "classifier": (
                "multinomial_logistic_regression"
            ),
            "penalty": "l2",
            "l1_ratio": 0.0,
            "regularisation_c": float(
                regularisation_c
            ),
            "class_weight": (
                class_weight
            ),
            "solver": "lbfgs",
            "maximum_iterations": int(
                maximum_iterations
            ),
            "tolerance": float(
                tolerance
            ),
            "seed": int(seed),
        },
        "metrics": {
            "locality_macro_f1": (
                macro_f1
            ),
            "balanced_accuracy": (
                balanced_accuracy
            ),
            "accuracy": accuracy,
            "validation_log_loss": (
                validation_log_loss
            ),
            "prediction_outside_validation_class_rate": float(
                1.0
                - validation_class_mask.mean()
            ),
            "training_majority_class": (
                train_majority_class
            ),
            "training_majority_baseline_accuracy": float(
                accuracy_score(
                    validation_y,
                    train_majority_predictions,
                )
            ),
            "training_majority_baseline_macro_f1": float(
                f1_score(
                    validation_y,
                    train_majority_predictions,
                    labels=validation_classes,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "optimisation": {
            "iterations": iterations,
            "converged_before_iteration_limit": (
                converged
            ),
        },
    }

    return LocalityProbeResult(
        pipeline=pipeline,
        summary=summary,
        per_class_metrics=per_class,
        confusion_matrix=confusion,
        predictions=prediction_table,
    )
