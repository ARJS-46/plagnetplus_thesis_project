#!/usr/bin/env python3
"""Fit a train-to-validation locality probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from plagnetplus.evaluation import (
    fit_locality_probe,
    load_embedding_bundle,
    load_embedding_manifest,
    sha256_file,
    validate_embedding_manifest_alignment,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-embedding",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation-embedding",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-id",
        required=True,
    )

    parser.add_argument(
        "--regularisation-c",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--maximum-iterations",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_bundle = load_embedding_bundle(
        args.train_embedding,
        expected_feature_dimension=512,
    )

    validation_bundle = load_embedding_bundle(
        args.validation_embedding,
        expected_feature_dimension=512,
    )

    train_manifest = load_embedding_manifest(
        args.train_manifest
    )

    validation_manifest = (
        load_embedding_manifest(
            args.validation_manifest
        )
    )

    train_alignment = (
        validate_embedding_manifest_alignment(
            train_bundle,
            train_manifest,
        )
    )

    validation_alignment = (
        validate_embedding_manifest_alignment(
            validation_bundle,
            validation_manifest,
        )
    )

    for name, manifest in [
        ("train", train_manifest),
        (
            "validation",
            validation_manifest,
        ),
    ]:
        if "locality" not in manifest.columns:
            raise ValueError(
                f"{name} manifest lacks locality."
            )

        if manifest[
            "locality"
        ].isna().any():
            raise ValueError(
                f"{name} locality contains nulls."
            )

    # Train the diagnostic probe on frozen training embeddings and score only the
    # validation representation; model weights are not updated.
    result = fit_locality_probe(
        train_bundle.features,
        train_manifest[
            "locality"
        ].astype(str).to_numpy(),
        validation_bundle.features,
        validation_manifest[
            "locality"
        ].astype(str).to_numpy(),
        regularisation_c=(
            args.regularisation_c
        ),
        class_weight="balanced",
        maximum_iterations=(
            args.maximum_iterations
        ),
        tolerance=args.tolerance,
        seed=args.seed,
    )

    predictions = result.predictions.copy()

    predictions.insert(
        0,
        "manifest_order",
        validation_bundle.manifest_orders,
    )

    predictions.insert(
        0,
        "crystal_id",
        validation_bundle.crystal_ids,
    )

    result.per_class_metrics.to_csv(
        args.output_root
        / "locality_per_class_metrics.csv",
        index=False,
    )

    result.confusion_matrix.to_csv(
        args.output_root
        / "locality_confusion_matrix.csv",
        index=True,
    )

    predictions.to_csv(
        args.output_root
        / "locality_validation_predictions.csv",
        index=False,
    )

    temporary_model = (
        args.output_root
        / "locality_probe.joblib.partial"
    )

    final_model = (
        args.output_root
        / "locality_probe.joblib"
    )

    joblib.dump(
        result.pipeline,
        temporary_model,
    )

    temporary_model.replace(
        final_model
    )

    checks = {
        "train_alignment_pass": bool(
            train_alignment[
                "ids_aligned"
            ]
            and train_alignment[
                "manifest_orders_aligned"
            ]
        ),
        "validation_alignment_pass": bool(
            validation_alignment[
                "ids_aligned"
            ]
            and validation_alignment[
                "manifest_orders_aligned"
            ]
        ),
        "train_rows_expected": (
            train_bundle.row_count
            == 101682
        ),
        "validation_rows_expected": (
            validation_bundle.row_count
            == 14919
        ),
        "feature_dimension_expected": (
            train_bundle.feature_dimension
            == validation_bundle.feature_dimension
            == 512
        ),
        "model_converged": bool(
            result.summary[
                "optimisation"
            ][
                "converged_before_iteration_limit"
            ]
        ),
        "metrics_finite": bool(
            np.isfinite(
                [
                    result.summary[
                        "metrics"
                    ][
                        "locality_macro_f1"
                    ],
                    result.summary[
                        "metrics"
                    ][
                        "balanced_accuracy"
                    ],
                    result.summary[
                        "metrics"
                    ]["accuracy"],
                    result.summary[
                        "metrics"
                    ][
                        "validation_log_loss"
                    ],
                ]
            ).all()
        ),
        "predictions_aligned": bool(
            np.array_equal(
                predictions[
                    "crystal_id"
                ].astype(str).to_numpy(),
                validation_bundle.crystal_ids,
            )
            and np.array_equal(
                predictions[
                    "manifest_order"
                ].to_numpy(
                    dtype=np.int64
                ),
                validation_bundle.manifest_orders,
            )
        ),
        "test_split_not_accessed": True,
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    report = {
        "schema_version": (
            "plagnetplus_locality_probe_v1"
        ),
        "model_id": args.model_id,
        "status": status,
        "fit_split": "train",
        "evaluation_split": "validation",
        "test_split_accessed": False,
        "inputs": {
            "train_embedding": {
                "path": str(
                    args.train_embedding.resolve()
                ),
                "sha256": sha256_file(
                    args.train_embedding
                ),
            },
            "train_manifest": {
                "path": str(
                    args.train_manifest.resolve()
                ),
                "sha256": sha256_file(
                    args.train_manifest
                ),
            },
            "validation_embedding": {
                "path": str(
                    args.validation_embedding.resolve()
                ),
                "sha256": sha256_file(
                    args.validation_embedding
                ),
            },
            "validation_manifest": {
                "path": str(
                    args.validation_manifest.resolve()
                ),
                "sha256": sha256_file(
                    args.validation_manifest
                ),
            },
        },
        "probe": result.summary,
        "outputs": {
            "model": str(
                final_model.resolve()
            ),
            "model_sha256": sha256_file(
                final_model
            ),
            "per_class_metrics": str(
                (
                    args.output_root
                    / "locality_per_class_metrics.csv"
                ).resolve()
            ),
            "confusion_matrix": str(
                (
                    args.output_root
                    / "locality_confusion_matrix.csv"
                ).resolve()
            ),
            "predictions": str(
                (
                    args.output_root
                    / "locality_validation_predictions.csv"
                ).resolve()
            ),
        },
        "checks": checks,
    }

    summary_path = (
        args.output_root
        / "locality_probe_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    print()
    print(
        result.per_class_metrics.to_string(
            index=False
        )
    )

    print()
    print(
        result.confusion_matrix.to_string()
    )

    print()
    print(
        "LOCALITY PROBE:",
        status,
    )

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
