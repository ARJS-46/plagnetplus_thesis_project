#!/usr/bin/env python3
"""Audit an exported backbone embedding matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plagnetplus.evaluation import (
    embedding_qc_summary,
    load_embedding_bundle,
    load_embedding_manifest,
    sha256_file,
    validate_embedding_manifest_alignment,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit embedding integrity, manifest "
            "alignment and representation health."
        )
    )

    parser.add_argument(
        "--embedding",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--completed-json",
        type=Path,
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
        "--split",
        default="validation",
    )

    parser.add_argument(
        "--expected-rows",
        type=int,
    )

    parser.add_argument(
        "--expected-dimension",
        type=int,
    )

    parser.add_argument(
        "--expected-fingerprint",
    )

    parser.add_argument(
        "--pair-count",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--pair-seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_completion_record(
    path: Path | None,
) -> dict[str, Any] | None:
    if path is None:
        return None

    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    args = parse_arguments()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load identifiers alongside features so quality checks can detect both
    # numerical problems and manifest-order mismatches.
    bundle = load_embedding_bundle(
        args.embedding,
        expected_feature_dimension=(
            args.expected_dimension
        ),
    )

    if args.expected_rows is not None:
        if bundle.row_count != args.expected_rows:
            raise ValueError(
                "Unexpected embedding row count: "
                f"{bundle.row_count} != "
                f"{args.expected_rows}."
            )

    manifest = load_embedding_manifest(
        args.manifest
    )

    alignment = (
        validate_embedding_manifest_alignment(
            bundle,
            manifest,
        )
    )

    qc, dimension_table = (
        embedding_qc_summary(
            bundle,
            pair_count=args.pair_count,
            pair_seed=args.pair_seed,
        )
    )

    completion = load_completion_record(
        args.completed_json
    )

    completion_checks: dict[str, bool] = {}

    if completion is not None:
        completion_checks = {
            "status_pass": (
                completion.get("status")
                == "PASS"
            ),
            "completed_embedding_hash_matches": (
                completion[
                    "validation_embeddings"
                ]["sha256"]
                == sha256_file(
                    args.embedding
                )
            ),
            "completed_shape_matches": (
                completion[
                    "validation_embeddings"
                ]["shape"]
                == [
                    bundle.row_count,
                    bundle.feature_dimension,
                ]
            ),
            "completed_ids_aligned": bool(
                completion[
                    "validation_embeddings"
                ]["ids_aligned"]
            ),
            "completed_orders_aligned": bool(
                completion[
                    "validation_embeddings"
                ][
                    "manifest_orders_aligned"
                ]
            ),
        }

        if args.expected_fingerprint:
            completion_checks[
                "fingerprint_matches"
            ] = (
                completion[
                    "manifest_fingerprint_sha256"
                ]
                == args.expected_fingerprint
            )

    required_qc_checks = {
        "all_finite": qc["all_finite"],
        "no_zero_variance_dimensions": (
            qc[
                "zero_variance_dimensions"
            ]
            == 0
        ),
        "no_zero_norm_rows": (
            qc["zero_norm_rows"]
            == 0
        ),
        "no_exact_duplicate_embeddings": (
            qc[
                "exact_duplicate_embedding_rows"
            ]
            == 0
        ),
    }

    all_checks = {
        **required_qc_checks,
        **completion_checks,
    }

    status = (
        "PASS"
        if all(all_checks.values())
        else "FAIL"
    )

    report = {
        "schema_version": (
            "plagnetplus_embedding_audit_v1"
        ),
        "model_id": args.model_id,
        "split": args.split,
        "status": status,
        "inputs": {
            "embedding_path": str(
                args.embedding.resolve()
            ),
            "embedding_sha256": (
                sha256_file(
                    args.embedding
                )
            ),
            "manifest_path": str(
                args.manifest.resolve()
            ),
            "manifest_sha256": (
                sha256_file(
                    args.manifest
                )
            ),
            "completed_json_path": (
                str(
                    args.completed_json.resolve()
                )
                if args.completed_json
                else None
            ),
        },
        "alignment": alignment,
        "representation_health": qc,
        "checks": all_checks,
    }

    report_path = (
        args.output_root
        / "embedding_integrity_summary.json"
    )

    dimensions_path = (
        args.output_root
        / "embedding_dimension_statistics.csv"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    dimension_table.to_csv(
        dimensions_path,
        index=False,
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
        "Summary:",
        report_path,
    )
    print(
        "Dimension table:",
        dimensions_path,
    )
    print()
    print(
        "EMBEDDING INTEGRITY AUDIT:",
        status,
    )

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
