#!/usr/bin/env python3
"""Evaluate exact cosine retrieval and source leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plagnetplus.evaluation import (
    categorical_enrichment_at_k,
    categorical_precision_at_k,
    exact_cosine_neighbours,
    load_embedding_bundle,
    load_embedding_manifest,
    prepare_cosine_features,
    sha256_file,
    validate_embedding_manifest_alignment,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-id",
        required=True,
    )

    parser.add_argument(
        "--maximum-neighbours",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = load_embedding_bundle(
        args.embedding,
        expected_feature_dimension=512,
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

    required_columns = {
        "locality",
        "split_group_id",
        "image_root_key",
        "source_image_relative_path",
        "source_tile_id",
        "scan_id",
    }

    missing = (
        required_columns
        - set(manifest.columns)
    )

    if missing:
        raise ValueError(
            "Manifest is missing retrieval columns: "
            f"{sorted(missing)}"
        )

    localities = (
        manifest["locality"]
        .astype(str)
        .to_numpy()
    )

    split_groups = (
        manifest["split_group_id"]
        .astype(str)
        .to_numpy()
    )

    _, split_group_codes = np.unique(
        split_groups,
        return_inverse=True,
    )

    source_image_keys = (
        manifest["image_root_key"].astype(str)
        + "::"
        + manifest[
            "source_image_relative_path"
        ].astype(str)
    ).to_numpy()

    non_aegina_mask = (
        manifest["image_root_key"]
        .astype(str)
        .eq("non_aegina")
        .to_numpy()
    )

    source_tile_values = (
        manifest["source_tile_id"]
        .astype("string")
    )

    source_tile_keys = np.where(
        non_aegina_mask,
        (
            manifest["scan_id"].astype(str)
            + "::"
            + source_tile_values.fillna(
                "<MISSING>"
            ).astype(str)
        ).to_numpy(),
        (
            "__NOT_APPLICABLE__::"
            + manifest["crystal_id"].astype(str)
        ).to_numpy(),
    )

    groups_per_locality = (
        manifest.groupby(
            "locality",
            observed=True,
        )["split_group_id"]
        .nunique()
    )

    eligible_localities = {
        str(locality)
        for locality, count
        in groups_per_locality.items()
        if int(count) >= 2
    }

    eligible_cross_sample_mask = np.asarray(
        [
            locality
            in eligible_localities
            for locality in localities
        ],
        dtype=bool,
    )

    representations = {
        "raw_cosine": prepare_cosine_features(
            bundle.features,
            centre=False,
        ),
        "centred_cosine": prepare_cosine_features(
            bundle.features,
            centre=True,
        ),
    }

    metric_rows: list[
        dict[str, Any]
    ] = []

    summaries: dict[
        str,
        Any,
    ] = {}

    for representation_name, features in representations.items():
        print()
        print(
            "=== Representation:",
            representation_name,
            "===",
        )

        # Compute unrestricted and cross-sample neighbours separately because they
        # answer different shortcut-learning and geological-retrieval questions.
        unrestricted_indices, unrestricted_similarities = (
            exact_cosine_neighbours(
                features,
                neighbour_count=(
                    args.maximum_neighbours
                ),
                chunk_size=args.chunk_size,
            )
        )

        cross_group_indices, cross_group_similarities = (
            exact_cosine_neighbours(
                features,
                neighbour_count=(
                    args.maximum_neighbours
                ),
                exclusion_codes=(
                    split_group_codes
                ),
                chunk_size=args.chunk_size,
            )
        )

        np.savez_compressed(
            args.output_root
            / (
                f"{representation_name}_"
                "neighbours_v1.npz"
            ),
            unrestricted_indices=(
                unrestricted_indices
            ),
            unrestricted_similarities=(
                unrestricted_similarities
            ),
            cross_group_indices=(
                cross_group_indices
            ),
            cross_group_similarities=(
                cross_group_similarities
            ),
            crystal_ids=bundle.crystal_ids,
            manifest_orders=(
                bundle.manifest_orders
            ),
        )

        representation_summary = {}

        for k in [5, 10, 20]:
            same_locality = (
                categorical_precision_at_k(
                    unrestricted_indices,
                    localities,
                    k=k,
                )
            )

            same_split_group = (
                categorical_precision_at_k(
                    unrestricted_indices,
                    split_groups,
                    k=k,
                )
            )

            cross_group_same_locality = (
                categorical_precision_at_k(
                    cross_group_indices,
                    localities,
                    k=k,
                )
            )

            eligible_cross_sample = (
                categorical_precision_at_k(
                    cross_group_indices,
                    localities,
                    k=k,
                    query_mask=(
                        eligible_cross_sample_mask
                    ),
                )
            )

            source_image = (
                categorical_enrichment_at_k(
                    unrestricted_indices,
                    source_image_keys,
                    k=k,
                )
            )

            source_tile = (
                categorical_enrichment_at_k(
                    unrestricted_indices,
                    source_tile_keys,
                    k=k,
                    query_mask=(
                        non_aegina_mask
                    ),
                )
            )

            row = {
                "representation": (
                    representation_name
                ),
                "k": k,
                "same_locality_precision": (
                    same_locality
                ),
                "same_split_group_precision": (
                    same_split_group
                ),
                "cross_group_same_locality_precision": (
                    cross_group_same_locality
                ),
                "eligible_cross_sample_same_locality_precision": (
                    eligible_cross_sample
                ),
                "same_source_image_precision": (
                    source_image[
                        "observed_precision"
                    ]
                ),
                "same_source_image_expected_precision": (
                    source_image[
                        "expected_precision"
                    ]
                ),
                "same_source_image_enrichment": (
                    source_image[
                        "enrichment"
                    ]
                ),
                "same_source_tile_precision_non_aegina": (
                    source_tile[
                        "observed_precision"
                    ]
                ),
                "same_source_tile_expected_precision_non_aegina": (
                    source_tile[
                        "expected_precision"
                    ]
                ),
                "same_source_tile_enrichment_non_aegina": (
                    source_tile[
                        "enrichment"
                    ]
                ),
            }

            metric_rows.append(row)

            representation_summary[
                f"k_{k}"
            ] = row

        summaries[
            representation_name
        ] = representation_summary

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        args.output_root
        / "retrieval_metrics_by_representation_and_k.csv",
        index=False,
    )

    raw_k10 = summaries[
        "raw_cosine"
    ]["k_10"]

    centred_k10 = summaries[
        "centred_cosine"
    ]["k_10"]

    checks = {
        "alignment_pass": (
            alignment["ids_aligned"]
            and alignment[
                "manifest_orders_aligned"
            ]
        ),
        "expected_rows": (
            bundle.row_count == 14919
        ),
        "expected_dimension": (
            bundle.feature_dimension == 512
        ),
        "eligible_localities_fct_only": (
            eligible_localities
            == {"FCT"}
        ),
        "metrics_finite": bool(
            np.isfinite(
                metrics.select_dtypes(
                    include=[np.number]
                ).to_numpy()
            ).all()
        ),
        "raw_source_image_enrichment_positive": (
            raw_k10[
                "same_source_image_enrichment"
            ]
            > 0.0
        ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    report = {
        "schema_version": (
            "plagnetplus_retrieval_evaluation_v1"
        ),
        "model_id": args.model_id,
        "split": "validation",
        "status": status,
        "input": {
            "embedding": str(
                args.embedding.resolve()
            ),
            "embedding_sha256": (
                sha256_file(
                    args.embedding
                )
            ),
            "manifest": str(
                args.manifest.resolve()
            ),
            "manifest_sha256": (
                sha256_file(
                    args.manifest
                )
            ),
            "rows": bundle.row_count,
            "dimensions": (
                bundle.feature_dimension
            ),
        },
        "eligible_cross_sample_localities": (
            sorted(
                eligible_localities
            )
        ),
        "eligible_cross_sample_rows": int(
            eligible_cross_sample_mask.sum()
        ),
        "headline_automatic_diagnostics": {
            "raw_cosine_fct_cross_sample_same_locality_precision_at_10": (
                raw_k10[
                    "eligible_cross_sample_same_locality_precision"
                ]
            ),
            "raw_cosine_same_source_image_enrichment_at_10": (
                raw_k10[
                    "same_source_image_enrichment"
                ]
            ),
            "raw_cosine_same_source_tile_enrichment_at_10_non_aegina": (
                raw_k10[
                    "same_source_tile_enrichment_non_aegina"
                ]
            ),
            "centred_cosine_fct_cross_sample_same_locality_precision_at_10": (
                centred_k10[
                    "eligible_cross_sample_same_locality_precision"
                ]
            ),
            "centred_cosine_same_source_image_enrichment_at_10": (
                centred_k10[
                    "same_source_image_enrichment"
                ]
            ),
        },
        "checks": checks,
    }

    (
        args.output_root
        / "retrieval_summary.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        metrics.to_string(
            index=False
        )
    )

    print()
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
        "RETRIEVAL EVALUATION:",
        status,
    )

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
