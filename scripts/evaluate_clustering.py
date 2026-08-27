#!/usr/bin/env python3
"""Evaluate clustering separation and stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plagnetplus.evaluation import (
    evaluate_kmeans_run,
    pairwise_clustering_stability,
    sha256_file,
)


DEFAULT_K_VALUES = (
    5,
    10,
    20,
    30,
    50,
)

DEFAULT_SEEDS = (
    11,
    22,
    33,
    44,
    55,
    66,
    77,
    88,
    99,
    110,
)


def comma_separated_integers(
    value: str,
) -> tuple[int, ...]:
    values = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one integer is required."
        )

    return values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pca-embedding",
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
        "--split",
        default="validation",
    )

    parser.add_argument(
        "--expected-rows",
        type=int,
        default=14919,
    )

    parser.add_argument(
        "--expected-dimensions",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--k-values",
        type=comma_separated_integers,
        default=DEFAULT_K_VALUES,
    )

    parser.add_argument(
        "--seeds",
        type=comma_separated_integers,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--silhouette-sample-seed",
        type=int,
        default=20260724,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignment_root = (
        args.output_root
        / "assignments"
    )

    assignment_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(
        args.pca_embedding,
        allow_pickle=False,
    ) as payload:
        required_keys = {
            "normalised_features",
            "crystal_ids",
            "manifest_orders",
        }

        missing = (
            required_keys
            - set(payload.files)
        )

        if missing:
            raise ValueError(
                "PCA archive is missing keys: "
                f"{sorted(missing)}"
            )

        features = np.asarray(
            payload["normalised_features"],
            dtype=np.float32,
        )

        crystal_ids = np.asarray(
            payload["crystal_ids"]
        ).astype(str)

        manifest_orders = np.asarray(
            payload["manifest_orders"],
            dtype=np.int64,
        )

    if features.shape != (
        args.expected_rows,
        args.expected_dimensions,
    ):
        raise ValueError(
            "Unexpected PCA feature shape: "
            f"{features.shape} != "
            f"{(args.expected_rows, args.expected_dimensions)}."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "PCA features contain non-finite values."
        )

    if crystal_ids.shape != (
        args.expected_rows,
    ):
        raise ValueError(
            "Unexpected crystal ID shape."
        )

    if manifest_orders.shape != (
        args.expected_rows,
    ):
        raise ValueError(
            "Unexpected manifest-order shape."
        )

    if np.unique(crystal_ids).size != len(
        crystal_ids
    ):
        raise ValueError(
            "Crystal IDs are not unique."
        )

    if np.unique(manifest_orders).size != len(
        manifest_orders
    ):
        raise ValueError(
            "Manifest orders are not unique."
        )

    if not 1 <= args.silhouette_sample_size <= len(
        features
    ):
        raise ValueError(
            "Invalid silhouette sample size."
        )

    sample_rng = np.random.default_rng(
        args.silhouette_sample_seed
    )

    # Draw the silhouette subset once from the fixed seed and reuse it for every
    # k/seed combination in this evaluation.
    silhouette_indices = np.sort(
        sample_rng.choice(
            len(features),
            size=args.silhouette_sample_size,
            replace=False,
        )
    ).astype(np.int64)

    np.savez_compressed(
        args.output_root
        / "silhouette_sample_v1.npz",
        indices=silhouette_indices,
        crystal_ids=crystal_ids[
            silhouette_indices
        ],
        manifest_orders=manifest_orders[
            silhouette_indices
        ],
    )

    run_rows: list[
        dict[str, Any]
    ] = []

    pair_tables: list[
        pd.DataFrame
    ] = []

    stability_rows: list[
        dict[str, Any]
    ] = []

    for number_of_clusters in args.k_values:
        print()
        print(
            f"=== k={number_of_clusters} ==="
        )

        labels_by_seed: list[
            np.ndarray
        ] = []

        centres_by_seed: list[
            np.ndarray
        ] = []

        for seed in args.seeds:
            labels, centres, metrics = (
                evaluate_kmeans_run(
                    features,
                    number_of_clusters=(
                        number_of_clusters
                    ),
                    seed=seed,
                    silhouette_indices=(
                        silhouette_indices
                    ),
                )
            )

            labels_by_seed.append(labels)
            centres_by_seed.append(
                centres
            )
            run_rows.append(metrics)

            print(
                "k={k} seed={seed} "
                "silhouette={silhouette:.6f} "
                "DB={db:.6f} "
                "entropy={entropy:.6f} "
                "min_fraction={minimum:.6f}".format(
                    k=number_of_clusters,
                    seed=seed,
                    silhouette=metrics[
                        "silhouette"
                    ],
                    db=metrics[
                        "davies_bouldin"
                    ],
                    entropy=metrics[
                        "normalised_cluster_entropy"
                    ],
                    minimum=metrics[
                        "minimum_cluster_fraction"
                    ],
                )
            )

        label_matrix = np.stack(
            labels_by_seed,
            axis=0,
        ).astype(
            np.int32,
            copy=False,
        )

        centre_matrix = np.stack(
            centres_by_seed,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        pair_table, stability = (
            pairwise_clustering_stability(
                label_matrix,
                seeds=args.seeds,
                number_of_clusters=(
                    number_of_clusters
                ),
            )
        )

        pair_tables.append(pair_table)
        stability_rows.append(stability)

        np.savez_compressed(
            assignment_root
            / (
                "validation_kmeans_"
                f"k{number_of_clusters:04d}.npz"
            ),
            labels=label_matrix,
            centres=centre_matrix,
            seeds=np.asarray(
                args.seeds,
                dtype=np.int64,
            ),
            crystal_ids=crystal_ids,
            manifest_orders=manifest_orders,
        )

        print(
            "stability median ARI:",
            stability[
                "median_pairwise_ari"
            ],
        )

    runs = pd.DataFrame(
        run_rows
    ).sort_values(
        [
            "k",
            "seed",
        ],
        kind="stable",
    )

    stability_pairs = pd.concat(
        pair_tables,
        ignore_index=True,
    ).sort_values(
        [
            "k",
            "seed_left",
            "seed_right",
        ],
        kind="stable",
    )

    stability_by_k = pd.DataFrame(
        stability_rows
    ).sort_values(
        "k",
        kind="stable",
    )

    separation_by_k = (
        runs.groupby(
            "k",
            observed=True,
        )
        .agg(
            median_silhouette=(
                "silhouette",
                "median",
            ),
            mean_silhouette=(
                "silhouette",
                "mean",
            ),
            minimum_silhouette=(
                "silhouette",
                "min",
            ),
            maximum_silhouette=(
                "silhouette",
                "max",
            ),
            median_davies_bouldin=(
                "davies_bouldin",
                "median",
            ),
            median_calinski_harabasz=(
                "calinski_harabasz",
                "median",
            ),
            median_cluster_entropy=(
                "normalised_cluster_entropy",
                "median",
            ),
            median_minimum_cluster_fraction=(
                "minimum_cluster_fraction",
                "median",
            ),
            minimum_occupied_clusters=(
                "occupied_clusters",
                "min",
            ),
            maximum_occupied_clusters=(
                "occupied_clusters",
                "max",
            ),
        )
        .reset_index()
    )

    summary_by_k = separation_by_k.merge(
        stability_by_k,
        on="k",
        validate="one_to_one",
    )

    headline_median_silhouette = float(
        np.median(
            summary_by_k[
                "median_silhouette"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )

    headline_stability_ari = float(
        np.median(
            summary_by_k[
                "median_pairwise_ari"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )

    numeric_run_columns = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "inertia",
        "inertia_per_row",
        "minimum_cluster_fraction",
        "maximum_cluster_fraction",
        "normalised_cluster_entropy",
    ]

    checks = {
        "feature_shape_expected": (
            features.shape
            == (
                args.expected_rows,
                args.expected_dimensions,
            )
        ),
        "features_finite": bool(
            np.isfinite(features).all()
        ),
        "crystal_ids_unique": (
            np.unique(crystal_ids).size
            == len(crystal_ids)
        ),
        "manifest_orders_unique": (
            np.unique(manifest_orders).size
            == len(manifest_orders)
        ),
        "expected_run_count": (
            len(runs)
            == (
                len(args.k_values)
                * len(args.seeds)
            )
        ),
        "all_run_metrics_finite": bool(
            np.isfinite(
                runs[
                    numeric_run_columns
                ].to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "all_requested_clusters_occupied": bool(
            (
                runs[
                    "occupied_clusters"
                ]
                == runs["k"]
            ).all()
        ),
        "expected_stability_pair_count": (
            len(stability_pairs)
            == (
                len(args.k_values)
                * (
                    len(args.seeds)
                    * (
                        len(args.seeds)
                        - 1
                    )
                    // 2
                )
            )
        ),
        "headline_silhouette_finite": bool(
            np.isfinite(
                headline_median_silhouette
            )
        ),
        "headline_stability_ari_finite": bool(
            np.isfinite(
                headline_stability_ari
            )
        ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    runs.to_csv(
        args.output_root
        / "clustering_runs.csv",
        index=False,
    )

    stability_pairs.to_csv(
        args.output_root
        / "clustering_stability_pairs.csv",
        index=False,
    )

    summary_by_k.to_csv(
        args.output_root
        / "clustering_summary_by_k.csv",
        index=False,
    )

    report = {
        "schema_version": (
            "plagnetplus_clustering_evaluation_v1"
        ),
        "model_id": args.model_id,
        "split": args.split,
        "status": status,
        "input": {
            "pca_embedding": str(
                args.pca_embedding.resolve()
            ),
            "pca_embedding_sha256": (
                sha256_file(
                    args.pca_embedding
                )
            ),
            "rows": int(
                features.shape[0]
            ),
            "dimensions": int(
                features.shape[1]
            ),
            "feature_key": (
                "normalised_features"
            ),
        },
        "clustering": {
            "algorithm": "kmeans",
            "initialisation": "k-means++",
            "n_init_per_seed": 1,
            "maximum_iterations": 300,
            "tolerance": 1e-4,
            "k_values": list(
                args.k_values
            ),
            "seeds": list(
                args.seeds
            ),
            "silhouette_sample_size": int(
                args.silhouette_sample_size
            ),
            "silhouette_sample_seed": int(
                args.silhouette_sample_seed
            ),
            "distance": "euclidean_on_l2_normalised_pca",
        },
        "headline_metrics": {
            "median_silhouette_across_k": (
                headline_median_silhouette
            ),
            "median_stability_ari_across_k": (
                headline_stability_ari
            ),
        },
        "checks": checks,
    }

    (
        args.output_root
        / "clustering_summary.json"
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
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    print()
    print(
        "CLUSTERING EVALUATION:",
        status,
    )

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
