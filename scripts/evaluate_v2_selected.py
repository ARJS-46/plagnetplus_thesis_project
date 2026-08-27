#!/usr/bin/env python3
"""Apply the common k=50 validation protocol to V2 S10, W05 and W12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from evaluate_v2_baselines import evaluate_run


EXPECTED_RUNS = {
    "simsiam_s10_lr0p1_plr0p025_wd0p0001_e100_seed42_v2": "SimSiam S10 v2",
    "swav_w05_lr0p3_p250_t0p1_eps0p05_e100_seed42_v2": "SwAV W05 v2",
    "swav_w12_lr0p6_p500_t0p1_eps0p05_e100_seed42_v2": "SwAV W12 v2",
}


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        required=True,
        help="Completed V2 selected-model directory; repeat exactly three times.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--silhouette-sample-size", type=int, default=5_000)
    parser.add_argument("--retrieval-chunk-size", type=int, default=256)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    project_root = arguments.project_root.expanduser().resolve()
    benchmark_root = arguments.benchmark_root.expanduser().resolve()
    output_root = arguments.output_root.expanduser().resolve()
    run_roots = [path.expanduser().resolve() for path in arguments.run_root]
    if len(run_roots) != 3:
        raise ValueError("Supply exactly three --run-root values.")
    if {path.name for path in run_roots} != set(EXPECTED_RUNS):
        raise RuntimeError(
            "The supplied directories must be the frozen S10, W05 and W12 V2 runs."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root / "src"))
    from plagnetplus.evaluation import (  # noqa: PLC0415
        categorical_enrichment_at_k,
        categorical_precision_at_k,
        entropy_effective_rank,
        evaluate_kmeans_run,
        exact_cosine_neighbours,
        fit_deterministic_pca,
        fit_locality_probe,
        l2_normalise_rows,
        load_embedding_bundle,
        load_embedding_manifest,
        pairwise_clustering_stability,
        prepare_cosine_features,
        validate_embedding_manifest_alignment,
    )

    tools = {
        name: value
        for name, value in locals().items()
        if name
        in {
            "categorical_enrichment_at_k",
            "categorical_precision_at_k",
            "entropy_effective_rank",
            "evaluate_kmeans_run",
            "exact_cosine_neighbours",
            "fit_deterministic_pca",
            "fit_locality_probe",
            "l2_normalise_rows",
            "load_embedding_bundle",
            "load_embedding_manifest",
            "pairwise_clustering_stability",
            "prepare_cosine_features",
            "validate_embedding_manifest_alignment",
        }
    }
    train_manifest = benchmark_root / "manifests" / "crystals_train.csv"
    validation_manifest = benchmark_root / "manifests" / "crystals_validation.csv"
    for path in (train_manifest, validation_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Reuse the common V2 baseline evaluator implementation so S10, W05 and W12
    # receive exactly the same validation protocol.
    results = [
        evaluate_run(
            run_root,
            train_manifest,
            validation_manifest,
            output_root,
            arguments.silhouette_sample_size,
            arguments.retrieval_chunk_size,
            tools,
        )
        for run_root in run_roots
    ]
    rows = [
        {
            "Model": EXPECTED_RUNS[result["run_name"]],
            **{
                column: result[column]
                for column in (
                    "Median silhouette score",
                    "Stability ARI",
                    "FCT cross-sample P@10",
                    "Location-probe macro-F1",
                    "Source-image enrichment@10",
                    "Effective rank",
                    "GPU hours",
                )
            },
            "Selected k": 50,
            "Status": "complete",
        }
        for result in results
    ]
    order = {name: index for index, name in enumerate(EXPECTED_RUNS.values())}
    # Write one ordered summary table from validation metrics only; the test
    # manifest is never loaded by this selected-model comparison.
    table = pd.DataFrame(rows).sort_values(
        "Model", key=lambda values: values.map(order)
    )
    table.to_csv(output_root / "selected_model_metrics.csv", index=False)
    write_json(output_root / "selected_model_metrics.json", {"runs": results})
    print(table.to_string(index=False))
    print("V2 SELECTED COMMON EVALUATION: PASS")
    print("TEST MANIFEST LOADED: NO")
    print("TEST IMAGES ACCESSED: NO")


if __name__ == "__main__":
    main()
