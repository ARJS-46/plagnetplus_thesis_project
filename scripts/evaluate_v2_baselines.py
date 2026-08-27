#!/usr/bin/env python3
"""Apply one test-free validation protocol to all four V2 baselines."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


CLUSTER_COUNT = 50
CLUSTER_SEEDS = (42, 43, 44)
EXPECTED_SHAPES = {"train": (91_357, 512), "validation": (14_919, 512)}


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
        help="Completed V2 baseline directory; repeat exactly four times.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--silhouette-sample-size", type=int, default=5_000)
    parser.add_argument("--retrieval-chunk-size", type=int, default=256)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(f"None of {candidates!r} occurs in {frame.columns.tolist()!r}.")


def align_manifest(bundle: Any, manifest: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(
        {
            "crystal_id": bundle.crystal_ids.astype(str),
            "manifest_order": bundle.manifest_orders.astype(np.int64),
        }
    )
    aligned = keys.merge(
        manifest,
        on=["crystal_id", "manifest_order"],
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if not bool((aligned["_merge"] == "both").all()):
        raise RuntimeError("Embedding-to-manifest alignment failed.")
    return aligned.drop(columns="_merge")


def nested_value(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def read_run_record(run_root: Path) -> dict[str, Any]:
    path = run_root / "metadata" / "run_record.json"
    return read_json(path) if path.is_file() else {}


def infer_run_identity(
    summary: dict[str, Any], run_record: dict[str, Any], run_root: Path
) -> dict[str, Any]:
    run_name = (
        summary.get("run_name")
        or nested_value(run_record, "config", "run_name")
        or run_root.name
    )
    method = (
        summary.get("method")
        or nested_value(summary, "model", "method")
        or nested_value(run_record, "model", "method")
    )
    if not method:
        lowered = str(run_name).lower()
        method = next(
            (name for name in ("simclr", "deepcluster", "simsiam", "swav") if lowered.startswith(name)),
            None,
        )
    benchmark_id = summary.get("benchmark_id") or nested_value(
        run_record, "config", "benchmark_id"
    )
    if not method or not benchmark_id:
        raise RuntimeError(f"Could not infer run identity from {run_root}.")
    return {
        "run_name": str(run_name),
        "method": str(method).lower(),
        "benchmark_id": str(benchmark_id),
    }


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"Invalid ISO-8601 timestamp: {value!r}") from error


def infer_gpu_hours(summary: dict[str, Any], run_record: dict[str, Any]) -> float:
    if summary.get("gpu_hours") is not None:
        return float(summary["gpu_hours"])
    if summary.get("elapsed_seconds") is not None:
        return float(summary["elapsed_seconds"]) / 3600.0
    started = parse_utc(run_record.get("created_utc"))
    completed = parse_utc(summary.get("completed_utc"))
    if started and completed:
        elapsed = (completed - started).total_seconds()
        if elapsed > 0:
            return elapsed / 3600.0
    raise RuntimeError("GPU hours cannot be inferred from the run metadata.")


def validate_completion(
    summary: dict[str, Any], identity: dict[str, Any], run_root: Path
) -> None:
    if summary.get("status") != "PASS":
        raise RuntimeError(f"Run is not PASS: {run_root}")
    method = identity["method"]
    if method == "deepcluster":
        completed = summary.get("completed_cycles", summary.get("cycles_completed"))
        if int(completed or -1) != 50:
            raise RuntimeError(f"DeepCluster did not complete 50 cycles: {run_root}")
    elif int(summary.get("completed_epochs", -1)) != 100:
        raise RuntimeError(f"Run did not complete 100 epochs: {run_root}")


def evaluate_run(
    run_root: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    output_root: Path,
    silhouette_sample_size: int,
    retrieval_chunk_size: int,
    tools: dict[str, Any],
) -> dict[str, Any]:
    # Start from the frozen run metadata and reject incomplete runs before
    # loading any embeddings into the common comparison.
    summary = read_json(run_root / "run_summary.json")
    run_record = read_run_record(run_root)
    identity = infer_run_identity(summary, run_record, run_root)
    validate_completion(summary, identity, run_root)
    method = identity["method"]
    bundles = {
        split: tools["load_embedding_bundle"](
            run_root / "embeddings" / f"{split}_backbone_embeddings.npz",
            expected_feature_dimension=512,
        )
        for split in EXPECTED_SHAPES
    }
    raw_manifests = {
        "train": tools["load_embedding_manifest"](train_manifest_path),
        "validation": tools["load_embedding_manifest"](validation_manifest_path),
    }
    manifests: dict[str, pd.DataFrame] = {}
    for split in EXPECTED_SHAPES:
        if bundles[split].features.shape != EXPECTED_SHAPES[split]:
            raise RuntimeError(f"Unexpected {split} shape in {run_root}.")
        if not np.isfinite(bundles[split].features).all():
            raise RuntimeError(f"Non-finite {split} embeddings in {run_root}.")
        if not tools["validate_embedding_manifest_alignment"](
            bundles[split], raw_manifests[split]
        ):
            raise RuntimeError(f"{split} alignment failed in {run_root}.")
        manifests[split] = align_manifest(bundles[split], raw_manifests[split])

    # All four baselines pass through the same representation-level metrics;
    # method-specific training losses are intentionally not compared here.
    effective_rank, participation_ratio = tools["entropy_effective_rank"](
        bundles["validation"].features
    )
    # Cluster the common 50-D PCA representation and reuse one fixed silhouette
    # subset so model scores are directly comparable.
    pca = tools["fit_deterministic_pca"](
        tools["l2_normalise_rows"](bundles["validation"].features),
        retained_components=50,
    )
    rng = np.random.default_rng(42)
    silhouette_indices = np.sort(
        rng.choice(
            len(pca.coordinates),
            size=min(silhouette_sample_size, len(pca.coordinates)),
            replace=False,
        )
    )
    labels_by_seed: list[np.ndarray] = []
    clustering_rows: list[dict[str, Any]] = []
    for seed in CLUSTER_SEEDS:
        labels, _, metrics = tools["evaluate_kmeans_run"](
            pca.coordinates,
            number_of_clusters=CLUSTER_COUNT,
            seed=seed,
            silhouette_indices=silhouette_indices,
        )
        labels_by_seed.append(labels)
        clustering_rows.append({"seed": seed, **metrics})
    _, stability = tools["pairwise_clustering_stability"](
        np.stack(labels_by_seed),
        seeds=CLUSTER_SEEDS,
        number_of_clusters=CLUSTER_COUNT,
    )
    ari_values = [
        float(value)
        for key, value in stability.items()
        if "ari" in key.lower() and "median" in key.lower()
    ]
    if len(ari_values) != 1:
        raise RuntimeError(f"Expected one median ARI field; found {stability!r}.")

    validation = manifests["validation"]
    location_column = resolve_column(validation, ("locality", "location", "volcano", "site"))
    sample_column = resolve_column(
        validation,
        ("acquisition_id", "sample_id", "sample", "sample_name", "specimen_id"),
    )
    source_column = resolve_column(
        validation,
        (
            "source_image_relative_path",
            "source_image",
            "source_image_id",
            "source_scan",
            "parent_image",
            "image_source",
        ),
    )
    location_labels = validation[location_column].astype(str).to_numpy()
    sample_labels = validation[sample_column].astype(str).to_numpy()
    cosine = tools["prepare_cosine_features"](bundles["validation"].features)
    raw_neighbours, _ = tools["exact_cosine_neighbours"](
        cosine, neighbour_count=20, chunk_size=retrieval_chunk_size
    )
    sample_codes = pd.factorize(sample_labels, sort=True)[0].astype(np.int64)
    cross_sample_neighbours, _ = tools["exact_cosine_neighbours"](
        cosine,
        neighbour_count=10,
        exclusion_codes=sample_codes,
        chunk_size=retrieval_chunk_size,
    )
    fct_mask = np.char.upper(location_labels.astype(str)) == "FCT"
    fct_precision = tools["categorical_precision_at_k"](
        cross_sample_neighbours,
        location_labels,
        k=10,
        query_mask=fct_mask,
    )
    source_enrichment = tools["categorical_enrichment_at_k"](
        raw_neighbours,
        validation[source_column].astype(str).to_numpy(),
        k=10,
    )

    train = manifests["train"]
    train_location_column = resolve_column(
        train, ("locality", "location", "volcano", "site")
    )
    # Fit the location probe from training embeddings to validation embeddings;
    # the test manifest is not loaded by this evaluator.
    probe = tools["fit_locality_probe"](
        bundles["train"].features,
        train[train_location_column].astype(str).to_numpy(),
        bundles["validation"].features,
        location_labels,
        regularisation_c=1.0,
        class_weight="balanced",
        maximum_iterations=500,
        tolerance=1e-5,
        seed=42,
    )

    run_output = output_root / identity["run_name"]
    run_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clustering_rows).to_csv(run_output / "clustering_runs.csv", index=False)
    probe.per_class_metrics.to_csv(run_output / "location_probe_per_class.csv", index=False)
    probe.confusion_matrix.to_csv(run_output / "location_probe_confusion_matrix.csv")
    result = {
        "schema_version": "plagnetplus_v2_common_metrics_v1",
        "status": "PASS",
        "method": method,
        "run_name": identity["run_name"],
        "benchmark_id": identity["benchmark_id"],
        "test_manifest_loaded": False,
        "test_images_accessed": False,
        "cluster_count": CLUSTER_COUNT,
        "cluster_seeds": list(CLUSTER_SEEDS),
        "Median silhouette score": float(
            pd.DataFrame(clustering_rows)["silhouette"].median()
        ),
        "Stability ARI": ari_values[0],
        "FCT cross-sample P@10": float(fct_precision),
        "Location-probe macro-F1": float(
            probe.summary["metrics"]["locality_macro_f1"]
        ),
        "Source-image enrichment@10": float(source_enrichment["enrichment"]),
        "Effective rank": float(effective_rank),
        "Participation ratio": float(participation_ratio),
        "GPU hours": infer_gpu_hours(summary, run_record),
    }
    write_json(run_output / "analysis_summary.json", result)
    return result


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.expanduser().resolve()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_roots = [path.expanduser().resolve() for path in args.run_root]
    if len(run_roots) != 4:
        raise ValueError("Supply exactly four --run-root values.")
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

    results = [
        evaluate_run(
            run_root,
            train_manifest,
            validation_manifest,
            output_root,
            args.silhouette_sample_size,
            args.retrieval_chunk_size,
            tools,
        )
        for run_root in run_roots
    ]
    if {result["method"] for result in results} != {
        "simclr",
        "deepcluster",
        "simsiam",
        "swav",
    }:
        raise RuntimeError("The four supplied runs do not cover the four baseline methods.")
    rows = [
        {
            "Model": result["method"].replace("simclr", "SimCLR").replace(
                "simsiam", "SimSiam"
            ).replace("deepcluster", "DeepCluster").replace("swav", "SwAV"),
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
            "Status": "complete",
        }
        for result in results
    ]
    order = {name: index for index, name in enumerate(("SimCLR", "DeepCluster", "SimSiam", "SwAV"))}
    table = pd.DataFrame(rows).sort_values("Model", key=lambda values: values.map(order))
    table.to_csv(output_root / "model_metrics.csv", index=False)
    write_json(output_root / "baseline_metrics.json", {"runs": results})
    print(table.to_string(index=False))
    print("V2 BASELINE COMMON EVALUATION: PASS")
    print("TEST MANIFEST LOADED: NO")
    print("TEST IMAGES ACCESSED: NO")


if __name__ == "__main__":
    main()
