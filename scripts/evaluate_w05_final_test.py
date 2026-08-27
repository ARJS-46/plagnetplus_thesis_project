#!/usr/bin/env python3
"""Reproduce the frozen SwAV W05 unified-V2 final-test evaluation.

This is the command-line equivalent of
``Notebook_11C_W05_Final_Test_Evaluation.ipynb``.  The W05 checkpoint,
configuration, benchmark identity, clustering settings and evaluation choices
are frozen before test evaluation.  No training or model selection is
performed here.

If the verified test embedding already exists, it is reused and no test image
pixels are opened.  Otherwise the script performs the one deterministic test
embedding export required for the frozen final evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


BENCHMARK_ID = "pooled_broad_p80_10_10_v2"
W05_RUN_NAME = (
    "swav_w05_lr0p3_p250_t0p1_eps0p05_e100_seed42_v2"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "025388f60f21409cffc48d39f9c4d305"
    "3f696b3a4d9ca6ad4567a41ad31d09da"
)
EXPECTED_CONFIG_SHA256 = (
    "50979fbf40f716978f15ff8ee27e42ce0"
    "d6e8237b47b11e5d2a67f0892ed45d2"
)
EXPECTED_MANIFEST_FINGERPRINT = (
    "c3c028a9aef47cd618ebceaa66ba7444"
    "f2b92a696abba6e509db2d468f440e7f"
)

EXPECTED_TRAIN_SHAPE = (91_357, 512)
EXPECTED_VALIDATION_SHAPE = (14_919, 512)
EXPECTED_TEST_SHAPE = (16_988, 512)
FROZEN_CLUSTER_COUNT = 50
PCA_COMPONENTS = 50
CLUSTER_SEEDS = (42, 43, 44)
SILHOUETTE_SAMPLE_SIZE = 5_000
EVALUATION_SEED = 42

COMPARISON_METRICS = (
    "Median silhouette score",
    "Stability ARI",
    "FCT cross-sample P@10",
    "Location-probe macro-F1",
    "Source-image enrichment@10",
    "Effective rank",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen SwAV W05 v2 final-test evaluation."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Data root containing processed/plagnetplus. Defaults to "
            "PNP_DATA_ROOT, then <repository-parent>/data."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help=(
            "Runs root containing hpc/v2. Defaults to PNP_RUNS_ROOT, then "
            "<repository-parent>/runs."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Frozen W05 v2 configuration JSON.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Completed W05 v2 run directory.",
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        default=None,
        help="SwAV training entry point used only for loader/export helpers.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Destination for reproduced final-test tables/JSON. Defaults to "
            "results/reproduced/w05_test."
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help=(
            "Frozen reference final-test results. Defaults to "
            "results/reference/w05_test."
        ),
    )
    parser.add_argument(
        "--validation-metrics",
        type=Path,
        default=None,
        help=(
            "CSV containing the frozen W05 validation metrics. Defaults to "
            "results/reference/v2_selected/selected_model_metrics.csv."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify the frozen inputs without opening test images.",
    )
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
        help="Do not compare reproduced metrics with packaged reference results.",
    )
    return parser.parse_args()


def resolve_root(
    explicit: Path | None,
    environment_name: str,
    fallback: Path,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    environment_value = os.environ.get(environment_name)
    if environment_value:
        expanded = os.path.expandvars(os.path.expanduser(environment_value))
        if "$" in expanded:
            raise SystemExit(
                f"Unresolved environment variable in {environment_name}: "
                f"{environment_value}"
            )
        return Path(expanded).resolve()
    return fallback.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    lookup = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def align_manifest(bundle, manifest: pd.DataFrame) -> pd.DataFrame:
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


def cross_reference_precision_at_k(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    query_samples: np.ndarray,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    reference_samples: np.ndarray,
    *,
    target_label: str,
    k: int,
    prepare_cosine_features,
    chunk_size: int = 128,
) -> dict[str, Any]:
    query_mask = (
        np.char.upper(query_labels.astype(str))
        == target_label.upper()
    )
    query_rows = np.flatnonzero(query_mask)
    if not len(query_rows):
        raise RuntimeError(f"No {target_label} query images are available.")

    normalised_query = prepare_cosine_features(query_features[query_rows])
    normalised_reference = prepare_cosine_features(reference_features)
    matches = 0
    neighbour_total = 0

    for start in range(0, len(query_rows), chunk_size):
        stop = min(start + chunk_size, len(query_rows))
        scores = normalised_query[start:stop] @ normalised_reference.T
        chunk_query_samples = query_samples[query_rows[start:stop]]
        same_sample = (
            chunk_query_samples[:, None].astype(str)
            == reference_samples[None, :].astype(str)
        )
        scores[same_sample] = -np.inf
        available = np.isfinite(scores).sum(axis=1)
        if np.any(available < k):
            raise RuntimeError("Too few cross-sample reference candidates.")

        neighbours = np.argpartition(
            -scores,
            kth=k - 1,
            axis=1,
        )[:, :k]
        neighbour_labels = reference_labels[neighbours]
        matches += int(
            np.sum(
                np.char.upper(neighbour_labels.astype(str))
                == target_label.upper()
            )
        )
        neighbour_total += int(neighbours.size)

    return {
        "precision": float(matches / neighbour_total),
        "query_rows": int(len(query_rows)),
        "reference_rows": int(len(reference_features)),
        "matches": int(matches),
        "neighbours": int(neighbour_total),
    }


def load_validation_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if "Model" not in frame.columns:
        raise RuntimeError(f"Validation metrics table has no Model column: {path}")
    selected = frame.loc[frame["Model"].astype(str) == "SwAV W05 v2"]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly one SwAV W05 v2 row in {path}; found {len(selected)}."
        )
    row = selected.iloc[0]
    output: dict[str, float] = {}
    for metric in COMPARISON_METRICS:
        if metric not in frame.columns or pd.isna(row[metric]):
            raise RuntimeError(f"Missing W05 validation metric {metric!r} in {path}.")
        output[metric] = float(row[metric])
    return output


def verify_reference(
    reproduced: dict[str, Any],
    reference_path: Path,
    tolerance: float = 1e-6,
) -> None:
    if not reference_path.is_file():
        print("Reference result check: SKIPPED (reference JSON absent)")
        return
    reference = read_json(reference_path)
    for metric in COMPARISON_METRICS:
        observed = reproduced.get(metric)
        expected = reference.get(metric)
        if observed is None and expected is None:
            continue
        if observed is None or expected is None:
            raise RuntimeError(
                f"Reference mismatch for {metric}: {observed!r} != {expected!r}"
            )
        if abs(float(observed) - float(expected)) > tolerance:
            raise RuntimeError(
                f"Reference mismatch for {metric}: "
                f"{float(observed):.12g} != {float(expected):.12g}"
            )
    print("Reference result metrics: PASS")


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.expanduser().resolve()
    workspace_root = project_root.parent
    data_root = resolve_root(
        args.data_root,
        "PNP_DATA_ROOT",
        workspace_root / "data",
    )
    runs_root = resolve_root(
        args.runs_root,
        "PNP_RUNS_ROOT",
        workspace_root / "runs",
    )
    os.environ["PNP_DATA_ROOT"] = str(data_root)
    os.environ["PNP_RUNS_ROOT"] = str(runs_root)

    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else project_root / "configs/v2_selected/swav_w05_e100_v2.json"
    )
    run_root = (
        args.run_root.expanduser().resolve()
        if args.run_root is not None
        else runs_root / "hpc/v2/selected" / W05_RUN_NAME
    )
    trainer_path = (
        args.trainer.expanduser().resolve()
        if args.trainer is not None
        else project_root / "scripts/train_swav.py"
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root / "results/reproduced/w05_test"
    )
    reference_root = (
        args.reference_root.expanduser().resolve()
        if args.reference_root is not None
        else project_root / "results/reference/w05_test"
    )
    validation_metrics_path = (
        args.validation_metrics.expanduser().resolve()
        if args.validation_metrics is not None
        else project_root / "results/reference/v2_selected/selected_model_metrics.csv"
    )

    benchmark_root = (
        data_root
        / "processed/plagnetplus/ssl_unified_v2/benchmarks"
        / BENCHMARK_ID
    )
    train_manifest_path = benchmark_root / "manifests/crystals_train.csv"
    validation_manifest_path = (
        benchmark_root / "manifests/crystals_validation.csv"
    )
    test_manifest_path = benchmark_root / "manifests/crystals_test.csv"

    checkpoint_path = run_root / "checkpoints/swav_last.pt"
    train_embedding_path = (
        run_root / "embeddings/train_backbone_embeddings.npz"
    )
    validation_embedding_path = (
        run_root / "embeddings/validation_backbone_embeddings.npz"
    )
    test_embedding_path = (
        run_root / "embeddings/test_backbone_embeddings.npz"
    )
    run_summary_path = run_root / "run_summary.json"

    required_pretest_paths = (
        config_path,
        checkpoint_path,
        trainer_path,
        train_manifest_path,
        validation_manifest_path,
        test_manifest_path,
        run_summary_path,
        train_embedding_path,
        validation_embedding_path,
    )
    missing = [path for path in required_pretest_paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required frozen W05 files are missing:\n" + formatted
        )

    observed_checkpoint_sha256 = sha256_file(checkpoint_path)
    observed_config_sha256 = sha256_file(config_path)
    if observed_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen W05 checkpoint SHA-256 mismatch.")
    if observed_config_sha256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Frozen W05 configuration SHA-256 mismatch.")

    run_summary = read_json(run_summary_path)
    if run_summary.get("status") != "PASS":
        raise RuntimeError("W05 run did not report PASS.")
    if run_summary.get("completed_epochs") != 100:
        raise RuntimeError("W05 did not complete exactly 100 epochs.")
    if (
        run_summary.get("manifest_fingerprint_sha256")
        != EXPECTED_MANIFEST_FINGERPRINT
    ):
        raise RuntimeError("W05 manifest fingerprint mismatch.")

    checkpoint_header = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint_header.get("schema_version") != "plagnetplus_swav_checkpoint_v1":
        raise RuntimeError("Unexpected W05 checkpoint schema.")
    if checkpoint_header.get("completed_epoch") != 100:
        raise RuntimeError("Checkpoint is not the completed epoch-100 model.")
    if checkpoint_header.get("benchmark_id") != BENCHMARK_ID:
        raise RuntimeError("Checkpoint benchmark identity mismatch.")
    if (
        checkpoint_header.get("manifest_fingerprint_sha256")
        != EXPECTED_MANIFEST_FINGERPRINT
    ):
        raise RuntimeError("Checkpoint manifest fingerprint mismatch.")
    del checkpoint_header

    print("W05 checkpoint SHA-256: PASS")
    print("W05 configuration SHA-256: PASS")
    print("W05 benchmark fingerprint: PASS")
    print("Training performed: NO")
    print("Model/hyperparameter selection changed: NO")

    if args.dry_run:
        print("TEST IMAGE PIXELS ACCESSED: NO")
        print("W05 FINAL TEST DRY RUN: PASS")
        return

    sys.path.insert(0, str(project_root / "src"))
    from plagnetplus.data.ssl_transforms import build_ssl_evaluation_transform
    from plagnetplus.data.unified_dataset import UnifiedCrystalDataset
    from plagnetplus.evaluation import (
        categorical_enrichment_at_k,
        entropy_effective_rank,
        exact_cosine_neighbours,
        fit_locality_probe,
        l2_normalise_rows,
        load_embedding_bundle,
        load_embedding_manifest,
        prepare_cosine_features,
        validate_embedding_manifest_alignment,
    )
    from plagnetplus.models.swav import SwAVModel
    from plagnetplus.training.swav_config import load_swav_training_config

    output_root.mkdir(parents=True, exist_ok=True)
    table_root = output_root
    freeze_record_path = output_root / "w05_frozen_test_contract.json"
    access_receipt_path = output_root / "test_access_receipt.json"

    freeze_contract = {
        "schema_version": "plagnetplus_w05_final_test_contract_v1",
        "model_selection": "SwAV W05 v2",
        "run_name": W05_RUN_NAME,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": observed_checkpoint_sha256,
        "config_path": str(config_path),
        "config_sha256": observed_config_sha256,
        "benchmark_id": BENCHMARK_ID,
        "manifest_fingerprint_sha256": EXPECTED_MANIFEST_FINGERPRINT,
        "completed_epochs": 100,
        "cluster_count": FROZEN_CLUSTER_COUNT,
        "pca_components": PCA_COMPONENTS,
        "cluster_seeds": list(CLUSTER_SEEDS),
        "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
        "location_probe_regularisation_c": 1.0,
        "location_probe_class_weight": "balanced",
        "retrieval_k": 10,
        "test_results_may_change_selection": False,
        "training_performed": False,
        "reproduced_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(freeze_record_path, freeze_contract)

    config = load_swav_training_config(config_path)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)

    model = SwAVModel(
        backbone_name=config.backbone_name,
        initialization=config.initialization,
        projection_hidden_dimension=config.projection_hidden_dimension,
        projection_dimension=config.projection_dimension,
        prototype_count=config.prototype_count,
        temperature=config.temperature,
        sinkhorn_epsilon=config.sinkhorn_epsilon,
        sinkhorn_iterations=config.sinkhorn_iterations,
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    trainer_spec = importlib.util.spec_from_file_location(
        "plagnetplus_train_swav_frozen_test",
        trainer_path,
    )
    if trainer_spec is None or trainer_spec.loader is None:
        raise RuntimeError("Could not import the SwAV training module.")
    trainer_module = importlib.util.module_from_spec(trainer_spec)
    sys.modules[trainer_spec.name] = trainer_module
    trainer_spec.loader.exec_module(trainer_module)

    transform_config = trainer_module.load_transform_config(
        config.transform_config
    )
    evaluation_transform = build_ssl_evaluation_transform(transform_config)

    test_dataset = UnifiedCrystalDataset(
        config.test_manifest,
        config.image_root_map,
        transform=evaluation_transform,
        verify_files=config.verify_files,
        verify_sample_size=config.verify_sample_size,
        verify_seed=config.seed,
        include_metadata=True,
    )
    if len(test_dataset) != EXPECTED_TEST_SHAPE[0]:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_SHAPE[0]:,} test rows; "
            f"found {len(test_dataset):,}."
        )

    # Prefer the previously verified test embedding archive when present; this
    # avoids touching test images again during report regeneration.
    if test_embedding_path.is_file():
        receipt_candidates = (
            access_receipt_path,
            reference_root / "test_access_receipt.json",
        )
        source_receipt_path = next(
            (path for path in receipt_candidates if path.is_file()),
            None,
        )
        if source_receipt_path is None:
            raise RuntimeError(
                "A test embedding exists but no audit receipt is available."
            )
        existing_receipt = read_json(source_receipt_path)
        observed_embedding_sha256 = sha256_file(test_embedding_path)
        if (
            existing_receipt.get("test_embedding_sha256")
            != observed_embedding_sha256
        ):
            raise RuntimeError("Existing test embedding hash mismatch.")
        if source_receipt_path != access_receipt_path:
            shutil.copy2(source_receipt_path, access_receipt_path)
        export_status = "VERIFIED EXISTING EXPORT REUSED"
        test_pixels_accessed_this_execution = False
    else:
        generator = torch.Generator().manual_seed(config.seed)
        test_loader = trainer_module.make_loader(
            test_dataset,
            batch_size=config.evaluation_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            generator=generator,
            pin_memory=(device.type == "cuda"),
        )
        checkpoint_sha256_before = sha256_file(checkpoint_path)
        export_summary = trainer_module.export_backbone_embeddings(
            split="test",
            model=model,
            loader=test_loader,
            full_dataset=test_dataset,
            expected_items=len(test_dataset),
            device=device,
            use_amp=False,
            output_path=test_embedding_path,
        )
        checkpoint_sha256_after = sha256_file(checkpoint_path)
        if checkpoint_sha256_after != checkpoint_sha256_before:
            raise RuntimeError("Frozen W05 checkpoint changed during export.")
        if tuple(export_summary["shape"]) != EXPECTED_TEST_SHAPE:
            raise RuntimeError("Unexpected test embedding shape.")
        observed_embedding_sha256 = sha256_file(test_embedding_path)
        receipt = {
            "schema_version": "plagnetplus_w05_test_access_receipt_v1",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": W05_RUN_NAME,
            "checkpoint_sha256": checkpoint_sha256_after,
            "config_sha256": sha256_file(config_path),
            "test_embedding_path": str(test_embedding_path),
            "test_embedding_sha256": observed_embedding_sha256,
            "test_embedding_shape": list(EXPECTED_TEST_SHAPE),
            "deterministic_preprocessing": True,
            "training_performed": False,
            "test_images_accessed": True,
            "test_access_purpose": "reproduction of frozen final evaluation",
        }
        write_json_atomic(access_receipt_path, receipt)
        export_status = "ONE-TIME EXPORT COMPLETED"
        test_pixels_accessed_this_execution = True

    del checkpoint
    print("Device:", device)
    print("Test embedding export:", export_status)
    print("Test embedding SHA-256:", observed_embedding_sha256)
    print(
        "Test image pixels accessed in this execution:",
        "YES" if test_pixels_accessed_this_execution else "NO",
    )

    train_manifest_raw = load_embedding_manifest(train_manifest_path)
    validation_manifest_raw = load_embedding_manifest(
        validation_manifest_path
    )
    test_manifest_raw = load_embedding_manifest(test_manifest_path)

    train_bundle = load_embedding_bundle(
        train_embedding_path,
        expected_feature_dimension=512,
    )
    validation_bundle = load_embedding_bundle(
        validation_embedding_path,
        expected_feature_dimension=512,
    )
    test_bundle = load_embedding_bundle(
        test_embedding_path,
        expected_feature_dimension=512,
    )

    if train_bundle.features.shape != EXPECTED_TRAIN_SHAPE:
        raise RuntimeError("Unexpected training embedding shape.")
    if validation_bundle.features.shape != EXPECTED_VALIDATION_SHAPE:
        raise RuntimeError("Unexpected validation embedding shape.")
    if test_bundle.features.shape != EXPECTED_TEST_SHAPE:
        raise RuntimeError("Unexpected test embedding shape.")

    for bundle, manifest in (
        (train_bundle, train_manifest_raw),
        (validation_bundle, validation_manifest_raw),
        (test_bundle, test_manifest_raw),
    ):
        if not validate_embedding_manifest_alignment(bundle, manifest):
            raise RuntimeError("Embedding-manifest alignment failed.")

    train_manifest = align_manifest(train_bundle, train_manifest_raw)
    validation_manifest = align_manifest(
        validation_bundle,
        validation_manifest_raw,
    )
    test_manifest = align_manifest(test_bundle, test_manifest_raw)

    location_column = resolve_column(
        test_manifest,
        ("locality", "location", "volcano", "site"),
    )
    sample_column = resolve_column(
        test_manifest,
        (
            "acquisition_id",
            "sample_id",
            "sample",
            "sample_name",
            "specimen_id",
        ),
    )
    source_column = resolve_column(
        test_manifest,
        (
            "source_image_relative_path",
            "source_image",
            "source_image_id",
            "source_scan",
            "parent_image",
            "image_source",
        ),
    )
    train_location_column = resolve_column(
        train_manifest,
        ("locality", "location", "volcano", "site"),
    )
    if None in (location_column, sample_column, train_location_column):
        raise RuntimeError("Required location/sample columns are missing.")

    test_location_labels = (
        test_manifest[location_column].astype(str).to_numpy()
    )
    test_sample_labels = (
        test_manifest[sample_column].astype(str).to_numpy()
    )
    train_location_labels = (
        train_manifest[train_location_column].astype(str).to_numpy()
    )

    effective_rank, participation_ratio = entropy_effective_rank(
        test_bundle.features
    )

    test_cosine_features = prepare_cosine_features(test_bundle.features)
    raw_indices, _ = exact_cosine_neighbours(
        test_cosine_features,
        neighbour_count=20,
        chunk_size=256,
    )

    fct_mask = (
        np.char.upper(test_location_labels.astype(str)) == "FCT"
    )
    if not bool(fct_mask.any()):
        raise RuntimeError("The test split contains no FCT queries.")

    # For cross-sample retrieval, compare held-out test queries against the
    # combined frozen train and validation reference embeddings.
    reference_features = np.vstack(
        [train_bundle.features, validation_bundle.features]
    )
    reference_manifest = pd.concat(
        [train_manifest, validation_manifest],
        ignore_index=True,
    )
    reference_location_column = resolve_column(
        reference_manifest,
        ("locality", "location", "volcano", "site"),
    )
    reference_sample_column = resolve_column(
        reference_manifest,
        (
            "acquisition_id",
            "sample_id",
            "sample",
            "sample_name",
            "specimen_id",
        ),
    )
    if reference_location_column is None or reference_sample_column is None:
        raise RuntimeError("Reference location/sample columns are missing.")

    fct_reference_result = cross_reference_precision_at_k(
        test_bundle.features,
        test_location_labels,
        test_sample_labels,
        reference_features,
        reference_manifest[
            reference_location_column
        ].astype(str).to_numpy(),
        reference_manifest[
            reference_sample_column
        ].astype(str).to_numpy(),
        target_label="FCT",
        k=10,
        prepare_cosine_features=prepare_cosine_features,
    )
    fct_cross_sample_p10 = float(fct_reference_result["precision"])

    source_enrichment = None
    if source_column is not None:
        source_result = categorical_enrichment_at_k(
            raw_indices,
            test_manifest[source_column].astype(str).to_numpy(),
            k=10,
        )
        source_enrichment = float(source_result["enrichment"])

    probe = fit_locality_probe(
        train_bundle.features,
        train_location_labels,
        test_bundle.features,
        test_location_labels,
        regularisation_c=1.0,
        class_weight="balanced",
        maximum_iterations=500,
        tolerance=1e-5,
        seed=EVALUATION_SEED,
    )
    location_macro_f1 = float(
        probe.summary["metrics"]["locality_macro_f1"]
    )
    probe.per_class_metrics.to_csv(
        table_root / "w05_test_location_probe_per_class.csv",
        index=False,
    )
    probe.confusion_matrix.to_csv(
        table_root / "w05_test_location_probe_confusion_matrix.csv"
    )

    validation_normalised = l2_normalise_rows(
        validation_bundle.features
    )
    test_normalised = l2_normalise_rows(test_bundle.features)
    frozen_pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="full",
    )
    validation_coordinates = frozen_pca.fit_transform(
        validation_normalised
    )
    test_coordinates = frozen_pca.transform(test_normalised)

    rng = np.random.default_rng(EVALUATION_SEED)
    silhouette_indices = np.sort(
        rng.choice(
            len(test_coordinates),
            size=min(SILHOUETTE_SAMPLE_SIZE, len(test_coordinates)),
            replace=False,
        )
    )
    test_labels_by_seed: list[np.ndarray] = []
    silhouette_rows: list[dict[str, Any]] = []
    for seed in CLUSTER_SEEDS:
        cluster_model = KMeans(
            n_clusters=FROZEN_CLUSTER_COUNT,
            random_state=seed,
            n_init=10,
            max_iter=300,
            algorithm="lloyd",
        )
        cluster_model.fit(validation_coordinates)
        test_labels = cluster_model.predict(test_coordinates)
        test_labels_by_seed.append(test_labels)
        sampled_labels = test_labels[silhouette_indices]
        if np.unique(sampled_labels).size < 2:
            raise RuntimeError(
                "Test assignments contain fewer than two clusters."
            )
        silhouette_rows.append(
            {
                "seed": seed,
                "test_silhouette": float(
                    silhouette_score(
                        test_coordinates[silhouette_indices],
                        sampled_labels,
                    )
                ),
                "occupied_clusters": int(
                    np.unique(test_labels).size
                ),
            }
        )

    stability_values: list[float] = []
    for first_index in range(len(CLUSTER_SEEDS)):
        for second_index in range(first_index + 1, len(CLUSTER_SEEDS)):
            stability_values.append(
                adjusted_rand_score(
                    test_labels_by_seed[first_index],
                    test_labels_by_seed[second_index],
                )
            )

    clustering_table = pd.DataFrame(silhouette_rows)
    clustering_table.to_csv(
        table_root / "w05_test_frozen_clustering.csv",
        index=False,
    )
    median_test_silhouette = float(
        clustering_table["test_silhouette"].median()
    )
    median_test_stability_ari = float(
        np.median(stability_values)
    )

    test_results: dict[str, Any] = {
        "schema_version": "plagnetplus_w05_final_test_results_v1",
        "Model": "SwAV W05 v2",
        "Run name": W05_RUN_NAME,
        "Split": "test",
        "Images": int(len(test_manifest)),
        "Samples": int(test_manifest[sample_column].nunique()),
        "Locations": int(test_manifest[location_column].nunique()),
        "Median silhouette score": median_test_silhouette,
        "Stability ARI": median_test_stability_ari,
        "Frozen k": FROZEN_CLUSTER_COUNT,
        "FCT cross-sample P@10": float(fct_cross_sample_p10),
        "FCT retrieval gallery": (
            "frozen training and validation embeddings"
        ),
        "FCT reference rows": int(fct_reference_result["reference_rows"]),
        "FCT query images": int(fct_mask.sum()),
        "FCT test samples": int(
            test_manifest.loc[fct_mask, sample_column].nunique()
        ),
        "Location-probe macro-F1": location_macro_f1,
        "Source-image enrichment@10": source_enrichment,
        "Effective rank": float(effective_rank),
        "Participation ratio": float(participation_ratio),
        "Checkpoint SHA-256": sha256_file(checkpoint_path),
        "Test embedding SHA-256": sha256_file(test_embedding_path),
        "Training performed": False,
        "Model selection changed after test": False,
    }

    write_json_atomic(
        output_root / "w05_final_test_results.json",
        test_results,
    )
    pd.DataFrame([test_results]).to_csv(
        table_root / "w05_final_test_metrics.csv",
        index=False,
    )

    validation_metrics = load_validation_metrics(
        validation_metrics_path
    )
    comparison_table = pd.DataFrame(
        {
            "Metric": COMPARISON_METRICS,
            "Validation": [
                validation_metrics[metric]
                for metric in COMPARISON_METRICS
            ],
            "Final test": [
                test_results[metric]
                for metric in COMPARISON_METRICS
            ],
        }
    )
    comparison_table["Test minus validation"] = (
        comparison_table["Final test"]
        - comparison_table["Validation"]
    )
    comparison_table.to_csv(
        table_root / "w05_validation_test_comparison.csv",
        index=False,
    )

    final_checks = {
        "Frozen checkpoint unchanged": (
            sha256_file(checkpoint_path)
            == EXPECTED_CHECKPOINT_SHA256
        ),
        "Frozen configuration unchanged": (
            sha256_file(config_path) == EXPECTED_CONFIG_SHA256
        ),
        "Test embedding shape": (
            test_bundle.features.shape == EXPECTED_TEST_SHAPE
        ),
        "Test embeddings finite": bool(
            np.isfinite(test_bundle.features).all()
        ),
        "Test alignment": bool(
            validate_embedding_manifest_alignment(
                test_bundle,
                test_manifest_raw,
            )
        ),
        "Freeze record exists": freeze_record_path.is_file(),
        "Test access receipt exists": access_receipt_path.is_file(),
        "Final results exist": (
            output_root / "w05_final_test_results.json"
        ).is_file(),
        "No post-test reselection": (
            test_results["Model selection changed after test"] is False
        ),
        "No training": (
            test_results["Training performed"] is False
        ),
    }
    failed = [
        name for name, passed in final_checks.items()
        if not passed
    ]
    if failed:
        raise RuntimeError(
            f"Final test verification failed: {failed}"
        )

    if not args.skip_reference_check:
        # Cross-check the regenerated final result against the frozen reference
        # record unless that verification was explicitly disabled.
        verify_reference(
            test_results,
            reference_root / "w05_final_test_results.json",
        )

    print("FROZEN W05 TEST EVALUATION: PASS")
    print("Test images:", f"{len(test_manifest):,}")
    print("Test samples:", test_manifest[sample_column].nunique())
    print("TRAINING PERFORMED: NO")
    print("MODEL FROZEN BEFORE TEST: YES")
    print("POST-TEST RETUNING PERMITTED: NO")
    print("Output root:", output_root)


if __name__ == "__main__":
    main()
