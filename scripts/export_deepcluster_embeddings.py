#!/usr/bin/env python3
"""Export deterministic backbone embeddings from a DeepCluster checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from plagnetplus.data.ssl_transforms import (
    SSLTransformConfig,
    build_ssl_evaluation_transform,
)
from plagnetplus.data.unified_dataset import UnifiedCrystalDataset
from plagnetplus.models import DeepClusterModel
from plagnetplus.training.deepcluster_engine import extract_backbone_features

EXPECTED_BENCHMARK_ID = "pooled_broad_p80_10_10_v1"
EXPECTED_BENCHMARK_FINGERPRINT = (
    "a2814f1c33b5adc45e9d4c8c90c7af"
    "7577e237ca74d112473a274dbd949ab582"
)
EXPECTED_CHECKPOINT_SCHEMA = "plagnetplus_deepcluster_checkpoint_v1"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export deterministic train or validation backbone embeddings "
            "from a frozen DeepCluster checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root-map", type=Path, required=True)
    parser.add_argument("--transform-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument(
        "--expected-benchmark-id",
        default=EXPECTED_BENCHMARK_ID,
    )
    parser.add_argument(
        "--expected-benchmark-fingerprint",
        default=EXPECTED_BENCHMARK_FINGERPRINT,
    )
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verify-files", default="sample")
    parser.add_argument("--verify-sample-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_npz_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    normalised = requested.strip().lower()
    if normalised == "auto":
        normalised = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalised)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def load_transform_config(path: Path) -> SSLTransformConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    blur_sigma = payload["blur_sigma"]
    return SSLTransformConfig(
        image_size=int(payload["image_size"]),
        horizontal_flip_probability=float(payload["horizontal_flip_probability"]),
        vertical_flip_probability=float(payload["vertical_flip_probability"]),
        jitter_probability=float(payload["jitter_probability"]),
        brightness=float(payload["brightness"]),
        contrast=float(payload["contrast"]),
        blur_probability=float(payload["blur_probability"]),
        blur_kernel_size=int(payload["blur_kernel_size"]),
        blur_sigma_min=float(blur_sigma[0]),
        blur_sigma_max=float(blur_sigma[1]),
        interpolation=str(payload["interpolation"]),
        normalization_mean=tuple(
            float(value) for value in payload["normalization_mean"]
        ),
        normalization_std=tuple(
            float(value) for value in payload["normalization_std"]
        ),
    )


def limited_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("limit-items must be positive.")
    return Subset(dataset, range(min(limit, len(dataset))))


def manifest_arrays(
    dataset: UnifiedCrystalDataset,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = dataset.manifest
    if limit is not None:
        frame = frame.iloc[: min(limit, len(frame))]
    crystal_ids = frame["crystal_id"].astype(str).to_numpy()
    manifest_orders = frame["manifest_order"].astype(np.int64).to_numpy()
    return crystal_ids, manifest_orders


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": None,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        arguments["prefetch_factor"] = 2
    return DataLoader(**arguments)


def validate_checkpoint(
    checkpoint: object,
    *,
    expected_benchmark_id: str,
    expected_fingerprint: str,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError("DeepCluster checkpoint must be a mapping.")
    required = {
        "schema_version",
        "cycle",
        "config",
        "model_state_dict",
        "benchmark_fingerprint",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError("Checkpoint is missing keys: " + ", ".join(missing))
    if checkpoint["schema_version"] != EXPECTED_CHECKPOINT_SCHEMA:
        raise RuntimeError(
            "Unexpected checkpoint schema: "
            f"{checkpoint['schema_version']!r}"
        )
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise TypeError("Checkpoint config must be a mapping.")
    if str(config.get("benchmark_id")) != expected_benchmark_id:
        raise RuntimeError("Checkpoint benchmark ID mismatch.")
    if str(checkpoint["benchmark_fingerprint"]) != expected_fingerprint:
        raise RuntimeError("Checkpoint benchmark fingerprint mismatch.")
    if int(checkpoint["cycle"]) < 1:
        raise RuntimeError("Checkpoint cycle must be positive.")
    return checkpoint


def main() -> None:
    arguments = parse_arguments()
    checkpoint_path = resolve_path(arguments.checkpoint)
    manifest_path = resolve_path(arguments.manifest)
    image_root_map_path = resolve_path(arguments.image_root_map)
    transform_config_path = resolve_path(arguments.transform_config)
    output_path = resolve_path(arguments.output)
    data_root = (
        resolve_path(arguments.data_root)
        if arguments.data_root is not None
        else None
    )

    if output_path.suffix.lower() != ".npz":
        raise ValueError("Output path must end in .npz.")
    summary_path = output_path.with_suffix(".summary.json")
    if not arguments.overwrite and (output_path.exists() or summary_path.exists()):
        raise FileExistsError(
            "Output already exists. Use --overwrite to replace it."
        )
    for required_path in (
        checkpoint_path,
        manifest_path,
        image_root_map_path,
        transform_config_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    if "test" in manifest_path.stem.lower():
        raise RuntimeError(
            "Test-manifest export is forbidden during model development."
        )

    seed_everything(arguments.seed)
    device = resolve_device(arguments.device)
    # Load the completed DeepCluster checkpoint and rebuild only the feature
    # extractor needed for deterministic embedding export.
    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = validate_checkpoint(
        raw_checkpoint,
        expected_benchmark_id=arguments.expected_benchmark_id,
        expected_fingerprint=arguments.expected_benchmark_fingerprint,
    )
    config = checkpoint["config"]

    model = DeepClusterModel(
        backbone_name=str(config["backbone_name"]),
        cluster_count=int(config["cluster_count"]),
        initialization=str(config["initialization"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    transform_config = load_transform_config(transform_config_path)
    transform = build_ssl_evaluation_transform(transform_config)
    dataset_full = UnifiedCrystalDataset(
        manifest_path,
        image_root_map_path,
        data_root=data_root,
        transform=transform,
        sort_by="manifest_order",
        verify_files=arguments.verify_files,
        verify_sample_size=arguments.verify_sample_size,
        verify_seed=arguments.seed,
        include_metadata=True,
    )
    if len(dataset_full) != arguments.expected_rows:
        raise RuntimeError(
            "Unexpected full manifest size: "
            f"observed={len(dataset_full)}, expected={arguments.expected_rows}"
        )

    dataset = limited_dataset(dataset_full, arguments.limit_items)
    expected_crystal_ids, expected_manifest_orders = manifest_arrays(
        dataset_full,
        arguments.limit_items,
    )
    loader = make_loader(
        dataset,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=device,
        seed=arguments.seed + 90_000,
    )
    use_amp = bool(arguments.amp and device.type == "cuda")
    # Export backbone features in manifest order without performing another
    # clustering or pseudo-label training cycle.
    result = extract_backbone_features(
        model=model,
        loader=loader,
        expected_crystal_ids=expected_crystal_ids,
        expected_manifest_orders=expected_manifest_orders,
        device=device,
        use_amp=use_amp,
    )

    features = np.asarray(result.features)
    crystal_ids = np.asarray(result.crystal_ids)
    manifest_orders = np.asarray(result.manifest_orders)
    expected_export_rows = len(dataset)
    checks = {
        "checkpoint_schema_expected": (
            checkpoint["schema_version"] == EXPECTED_CHECKPOINT_SCHEMA
        ),
        "benchmark_id_expected": (
            str(config["benchmark_id"]) == arguments.expected_benchmark_id
        ),
        "benchmark_fingerprint_expected": (
            str(checkpoint["benchmark_fingerprint"])
            == arguments.expected_benchmark_fingerprint
        ),
        "row_count_expected": features.shape[0] == expected_export_rows,
        "embedding_dimension_expected": (
            features.ndim == 2 and features.shape[1] == 512
        ),
        "features_finite": bool(np.isfinite(features).all()),
        "crystal_ids_aligned": bool(
            np.array_equal(crystal_ids, expected_crystal_ids)
        ),
        "manifest_orders_aligned": bool(
            np.array_equal(manifest_orders, expected_manifest_orders)
        ),
        "test_split_untouched": arguments.split in {"train", "validation"},
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "Embedding export checks failed: " + ", ".join(failed)
        )

    save_npz_atomic(
        output_path,
        features=features,
        crystal_ids=crystal_ids,
        manifest_orders=manifest_orders,
    )
    summary = {
        "schema_version": "plagnetplus_deepcluster_embedding_export_v1",
        "status": "PASS",
        "created_utc": (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        ),
        "model_family": "deepcluster",
        "model_id": f"deepcluster_cycle_{int(checkpoint['cycle']):03d}",
        "split": arguments.split,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "cycle": int(checkpoint["cycle"]),
            "schema_version": checkpoint["schema_version"],
        },
        "benchmark": {
            "benchmark_id": arguments.expected_benchmark_id,
            "manifest_fingerprint_sha256": (
                arguments.expected_benchmark_fingerprint
            ),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "full_rows": len(dataset_full),
            "exported_rows": expected_export_rows,
        },
        "image_root_map": {
            "path": str(image_root_map_path),
            "sha256": sha256_file(image_root_map_path),
        },
        "transform": {
            "path": str(transform_config_path),
            "sha256": sha256_file(transform_config_path),
            "config": transform_config.to_dict(),
        },
        "model": {
            "backbone_name": str(config["backbone_name"]),
            "initialization": str(config["initialization"]),
            "cluster_count": int(config["cluster_count"]),
            "embedding_dimension": int(features.shape[1]),
        },
        "runtime": {
            "device": str(device),
            "amp_enabled": use_amp,
            "batch_size": arguments.batch_size,
            "num_workers": arguments.num_workers,
            "seed": arguments.seed,
            "limited_export": arguments.limit_items is not None,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "shape": [int(value) for value in features.shape],
            "dtype": str(features.dtype),
        },
        "checks": checks,
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print()
    print("DEEPCLUSTER EMBEDDING EXPORT: PASS")


if __name__ == "__main__":
    main()
