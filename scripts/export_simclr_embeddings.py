#!/usr/bin/env python3
"""Export deterministic SimCLR backbone embeddings for any frozen split."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
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
from plagnetplus.evaluation import sha256_file
from plagnetplus.models.simclr import SimCLRModel


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--image-root-map",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--transform-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--split",
        required=True,
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--expected-benchmark-id",
        required=True,
    )
    parser.add_argument(
        "--expected-manifest-fingerprint",
        required=True,
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
    )
    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--verify-files",
        choices=[
            "none",
            "sample",
            "all",
        ],
        default="sample",
    )
    parser.add_argument(
        "--verify-sample-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--limit-items",
        type=int,
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".partial"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (
        torch.initial_seed()
        + worker_id
    ) % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device("cuda")

    return torch.device("cpu")


def load_transform_config(
    path: Path,
) -> SSLTransformConfig:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    blur_sigma = payload[
        "blur_sigma"
    ]

    return SSLTransformConfig(
        image_size=int(
            payload["image_size"]
        ),
        horizontal_flip_probability=float(
            payload[
                "horizontal_flip_probability"
            ]
        ),
        vertical_flip_probability=float(
            payload[
                "vertical_flip_probability"
            ]
        ),
        jitter_probability=float(
            payload["jitter_probability"]
        ),
        brightness=float(
            payload["brightness"]
        ),
        contrast=float(
            payload["contrast"]
        ),
        blur_probability=float(
            payload["blur_probability"]
        ),
        blur_kernel_size=int(
            payload["blur_kernel_size"]
        ),
        blur_sigma_min=float(
            blur_sigma[0]
        ),
        blur_sigma_max=float(
            blur_sigma[1]
        ),
        interpolation=str(
            payload["interpolation"]
        ),
        normalization_mean=tuple(
            float(value)
            for value in payload[
                "normalization_mean"
            ]
        ),
        normalization_std=tuple(
            float(value)
            for value in payload[
                "normalization_std"
            ]
        ),
    )


def metadata_values(
    value: object,
) -> list[object]:
    if isinstance(
        value,
        torch.Tensor,
    ):
        result = value.detach().cpu().tolist()

        if isinstance(result, list):
            return result

        return [result]

    if isinstance(value, np.ndarray):
        result = value.tolist()

        if isinstance(result, list):
            return result

        return [result]

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return list(value)

    return [value]


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if num_workers > 0:
        arguments[
            "prefetch_factor"
        ] = 2

    return DataLoader(**arguments)


def build_model(
    checkpoint: dict[str, Any],
) -> SimCLRModel:
    config = checkpoint["config"]

    model = SimCLRModel(
        backbone_name=str(
            config["backbone_name"]
        ),
        projection_dimension=int(
            config[
                "projection_dimension"
            ]
        ),
        projection_hidden_dimension=int(
            config[
                "projection_hidden_dimension"
            ]
        ),
        temperature=float(
            config["temperature"]
        ),
        initialization=str(
            config["initialization"]
        ),
    )

    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Strict model-state loading produced "
            "incompatible keys."
        )

    return model


def main() -> None:
    args = parse_arguments()

    for path in [
        args.checkpoint,
        args.manifest,
        args.image_root_map,
        args.transform_config,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.batch_size < 1:
        raise ValueError(
            "batch-size must be positive."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers cannot be negative."
        )

    seed_everything(args.seed)

    checkpoint_sha256 = sha256_file(
        args.checkpoint
    )

    manifest_sha256 = sha256_file(
        args.manifest
    )

    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha256
        != args.expected_checkpoint_sha256
    ):
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch."
        )

    if (
        args.expected_manifest_sha256
        and manifest_sha256
        != args.expected_manifest_sha256
    ):
        raise RuntimeError(
            "Manifest SHA-256 mismatch."
        )

    # Load the frozen checkpoint for inference only; this script exports
    # representations and never resumes optimisation.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint must contain a dictionary."
        )

    if (
        checkpoint.get("schema_version")
        != "plagnetplus_simclr_checkpoint_v1"
    ):
        raise RuntimeError(
            "Unexpected checkpoint schema."
        )

    if (
        checkpoint.get("benchmark_id")
        != args.expected_benchmark_id
    ):
        raise RuntimeError(
            "Checkpoint benchmark mismatch."
        )

    if (
        checkpoint.get(
            "manifest_fingerprint_sha256"
        )
        != args.expected_manifest_fingerprint
    ):
        raise RuntimeError(
            "Checkpoint manifest fingerprint mismatch."
        )

    transform_config = (
        load_transform_config(
            args.transform_config
        )
    )

    evaluation_transform = (
        build_ssl_evaluation_transform(
            transform_config
        )
    )

    full_dataset = UnifiedCrystalDataset(
        args.manifest,
        args.image_root_map,
        data_root=os.environ.get(
            "PNP_DATA_ROOT"
        ),
        transform=evaluation_transform,
        verify_files=args.verify_files,
        verify_sample_size=(
            args.verify_sample_size
        ),
        verify_seed=args.seed,
        include_metadata=True,
    )

    if len(full_dataset) != args.expected_rows:
        raise RuntimeError(
            "Dataset row count mismatch: "
            f"{len(full_dataset)} != "
            f"{args.expected_rows}."
        )

    expected_items = (
        len(full_dataset)
        if args.limit_items is None
        else min(
            int(args.limit_items),
            len(full_dataset),
        )
    )

    if expected_items < 1:
        raise ValueError(
            "No items were selected."
        )

    extraction_dataset: Dataset = (
        full_dataset
        if expected_items
        == len(full_dataset)
        else Subset(
            full_dataset,
            range(expected_items),
        )
    )

    device = resolve_device(
        args.device
    )

    loader = make_loader(
        extraction_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 2,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = build_model(
        checkpoint
    ).to(device)

    model.eval()

    amp_enabled = bool(
        args.use_amp
        and device.type == "cuda"
    )

    feature_blocks: list[
        np.ndarray
    ] = []

    crystal_ids: list[str] = []
    manifest_orders: list[int] = []

    started = time.perf_counter()

    # Embedding export uses deterministic evaluation preprocessing and disables
    # gradient tracking to keep inference repeatable and lightweight.
    with torch.inference_mode():
        for batch_number, batch in enumerate(
            loader,
            start=1,
        ):
            images = batch["image"].to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                features = model.encode(
                    images
                )

            if not torch.isfinite(
                features
            ).all():
                raise RuntimeError(
                    "Encountered non-finite embeddings."
                )

            feature_blocks.append(
                features.detach()
                .to(
                    dtype=torch.float32
                )
                .cpu()
                .numpy()
            )

            metadata = batch["metadata"]

            crystal_ids.extend(
                str(value)
                for value in metadata_values(
                    metadata["crystal_id"]
                )
            )

            manifest_orders.extend(
                int(value)
                for value in metadata_values(
                    metadata[
                        "manifest_order"
                    ]
                )
            )

            if (
                batch_number == 1
                or batch_number % 50 == 0
                or len(crystal_ids)
                == expected_items
            ):
                print(
                    f"embedded={len(crystal_ids)}/"
                    f"{expected_items}"
                )

    feature_array = np.concatenate(
        feature_blocks,
        axis=0,
    )

    expected_ids = (
        full_dataset.manifest[
            "crystal_id"
        ]
        .iloc[:expected_items]
        .astype(str)
        .tolist()
    )

    expected_orders = (
        full_dataset.manifest[
            "manifest_order"
        ]
        .iloc[:expected_items]
        .astype(int)
        .tolist()
    )

    if feature_array.shape != (
        expected_items,
        model.feature_dimension,
    ):
        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{feature_array.shape}."
        )

    if crystal_ids != expected_ids:
        raise RuntimeError(
            "Crystal IDs are misaligned."
        )

    if manifest_orders != expected_orders:
        raise RuntimeError(
            "Manifest orders are misaligned."
        )

    if not np.isfinite(
        feature_array
    ).all():
        raise RuntimeError(
            "Export contains non-finite values."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = (
        args.output.with_suffix(
            args.output.suffix
            + ".partial"
        )
    )

    with temporary_output.open(
        "wb"
    ) as handle:
        np.savez_compressed(
            handle,
            features=feature_array,
            crystal_ids=np.asarray(
                crystal_ids,
                dtype=str,
            ),
            manifest_orders=np.asarray(
                manifest_orders,
                dtype=np.int64,
            ),
        )

    temporary_output.replace(
        args.output
    )

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    summary = {
        "schema_version": (
            "plagnetplus_simclr_embedding_export_v1"
        ),
        "status": "PASS",
        "split": args.split,
        "limited_export": (
            expected_items
            != len(full_dataset)
        ),
        "checkpoint": {
            "path": str(
                args.checkpoint.resolve()
            ),
            "sha256": checkpoint_sha256,
            "completed_epoch": int(
                checkpoint[
                    "completed_epoch"
                ]
            ),
        },
        "manifest": {
            "path": str(
                args.manifest.resolve()
            ),
            "sha256": manifest_sha256,
            "full_rows": int(
                len(full_dataset)
            ),
            "exported_rows": int(
                expected_items
            ),
        },
        "benchmark": {
            "benchmark_id": (
                args.expected_benchmark_id
            ),
            "manifest_fingerprint_sha256": (
                args
                .expected_manifest_fingerprint
            ),
        },
        "model": {
            **model.model_metadata(),
        },
        "transform": {
            "path": str(
                args.transform_config.resolve()
            ),
            "sha256": sha256_file(
                args.transform_config
            ),
            "deterministic": True,
        },
        "image_root_map": {
            "path": str(
                args.image_root_map.resolve()
            ),
            "sha256": sha256_file(
                args.image_root_map
            ),
        },
        "runtime": {
            "device": str(device),
            "amp_enabled": amp_enabled,
            "batch_size": int(
                args.batch_size
            ),
            "num_workers": int(
                args.num_workers
            ),
            "seconds": float(
                elapsed_seconds
            ),
        },
        "output": {
            "path": str(
                args.output.resolve()
            ),
            "sha256": sha256_file(
                args.output
            ),
            "shape": list(
                feature_array.shape
            ),
            "dtype": str(
                feature_array.dtype
            ),
            "all_finite": True,
            "ids_aligned": True,
            "manifest_orders_aligned": True,
        },
    }

    write_json_atomic(
        args.summary,
        summary,
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    print()
    print(
        "SIMCLR EMBEDDING EXPORT: PASS"
    )


if __name__ == "__main__":
    main()
