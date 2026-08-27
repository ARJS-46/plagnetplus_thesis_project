#!/usr/bin/env python3

"""Train SimCLR on a frozen PlagNetPlus benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from plagnetplus.data.ssl_transforms import (
    SSLTransformConfig,
    build_ssl_evaluation_transform,
    build_ssl_train_transform,
)
from plagnetplus.data.transforms import MultiViewTransform
from plagnetplus.data.unified_dataset import UnifiedCrystalDataset
from plagnetplus.models import SimCLRModel
from plagnetplus.training import (
    evaluate_contrastive_epoch,
    learning_rate_for_epoch,
    load_checkpoint,
    load_simclr_training_config,
    save_checkpoint_atomic,
    train_one_epoch,
    write_history_csv_atomic,
)
from plagnetplus.training.provenance import save_run_record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit-train-items",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit-validation-items",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--export-validation-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def write_json_atomic(
    path: Path,
    payload: object,
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
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(
    worker_id: int,
) -> None:
    del worker_id

    worker_seed = (
        torch.initial_seed()
        % 2**32
    )

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if torch.cuda.is_available():
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
            payload[
                "jitter_probability"
            ]
        ),
        brightness=float(
            payload["brightness"]
        ),
        contrast=float(
            payload["contrast"]
        ),
        blur_probability=float(
            payload[
                "blur_probability"
            ]
        ),
        blur_kernel_size=int(
            payload[
                "blur_kernel_size"
            ]
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
            for value
            in payload[
                "normalization_mean"
            ]
        ),
        normalization_std=tuple(
            float(value)
            for value
            in payload[
                "normalization_std"
            ]
        ),
    )


def manifest_row_count(
    path: Path,
) -> int:
    frame = pd.read_csv(
        path,
        usecols=["crystal_id"],
        low_memory=False,
    )

    return len(frame)


def combined_manifest_fingerprint(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
) -> tuple[str, dict[str, object]]:
    paths = {
        "train": train_manifest,
        "validation": (
            validation_manifest
        ),
        "test": test_manifest,
    }

    details: dict[
        str,
        object,
    ] = {}

    digest = hashlib.sha256()

    for split in (
        "train",
        "validation",
        "test",
    ):
        path = paths[split]
        file_hash = sha256_file(path)
        rows = manifest_row_count(
            path
        )

        details[split] = {
            "path": str(path),
            "sha256": file_hash,
            "rows": rows,
        }

        digest.update(
            split.encode("utf-8")
        )

        digest.update(
            file_hash.encode(
                "utf-8"
            )
        )

        digest.update(
            str(rows).encode(
                "utf-8"
            )
        )

    return (
        digest.hexdigest(),
        details,
    )


def limited_dataset(
    dataset: Dataset,
    limit: int | None,
) -> Dataset:
    if limit is None:
        return dataset

    if limit <= 0:
        raise ValueError(
            "Dataset limits must be positive."
        )

    item_count = min(
        limit,
        len(dataset),
    )

    return Subset(
        dataset,
        range(item_count),
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    generator: torch.Generator,
    pin_memory: bool,
) -> DataLoader:
    arguments: dict[
        str,
        object,
    ] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
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

    return DataLoader(
        **arguments
    )


def metadata_values(
    value: object,
) -> list[object]:
    if isinstance(
        value,
        torch.Tensor,
    ):
        return value.detach().cpu().tolist()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        (tuple, list),
    ):
        return list(value)

    return [value]


def export_validation_embeddings(
    *,
    model: SimCLRModel,
    loader: DataLoader,
    full_dataset: UnifiedCrystalDataset,
    expected_items: int,
    device: torch.device,
    use_amp: bool,
    output_path: Path,
) -> dict[str, object]:
    model.eval()

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    feature_blocks: list[
        np.ndarray
    ] = []

    crystal_ids: list[str] = []
    manifest_orders: list[int] = []

    started = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            images = batch[
                "image"
            ].to(
                device,
                non_blocking=(
                    device.type
                    == "cuda"
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
                features
                .detach()
                .to(
                    dtype=torch.float32
                )
                .cpu()
                .numpy()
            )

            metadata = batch[
                "metadata"
            ]

            crystal_ids.extend(
                str(value)
                for value in metadata_values(
                    metadata[
                        "crystal_id"
                    ]
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

    feature_array = np.concatenate(
        feature_blocks,
        axis=0,
    )

    if feature_array.shape[0] != (
        expected_items
    ):
        raise RuntimeError(
            "Embedding row count mismatch: "
            f"{feature_array.shape[0]} != "
            f"{expected_items}."
        )

    expected_ids = (
        full_dataset
        .manifest[
            "crystal_id"
        ]
        .iloc[:expected_items]
        .astype(str)
        .tolist()
    )

    expected_orders = (
        full_dataset
        .manifest[
            "manifest_order"
        ]
        .iloc[:expected_items]
        .astype(int)
        .tolist()
    )

    if crystal_ids != expected_ids:
        raise RuntimeError(
            "Embedding crystal IDs are misaligned."
        )

    if manifest_orders != (
        expected_orders
    ):
        raise RuntimeError(
            "Embedding manifest ordering is misaligned."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_path.with_suffix(
        output_path.suffix
        + ".partial"
    )

    with temporary.open(
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

    temporary.replace(
        output_path
    )

    return {
        "path": str(
            output_path
        ),
        "sha256": sha256_file(
            output_path
        ),
        "shape": list(
            feature_array.shape
        ),
        "all_finite": bool(
            np.isfinite(
                feature_array
            ).all()
        ),
        "ids_aligned": True,
        "manifest_orders_aligned": True,
        "seconds": (
            time.perf_counter()
            - started
        ),
    }


def model_parameter_count(
    model: torch.nn.Module,
) -> dict[str, int]:
    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": int(total),
        "trainable": int(
            trainable
        ),
    }


def history_dicts(
    rows: Iterable[object],
) -> list[dict[str, object]]:
    output = []

    for row in rows:
        if isinstance(
            row,
            dict,
        ):
            output.append(
                dict(row)
            )
        else:
            output.append(
                row.to_dict()
            )

    return output


def main() -> None:
    arguments = parse_arguments()

    config = (
        load_simclr_training_config(
            arguments.config
        )
    )

    replacements: dict[
        str,
        object,
    ] = {}

    if arguments.run_name:
        replacements[
            "run_name"
        ] = arguments.run_name

    if arguments.epochs is not None:
        replacements[
            "epochs"
        ] = arguments.epochs

        if (
            config.warmup_epochs
            > arguments.epochs
        ):
            replacements[
                "warmup_epochs"
            ] = min(
                1,
                arguments.epochs,
            )

    if arguments.batch_size is not None:
        replacements[
            "batch_size"
        ] = arguments.batch_size

    if (
        arguments
        .evaluation_batch_size
        is not None
    ):
        replacements[
            "evaluation_batch_size"
        ] = (
            arguments
            .evaluation_batch_size
        )

    if arguments.num_workers is not None:
        replacements[
            "num_workers"
        ] = arguments.num_workers

    if replacements:
        config = replace(
            config,
            **replacements,
        )

    config.validate_input_paths()

    device = resolve_device(
        arguments.device
    )

    seed_everything(
        config.seed
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    # Fingerprint the frozen train/validation/test manifests before training
    # so the run record can prove which benchmark definition was used.
    fingerprint, manifest_details = (
        combined_manifest_fingerprint(
            train_manifest=(
                config.train_manifest
            ),
            validation_manifest=(
                config
                .validation_manifest
            ),
            test_manifest=(
                config.test_manifest
            ),
        )
    )

    expected_counts = {
        "train": (
            config.expected_train_rows
        ),
        "validation": (
            config
            .expected_validation_rows
        ),
        "test": (
            config.expected_test_rows
        ),
    }

    for split, expected in (
        expected_counts.items()
    ):
        observed = int(
            manifest_details[
                split
            ]["rows"]
        )

        if observed != expected:
            raise RuntimeError(
                f"{split} row count mismatch: "
                f"{observed} != {expected}."
            )

    run_directory = (
        config.output_root
        / config.run_name
    )

    if (
        run_directory.exists()
        and any(
            run_directory.iterdir()
        )
        and arguments.resume is None
    ):
        raise FileExistsError(
            "Run directory already exists and is not empty: "
            f"{run_directory}"
        )

    checkpoint_directory = (
        run_directory
        / "checkpoints"
    )

    embedding_directory = (
        run_directory
        / "embeddings"
    )

    metadata_directory = (
        run_directory
        / "metadata"
    )

    for directory in (
        run_directory,
        checkpoint_directory,
        embedding_directory,
        metadata_directory,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Build the recorded SimCLR preprocessing from configuration rather than
    # embedding augmentation choices directly in the training loop.
    transform_config = (
        load_transform_config(
            config.transform_config
        )
    )

    train_transform = (
        MultiViewTransform(
            build_ssl_train_transform(
                transform_config
            ),
            views=2,
        )
    )

    validation_contrastive_transform = (
        MultiViewTransform(
            build_ssl_train_transform(
                transform_config
            ),
            views=2,
        )
    )

    evaluation_transform = (
        build_ssl_evaluation_transform(
            transform_config
        )
    )

    train_full = (
        UnifiedCrystalDataset(
            config.train_manifest,
            config.image_root_map,
            transform=train_transform,
            verify_files=(
                config.verify_files
            ),
            verify_sample_size=(
                config
                .verify_sample_size
            ),
            verify_seed=config.seed,
            include_metadata=False,
        )
    )

    validation_contrastive_full = (
        UnifiedCrystalDataset(
            config.validation_manifest,
            config.image_root_map,
            transform=(
                validation_contrastive_transform
            ),
            verify_files=(
                config.verify_files
            ),
            verify_sample_size=(
                config
                .verify_sample_size
            ),
            verify_seed=config.seed,
            include_metadata=False,
        )
    )

    validation_embedding_full = (
        UnifiedCrystalDataset(
            config.validation_manifest,
            config.image_root_map,
            transform=(
                evaluation_transform
            ),
            verify_files=(
                config.verify_files
            ),
            verify_sample_size=(
                config
                .verify_sample_size
            ),
            verify_seed=config.seed,
            include_metadata=True,
        )
    )

    if len(train_full) != (
        config.expected_train_rows
    ):
        raise RuntimeError(
            "Training dataset row count mismatch."
        )

    if len(
        validation_contrastive_full
    ) != (
        config.expected_validation_rows
    ):
        raise RuntimeError(
            "Validation dataset row count mismatch."
        )

    train_dataset = limited_dataset(
        train_full,
        arguments.limit_train_items,
    )

    validation_contrastive_dataset = (
        limited_dataset(
            validation_contrastive_full,
            arguments
            .limit_validation_items,
        )
    )

    validation_embedding_dataset = (
        limited_dataset(
            validation_embedding_full,
            arguments
            .limit_validation_items,
        )
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        config.seed
    )

    validation_generator = (
        torch.Generator()
    )

    validation_generator.manual_seed(
        config.seed + 1
    )

    embedding_generator = (
        torch.Generator()
    )

    embedding_generator.manual_seed(
        config.seed + 2
    )

    # Use separately seeded generators for training, validation, and embedding
    # export so each data stream is reproducible without sharing RNG state.
    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=(
            config.num_workers
        ),
        generator=train_generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = make_loader(
        validation_contrastive_dataset,
        batch_size=(
            config
            .evaluation_batch_size
        ),
        shuffle=False,
        drop_last=False,
        num_workers=(
            config.num_workers
        ),
        generator=validation_generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    embedding_loader = make_loader(
        validation_embedding_dataset,
        batch_size=(
            config
            .evaluation_batch_size
        ),
        shuffle=False,
        drop_last=False,
        num_workers=(
            config.num_workers
        ),
        generator=embedding_generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # Construct the model only after the benchmark and loaders have passed
    # their validation checks, avoiding partially initialised run artifacts.
    model = SimCLRModel(
        backbone_name=(
            config.backbone_name
        ),
        projection_hidden_dimension=(
            config
            .projection_hidden_dimension
        ),
        projection_dimension=(
            config
            .projection_dimension
        ),
        temperature=(
            config.temperature
        ),
        initialization=(
            config.initialization
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=(
            config.weight_decay
        ),
    )

    amp_enabled = bool(
        config.use_amp
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    resolved_config = (
        config.to_dict()
    )

    resolved_config[
        "device"
    ] = str(device)

    resolved_config[
        "amp_enabled"
    ] = amp_enabled

    resolved_config[
        "limit_train_items"
    ] = arguments.limit_train_items

    resolved_config[
        "limit_validation_items"
    ] = (
        arguments
        .limit_validation_items
    )

    write_json_atomic(
        metadata_directory
        / "resolved_config.json",
        resolved_config,
    )

    save_run_record(
        metadata_directory
        / "run_record.json",
        model=(
            f"simclr_{config.backbone_name}"
        ),
        manifest=(
            config.train_manifest
        ),
        seed=config.seed,
        config=resolved_config,
    )

    environment = {
        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "hostname": (
            platform.node()
        ),
        "python": (
            platform.python_version()
        ),
        "pytorch": (
            torch.__version__
        ),
        "cuda_runtime": (
            torch.version.cuda
        ),
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "device": str(device),
        "gpu": (
            torch.cuda
            .get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "pbs_job_id": (
            os.environ.get(
                "PBS_JOBID"
            )
        ),
        "model_parameters": (
            model_parameter_count(
                model
            )
        ),
        "model": (
            model.model_metadata()
        ),
        "manifest_fingerprint_sha256": (
            fingerprint
        ),
        "manifests": (
            manifest_details
        ),
    }

    write_json_atomic(
        metadata_directory
        / "environment.json",
        environment,
    )

    history: list[
        dict[str, object]
    ] = []

    start_epoch = 1

    if arguments.resume is not None:
        checkpoint = load_checkpoint(
            arguments.resume,
            model=model,
            optimizer=optimizer,
            device=device,
            expected_benchmark_id=(
                config.benchmark_id
            ),
            expected_manifest_fingerprint_sha256=(
                fingerprint
            ),
            scaler=scaler,
            train_generator=(
                train_generator
            ),
            restore_rng=True,
        )

        start_epoch = (
            int(
                checkpoint[
                    "completed_epoch"
                ]
            )
            + 1
        )

        history = history_dicts(
            checkpoint.get(
                "history",
                [],
            )
        )

        print(
            "Resuming from epoch "
            f"{start_epoch}.",
            flush=True,
        )

    started = time.perf_counter()

    # Run the fixed epoch schedule, recording validation metrics and writing
    # recoverable checkpoints without consulting the held-out test images.
    for epoch in range(
        start_epoch,
        config.epochs + 1,
    ):
        learning_rate = (
            learning_rate_for_epoch(
                epoch_index=epoch - 1,
                total_epochs=(
                    config.epochs
                ),
                warmup_epochs=(
                    config
                    .warmup_epochs
                ),
                base_learning_rate=(
                    config
                    .learning_rate
                ),
                minimum_learning_rate=(
                    config
                    .minimum_learning_rate
                ),
            )
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            learning_rate=(
                learning_rate
            ),
            use_amp=amp_enabled,
            scaler=scaler,
            gradient_clip_norm=(
                config
                .gradient_clip_norm
            ),
        )

        history.append(
            train_metrics.to_dict()
        )

        print(
            f"epoch={epoch} "
            f"split=train "
            f"loss={train_metrics.loss:.6f} "
            "positive_pair_accuracy="
            f"{train_metrics.positive_pair_accuracy:.6f} "
            f"lr={learning_rate:.8f} "
            f"seconds={train_metrics.seconds:.2f}",
            flush=True,
        )

        if (
            epoch
            % config.validation_every
            == 0
        ):
            validation_metrics = (
                evaluate_contrastive_epoch(
                    model=model,
                    loader=(
                        validation_loader
                    ),
                    device=device,
                    epoch=epoch,
                    learning_rate=(
                        learning_rate
                    ),
                    use_amp=(
                        amp_enabled
                    ),
                )
            )

            history.append(
                validation_metrics.to_dict()
            )

            print(
                f"epoch={epoch} "
                "split=validation "
                f"loss={validation_metrics.loss:.6f} "
                "positive_pair_accuracy="
                f"{validation_metrics.positive_pair_accuracy:.6f} "
                f"seconds={validation_metrics.seconds:.2f}",
                flush=True,
            )

        write_history_csv_atomic(
            run_directory
            / "training_history.csv",
            history,
        )

        checkpoint_arguments = {
            "completed_epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "scaler": scaler,
            "history": history,
            "config": resolved_config,
            "benchmark_id": (
                config.benchmark_id
            ),
            "manifest_fingerprint_sha256": (
                fingerprint
            ),
            "train_manifest_sha256": (
                sha256_file(
                    config.train_manifest
                )
            ),
            "train_generator": (
                train_generator
            ),
        }

        save_checkpoint_atomic(
            checkpoint_directory
            / "simclr_last.pt",
            **checkpoint_arguments,
        )

        if (
            epoch
            % config.checkpoint_every
            == 0
            or epoch == config.epochs
        ):
            save_checkpoint_atomic(
                checkpoint_directory
                / (
                    "simclr_epoch_"
                    f"{epoch:04d}.pt"
                ),
                **checkpoint_arguments,
            )

    embedding_summary = None

    if (
        arguments
        .export_validation_embeddings
    ):
        embedding_summary = (
            export_validation_embeddings(
                model=model,
                loader=embedding_loader,
                full_dataset=(
                    validation_embedding_full
                ),
                expected_items=len(
                    validation_embedding_dataset
                ),
                device=device,
                use_amp=amp_enabled,
                output_path=(
                    embedding_directory
                    / (
                        "validation_"
                        "backbone_embeddings.npz"
                    )
                ),
            )
        )

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    completed = {
        "schema_version": (
            "plagnetplus_simclr_run_v1"
        ),
        "status": "PASS",
        "completed_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "run_name": (
            config.run_name
        ),
        "benchmark_id": (
            config.benchmark_id
        ),
        "manifest_fingerprint_sha256": (
            fingerprint
        ),
        "model": (
            model.model_metadata()
        ),
        "device": str(device),
        "amp_enabled": (
            amp_enabled
        ),
        "completed_epochs": (
            config.epochs
        ),
        "history_rows": len(
            history
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "checkpoint": {
            "path": str(
                checkpoint_directory
                / "simclr_last.pt"
            ),
            "sha256": sha256_file(
                checkpoint_directory
                / "simclr_last.pt"
            ),
        },
        "validation_embeddings": (
            embedding_summary
        ),
    }

    write_json_atomic(
        run_directory
        / "COMPLETED.json",
        completed,
    )

    print()
    print(
        json.dumps(
            completed,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "SIMCLR TRAINING RUN: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
