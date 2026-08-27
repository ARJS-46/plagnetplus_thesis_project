#!/usr/bin/env python3
"""Train native PlagNetPlus SwAV on a frozen benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)

from plagnetplus.data.ssl_transforms import (
    SSLTransformConfig,
    build_ssl_evaluation_transform,
    build_ssl_train_transform,
)
from plagnetplus.data.transforms import (
    MultiCropTransform,
)
from plagnetplus.data.unified_dataset import (
    UnifiedCrystalDataset,
)
from plagnetplus.models.swav import (
    SwAVModel,
)
from plagnetplus.training.swav_config import (
    load_swav_training_config,
)
from plagnetplus.training.swav_engine import (
    build_swav_optimizer,
    evaluate_one_epoch,
    learning_rate_for_epoch,
    load_checkpoint,
    save_checkpoint_atomic,
    train_one_epoch,
    write_history_csv_atomic,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train native PlagNetPlus SwAV "
            "on the frozen pooled benchmark."
        )
    )

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
        "--export-train-embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--export-validation-embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    return parser.parse_args()


def write_json_atomic(
    path: Path,
    payload: object,
) -> None:
    """Write formatted JSON using an atomic replacement."""

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
    """Return the SHA-256 digest of one file."""

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
    """Seed Python, NumPy and Torch."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(
    worker_id: int,
) -> None:
    """Seed one DataLoader worker."""

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
    """Resolve an explicit or automatic training device."""

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
    """Load the frozen SSL transformation configuration."""

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
    """Return the number of crystals in one manifest."""

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
) -> tuple[
    str,
    dict[str, dict[str, object]],
]:
    """Hash all split manifests without accessing test images."""

    paths = {
        "train": train_manifest,
        "validation": validation_manifest,
        "test": test_manifest,
    }

    details: dict[
        str,
        dict[str, object],
    ] = {}

    digest = hashlib.sha256()

    for split in (
        "train",
        "validation",
        "test",
    ):
        path = paths[split]

        file_hash = sha256_file(path)
        rows = manifest_row_count(path)

        details[split] = {
            "path": str(path),
            "sha256": file_hash,
            "rows": rows,
        }

        digest.update(
            split.encode("utf-8")
        )

        digest.update(
            file_hash.encode("utf-8")
        )

        digest.update(
            str(rows).encode("utf-8")
        )

    return (
        digest.hexdigest(),
        details,
    )


def limited_dataset(
    dataset: Dataset,
    limit: int | None,
) -> Dataset:
    """Return the complete dataset or its deterministic prefix."""

    if limit is None:
        return dataset

    if limit <= 0:
        raise ValueError(
            "Dataset limits must be positive."
        )

    return Subset(
        dataset,
        range(
            min(
                limit,
                len(dataset),
            )
        ),
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
    """Build a deterministic DataLoader."""

    arguments: dict[str, object] = {
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
    """Convert collated metadata to a Python list."""

    if isinstance(
        value,
        torch.Tensor,
    ):
        return (
            value.detach()
            .cpu()
            .tolist()
        )

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


def export_backbone_embeddings(
    *,
    split: str,
    model: SwAVModel,
    loader: DataLoader,
    full_dataset: UnifiedCrystalDataset,
    expected_items: int,
    device: torch.device,
    use_amp: bool,
    output_path: Path,
) -> dict[str, object]:
    """Export aligned deterministic backbone embeddings."""

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
                    "Encountered non-finite "
                    f"{split} embeddings."
                )

            feature_blocks.append(
                features.detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )

            metadata = batch[
                "metadata"
            ]

            crystal_ids.extend(
                str(value)
                for value
                in metadata_values(
                    metadata["crystal_id"]
                )
            )

            manifest_orders.extend(
                int(value)
                for value
                in metadata_values(
                    metadata[
                        "manifest_order"
                    ]
                )
            )

    if not feature_blocks:
        raise RuntimeError(
            f"No {split} embeddings were produced."
        )

    feature_array = np.concatenate(
        feature_blocks,
        axis=0,
    )

    if feature_array.shape[0] != expected_items:
        raise RuntimeError(
            f"{split} embedding row-count mismatch: "
            f"{feature_array.shape[0]} != "
            f"{expected_items}."
        )

    expected_ids = (
        full_dataset
        .manifest["crystal_id"]
        .iloc[:expected_items]
        .astype(str)
        .tolist()
    )

    expected_orders = (
        full_dataset
        .manifest["manifest_order"]
        .iloc[:expected_items]
        .astype(int)
        .tolist()
    )

    if crystal_ids != expected_ids:
        raise RuntimeError(
            f"{split} embedding crystal IDs "
            "are misaligned."
        )

    if manifest_orders != expected_orders:
        raise RuntimeError(
            f"{split} embedding manifest ordering "
            "is misaligned."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_path.with_suffix(
        output_path.suffix
        + ".partial"
    )

    with temporary.open("wb") as handle:
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

    temporary.replace(output_path)

    return {
        "split": split,
        "path": str(output_path),
        "sha256": sha256_file(
            output_path
        ),
        "shape": list(
            feature_array.shape
        ),
        "dtype": str(
            feature_array.dtype
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



def export_prototype_weights(
    *,
    model: SwAVModel,
    output_path: Path,
) -> dict[str, object]:
    """Export final normalised SwAV prototype vectors."""

    model.normalise_prototypes()

    weights = (
        model.prototypes
        .weight
        .detach()
        .to(
            dtype=torch.float32
        )
        .cpu()
        .numpy()
    )

    expected_shape = (
        model.prototype_count,
        model.projection_dimension,
    )

    if weights.shape != expected_shape:
        raise RuntimeError(
            "Prototype-weight shape mismatch: "
            f"{weights.shape} != "
            f"{expected_shape}."
        )

    if not np.isfinite(
        weights
    ).all():
        raise RuntimeError(
            "Prototype weights contain "
            "non-finite values."
        )

    norms = np.linalg.norm(
        weights,
        axis=1,
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
            weights=weights,
            prototype_indices=np.arange(
                model.prototype_count,
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
            weights.shape
        ),
        "dtype": str(
            weights.dtype
        ),
        "all_finite": True,
        "minimum_norm": float(
            norms.min()
        ),
        "mean_norm": float(
            norms.mean()
        ),
        "maximum_norm": float(
            norms.max()
        ),
    }

def model_parameter_count(
    model: torch.nn.Module,
) -> dict[str, int]:
    """Count total and trainable model parameters."""

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
        "trainable": int(trainable),
    }


def history_dicts(
    rows: Iterable[object],
) -> list[dict[str, object]]:
    """Convert dataclass metrics and dictionaries consistently."""

    output: list[
        dict[str, object]
    ] = []

    for row in rows:
        if isinstance(row, dict):
            output.append(
                dict(row)
            )
        else:
            output.append(
                row.to_dict()
            )

    return output


def main() -> None:
    """Execute one SwAV training run."""

    arguments = parse_arguments()

    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )

    config = (
        load_swav_training_config(
            config_path
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
            >= arguments.epochs
        ):
            replacements[
                "warmup_epochs"
            ] = max(
                0,
                arguments.epochs - 1,
            )

        if (
            config.prototype_freeze_epochs
            >= arguments.epochs
        ):
            replacements[
                "prototype_freeze_epochs"
            ] = max(
                0,
                arguments.epochs - 1,
            )

    if arguments.batch_size is not None:
        replacements[
            "batch_size"
        ] = arguments.batch_size

    if (
        arguments.evaluation_batch_size
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

    if arguments.limit_train_items is not None:
        replacements[
            "limit_train_items"
        ] = arguments.limit_train_items

    if (
        arguments.limit_validation_items
        is not None
    ):
        replacements[
            "limit_validation_items"
        ] = (
            arguments
            .limit_validation_items
        )

    if (
        arguments.export_train_embeddings
        is not None
    ):
        replacements[
            "export_train_embeddings"
        ] = (
            arguments
            .export_train_embeddings
        )

    if (
        arguments
        .export_validation_embeddings
        is not None
    ):
        replacements[
            "export_validation_embeddings"
        ] = (
            arguments
            .export_validation_embeddings
        )

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

    # Fingerprint the frozen train/validation/test manifests before training
    # so the run record can prove which benchmark definition was used.
    fingerprint, manifest_details = (
        combined_manifest_fingerprint(
            train_manifest=(
                config.train_manifest
            ),
            validation_manifest=(
                config.validation_manifest
            ),
            test_manifest=(
                config.test_manifest
            ),
        )
    )

    expected_rows = {
        "train": (
            config.expected_train_rows
        ),
        "validation": (
            config.expected_validation_rows
        ),
        "test": (
            config.expected_test_rows
        ),
    }

    for split, expected in (
        expected_rows.items()
    ):
        observed = int(
            manifest_details[
                split
            ]["rows"]
        )

        if observed != expected:
            raise RuntimeError(
                f"{split} manifest row-count mismatch: "
                f"{observed} != {expected}."
            )

    if (
        fingerprint
        != config.expected_manifest_fingerprint_sha256
    ):
        raise RuntimeError(
            "Frozen benchmark fingerprint mismatch: "
            f"{fingerprint} != "
            f"{config.expected_manifest_fingerprint_sha256}."
        )

    run_directory = (
        config.output_root
        / config.run_name
    )

    if arguments.resume is not None:
        resume_path = (
            arguments.resume
            .expanduser()
            .resolve()
        )

        if not resume_path.is_file():
            raise FileNotFoundError(
                resume_path
            )

        resume_run_directory = (
            resume_path.parent.parent
        )

        if (
            run_directory.resolve()
            != resume_run_directory.resolve()
        ):
            raise ValueError(
                "Resume checkpoint does not belong "
                "to the configured run directory."
            )
    else:
        resume_path = None

        if (
            run_directory.exists()
            and any(
                run_directory.iterdir()
            )
        ):
            raise FileExistsError(
                "Run directory is not empty: "
                f"{run_directory}"
            )

    checkpoint_directory = (
        run_directory / "checkpoints"
    )

    embedding_directory = (
        run_directory / "embeddings"
    )

    metadata_directory = (
        run_directory / "metadata"
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

    resolved_config_path = (
        metadata_directory
        / "resolved_config.json"
    )

    write_json_atomic(
        resolved_config_path,
        config.to_dict(),
    )

    write_json_atomic(
        metadata_directory
        / "manifest_fingerprints.json",
        {
            "combined_sha256": (
                fingerprint
            ),
            "expected_combined_sha256": (
                config.expected_manifest_fingerprint_sha256
            ),
            "splits": manifest_details,
            "test_images_accessed": False,
        },
    )

    # Build the recorded SwAV preprocessing from configuration rather than
    # embedding augmentation choices directly in the training loop.
    transform_config = (
        load_transform_config(
            config.transform_config
        )
    )

    train_transform = MultiCropTransform(
        build_ssl_train_transform(
            transform_config
        ),
        global_crops=config.global_crops,
    )

    validation_transform = (
        MultiCropTransform(
            build_ssl_train_transform(
                transform_config
            ),
            global_crops=config.global_crops,
        )
    )

    evaluation_transform = (
        build_ssl_evaluation_transform(
            transform_config
        )
    )

    train_full = UnifiedCrystalDataset(
        config.train_manifest,
        config.image_root_map,
        transform=train_transform,
        verify_files=config.verify_files,
        verify_sample_size=(
            config.verify_sample_size
        ),
        verify_seed=config.seed,
        include_metadata=False,
    )

    validation_full = (
        UnifiedCrystalDataset(
            config.validation_manifest,
            config.image_root_map,
            transform=validation_transform,
            verify_files=(
                config.verify_files
            ),
            verify_sample_size=(
                config.verify_sample_size
            ),
            verify_seed=config.seed,
            include_metadata=False,
        )
    )

    train_embedding_full = (
        UnifiedCrystalDataset(
            config.train_manifest,
            config.image_root_map,
            transform=evaluation_transform,
            verify_files=(
                config.verify_files
            ),
            verify_sample_size=(
                config.verify_sample_size
            ),
            verify_seed=config.seed,
            include_metadata=True,
        )
    )

    validation_embedding_full = (
        UnifiedCrystalDataset(
            config.validation_manifest,
            config.image_root_map,
            transform=evaluation_transform,
            verify_files=(
                config.verify_files
            ),
            verify_sample_size=(
                config.verify_sample_size
            ),
            verify_seed=config.seed,
            include_metadata=True,
        )
    )

    if len(train_full) != (
        config.expected_train_rows
    ):
        raise RuntimeError(
            "Training dataset row-count mismatch."
        )

    if len(validation_full) != (
        config.expected_validation_rows
    ):
        raise RuntimeError(
            "Validation dataset row-count mismatch."
        )

    train_dataset = limited_dataset(
        train_full,
        config.limit_train_items,
    )

    validation_dataset = (
        limited_dataset(
            validation_full,
            config.limit_validation_items,
        )
    )

    train_embedding_dataset = (
        limited_dataset(
            train_embedding_full,
            config.limit_train_items,
        )
    )

    validation_embedding_dataset = (
        limited_dataset(
            validation_embedding_full,
            config.limit_validation_items,
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

    train_embedding_generator = (
        torch.Generator()
    )

    train_embedding_generator.manual_seed(
        config.seed + 2
    )

    validation_embedding_generator = (
        torch.Generator()
    )

    validation_embedding_generator.manual_seed(
        config.seed + 3
    )

    # Use separately seeded generators for training, validation, and embedding
    # export so each data stream is reproducible without sharing RNG state.
    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        generator=train_generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = make_loader(
        validation_dataset,
        batch_size=(
            config.evaluation_batch_size
        ),
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        generator=validation_generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    train_embedding_loader = (
        make_loader(
            train_embedding_dataset,
            batch_size=(
                config
                .evaluation_batch_size
            ),
            shuffle=False,
            drop_last=False,
            num_workers=(
                config.num_workers
            ),
            generator=(
                train_embedding_generator
            ),
            pin_memory=(
                device.type == "cuda"
            ),
        )
    )

    validation_embedding_loader = (
        make_loader(
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
            generator=(
                validation_embedding_generator
            ),
            pin_memory=(
                device.type == "cuda"
            ),
        )
    )

    # Construct the model only after the benchmark and loaders have passed
    # their validation checks, avoiding partially initialised run artifacts.
    model = SwAVModel(
        backbone_name=(
            config.backbone_name
        ),
        initialization=(
            config.initialization
        ),
        projection_hidden_dimension=(
            config
            .projection_hidden_dimension
        ),
        projection_dimension=(
            config.projection_dimension
        ),
        prototype_count=(
            config.prototype_count
        ),
        temperature=(
            config.temperature
        ),
        sinkhorn_epsilon=(
            config.sinkhorn_epsilon
        ),
        sinkhorn_iterations=(
            config.sinkhorn_iterations
        ),
    )

    model.to(device)

    optimizer = build_swav_optimizer(
        model=model,
        learning_rate=(
            config.learning_rate
        ),
        momentum=config.momentum,
        weight_decay=(
            config.weight_decay
        ),
    )

    amp_enabled = bool(
        config.use_amp
        and device.type == "cuda"
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda"
        )
        if amp_enabled
        else None
    )

    history: list[
        dict[str, object]
    ] = []

    start_epoch = 1

    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
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

        completed_epoch = int(
            checkpoint[
                "completed_epoch"
            ]
        )

        history = [
            dict(row)
            for row
            in checkpoint.get(
                "history",
                [],
            )
        ]

        start_epoch = (
            completed_epoch + 1
        )

    write_json_atomic(
        metadata_directory
        / "environment.json",
        {
            "created_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "python": (
                platform.python_version()
            ),
            "platform": (
                platform.platform()
            ),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": str(device),
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),
            "amp_enabled": (
                amp_enabled
            ),
        },
    )

    write_json_atomic(
        metadata_directory
        / "run_initialisation.json",
        {
            "schema_version": (
                "plagnetplus_swav_initialisation_v1"
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
            "config_path": str(
                config_path
            ),
            "config_sha256": (
                sha256_file(
                    config_path
                )
            ),
            "resolved_config_sha256": (
                sha256_file(
                    resolved_config_path
                )
            ),
            "transform_config_sha256": (
                sha256_file(
                    config.transform_config
                )
            ),
            "parameters": (
                model_parameter_count(
                    model
                )
            ),
            "train_items": len(
                train_dataset
            ),
            "validation_items": len(
                validation_dataset
            ),
            "start_epoch": (
                start_epoch
            ),
            "test_images_accessed": False,
        },
    )

    started = time.perf_counter()

    # Run the fixed epoch schedule, recording validation metrics and writing
    # recoverable checkpoints without consulting the held-out test images.
    for epoch in range(
        start_epoch,
        config.epochs + 1,
    ):
        base_learning_rate = (
            learning_rate_for_epoch(
                epoch_index=epoch - 1,
                total_epochs=(
                    config.epochs
                ),
                warmup_epochs=(
                    config.warmup_epochs
                ),
                base_learning_rate=(
                    config.learning_rate
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
                base_learning_rate
            ),
            prototype_freeze_epochs=(
                config
                .prototype_freeze_epochs
            ),
            use_amp=amp_enabled,
            scaler=scaler,
            gradient_clip_norm=(
                config
                .gradient_clip_norm
            ),
            require_finite_loss=(
                config
                .require_finite_loss
            ),
            require_finite_gradients=(
                config
                .require_finite_gradients
            ),
        )

        history.append(
            train_metrics.to_dict()
        )

        validation_metrics = None

        if (
            epoch
            % config.validation_every
            == 0
            or epoch == config.epochs
        ):
            validation_metrics = (
                evaluate_one_epoch(
                    model=model,
                    loader=(
                        validation_loader
                    ),
                    device=device,
                    epoch=epoch,
                    learning_rate=(
                        base_learning_rate
                    ),
                    prototype_freeze_epochs=(
                        config
                        .prototype_freeze_epochs
                    ),
                    use_amp=amp_enabled,
                    require_finite_loss=(
                        config
                        .require_finite_loss
                    ),
                )
            )

            history.append(
                validation_metrics.to_dict()
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
            "config": (
                config.to_dict()
            ),
            "benchmark_id": (
                config.benchmark_id
            ),
            "manifest_fingerprint_sha256": (
                fingerprint
            ),
            "train_manifest_sha256": (
                str(
                    manifest_details[
                        "train"
                    ]["sha256"]
                )
            ),
            "train_generator": (
                train_generator
            ),
        }

        save_checkpoint_atomic(
            checkpoint_directory
            / "swav_last.pt",
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
                    "swav_epoch_"
                    f"{epoch:04d}.pt"
                ),
                **checkpoint_arguments,
            )

        message = (
            f"Epoch {epoch:04d}/"
            f"{config.epochs:04d} | "
            f"train loss "
            f"{train_metrics.loss:.6f} | "
            f"occupied "
            f"{train_metrics.occupied_prototype_fraction:.4f} | "
            f"largest "
            f"{train_metrics.largest_prototype_fraction:.4f} | "
            f"agreement "
            f"{train_metrics.assignment_agreement:.4f} | "
            f"lr "
            f"{base_learning_rate:.8f} | "
            f"prototype freeze "
            f"{train_metrics.prototype_gradients_frozen}"
        )

        if validation_metrics is not None:
            message += (
                " | validation loss "
                f"{validation_metrics.loss:.6f}"
                " | validation occupied "
                f"{validation_metrics.occupied_prototype_fraction:.4f}"
                " | validation largest "
                f"{validation_metrics.largest_prototype_fraction:.4f}"
                " | validation agreement "
                f"{validation_metrics.assignment_agreement:.4f}"
            )

        print(
            message,
            flush=True,
        )

    train_embedding_summary = None
    validation_embedding_summary = None

    if config.export_train_embeddings:
        train_embedding_summary = (
            export_backbone_embeddings(
                split="train",
                model=model,
                loader=(
                    train_embedding_loader
                ),
                full_dataset=(
                    train_embedding_full
                ),
                expected_items=len(
                    train_embedding_dataset
                ),
                device=device,
                use_amp=amp_enabled,
                output_path=(
                    embedding_directory
                    / (
                        "train_"
                        "backbone_embeddings.npz"
                    )
                ),
            )
        )

    if (
        config
        .export_validation_embeddings
    ):
        validation_embedding_summary = (
            export_backbone_embeddings(
                split="validation",
                model=model,
                loader=(
                    validation_embedding_loader
                ),
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

    prototype_summary = (
        export_prototype_weights(
            model=model,
            output_path=(
                embedding_directory
                / "prototype_weights.npz"
            ),
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    train_history = [
        row
        for row in history
        if row["split"] == "train"
    ]

    validation_history = [
        row
        for row in history
        if row["split"] == "validation"
    ]

    final_train = (
        train_history[-1]
        if train_history
        else None
    )

    final_validation = (
        validation_history[-1]
        if validation_history
        else None
    )

    checkpoint_path = (
        checkpoint_directory
        / "swav_last.pt"
    )

    completed = {
        "schema_version": (
            "plagnetplus_swav_run_v1"
        ),
        "status": "PASS",
        "completed_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "run_name": config.run_name,
        "method": "swav",
        "benchmark_id": (
            config.benchmark_id
        ),
        "manifest_fingerprint_sha256": (
            fingerprint
        ),
        "model": {
            **model.model_metadata(),
            "parameters": (
                model_parameter_count(
                    model
                )
            ),
        },
        "device": str(device),
        "amp_enabled": amp_enabled,
        "completed_epochs": (
            config.epochs
        ),
        "history_rows": len(history),
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "gpu_hours": (
            elapsed_seconds / 3600.0
            if device.type == "cuda"
            else None
        ),
        "final_train_metrics": (
            final_train
        ),
        "final_validation_metrics": (
            final_validation
        ),
        "checkpoint": {
            "path": str(
                checkpoint_path
            ),
            "sha256": sha256_file(
                checkpoint_path
            ),
        },
        "train_embeddings": (
            train_embedding_summary
        ),
        "validation_embeddings": (
            validation_embedding_summary
        ),
        "prototypes": (
            prototype_summary
        ),
        "checks": {
            "manifest_fingerprint_matches": (
                fingerprint
                == config.expected_manifest_fingerprint_sha256
            ),
            "train_embeddings_exported": (
                train_embedding_summary
                is not None
            ),
            "validation_embeddings_exported": (
                validation_embedding_summary
                is not None
            ),
            "all_history_metrics_finite": bool(
                all(
                    np.isfinite(
                        float(
                            row["loss"]
                        )
                    )
                    for row in history
                )
            ),
            "prototype_weights_exported": (
                prototype_summary
                is not None
            ),
            "prototype_weights_finite": (
                bool(
                    prototype_summary[
                        "all_finite"
                    ]
                )
            ),
            "test_manifest_hashed_only": True,
            "test_images_accessed": False,
            "test_embeddings_exported": False,
        },
    }

    write_json_atomic(
        run_directory
        / "run_summary.json",
        completed,
    )

    print()
    print(
        "Run directory:",
        run_directory,
    )

    print(
        "Status:",
        completed["status"],
    )

    print(
        "Train embedding shape:",
        (
            None
            if train_embedding_summary
            is None
            else train_embedding_summary[
                "shape"
            ]
        ),
    )

    print(
        "Validation embedding shape:",
        (
            None
            if validation_embedding_summary
            is None
            else validation_embedding_summary[
                "shape"
            ]
        ),
    )

    print(
        "Prototype shape:",
        prototype_summary[
            "shape"
        ],
    )

    print(
        "Test images accessed:",
        completed["checks"][
            "test_images_accessed"
        ],
    )


if __name__ == "__main__":
    main()
