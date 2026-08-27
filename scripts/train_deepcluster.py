#!/usr/bin/env python3
"""Train a DeepCluster model on a frozen PlagNetPlus benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, Subset

from plagnetplus.data.ssl_transforms import (
    SSLTransformConfig,
    build_ssl_evaluation_transform,
    build_ssl_train_transform,
)
from plagnetplus.data.unified_dataset import (
    UnifiedCrystalDataset,
)
from plagnetplus.models import DeepClusterModel
from plagnetplus.training.deepcluster_config import (
    DeepClusterTrainingConfig,
    load_deepcluster_training_config,
)
from plagnetplus.training.deepcluster_engine import (
    PseudoLabelDataset,
    array_sha256,
    assignment_stability,
    build_balanced_sampler,
    cycle_learning_rate,
    extract_backbone_features,
    fit_kmeans,
    preprocess_features,
    set_optimizer_learning_rate,
    train_pseudo_label_epoch,
    validate_cluster_health,
)
from plagnetplus.training.provenance import (
    save_run_record,
)


EXPECTED_BENCHMARK_FINGERPRINTS_BY_TRAIN_ROWS = {
    101_682: (
        "a2814f1c33b5adc45e9d4c8c90c7af7577e237ca74d112473a274dbd949ab582"
    ),
    91_357: (
        "c3c028a9aef47cd618ebceaa66ba7444f2b92a696abba6e509db2d468f440e7f"
    ),
}


def expected_benchmark_fingerprint_for_train_rows(
    train_rows: int,
) -> str:
    try:
        return (
            EXPECTED_BENCHMARK_FINGERPRINTS_BY_TRAIN_ROWS[
                int(train_rows)
            ]
        )
    except KeyError as error:
        raise RuntimeError(
            "No frozen DeepCluster benchmark fingerprint "
            f"is registered for {train_rows:,} training rows."
        ) from error


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
        help=(
            "Resume from a completed cycle checkpoint. "
            "Only cycle-boundary resume is supported."
        ),
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
        default=None,
    )

    parser.add_argument(
        "--cycles",
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
        "--overwrite",
        action="store_true",
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
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".partial"
    )

    pd.DataFrame(rows).to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def save_npz_atomic(
    path: Path,
    **arrays: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".partial"
    )

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
        )

    temporary.replace(path)


def save_checkpoint_atomic(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".partial"
    )

    torch.save(
        payload,
        temporary,
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
                "CUDA was requested but is unavailable."
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
        "validation": validation_manifest,
        "test": test_manifest,
    }

    details: dict[str, object] = {}

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
    if limit is None:
        return dataset

    if limit < 1:
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
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
    sampler: object | None = None,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": (
            shuffle
            if sampler is None
            else False
        ),
        "sampler": sampler,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": (
            device.type == "cuda"
        ),
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


def manifest_arrays(
    dataset: UnifiedCrystalDataset,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = dataset.manifest

    if limit is not None:
        frame = frame.iloc[
            : min(
                limit,
                len(frame),
            )
        ]

    crystal_ids = (
        frame["crystal_id"]
        .astype(str)
        .to_numpy()
    )

    manifest_orders = (
        frame["manifest_order"]
        .astype(np.int64)
        .to_numpy()
    )

    return (
        crystal_ids,
        manifest_orders,
    )


def config_with_overrides(
    config: DeepClusterTrainingConfig,
    arguments: argparse.Namespace,
) -> DeepClusterTrainingConfig:
    updates: dict[str, object] = {}

    for argument_name, config_name in (
        ("run_name", "run_name"),
        ("device", "device"),
        ("cycles", "cycles"),
        ("batch_size", "batch_size"),
        (
            "evaluation_batch_size",
            "evaluation_batch_size",
        ),
        ("num_workers", "num_workers"),
        (
            "limit_train_items",
            "limit_train_items",
        ),
    ):
        value = getattr(
            arguments,
            argument_name,
        )

        if value is not None:
            updates[config_name] = value

    if not updates:
        return config

    return replace(
        config,
        **updates,
    )



def restore_rng_state(
    checkpoint: dict[str, object],
) -> None:
    """Restore Python, NumPy and PyTorch random states."""
    random.setstate(
        checkpoint[
            "python_random_state"
        ]
    )

    np.random.set_state(
        checkpoint[
            "numpy_random_state"
        ]
    )

    torch.set_rng_state(
        checkpoint[
            "torch_random_state"
        ]
    )

    cuda_state = checkpoint.get(
        "torch_cuda_random_state"
    )

    if (
        cuda_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_state
        )


def load_existing_history(
    path: Path,
) -> list[dict[str, object]]:
    """Load completed cycle history for a resumed run."""
    if not path.is_file():
        raise FileNotFoundError(
            "Resume history does not exist: "
            f"{path}"
        )

    frame = pd.read_csv(
        path
    )

    if frame.empty:
        raise RuntimeError(
            "Resume history is empty."
        )

    return frame.to_dict(
        orient="records"
    )


def validate_resume_configuration(
    *,
    checkpoint_config: dict[str, object],
    config: DeepClusterTrainingConfig,
) -> None:
    """Validate fields that cannot change across resume."""
    expected = {
        "run_name": config.run_name,
        "benchmark_id": (
            config.benchmark_id
        ),
        "sampling": config.sampling,
        "backbone_name": (
            config.backbone_name
        ),
        "initialization": (
            config.initialization
        ),
        "cluster_count": (
            config.cluster_count
        ),
        "pca_components": (
            config.pca_components
        ),
        "epochs_per_cycle": (
            config.epochs_per_cycle
        ),
        "seed": config.seed,
    }

    mismatches = {
        key: {
            "checkpoint": (
                checkpoint_config.get(
                    key
                )
            ),
            "current": value,
        }
        for key, value
        in expected.items()
        if checkpoint_config.get(
            key
        )
        != value
    }

    if mismatches:
        raise RuntimeError(
            "Resume configuration differs "
            "from the checkpoint: "
            f"{mismatches}"
        )


def load_resume_state(
    *,
    checkpoint_path: Path,
    run_root: Path,
    model: DeepClusterModel,
    config: DeepClusterTrainingConfig,
    benchmark_fingerprint: str,
    expected_crystal_ids: np.ndarray,
    expected_manifest_orders: np.ndarray,
) -> tuple[
    int,
    list[dict[str, object]],
    np.ndarray,
    dict[str, object],
]:
    """Load a completed-cycle checkpoint and aligned assignments."""
    selected = (
        checkpoint_path
        .expanduser()
        .resolve()
    )

    if not selected.is_file():
        raise FileNotFoundError(
            selected
        )

    checkpoint_root = (
        run_root
        / "checkpoints"
    ).resolve()

    if not selected.is_relative_to(
        checkpoint_root
    ):
        raise ValueError(
            "Resume checkpoint must belong "
            "to the selected run directory."
        )

    checkpoint = torch.load(
        selected,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get(
        "schema_version"
    ) != (
        "plagnetplus_"
        "deepcluster_"
        "checkpoint_v1"
    ):
        raise RuntimeError(
            "Unsupported checkpoint schema."
        )

    completed_cycle = int(
        checkpoint["cycle"]
    )

    if completed_cycle < 1:
        raise RuntimeError(
            "Checkpoint cycle is invalid."
        )

    if completed_cycle >= config.cycles:
        raise RuntimeError(
            "Resume target must contain at "
            "least one cycle after the checkpoint."
        )

    if checkpoint.get(
        "benchmark_fingerprint"
    ) != benchmark_fingerprint:
        raise RuntimeError(
            "Resume checkpoint benchmark "
            "fingerprint does not match."
        )

    checkpoint_config = dict(
        checkpoint["config"]
    )

    validate_resume_configuration(
        checkpoint_config=(
            checkpoint_config
        ),
        config=config,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    history = load_existing_history(
        run_root
        / "training_history.csv"
    )

    if len(history) != completed_cycle:
        raise RuntimeError(
            "History length does not match "
            "the checkpoint cycle."
        )

    recorded_cycles = [
        int(row["cycle"])
        for row in history
    ]

    expected_cycles = list(
        range(
            1,
            completed_cycle + 1,
        )
    )

    if recorded_cycles != expected_cycles:
        raise RuntimeError(
            "Resume history cycle sequence "
            "is incomplete or misordered."
        )

    assignment_path = (
        run_root
        / "cycles"
        / f"cycle_{completed_cycle:03d}"
        / "cluster_assignments.npz"
    )

    if not assignment_path.is_file():
        raise FileNotFoundError(
            assignment_path
        )

    with np.load(
        assignment_path,
        allow_pickle=False,
    ) as payload:
        crystal_ids = np.asarray(
            payload["crystal_ids"]
        ).astype(str)

        manifest_orders = np.asarray(
            payload["manifest_orders"],
            dtype=np.int64,
        )

        assignments = np.asarray(
            payload["assignments"],
            dtype=np.int32,
        )

    if not np.array_equal(
        crystal_ids,
        expected_crystal_ids,
    ):
        raise RuntimeError(
            "Resume assignment crystal IDs "
            "are misaligned."
        )

    if not np.array_equal(
        manifest_orders,
        expected_manifest_orders,
    ):
        raise RuntimeError(
            "Resume assignment manifest orders "
            "are misaligned."
        )

    assignment_hash = array_sha256(
        assignments
    )

    if assignment_hash != checkpoint.get(
        "assignment_sha256"
    ):
        raise RuntimeError(
            "Resume assignment hash does not "
            "match the checkpoint."
        )

    restore_rng_state(
        checkpoint
    )

    return (
        completed_cycle + 1,
        history,
        assignments.copy(),
        {
            "checkpoint_path": str(
                selected
            ),
            "checkpoint_sha256": (
                sha256_file(
                    selected
                )
            ),
            "completed_cycle": (
                completed_cycle
            ),
            "next_cycle": (
                completed_cycle + 1
            ),
            "assignment_path": str(
                assignment_path.resolve()
            ),
            "assignment_sha256": (
                assignment_hash
            ),
        },
    )


def main() -> None:
    arguments = parse_arguments()

    config = (
        load_deepcluster_training_config(
            arguments.config
        )
    )

    config = config_with_overrides(
        config,
        arguments,
    )

    seed_everything(
        config.seed
    )

    device = resolve_device(
        config.device
    )

    run_root = (
        config.output_root
        / config.run_name
    )

    resuming = (
        arguments.resume is not None
    )

    if resuming:
        if arguments.overwrite:
            raise ValueError(
                "--overwrite cannot be used "
                "together with --resume."
            )

        if not run_root.is_dir():
            raise FileNotFoundError(
                "Resume run directory does "
                "not exist: "
                f"{run_root}"
            )

    else:
        if run_root.exists():
            if not arguments.overwrite:
                raise FileExistsError(
                    "Run directory already exists. "
                    "Use --overwrite only for an "
                    "intentional fresh rerun: "
                    f"{run_root}"
                )

            shutil.rmtree(
                run_root
            )

        run_root.mkdir(
            parents=True,
            exist_ok=False,
        )

    metadata_root = (
        run_root / "metadata"
    )

    cycles_root = (
        run_root / "cycles"
    )

    checkpoints_root = (
        run_root / "checkpoints"
    )

    embeddings_root = (
        run_root / "embeddings"
    )

    for directory in (
        metadata_root,
        cycles_root,
        checkpoints_root,
        embeddings_root,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    row_checks = {
        "train": manifest_row_count(
            config.train_manifest
        ),
        "validation": manifest_row_count(
            config.validation_manifest
        ),
        "test": manifest_row_count(
            config.test_manifest
        ),
    }

    expected_row_checks = {
        "train": config.expected_train_rows,
        "validation": (
            config.expected_validation_rows
        ),
        "test": config.expected_test_rows,
    }

    # Verify the frozen benchmark identity before beginning the expensive
    # feature-extraction and reclustering cycle.
    expected_benchmark_fingerprint = (
        expected_benchmark_fingerprint_for_train_rows(
            config.expected_train_rows
        )
    )

    if row_checks != expected_row_checks:
        raise RuntimeError(
            "Manifest row counts differ from "
            "the frozen benchmark: "
            f"{row_checks} != "
            f"{expected_row_checks}"
        )

    (
        benchmark_fingerprint,
        manifest_details,
    ) = combined_manifest_fingerprint(
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

    if (
        benchmark_fingerprint
        != expected_benchmark_fingerprint
    ):
        raise RuntimeError(
            "Combined benchmark fingerprint "
            "does not match the frozen benchmark."
        )

    transform_config = (
        load_transform_config(
            config.transform_config
        )
    )

    deterministic_transform = (
        build_ssl_evaluation_transform(
            transform_config
        )
    )

    training_transform = (
        build_ssl_train_transform(
            transform_config
        )
    )

    evaluation_dataset_full = (
        UnifiedCrystalDataset(
            config.train_manifest,
            config.image_root_map,
            transform=(
                deterministic_transform
            ),
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

    training_dataset_full = (
        UnifiedCrystalDataset(
            config.train_manifest,
            config.image_root_map,
            transform=training_transform,
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

    evaluation_dataset = limited_dataset(
        evaluation_dataset_full,
        config.limit_train_items,
    )

    training_dataset = limited_dataset(
        training_dataset_full,
        config.limit_train_items,
    )

    expected_crystal_ids, expected_orders = (
        manifest_arrays(
            evaluation_dataset_full,
            config.limit_train_items,
        )
    )

    if len(evaluation_dataset) != len(
        training_dataset
    ):
        raise RuntimeError(
            "Evaluation and training dataset "
            "lengths differ."
        )

    if len(expected_crystal_ids) != len(
        evaluation_dataset
    ):
        raise RuntimeError(
            "Expected metadata length differs "
            "from limited dataset length."
        )

    if config.cluster_count > len(
        training_dataset
    ):
        raise RuntimeError(
            "cluster_count exceeds the "
            "available training rows."
        )

    if config.pca_components > min(
        len(training_dataset),
        512,
    ):
        raise RuntimeError(
            "pca_components exceeds the "
            "smoke-run feature matrix."
        )

    # Initialise the shared backbone once; only the pseudo-label classifier is
    # reset as cluster assignments change between cycles.
    model = DeepClusterModel(
        backbone_name=(
            config.backbone_name
        ),
        cluster_count=(
            config.cluster_count
        ),
        initialization=(
            config.initialization
        ),
    ).to(device)

    if not resuming:
        save_run_record(
            metadata_root
            / "run_record.json",
            model=model.model_metadata(),
            manifest=config.train_manifest,
            seed=config.seed,
            config=config.to_dict(),
        )

        write_json_atomic(
            metadata_root
            / "benchmark_provenance.json",
            {
                "benchmark_id": (
                    config.benchmark_id
                ),
                "combined_manifest_fingerprint": (
                    benchmark_fingerprint
                ),
                "expected_fingerprint": (
                    expected_benchmark_fingerprint
                ),
                "manifest_details": (
                    manifest_details
                ),
                "test_split_accessed": False,
                "test_images_accessed": False,
            },
        )

        write_json_atomic(
            metadata_root
            / "environment.json",
            {
                "python": (
                    platform.python_version()
                ),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "cuda_available": (
                    torch.cuda.is_available()
                ),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0)
                    if device.type == "cuda"
                    else None
                ),
                "created_utc": (
                    datetime.now(
                        timezone.utc
                    )
                    .replace(
                        microsecond=0
                    )
                    .isoformat()
                ),
            },
        )

    history: list[
        dict[str, object]
    ] = []

    previous_assignments: (
        np.ndarray | None
    ) = None

    start_cycle = 1
    resume_record = None

    if arguments.resume is not None:
        (
            start_cycle,
            history,
            previous_assignments,
            resume_record,
        ) = load_resume_state(
            checkpoint_path=(
                arguments.resume
            ),
            run_root=run_root,
            model=model,
            config=config,
            benchmark_fingerprint=(
                benchmark_fingerprint
            ),
            expected_crystal_ids=(
                expected_crystal_ids
            ),
            expected_manifest_orders=(
                expected_orders
            ),
        )

        write_json_atomic(
            metadata_root
            / (
                "resume_from_cycle_"
                f"{start_cycle - 1:03d}.json"
            ),
            {
                **resume_record,
                "resumed_utc": (
                    datetime.now(
                        timezone.utc
                    )
                    .replace(
                        microsecond=0
                    )
                    .isoformat()
                ),
                "target_cycles": (
                    config.cycles
                ),
            },
        )

    print("=== DEEPCLUSTER RUN ===")
    print("Run root:", run_root)
    print("Device:", device)
    print(
        "Training rows:",
        len(training_dataset),
    )
    print(
        "Validation export limit:",
        arguments.limit_validation_items,
    )
    print("Cycles:", config.cycles)
    print(
        "Starting cycle:",
        start_cycle,
    )
    print(
        "Resuming:",
        resuming,
    )
    print(
        "Cluster count:",
        config.cluster_count,
    )

    # Each DeepCluster cycle extracts current features, reclusters them, and then
    # trains for one epoch against the newly generated pseudo-labels.
    for cycle in range(
        start_cycle,
        config.cycles + 1,
    ):
        print()
        print(
            f"=== CYCLE "
            f"{cycle}/{config.cycles} ==="
        )

        cycle_root = (
            cycles_root
            / f"cycle_{cycle:03d}"
        )

        cycle_root.mkdir(
            parents=True,
            exist_ok=False,
        )

        evaluation_loader = make_loader(
            evaluation_dataset,
            batch_size=(
                config.evaluation_batch_size
            ),
            num_workers=(
                config.num_workers
            ),
            device=device,
            shuffle=False,
            sampler=None,
            seed=(
                config.seed
                + cycle
            ),
        )

        extracted = (
            extract_backbone_features(
                model=model,
                loader=evaluation_loader,
                expected_crystal_ids=(
                    expected_crystal_ids
                ),
                expected_manifest_orders=(
                    expected_orders
                ),
                device=device,
                use_amp=config.use_amp,
            )
        )

        preprocessing = (
            preprocess_features(
                extracted.features,
                pca_components=(
                    config.pca_components
                ),
                whiten=(
                    config.pca_whiten
                ),
                solver=(
                    config.pca_solver
                ),
                seed=(
                    config.seed
                    + cycle
                ),
            )
        )

        clustering = fit_kmeans(
            preprocessing.features,
            cluster_count=(
                config.cluster_count
            ),
            seed=(
                config.seed
                + cycle
            ),
            initialization=(
                config.kmeans_initialization
            ),
            n_init=config.kmeans_n_init,
            maximum_iterations=(
                config
                .kmeans_maximum_iterations
            ),
            tolerance=(
                config.kmeans_tolerance
            ),
            algorithm=(
                config.kmeans_algorithm
            ),
        )

        validate_cluster_health(
            clustering.statistics,
            required_occupied_cluster_fraction=(
                config
                .required_occupied_cluster_fraction
            ),
            minimum_normalised_cluster_entropy=(
                config
                .minimum_normalised_cluster_entropy
            ),
            maximum_largest_cluster_fraction=(
                config
                .maximum_largest_cluster_fraction
            ),
        )

        stability = None

        if previous_assignments is not None:
            stability = assignment_stability(
                clustering.assignments,
                previous_assignments,
            )

        save_npz_atomic(
            cycle_root
            / "cluster_assignments.npz",
            crystal_ids=(
                extracted.crystal_ids
            ),
            manifest_orders=(
                extracted.manifest_orders
            ),
            assignments=(
                clustering.assignments
            ),
        )

        save_npz_atomic(
            cycle_root
            / "clustering_state.npz",
            centres=clustering.centres,
            pca_components=(
                preprocessing
                .pca
                .components_
            ),
            pca_mean=(
                preprocessing.pca.mean_
            ),
            pca_explained_variance=(
                preprocessing
                .pca
                .explained_variance_
            ),
            pca_explained_variance_ratio=(
                preprocessing
                .pca
                .explained_variance_ratio_
            ),
        )

        model.reset_classifier(
            seed=(
                config.seed
                + 10_000
                + cycle
            )
        )

        learning_rate = (
            cycle_learning_rate(
                cycle=cycle,
                total_cycles=(
                    config.cycles
                ),
                warmup_cycles=(
                    config.warmup_cycles
                ),
                learning_rate=(
                    config.learning_rate
                ),
                minimum_learning_rate=(
                    config
                    .minimum_learning_rate
                ),
            )
        )

        # Create the cycle optimizer after reclustering so the reset classifier is
        # included with the current model parameters.
        optimizer = SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.momentum,
            weight_decay=(
                config.weight_decay
            ),
        )

        set_optimizer_learning_rate(
            optimizer,
            learning_rate,
        )

        pseudo_label_dataset = (
            PseudoLabelDataset(
                training_dataset,
                clustering.assignments,
            )
        )

        samples_per_epoch = (
            config.samples_per_epoch
            if (
                config.samples_per_epoch
                is not None
            )
            else len(
                pseudo_label_dataset
            )
        )

        if (
            config
            .inverse_cluster_frequency_sampling
        ):
            sampler = (
                build_balanced_sampler(
                    clustering.assignments,
                    cluster_count=(
                        config.cluster_count
                    ),
                    samples_per_epoch=(
                        samples_per_epoch
                    ),
                    seed=(
                        config.seed
                        + 20_000
                        + cycle
                    ),
                )
            )
        else:
            sampler = None

        training_loader = make_loader(
            pseudo_label_dataset,
            batch_size=config.batch_size,
            num_workers=(
                config.num_workers
            ),
            device=device,
            shuffle=(
                sampler is None
            ),
            sampler=sampler,
            seed=(
                config.seed
                + 30_000
                + cycle
            ),
        )

        amp_enabled = bool(
            config.use_amp
            and device.type == "cuda"
        )

        scaler = (
            torch.amp.GradScaler(
                "cuda",
                enabled=True,
            )
            if amp_enabled
            else None
        )

        epoch_metrics = []

        for pseudo_epoch in range(
            1,
            config.epochs_per_cycle + 1,
        ):
            metrics = (
                train_pseudo_label_epoch(
                    model=model,
                    loader=training_loader,
                    optimizer=optimizer,
                    device=device,
                    use_amp=(
                        config.use_amp
                    ),
                    gradient_clip_norm=(
                        config
                        .gradient_clip_norm
                    ),
                    scaler=scaler,
                )
            )

            epoch_metrics.append(
                {
                    "pseudo_epoch": (
                        pseudo_epoch
                    ),
                    "loss": metrics.loss,
                    "accuracy": (
                        metrics.accuracy
                    ),
                    "batches": (
                        metrics.batches
                    ),
                    "images": (
                        metrics.images
                    ),
                    "seconds": (
                        metrics.seconds
                    ),
                    "learning_rate": (
                        metrics
                        .learning_rate
                    ),
                }
            )

        cycle_record = {
            "cycle": cycle,
            "feature_rows": (
                extracted.row_count
            ),
            "feature_dimension": (
                extracted
                .feature_dimension
            ),
            "feature_seconds": (
                extracted.seconds
            ),
            "pca_components": (
                config.pca_components
            ),
            "pca_retained_variance": (
                preprocessing
                .retained_variance
            ),
            "pca_seconds": (
                preprocessing.seconds
            ),
            "cluster_count": (
                config.cluster_count
            ),
            "occupied_clusters": (
                clustering
                .statistics
                .occupied_clusters
            ),
            "empty_clusters": (
                clustering
                .statistics
                .empty_clusters
            ),
            "minimum_cluster_size": (
                clustering
                .statistics
                .minimum_cluster_size
            ),
            "median_cluster_size": (
                clustering
                .statistics
                .median_cluster_size
            ),
            "maximum_cluster_size": (
                clustering
                .statistics
                .maximum_cluster_size
            ),
            "largest_cluster_fraction": (
                clustering
                .statistics
                .largest_cluster_fraction
            ),
            "normalised_cluster_entropy": (
                clustering
                .statistics
                .normalised_cluster_entropy
            ),
            "kmeans_inertia": (
                clustering.inertia
            ),
            "kmeans_iterations": (
                clustering.iterations
            ),
            "kmeans_seconds": (
                clustering.seconds
            ),
            "assignment_sha256": (
                clustering
                .assignment_sha256
            ),
            "centres_sha256": (
                clustering
                .centres_sha256
            ),
            "assignment_stability": (
                stability
            ),
            "learning_rate": (
                learning_rate
            ),
            "pseudo_label_epochs": (
                epoch_metrics
            ),
        }

        write_json_atomic(
            cycle_root
            / "cycle_summary.json",
            cycle_record,
        )

        history_row = {
            "cycle": cycle,
            "feature_seconds": (
                extracted.seconds
            ),
            "pca_retained_variance": (
                preprocessing
                .retained_variance
            ),
            "pca_seconds": (
                preprocessing.seconds
            ),
            "occupied_clusters": (
                clustering
                .statistics
                .occupied_clusters
            ),
            "minimum_cluster_size": (
                clustering
                .statistics
                .minimum_cluster_size
            ),
            "median_cluster_size": (
                clustering
                .statistics
                .median_cluster_size
            ),
            "maximum_cluster_size": (
                clustering
                .statistics
                .maximum_cluster_size
            ),
            "largest_cluster_fraction": (
                clustering
                .statistics
                .largest_cluster_fraction
            ),
            "normalised_cluster_entropy": (
                clustering
                .statistics
                .normalised_cluster_entropy
            ),
            "kmeans_inertia": (
                clustering.inertia
            ),
            "kmeans_iterations": (
                clustering.iterations
            ),
            "kmeans_seconds": (
                clustering.seconds
            ),
            "assignment_ari_previous": (
                None
                if stability is None
                else stability[
                    "adjusted_rand_index"
                ]
            ),
            "assignment_nmi_previous": (
                None
                if stability is None
                else stability[
                    "normalised_mutual_information"
                ]
            ),
            "pseudo_label_loss": (
                epoch_metrics[-1][
                    "loss"
                ]
            ),
            "pseudo_label_accuracy": (
                epoch_metrics[-1][
                    "accuracy"
                ]
            ),
            "training_seconds": (
                sum(
                    float(item["seconds"])
                    for item
                    in epoch_metrics
                )
            ),
            "learning_rate": (
                learning_rate
            ),
            "assignment_sha256": (
                clustering
                .assignment_sha256
            ),
            "centres_sha256": (
                clustering
                .centres_sha256
            ),
        }

        history.append(
            history_row
        )

        write_csv_atomic(
            run_root
            / "training_history.csv",
            history,
        )

        if (
            cycle
            % config.checkpoint_every
            == 0
            or cycle == config.cycles
        ):
            checkpoint_path = (
                checkpoints_root
                / (
                    "deepcluster_"
                    f"cycle_{cycle:03d}.pt"
                )
            )

            save_checkpoint_atomic(
                checkpoint_path,
                {
                    "schema_version": (
                        "plagnetplus_"
                        "deepcluster_"
                        "checkpoint_v1"
                    ),
                    "cycle": cycle,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "config": (
                        config.to_dict()
                    ),
                    "benchmark_fingerprint": (
                        benchmark_fingerprint
                    ),
                    "assignment_sha256": (
                        clustering
                        .assignment_sha256
                    ),
                    "centres_sha256": (
                        clustering
                        .centres_sha256
                    ),
                    "cluster_statistics": (
                        clustering
                        .statistics
                        .to_dict()
                    ),
                    "numpy_random_state": (
                        np.random.get_state()
                    ),
                    "python_random_state": (
                        random.getstate()
                    ),
                    "torch_random_state": (
                        torch.get_rng_state()
                    ),
                    "torch_cuda_random_state": (
                        torch.cuda
                        .get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
            )

            print(
                "Checkpoint:",
                checkpoint_path,
            )

        previous_assignments = (
            clustering
            .assignments
            .copy()
        )

        print(
            "Occupied clusters:",
            clustering
            .statistics
            .occupied_clusters,
        )

        print(
            "Cluster entropy:",
            clustering
            .statistics
            .normalised_cluster_entropy,
        )

        print(
            "Pseudo-label loss:",
            epoch_metrics[-1][
                "loss"
            ],
        )

        print(
            "Pseudo-label accuracy:",
            epoch_metrics[-1][
                "accuracy"
            ],
        )

    validation_dataset_full = (
        UnifiedCrystalDataset(
            config.validation_manifest,
            config.image_root_map,
            transform=(
                deterministic_transform
            ),
            verify_files=(
                config.verify_files
            ),
            verify_sample_size=(
                min(
                    config.verify_sample_size,
                    256,
                )
            ),
            verify_seed=config.seed,
            include_metadata=True,
        )
    )

    validation_dataset = limited_dataset(
        validation_dataset_full,
        arguments.limit_validation_items,
    )

    (
        validation_ids,
        validation_orders,
    ) = manifest_arrays(
        validation_dataset_full,
        arguments.limit_validation_items,
    )

    validation_loader = make_loader(
        validation_dataset,
        batch_size=(
            config.evaluation_batch_size
        ),
        num_workers=(
            config.num_workers
        ),
        device=device,
        shuffle=False,
        sampler=None,
        seed=(
            config.seed
            + 90_000
        ),
    )

    validation_embeddings = (
        extract_backbone_features(
            model=model,
            loader=validation_loader,
            expected_crystal_ids=(
                validation_ids
            ),
            expected_manifest_orders=(
                validation_orders
            ),
            device=device,
            use_amp=config.use_amp,
        )
    )

    validation_embedding_path = (
        embeddings_root
        / "validation_backbone_embeddings.npz"
    )

    save_npz_atomic(
        validation_embedding_path,
        features=(
            validation_embeddings.features
        ),
        crystal_ids=(
            validation_embeddings.crystal_ids
        ),
        manifest_orders=(
            validation_embeddings
            .manifest_orders
        ),
    )

    final_checkpoint = (
        checkpoints_root
        / (
            "deepcluster_"
            f"cycle_{config.cycles:03d}.pt"
        )
    )

    checks = {
        "benchmark_fingerprint_expected": (
            benchmark_fingerprint
            == expected_benchmark_fingerprint
        ),
        "cycle_count_complete": (
            len(history)
            == config.cycles
        ),
        "all_clusters_occupied": all(
            int(row["occupied_clusters"])
            == config.cluster_count
            for row in history
        ),
        "all_losses_finite": all(
            np.isfinite(
                float(
                    row[
                        "pseudo_label_loss"
                    ]
                )
            )
            for row in history
        ),
        "validation_embeddings_finite": bool(
            np.isfinite(
                validation_embeddings
                .features
            ).all()
        ),
        "validation_ids_aligned": bool(
            np.array_equal(
                validation_embeddings
                .crystal_ids,
                validation_ids,
            )
        ),
        "validation_orders_aligned": bool(
            np.array_equal(
                validation_embeddings
                .manifest_orders,
                validation_orders,
            )
        ),
        "final_checkpoint_exists": (
            final_checkpoint.is_file()
        ),
        "test_split_untouched": True,
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    summary = {
        "schema_version": (
            "plagnetplus_"
            "deepcluster_run_summary_v1"
        ),
        "status": status,
        "run_name": config.run_name,
        "run_root": str(
            run_root.resolve()
        ),
        "device": str(device),
        "cycles_completed": len(
            history
        ),
        "training_rows": len(
            training_dataset
        ),
        "validation_embedding_rows": (
            validation_embeddings
            .row_count
        ),
        "validation_embedding_dimension": (
            validation_embeddings
            .feature_dimension
        ),
        "benchmark_fingerprint": (
            benchmark_fingerprint
        ),
        "test_split_accessed": False,
        "test_images_accessed": False,
        "resumed": resuming,
        "resume_record": resume_record,
        "starting_cycle": start_cycle,
        "final_checkpoint": {
            "path": str(
                final_checkpoint.resolve()
            ),
            "sha256": sha256_file(
                final_checkpoint
            ),
        },
        "validation_embeddings": {
            "path": str(
                validation_embedding_path
                .resolve()
            ),
            "sha256": sha256_file(
                validation_embedding_path
            ),
        },
        "checks": checks,
        "completed_utc": (
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat()
        ),
    }

    write_json_atomic(
        run_root
        / "run_summary.json",
        summary,
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "DEEPCLUSTER TRAINING RUN:",
        status,
    )

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
