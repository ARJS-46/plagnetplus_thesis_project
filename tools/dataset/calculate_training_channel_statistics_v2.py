#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from plagnetplus.data.ssl_transforms import (
    SSLTransformConfig,
    build_ssl_evaluation_transform,
)
from plagnetplus.data.unified_dataset import (
    UnifiedCrystalDataset,
)


EXPECTED_TRAIN_ROWS = 91_357


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--irp-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PNP_PROJECT_ROOT",
                str(Path(__file__).resolve().parents[3]),
            )
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id

    worker_seed = (
        torch.initial_seed()
        % 2**32
    )

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def write_json(
    path: Path,
    value: Any,
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
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".partial"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    temporary.replace(path)


def main() -> None:
    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive."
        )

    if arguments.num_workers < 0:
        raise ValueError(
            "num-workers cannot be negative."
        )

    irp_root = (
        arguments.irp_root
        .expanduser()
        .resolve()
    )

    output_root = (
        arguments.output_root
        .expanduser()
        .resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(
        arguments.seed
    )

    benchmark_root = (
        irp_root
        / "data/processed/plagnetplus"
        / "ssl_unified_v2"
        / "benchmarks"
        / "pooled_broad_p80_10_10_v2"
    )

    train_manifest = (
        benchmark_root
        / "manifests"
        / "crystals_train.csv"
    )

    root_map = (
        irp_root
        / "data/processed/plagnetplus"
        / "ssl_unified_v2"
        / "metadata"
        / "image_root_map.json"
    )

    transform_config = SSLTransformConfig(
        image_size=224,
        normalization_mean=None,
        normalization_std=None,
    )

    dataset = UnifiedCrystalDataset(
        train_manifest,
        root_map,
        transform=build_ssl_evaluation_transform(
            transform_config
        ),
        verify_files="sample",
        verify_sample_size=1_024,
        verify_seed=arguments.seed,
        include_metadata=False,
    )

    if len(dataset) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Unexpected training row count: "
            f"{len(dataset)}"
        )

    generator = torch.Generator()
    generator.manual_seed(
        arguments.seed
    )

    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=(
            arguments.num_workers > 0
        ),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    channel_sum = torch.zeros(
        3,
        dtype=torch.float64,
    )

    channel_sum_sq = torch.zeros(
        3,
        dtype=torch.float64,
    )

    foreground_sum = torch.zeros(
        3,
        dtype=torch.float64,
    )

    foreground_sum_sq = torch.zeros(
        3,
        dtype=torch.float64,
    )

    spatial_pixel_count = 0
    foreground_pixel_count = 0
    image_count = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            images = batch["image"].to(
                dtype=torch.float64
            )

            if images.ndim != 4:
                raise RuntimeError(
                    "Expected BCHW image tensor, "
                    f"received shape {tuple(images.shape)}"
                )

            if images.shape[1:] != (
                3,
                224,
                224,
            ):
                raise RuntimeError(
                    "Unexpected image shape: "
                    f"{tuple(images.shape)}"
                )

            if not torch.isfinite(
                images
            ).all():
                raise RuntimeError(
                    "Encountered non-finite image tensor."
                )

            batch_images = images.shape[0]
            batch_spatial_pixels = (
                batch_images
                * images.shape[2]
                * images.shape[3]
            )

            channel_sum += images.sum(
                dim=(0, 2, 3)
            )

            channel_sum_sq += (
                images
                .square()
                .sum(dim=(0, 2, 3))
            )

            foreground_mask = (
                images.amax(
                    dim=1
                )
                > 0.0
            )

            foreground_count = int(
                foreground_mask.sum().item()
            )

            if foreground_count > 0:
                foreground_values = (
                    images.permute(
                        0,
                        2,
                        3,
                        1,
                    )[
                        foreground_mask
                    ]
                )

                foreground_sum += (
                    foreground_values.sum(
                        dim=0
                    )
                )

                foreground_sum_sq += (
                    foreground_values
                    .square()
                    .sum(dim=0)
                )

            spatial_pixel_count += (
                batch_spatial_pixels
            )

            foreground_pixel_count += (
                foreground_count
            )

            image_count += batch_images

            if (
                batch_index
                % arguments.progress_every
                == 0
                or image_count
                == len(dataset)
            ):
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                rate = (
                    image_count / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"images={image_count:,}/"
                    f"{len(dataset):,} "
                    f"batches={batch_index:,}/"
                    f"{len(loader):,} "
                    f"rate={rate:.1f} images/s",
                    flush=True,
                )

    if image_count != len(dataset):
        raise RuntimeError(
            "Processed image count does not match "
            f"the dataset: {image_count}"
        )

    mean = (
        channel_sum
        / spatial_pixel_count
    )

    variance = (
        channel_sum_sq
        / spatial_pixel_count
        - mean.square()
    ).clamp_min(0.0)

    standard_deviation = (
        variance.sqrt()
    )

    if foreground_pixel_count <= 0:
        raise RuntimeError(
            "No nonblack foreground pixels detected."
        )

    foreground_mean = (
        foreground_sum
        / foreground_pixel_count
    )

    foreground_variance = (
        foreground_sum_sq
        / foreground_pixel_count
        - foreground_mean.square()
    ).clamp_min(0.0)

    foreground_standard_deviation = (
        foreground_variance.sqrt()
    )

    black_background_fraction = (
        1.0
        - (
            foreground_pixel_count
            / spatial_pixel_count
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    checks = {
        "training_rows_expected": (
            image_count
            == EXPECTED_TRAIN_ROWS
        ),
        "all_pixel_mean_finite": bool(
            torch.isfinite(
                mean
            ).all()
        ),
        "all_pixel_std_finite": bool(
            torch.isfinite(
                standard_deviation
            ).all()
        ),
        "all_pixel_std_positive": bool(
            (
                standard_deviation
                > 0
            ).all()
        ),
        "foreground_mean_finite": bool(
            torch.isfinite(
                foreground_mean
            ).all()
        ),
        "foreground_std_finite": bool(
            torch.isfinite(
                foreground_standard_deviation
            ).all()
        ),
        "foreground_std_positive": bool(
            (
                foreground_standard_deviation
                > 0
            ).all()
        ),
        "background_fraction_valid": (
            0.0
            <= black_background_fraction
            < 1.0
        ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    statistics = {
        "schema_version": (
            "plagnetplus_training_normalization_v2"
        ),
        "status": status,
        "benchmark_id": (
            "pooled_broad_p80_10_10_v2"
        ),
        "split_used": "train",
        "image_count": image_count,
        "image_size": 224,
        "spatial_pixel_count": (
            spatial_pixel_count
        ),
        "foreground_pixel_count": (
            foreground_pixel_count
        ),
        "black_background_fraction": (
            black_background_fraction
        ),
        "normalization_policy": (
            "pixel_weighted_all_pixels_after_"
            "deterministic_bicubic_resize_to_224"
        ),
        "normalization_mean": [
            float(value)
            for value in mean.tolist()
        ],
        "normalization_std": [
            float(value)
            for value
            in standard_deviation.tolist()
        ],
        "foreground_diagnostic_mean": [
            float(value)
            for value
            in foreground_mean.tolist()
        ],
        "foreground_diagnostic_std": [
            float(value)
            for value
            in foreground_standard_deviation.tolist()
        ],
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "images_per_second": (
            image_count
            / elapsed_seconds
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": (
                torch.version.cuda
            ),
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "numpy": np.__version__,
        },
        "source_manifest": str(
            train_manifest
        ),
        "source_manifest_sha256": (
            sha256_file(
                train_manifest
            )
        ),
        "image_root_map": str(
            root_map
        ),
        "image_root_map_sha256": (
            sha256_file(
                root_map
            )
        ),
        "checks": checks,
    }

    statistics_path = (
        output_root
        / "training_channel_statistics_v2.json"
    )

    write_json(
        statistics_path,
        statistics,
    )

    frozen_config = {
        "schema_version": (
            "plagnetplus_ssl_transform_v2"
        ),
        "benchmark_id": (
            "pooled_broad_p80_10_10_v2"
        ),
        "image_size": 224,
        "interpolation": "bicubic",
        "horizontal_flip_probability": 0.5,
        "vertical_flip_probability": 0.5,
        "quarter_turn_rotation": True,
        "jitter_probability": 0.8,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.0,
        "hue": 0.0,
        "blur_probability": 0.2,
        "blur_kernel_size": 23,
        "blur_sigma": [
            0.1,
            2.0,
        ],
        "normalization_mean": (
            statistics[
                "normalization_mean"
            ]
        ),
        "normalization_std": (
            statistics[
                "normalization_std"
            ]
        ),
        "normalization_source": (
            "training_channel_statistics_v2.json"
        ),
        "training_views": 2,
        "evaluation_is_deterministic": True,
    }

    config_path = (
        output_root
        / "ssl_transform_config_v2.json"
    )

    write_json(
        config_path,
        frozen_config,
    )

    readme = """# PlagNetPlus normalization v2

Statistics were calculated from the frozen pooled-broad training split only.

Images were deterministically resized to 224 × 224 using bicubic
interpolation, converted to float tensors in [0, 1], and accumulated using
float64 sums and squared sums.

The frozen normalization uses pixel-weighted statistics over all pixels,
including the black masked background, because those pixels are part of the
actual model input.

Foreground-only statistics are reported as diagnostics and are not used for
the primary normalization contract.
"""

    readme_path = (
        output_root
        / "README.md"
    )

    write_text(
        readme_path,
        readme,
    )

    checksum_targets = [
        statistics_path,
        config_path,
        readme_path,
    ]

    checksum_lines = []

    for path in sorted(
        checksum_targets
    ):
        checksum_lines.append(
            f"{sha256_file(path)}  "
            f"{path.name}\n"
        )

    write_text(
        output_root / "SHA256SUMS",
        "".join(checksum_lines),
    )

    print()
    print(
        json.dumps(
            statistics,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "TRAINING CHANNEL STATISTICS:",
        status,
    )

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
