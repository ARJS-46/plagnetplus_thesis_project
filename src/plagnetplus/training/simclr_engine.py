"""Reusable SimCLR training, evaluation, and checkpoint utilities."""

from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics for one training or validation epoch."""

    epoch: int
    split: str
    loss: float
    positive_pair_accuracy: float
    learning_rate: float
    batches: int
    images: int
    seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def learning_rate_for_epoch(
    *,
    epoch_index: int,
    total_epochs: int,
    warmup_epochs: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    """Return warm-up plus cosine-decay learning rate."""

    if total_epochs <= 0:
        raise ValueError(
            "total_epochs must be positive."
        )

    if not (
        0
        <= epoch_index
        < total_epochs
    ):
        raise ValueError(
            "epoch_index is outside the training range."
        )

    if not (
        0
        <= warmup_epochs
        <= total_epochs
    ):
        raise ValueError(
            "warmup_epochs must be between zero "
            "and total_epochs."
        )

    if base_learning_rate <= 0:
        raise ValueError(
            "base_learning_rate must be positive."
        )

    if not (
        0
        <= minimum_learning_rate
        <= base_learning_rate
    ):
        raise ValueError(
            "minimum_learning_rate must lie between "
            "zero and base_learning_rate."
        )

    if (
        warmup_epochs > 0
        and epoch_index < warmup_epochs
    ):
        return (
            base_learning_rate
            * float(
                epoch_index + 1
            )
            / float(
                warmup_epochs
            )
        )

    cosine_epochs = (
        total_epochs
        - warmup_epochs
    )

    if cosine_epochs <= 1:
        return base_learning_rate

    cosine_index = (
        epoch_index
        - warmup_epochs
    )

    progress = (
        float(cosine_index)
        / float(
            cosine_epochs - 1
        )
    )

    cosine_factor = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    return (
        minimum_learning_rate
        + (
            base_learning_rate
            - minimum_learning_rate
        )
        * cosine_factor
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    """Apply one learning rate to every optimizer group."""

    if learning_rate < 0:
        raise ValueError(
            "learning_rate cannot be negative."
        )

    for parameter_group in (
        optimizer.param_groups
    ):
        parameter_group[
            "lr"
        ] = learning_rate


def _extract_views(
    batch: Mapping[str, object],
) -> tuple[Tensor, Tensor]:
    images = batch.get(
        "image"
    )

    if not isinstance(
        images,
        (tuple, list),
    ):
        raise TypeError(
            "Expected batch['image'] to contain "
            "two augmented views."
        )

    if len(images) != 2:
        raise ValueError(
            "SimCLR requires exactly two views."
        )

    first = images[0]
    second = images[1]

    if not isinstance(
        first,
        Tensor,
    ) or not isinstance(
        second,
        Tensor,
    ):
        raise TypeError(
            "Both augmented views must be tensors."
        )

    if first.shape != second.shape:
        raise ValueError(
            "Augmented view batch shapes do not match."
        )

    return (
        first,
        second,
    )


def _objective(
    model: nn.Module,
    views: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    method = getattr(
        model,
        "contrastive_objective",
        None,
    )

    if not callable(method):
        raise TypeError(
            "Model must provide contrastive_objective()."
        )

    result = method(
        views
    )

    if not isinstance(
        result,
        tuple,
    ) or len(result) != 2:
        raise TypeError(
            "contrastive_objective() must return "
            "(loss, positive_pair_accuracy)."
        )

    loss = result[0]
    accuracy = result[1]

    if not isinstance(
        loss,
        Tensor,
    ) or not isinstance(
        accuracy,
        Tensor,
    ):
        raise TypeError(
            "Loss and accuracy must be tensors."
        )

    return (
        loss,
        accuracy,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    learning_rate: float,
    use_amp: bool,
    scaler: Any | None,
    gradient_clip_norm: float | None,
) -> EpochMetrics:
    """Train for one complete epoch."""

    set_optimizer_learning_rate(
        optimizer,
        learning_rate,
    )

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    total_images = 0
    batches = 0

    started = time.perf_counter()

    # Process the two independently augmented views together so positive pairs
    # remain aligned within each minibatch.
    for batch in loader:
        first_view, second_view = (
            _extract_views(
                batch
            )
        )

        first_view = first_view.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        second_view = second_view.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            loss, accuracy = (
                _objective(
                    model,
                    (
                        first_view,
                        second_view,
                    ),
                )
            )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Encountered non-finite training loss."
            )

        if not torch.isfinite(
            accuracy
        ):
            raise RuntimeError(
                "Encountered non-finite training accuracy."
            )

        if amp_enabled:
            if scaler is None:
                raise ValueError(
                    "AMP is enabled but no gradient "
                    "scaler was supplied."
                )

            scaler.scale(
                loss
            ).backward()

            if (
                gradient_clip_norm
                is not None
            ):
                # Unscale AMP gradients before clipping so the configured threshold
                # has the same meaning with and without mixed precision.
                scaler.unscale_(
                    optimizer
                )

                clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )

            scaler.step(
                optimizer
            )

            scaler.update()
        else:
            loss.backward()

            if (
                gradient_clip_norm
                is not None
            ):
                clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )

            optimizer.step()

        batch_size = int(
            first_view.shape[0]
        )

        total_loss += (
            float(
                loss.detach().item()
            )
            * batch_size
        )

        total_accuracy += (
            float(
                accuracy.detach().item()
            )
            * batch_size
        )

        total_images += batch_size
        batches += 1

    if total_images <= 0:
        raise RuntimeError(
            "Training loader produced no images."
        )

    seconds = (
        time.perf_counter()
        - started
    )

    return EpochMetrics(
        epoch=epoch,
        split="train",
        loss=(
            total_loss
            / total_images
        ),
        positive_pair_accuracy=(
            total_accuracy
            / total_images
        ),
        learning_rate=learning_rate,
        batches=batches,
        images=total_images,
        seconds=seconds,
    )


def evaluate_contrastive_epoch(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    learning_rate: float,
    use_amp: bool,
) -> EpochMetrics:
    """Evaluate the contrastive objective without optimization."""

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_images = 0
    batches = 0

    started = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            first_view, second_view = (
                _extract_views(
                    batch
                )
            )

            first_view = first_view.to(
                device,
                non_blocking=(
                    device.type
                    == "cuda"
                ),
            )

            second_view = second_view.to(
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
                loss, accuracy = (
                    _objective(
                        model,
                        (
                            first_view,
                            second_view,
                        ),
                    )
                )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Encountered non-finite validation loss."
                )

            if not torch.isfinite(
                accuracy
            ):
                raise RuntimeError(
                    "Encountered non-finite validation accuracy."
                )

            batch_size = int(
                first_view.shape[0]
            )

            total_loss += (
                float(
                    loss.item()
                )
                * batch_size
            )

            total_accuracy += (
                float(
                    accuracy.item()
                )
                * batch_size
            )

            total_images += (
                batch_size
            )

            batches += 1

    if total_images <= 0:
        raise RuntimeError(
            "Validation loader produced no images."
        )

    seconds = (
        time.perf_counter()
        - started
    )

    return EpochMetrics(
        epoch=epoch,
        split="validation",
        loss=(
            total_loss
            / total_images
        ),
        positive_pair_accuracy=(
            total_accuracy
            / total_images
        ),
        learning_rate=learning_rate,
        batches=batches,
        images=total_images,
        seconds=seconds,
    )


def capture_rng_state() -> dict[str, object]:
    """Capture Python, NumPy, and Torch RNG states."""

    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": (
            torch.get_rng_state()
        ),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: Mapping[str, object],
) -> None:
    """Restore Python, NumPy, and Torch RNG states."""

    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch.set_rng_state(
        state["torch_cpu"]
    )

    cuda_state = state.get(
        "torch_cuda"
    )

    if (
        cuda_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_state
        )


def save_checkpoint_atomic(
    path: str | Path,
    *,
    completed_epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    history: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    benchmark_id: str,
    manifest_fingerprint_sha256: str,
    train_manifest_sha256: str,
    train_generator: torch.Generator | None = None,
) -> None:
    """Atomically save a resumable training checkpoint."""

    destination = Path(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write checkpoints through a temporary file and atomic rename so an
    # interrupted job cannot leave a partially written checkpoint.
    temporary = destination.with_suffix(
        destination.suffix
        + ".partial"
    )

    checkpoint: dict[str, object] = {
        "schema_version": (
            "plagnetplus_simclr_checkpoint_v1"
        ),
        "completed_epoch": (
            completed_epoch
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "history": list(
            history
        ),
        "config": dict(
            config
        ),
        "benchmark_id": (
            benchmark_id
        ),
        "manifest_fingerprint_sha256": (
            manifest_fingerprint_sha256
        ),
        "train_manifest_sha256": (
            train_manifest_sha256
        ),
        "rng_state": (
            capture_rng_state()
        ),
    }

    if scaler is not None:
        checkpoint[
            "scaler_state_dict"
        ] = scaler.state_dict()

    if train_generator is not None:
        checkpoint[
            "train_generator_state"
        ] = train_generator.get_state()

    torch.save(
        checkpoint,
        temporary,
    )

    temporary.replace(
        destination
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_benchmark_id: str,
    expected_manifest_fingerprint_sha256: str,
    scaler: Any | None = None,
    train_generator: torch.Generator | None = None,
    restore_rng: bool = True,
) -> dict[str, object]:
    """Load a checkpoint after validating benchmark identity."""

    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    observed_benchmark = (
        checkpoint.get(
            "benchmark_id"
        )
    )

    if (
        observed_benchmark
        != expected_benchmark_id
    ):
        raise ValueError(
            "Checkpoint benchmark mismatch: "
            f"{observed_benchmark!r} != "
            f"{expected_benchmark_id!r}."
        )

    observed_fingerprint = (
        checkpoint.get(
            "manifest_fingerprint_sha256"
        )
    )

    if (
        observed_fingerprint
        != expected_manifest_fingerprint_sha256
    ):
        raise ValueError(
            "Checkpoint manifest fingerprint mismatch."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scaler_state = (
        checkpoint.get(
            "scaler_state_dict"
        )
    )

    if (
        scaler is not None
        and scaler_state is not None
    ):
        scaler.load_state_dict(
            scaler_state
        )

    generator_state = (
        checkpoint.get(
            "train_generator_state"
        )
    )

    if (
        train_generator is not None
        and generator_state
        is not None
    ):
        train_generator.set_state(
            generator_state
        )

    if restore_rng:
        rng_state = (
            checkpoint.get(
                "rng_state"
            )
        )

        if isinstance(
            rng_state,
            Mapping,
        ):
            restore_rng_state(
                rng_state
            )

    return checkpoint


def write_history_csv_atomic(
    path: str | Path,
    rows: Sequence[
        Mapping[str, object]
    ],
) -> None:
    """Atomically write training-history rows."""

    destination = Path(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix
        + ".partial"
    )

    fieldnames = [
        "epoch",
        "split",
        "loss",
        "positive_pair_accuracy",
        "learning_rate",
        "batches",
        "images",
        "seconds",
    ]

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                name: row.get(name)
                for name in fieldnames
            })

    temporary.replace(
        destination
    )
