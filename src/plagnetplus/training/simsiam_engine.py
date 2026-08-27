"""Training, validation and checkpoint utilities for native SimSiam."""

from __future__ import annotations

import csv
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


DIAGNOSTIC_NAMES = (
    "mean_cosine_similarity",
    "projection_standard_deviation",
    "collapse_level",
    "backbone_feature_norm",
    "projection_norm",
    "prediction_norm",
)


@dataclass(frozen=True)
class SimSiamEpochMetrics:
    """Aggregated measurements for one train or validation epoch."""

    epoch: int
    split: str
    loss: float
    mean_cosine_similarity: float
    projection_standard_deviation: float
    collapse_level: float
    backbone_feature_norm: float
    projection_norm: float
    prediction_norm: float
    backbone_projector_learning_rate: float
    predictor_learning_rate: float
    batches: int
    images: int
    seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable representation."""

        return asdict(self)


def learning_rate_for_epoch(
    *,
    epoch_index: int,
    total_epochs: int,
    warmup_epochs: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    """Return linear warm-up followed by cosine decay."""

    if total_epochs <= 0:
        raise ValueError(
            "total_epochs must be positive."
        )

    if not 0 <= epoch_index < total_epochs:
        raise ValueError(
            "epoch_index is outside the training range."
        )

    if not 0 <= warmup_epochs <= total_epochs:
        raise ValueError(
            "warmup_epochs must lie between zero and total_epochs."
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
            "minimum_learning_rate must lie between zero "
            "and base_learning_rate."
        )

    if (
        warmup_epochs > 0
        and epoch_index < warmup_epochs
    ):
        return base_learning_rate * (
            (epoch_index + 1)
            / warmup_epochs
        )

    decay_epochs = (
        total_epochs
        - warmup_epochs
    )

    if decay_epochs <= 1:
        return base_learning_rate

    decay_index = (
        epoch_index
        - warmup_epochs
    )

    progress = decay_index / (
        decay_epochs - 1
    )

    progress = min(
        1.0,
        max(0.0, progress),
    )

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )

    return (
        minimum_learning_rate
        + (
            base_learning_rate
            - minimum_learning_rate
        )
        * cosine
    )


def build_simsiam_optimizer(
    *,
    model: nn.Module,
    learning_rate: float,
    predictor_learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.SGD:
    """Build SGD with scheduled base and fixed predictor groups."""

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    if predictor_learning_rate <= 0:
        raise ValueError(
            "predictor_learning_rate must be positive."
        )

    predictor = getattr(
        model,
        "predictor",
        None,
    )

    if not isinstance(
        predictor,
        nn.Module,
    ):
        raise TypeError(
            "SimSiam model must expose a predictor module."
        )

    # Keep backbone/projector and predictor in separate optimizer groups because
    # SimSiam uses independently controlled learning rates for these components.
    predictor_parameters = [
        parameter
        for parameter in predictor.parameters()
        if parameter.requires_grad
    ]

    predictor_parameter_ids = {
        id(parameter)
        for parameter in predictor_parameters
    }

    backbone_projector_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter)
        not in predictor_parameter_ids
    ]

    if not backbone_projector_parameters:
        raise ValueError(
            "No trainable backbone/projector parameters were found."
        )

    if not predictor_parameters:
        raise ValueError(
            "No trainable predictor parameters were found."
        )

    all_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    grouped_parameter_ids = {
        id(parameter)
        for parameter in (
            backbone_projector_parameters
            + predictor_parameters
        )
    }

    if grouped_parameter_ids != all_parameter_ids:
        raise RuntimeError(
            "Optimizer parameter groups do not cover the model exactly."
        )

    return torch.optim.SGD(
        [
            {
                "params": (
                    backbone_projector_parameters
                ),
                "lr": learning_rate,
                "group_name": (
                    "backbone_projector"
                ),
            },
            {
                "params": predictor_parameters,
                "lr": predictor_learning_rate,
                "group_name": "predictor",
            },
        ],
        momentum=momentum,
        weight_decay=weight_decay,
    )


def set_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    backbone_projector_learning_rate: float,
    predictor_learning_rate: float,
) -> None:
    """Apply the scheduled and fixed learning rates."""

    if backbone_projector_learning_rate < 0:
        raise ValueError(
            "Backbone/projector learning rate cannot be negative."
        )

    if predictor_learning_rate <= 0:
        raise ValueError(
            "Predictor learning rate must be positive."
        )

    observed_groups: set[str] = set()

    for parameter_group in optimizer.param_groups:
        group_name = parameter_group.get(
            "group_name"
        )

        if group_name == "backbone_projector":
            parameter_group["lr"] = (
                backbone_projector_learning_rate
            )
        elif group_name == "predictor":
            parameter_group["lr"] = (
                predictor_learning_rate
            )
        else:
            raise ValueError(
                "Unexpected optimizer parameter group: "
                f"{group_name!r}."
            )

        observed_groups.add(
            str(group_name)
        )

    if observed_groups != {
        "backbone_projector",
        "predictor",
    }:
        raise ValueError(
            "Both SimSiam optimizer groups are required."
        )


def optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Return backbone/projector and predictor learning rates."""

    rates: dict[str, float] = {}

    for parameter_group in optimizer.param_groups:
        group_name = str(
            parameter_group.get(
                "group_name"
            )
        )

        rates[group_name] = float(
            parameter_group["lr"]
        )

    required = {
        "backbone_projector",
        "predictor",
    }

    if set(rates) != required:
        raise ValueError(
            "Optimizer does not contain the required SimSiam groups."
        )

    return (
        rates["backbone_projector"],
        rates["predictor"],
    )


def _extract_views(
    batch: Mapping[str, object],
) -> tuple[Tensor, Tensor]:
    images = batch.get("image")

    if not isinstance(
        images,
        (tuple, list),
    ):
        raise TypeError(
            "Expected batch['image'] to contain two augmented views."
        )

    if len(images) != 2:
        raise ValueError(
            "SimSiam requires exactly two augmented views."
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

    if first.ndim < 2:
        raise ValueError(
            "Augmented views must include a batch dimension."
        )

    return first, second


def _objective(
    model: nn.Module,
    views: Sequence[Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    method = getattr(
        model,
        "simsiam_objective",
        None,
    )

    if not callable(method):
        raise TypeError(
            "Model must provide simsiam_objective()."
        )

    result = method(views)

    if not isinstance(
        result,
        tuple,
    ) or len(result) != 2:
        raise TypeError(
            "simsiam_objective() must return "
            "(loss, diagnostics)."
        )

    loss = result[0]
    diagnostics = result[1]

    if not isinstance(
        loss,
        Tensor,
    ):
        raise TypeError(
            "SimSiam loss must be a tensor."
        )

    if loss.ndim != 0:
        raise ValueError(
            "SimSiam loss must be scalar."
        )

    if not isinstance(
        diagnostics,
        Mapping,
    ):
        raise TypeError(
            "SimSiam diagnostics must be a mapping."
        )

    validated: dict[str, Tensor] = {}

    for name in DIAGNOSTIC_NAMES:
        value = diagnostics.get(name)

        if not isinstance(
            value,
            Tensor,
        ):
            raise TypeError(
                f"Diagnostic {name!r} must be a tensor."
            )

        if value.ndim != 0:
            raise ValueError(
                f"Diagnostic {name!r} must be scalar."
            )

        validated[name] = value

    return loss, validated


def gradients_are_finite(
    model: nn.Module,
) -> bool:
    """Return whether every populated model gradient is finite."""

    found_gradient = False

    for parameter in model.parameters():
        gradient = parameter.grad

        if gradient is None:
            continue

        found_gradient = True

        if not torch.isfinite(
            gradient
        ).all():
            return False

    return found_gradient


def _aggregate_metrics(
    *,
    epoch: int,
    split: str,
    total_loss: float,
    diagnostic_totals: Mapping[str, float],
    batches: int,
    images: int,
    seconds: float,
    backbone_projector_learning_rate: float,
    predictor_learning_rate: float,
) -> SimSiamEpochMetrics:
    if batches <= 0 or images <= 0:
        raise RuntimeError(
            f"No batches were processed for split {split!r}."
        )

    return SimSiamEpochMetrics(
        epoch=epoch,
        split=split,
        loss=total_loss / images,
        mean_cosine_similarity=(
            diagnostic_totals[
                "mean_cosine_similarity"
            ]
            / images
        ),
        projection_standard_deviation=(
            diagnostic_totals[
                "projection_standard_deviation"
            ]
            / images
        ),
        collapse_level=(
            diagnostic_totals[
                "collapse_level"
            ]
            / images
        ),
        backbone_feature_norm=(
            diagnostic_totals[
                "backbone_feature_norm"
            ]
            / images
        ),
        projection_norm=(
            diagnostic_totals[
                "projection_norm"
            ]
            / images
        ),
        prediction_norm=(
            diagnostic_totals[
                "prediction_norm"
            ]
            / images
        ),
        backbone_projector_learning_rate=(
            backbone_projector_learning_rate
        ),
        predictor_learning_rate=(
            predictor_learning_rate
        ),
        batches=batches,
        images=images,
        seconds=seconds,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    backbone_projector_learning_rate: float,
    predictor_learning_rate: float,
    use_amp: bool,
    scaler: Any | None,
    gradient_clip_norm: float | None,
    require_finite_loss: bool = True,
    require_finite_gradients: bool = True,
) -> SimSiamEpochMetrics:
    """Train SimSiam for one complete epoch."""

    set_optimizer_learning_rates(
        optimizer,
        backbone_projector_learning_rate=(
            backbone_projector_learning_rate
        ),
        predictor_learning_rate=(
            predictor_learning_rate
        ),
    )

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    if amp_enabled and scaler is None:
        raise ValueError(
            "AMP training requires a gradient scaler."
        )

    model.train()

    total_loss = 0.0
    diagnostic_totals = {
        name: 0.0
        for name in DIAGNOSTIC_NAMES
    }

    total_images = 0
    batches = 0
    started = time.perf_counter()

    # Both views are required for the symmetric stop-gradient objective, so they
    # are transferred and evaluated as an aligned pair.
    for batch in loader:
        first_view, second_view = (
            _extract_views(batch)
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

        batch_size = int(
            first_view.shape[0]
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            loss, diagnostics = _objective(
                model,
                (
                    first_view,
                    second_view,
                ),
            )

        if (
            require_finite_loss
            and not torch.isfinite(loss)
        ):
            raise FloatingPointError(
                "Non-finite SimSiam training loss encountered."
            )

        for name, value in diagnostics.items():
            if not torch.isfinite(value):
                raise FloatingPointError(
                    "Non-finite SimSiam diagnostic "
                    f"encountered: {name}."
                )

        if amp_enabled:
            scaler.scale(loss).backward()
            # Unscale AMP gradients before finite checks and clipping so thresholds
            # remain consistent across precision modes.
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if (
            require_finite_gradients
            and not gradients_are_finite(
                model
            )
        ):
            raise FloatingPointError(
                "Non-finite or absent SimSiam gradients encountered."
            )

        if gradient_clip_norm is not None:
            clipped_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                )
            )

            if not torch.isfinite(
                clipped_norm
            ):
                raise FloatingPointError(
                    "Gradient clipping returned a non-finite norm."
                )

        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += (
            float(
                loss.detach().cpu()
            )
            * batch_size
        )

        for name, value in diagnostics.items():
            diagnostic_totals[name] += (
                float(
                    value.detach().cpu()
                )
                * batch_size
            )

        total_images += batch_size
        batches += 1

    elapsed = (
        time.perf_counter()
        - started
    )

    observed_base_lr, observed_predictor_lr = (
        optimizer_learning_rates(
            optimizer
        )
    )

    return _aggregate_metrics(
        epoch=epoch,
        split="train",
        total_loss=total_loss,
        diagnostic_totals=(
            diagnostic_totals
        ),
        batches=batches,
        images=total_images,
        seconds=elapsed,
        backbone_projector_learning_rate=(
            observed_base_lr
        ),
        predictor_learning_rate=(
            observed_predictor_lr
        ),
    )


def evaluate_one_epoch(
    *,
    model: nn.Module,
    loader: Any,
    device: torch.device,
    epoch: int,
    backbone_projector_learning_rate: float,
    predictor_learning_rate: float,
    use_amp: bool,
    require_finite_loss: bool = True,
) -> SimSiamEpochMetrics:
    """Evaluate the self-supervised objective without optimisation."""

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    model.eval()

    total_loss = 0.0
    diagnostic_totals = {
        name: 0.0
        for name in DIAGNOSTIC_NAMES
    }

    total_images = 0
    batches = 0
    started = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            first_view, second_view = (
                _extract_views(batch)
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

            batch_size = int(
                first_view.shape[0]
            )

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                loss, diagnostics = (
                    _objective(
                        model,
                        (
                            first_view,
                            second_view,
                        ),
                    )
                )

            if (
                require_finite_loss
                and not torch.isfinite(loss)
            ):
                raise FloatingPointError(
                    "Non-finite SimSiam validation loss encountered."
                )

            for name, value in diagnostics.items():
                if not torch.isfinite(value):
                    raise FloatingPointError(
                        "Non-finite SimSiam validation "
                        f"diagnostic encountered: {name}."
                    )

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * batch_size
            )

            for name, value in diagnostics.items():
                diagnostic_totals[name] += (
                    float(
                        value.detach().cpu()
                    )
                    * batch_size
                )

            total_images += batch_size
            batches += 1

    elapsed = (
        time.perf_counter()
        - started
    )

    return _aggregate_metrics(
        epoch=epoch,
        split="validation",
        total_loss=total_loss,
        diagnostic_totals=(
            diagnostic_totals
        ),
        batches=batches,
        images=total_images,
        seconds=elapsed,
        backbone_projector_learning_rate=(
            backbone_projector_learning_rate
        ),
        predictor_learning_rate=(
            predictor_learning_rate
        ),
    )


def capture_rng_state() -> dict[str, object]:
    """Capture Python, NumPy and Torch RNG states."""

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
    """Restore Python, NumPy and Torch RNG states."""

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
    """Atomically save a resumable SimSiam checkpoint."""

    if completed_epoch <= 0:
        raise ValueError(
            "completed_epoch must be positive."
        )

    destination = Path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix
        + ".partial"
    )

    if temporary.exists():
        temporary.unlink()

    checkpoint: dict[str, object] = {
        "schema_version": (
            "plagnetplus_simsiam_checkpoint_v1"
        ),
        "completed_epoch": completed_epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scaler_state_dict": (
            None
            if scaler is None
            else scaler.state_dict()
        ),
        "history": [
            dict(row)
            for row in history
        ],
        "config": dict(config),
        "benchmark_id": benchmark_id,
        "manifest_fingerprint_sha256": (
            manifest_fingerprint_sha256
        ),
        "train_manifest_sha256": (
            train_manifest_sha256
        ),
        "rng_state": capture_rng_state(),
        "train_generator_state": (
            None
            if train_generator is None
            else train_generator.get_state()
        ),
    }

    torch.save(
        checkpoint,
        temporary,
    )

    temporary.replace(destination)


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
    """Load a checkpoint after validating experiment identity."""

    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    if (
        checkpoint.get(
            "schema_version"
        )
        != "plagnetplus_simsiam_checkpoint_v1"
    ):
        raise ValueError(
            "Unsupported SimSiam checkpoint schema."
        )

    if (
        checkpoint.get(
            "benchmark_id"
        )
        != expected_benchmark_id
    ):
        raise ValueError(
            "Checkpoint benchmark mismatch."
        )

    if (
        checkpoint.get(
            "manifest_fingerprint_sha256"
        )
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

    scaler_state = checkpoint.get(
        "scaler_state_dict"
    )

    if (
        scaler is not None
        and scaler_state is not None
    ):
        scaler.load_state_dict(
            scaler_state
        )

    generator_state = checkpoint.get(
        "train_generator_state"
    )

    if (
        train_generator is not None
        and generator_state is not None
    ):
        train_generator.set_state(
            generator_state
        )

    if restore_rng:
        restore_rng_state(
            checkpoint["rng_state"]
        )

    return checkpoint


def write_history_csv_atomic(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Atomically write SimSiam training-history rows."""

    destination = Path(path)

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
        "mean_cosine_similarity",
        "projection_standard_deviation",
        "collapse_level",
        "backbone_feature_norm",
        "projection_norm",
        "prediction_norm",
        "backbone_projector_learning_rate",
        "predictor_learning_rate",
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
            writer.writerow(
                {
                    name: row[name]
                    for name in fieldnames
                }
            )

    temporary.replace(destination)


__all__ = [
    "DIAGNOSTIC_NAMES",
    "SimSiamEpochMetrics",
    "build_simsiam_optimizer",
    "capture_rng_state",
    "evaluate_one_epoch",
    "gradients_are_finite",
    "learning_rate_for_epoch",
    "load_checkpoint",
    "optimizer_learning_rates",
    "restore_rng_state",
    "save_checkpoint_atomic",
    "set_optimizer_learning_rates",
    "train_one_epoch",
    "write_history_csv_atomic",
]
