"""Shared image transformations for PlagNetPlus SSL experiments."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ChannelStatistics = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class SSLTransformConfig:
    """Configuration for the common PlagNetPlus SSL image contract."""

    image_size: int = 224
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    jitter_probability: float = 0.8
    brightness: float = 0.2
    contrast: float = 0.2
    blur_probability: float = 0.2
    blur_kernel_size: int = 23
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0
    interpolation: str = "bicubic"
    normalization_mean: ChannelStatistics | None = None
    normalization_std: ChannelStatistics | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RandomQuarterTurn:
    """Rotate by 0, 90, 180, or 270 degrees with equal probability."""

    _OPERATIONS = (
        None,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
    )

    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:
        operation = random.choice(
            self._OPERATIONS
        )

        if operation is None:
            return image.copy()

        return image.transpose(operation)


def _validate_config(
    config: SSLTransformConfig,
) -> None:
    if config.image_size <= 0:
        raise ValueError(
            "image_size must be positive."
        )

    probabilities = {
        "horizontal_flip_probability":
        config.horizontal_flip_probability,
        "vertical_flip_probability":
        config.vertical_flip_probability,
        "jitter_probability":
        config.jitter_probability,
        "blur_probability":
        config.blur_probability,
    }

    for name, value in probabilities.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    if (
        config.blur_kernel_size < 3
        or config.blur_kernel_size % 2 == 0
    ):
        raise ValueError(
            "blur_kernel_size must be an odd integer "
            "greater than or equal to 3."
        )

    if (
        config.normalization_mean is None
    ) != (
        config.normalization_std is None
    ):
        raise ValueError(
            "normalization_mean and normalization_std "
            "must either both be provided or both be None."
        )

    if config.normalization_std is not None:
        if any(
            value <= 0
            for value in config.normalization_std
        ):
            raise ValueError(
                "All normalization standard deviations "
                "must be positive."
            )


def _normalization_steps(
    config: SSLTransformConfig,
) -> list[Callable]:
    if (
        config.normalization_mean is None
        or config.normalization_std is None
    ):
        return []

    return [
        transforms.Normalize(
            mean=config.normalization_mean,
            std=config.normalization_std,
        )
    ]


def build_ssl_train_transform(
    config: SSLTransformConfig | None = None,
) -> transforms.Compose:
    """Build one stochastic SSL view."""

    selected = (
        config
        if config is not None
        else SSLTransformConfig()
    )

    _validate_config(selected)

    operations: list[Callable] = [
        transforms.Resize(
            (
                selected.image_size,
                selected.image_size,
            ),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        RandomQuarterTurn(),
        transforms.RandomHorizontalFlip(
            p=selected.horizontal_flip_probability
        ),
        transforms.RandomVerticalFlip(
            p=selected.vertical_flip_probability
        ),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=selected.brightness,
                    contrast=selected.contrast,
                    saturation=0.0,
                    hue=0.0,
                )
            ],
            p=selected.jitter_probability,
        ),
        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=(
                        selected.blur_kernel_size
                    ),
                    sigma=(
                        selected.blur_sigma_min,
                        selected.blur_sigma_max,
                    ),
                )
            ],
            p=selected.blur_probability,
        ),
        transforms.ToTensor(),
        *_normalization_steps(selected),
    ]

    return transforms.Compose(
        operations
    )


def build_ssl_evaluation_transform(
    config: SSLTransformConfig | None = None,
) -> transforms.Compose:
    """Build the deterministic validation and embedding transform."""

    selected = (
        config
        if config is not None
        else SSLTransformConfig()
    )

    _validate_config(selected)

    operations: list[Callable] = [
        transforms.Resize(
            (
                selected.image_size,
                selected.image_size,
            ),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        *_normalization_steps(selected),
    ]

    return transforms.Compose(
        operations
    )
