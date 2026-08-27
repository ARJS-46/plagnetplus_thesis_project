import random

import pytest
import torch
from PIL import Image

from plagnetplus.data.ssl_transforms import (
    RandomQuarterTurn,
    SSLTransformConfig,
    build_ssl_evaluation_transform,
    build_ssl_train_transform,
)
from plagnetplus.data.transforms import (
    MultiCropTransform,
    MultiViewTransform,
)


def _gradient_image(
    size: int = 32,
) -> Image.Image:
    image = Image.new(
        "RGB",
        (size, size),
    )

    pixels = image.load()

    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                x % 256,
                y % 256,
                (x + y) % 256,
            )

    return image




def test_multiview_ssl_transform_returns_two_tensors():
    random.seed(42)
    torch.manual_seed(42)

    one_view = build_ssl_train_transform(
        SSLTransformConfig(
            image_size=64,
            blur_kernel_size=7,
        )
    )

    transform = MultiViewTransform(
        one_view,
        views=2,
    )

    views = transform(
        _gradient_image()
    )

    assert len(views) == 2

    for view in views:
        assert view.shape == (
            3,
            64,
            64,
        )
        assert torch.isfinite(view).all()
