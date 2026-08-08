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


def test_evaluation_transform_shape_and_range():
    transform = (
        build_ssl_evaluation_transform(
            SSLTransformConfig(
                image_size=64
            )
        )
    )

    output = transform(
        _gradient_image()
    )

    assert output.shape == (
        3,
        64,
        64,
    )
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert output.min() >= 0
    assert output.max() <= 1


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


def test_quarter_turn_preserves_dimensions():
    image = _gradient_image(
        size=31
    )

    transform = RandomQuarterTurn()

    for seed in range(20):
        random.seed(seed)
        output = transform(image)

        assert output.size == image.size
        assert output.mode == image.mode


def test_normalization_requires_mean_and_std():
    with pytest.raises(
        ValueError,
        match="must either both be provided",
    ):
        build_ssl_train_transform(
            SSLTransformConfig(
                normalization_mean=(
                    0.1,
                    0.2,
                    0.3,
                ),
                normalization_std=None,
            )
        )


def test_normalized_evaluation_is_finite():
    transform = (
        build_ssl_evaluation_transform(
            SSLTransformConfig(
                image_size=32,
                normalization_mean=(
                    0.4,
                    0.4,
                    0.4,
                ),
                normalization_std=(
                    0.2,
                    0.2,
                    0.2,
                ),
            )
        )
    )

    output = transform(
        _gradient_image()
    )

    assert output.shape == (
        3,
        32,
        32,
    )
    assert torch.isfinite(output).all()


def test_multicrop_transform_orders_global_before_local():
    image = _gradient_image()

    def global_transform(value):
        return (
            "global",
            value.size,
        )

    def local_transform(value):
        return (
            "local",
            value.size,
        )

    transform = MultiCropTransform(
        global_transform,
        global_crops=2,
        local_transform=local_transform,
        local_crops=3,
    )

    views = transform(image)

    assert transform.views == 5

    assert views == (
        ("global", image.size),
        ("global", image.size),
        ("local", image.size),
        ("local", image.size),
        ("local", image.size),
    )


def test_multicrop_transform_supports_two_global_views_only():
    one_view = build_ssl_train_transform(
        SSLTransformConfig(
            image_size=64,
            blur_kernel_size=7,
        )
    )

    transform = MultiCropTransform(
        one_view,
        global_crops=2,
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

        assert torch.isfinite(
            view
        ).all()


@pytest.mark.parametrize(
    (
        "global_crops",
        "local_transform",
        "local_crops",
    ),
    [
        (0, None, 0),
        (2, None, -1),
        (2, None, 1),
    ],
)
def test_multicrop_transform_rejects_invalid_configuration(
    global_crops,
    local_transform,
    local_crops,
):
    with pytest.raises(
        ValueError
    ):
        MultiCropTransform(
            lambda image: image,
            global_crops=global_crops,
            local_transform=local_transform,
            local_crops=local_crops,
        )
