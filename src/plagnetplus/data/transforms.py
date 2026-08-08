"""Small transform wrappers shared across crystal-image models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image, ImageOps


class PadToSquare:
    """Pad an image to a square without distorting crystal morphology."""

    def __init__(self, fill: int | tuple[int, int, int] = 0) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)

        horizontal = side - width
        vertical = side - height

        left = horizontal // 2
        right = horizontal - left
        top = vertical // 2
        bottom = vertical - top

        return ImageOps.expand(
            image,
            border=(left, top, right, bottom),
            fill=self.fill,
        )


class MultiViewTransform:
    """Apply one stochastic transform independently a fixed number of times."""

    def __init__(
        self,
        transform: Callable[[Image.Image], Any],
        views: int = 2,
    ) -> None:
        if views < 1:
            raise ValueError("views must be at least one.")
        self.transform = transform
        self.views = views

    def __call__(self, image: Image.Image) -> tuple[Any, ...]:
        return tuple(self.transform(image.copy()) for _ in range(self.views))


class MultiCropTransform:
    """Apply distinct stochastic pipelines to global and local crops."""

    def __init__(
        self,
        global_transform: Callable[[Image.Image], Any],
        *,
        global_crops: int = 2,
        local_transform: Callable[[Image.Image], Any] | None = None,
        local_crops: int = 0,
    ) -> None:
        if global_crops < 1:
            raise ValueError(
                "global_crops must be at least one."
            )

        if local_crops < 0:
            raise ValueError(
                "local_crops cannot be negative."
            )

        if (
            local_crops > 0
            and local_transform is None
        ):
            raise ValueError(
                "local_transform is required when "
                "local_crops is positive."
            )

        self.global_transform = global_transform
        self.global_crops = int(global_crops)
        self.local_transform = local_transform
        self.local_crops = int(local_crops)

    @property
    def views(
        self,
    ) -> int:
        """Return the total number of generated views."""

        return self.global_crops + self.local_crops

    def __call__(
        self,
        image: Image.Image,
    ) -> tuple[Any, ...]:
        global_views = tuple(
            self.global_transform(
                image.copy()
            )
            for _ in range(
                self.global_crops
            )
        )

        if self.local_transform is None:
            return global_views

        local_views = tuple(
            self.local_transform(
                image.copy()
            )
            for _ in range(
                self.local_crops
            )
        )

        return global_views + local_views
