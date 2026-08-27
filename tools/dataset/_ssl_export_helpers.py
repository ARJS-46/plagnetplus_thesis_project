"""Shared mask and crop helpers used by the production PlagNet SSL exporter."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
import hashlib
import math
import tarfile

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def normalise_member_name(value: str) -> str:
    return PurePosixPath(
        value.replace("\\", "/")
    ).as_posix().lstrip("./")


def build_member_indices(
    archive: tarfile.TarFile,
) -> tuple[
    dict[str, tarfile.TarInfo],
    dict[str, list[str]],
]:
    members: dict[str, tarfile.TarInfo] = {}
    by_basename: dict[str, list[str]] = defaultdict(list)

    for member in archive.getmembers():
        if not member.isfile():
            continue

        name = normalise_member_name(
            member.name
        )

        members[name] = member

        by_basename[
            PurePosixPath(name).name
        ].append(name)

    return members, by_basename


def resolve_mask_member(
    *,
    member_names: set[str],
    members_by_basename: dict[str, list[str]],
    mask_path: str,
    tile_stem: str,
) -> str:
    normalised = normalise_member_name(
        mask_path
    )

    basename = PurePosixPath(
        normalised
    ).name

    direct_candidates = (
        normalised,
        f"{tile_stem}/{basename}",
    )

    for candidate in direct_candidates:
        if candidate in member_names:
            return candidate

    tile_matches = [
        name
        for name in members_by_basename.get(
            basename,
            [],
        )
        if tile_stem in name
    ]

    if len(tile_matches) == 1:
        return tile_matches[0]

    suffix_matches = [
        name
        for name in member_names
        if name.endswith(normalised)
    ]

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise RuntimeError(
        "Could not uniquely resolve PlagNet mask: "
        f"tile={tile_stem!r}; "
        f"mask_path={mask_path!r}; "
        f"tile_matches={tile_matches!r}; "
        f"suffix_matches={suffix_matches!r}"
    )


def read_mask(
    *,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> tuple[np.ndarray, str]:
    extracted = archive.extractfile(member)

    if extracted is None:
        raise RuntimeError(
            f"Could not extract {member.name}"
        )

    data = extracted.read()

    with Image.open(
        BytesIO(data)
    ) as image:
        mode = image.mode
        array = np.asarray(image)

    if array.ndim != 2:
        raise RuntimeError(
            f"Mask is not two-dimensional: "
            f"{member.name}: {array.shape}"
        )

    return array != 0, mode


def calculate_exact_geometry(
    mask: np.ndarray,
) -> dict[str, Any]:
    area = int(mask.sum())

    if area == 0:
        raise RuntimeError(
            "Encountered empty exact mask"
        )

    # Derive geometry from the exact binary mask rather than trusting detector
    # bounding-box metadata, which keeps occupancy calculations internally consistent.
    ys, xs = np.nonzero(mask)

    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max_exclusive = int(
        xs.max() + 1
    )
    y_max_exclusive = int(
        ys.max() + 1
    )

    height, width = mask.shape

    border_distance = min(
        x_min,
        y_min,
        width - x_max_exclusive,
        height - y_max_exclusive,
    )

    return {
        "area_px": area,
        "x_min": x_min,
        "y_min": y_min,
        "x_max_exclusive": (
            x_max_exclusive
        ),
        "y_max_exclusive": (
            y_max_exclusive
        ),
        "width_px": (
            x_max_exclusive - x_min
        ),
        "height_px": (
            y_max_exclusive - y_min
        ),
        "centroid_x_px": float(
            xs.mean()
        ),
        "centroid_y_px": float(
            ys.mean()
        ),
        "border_distance_px": (
            border_distance
        ),
        "touches_tile_border": (
            border_distance == 0
        ),
    }


def expanded_crop_box(
    *,
    geometry: dict[str, Any],
    image_width: int,
    image_height: int,
    margin_fraction: float,
    minimum_margin_px: int,
) -> tuple[int, int, int, int, int]:
    object_side = max(
        int(geometry["width_px"]),
        int(geometry["height_px"]),
    )

    # Expand the exact mask bounding box by the configured margin before square
    # padding, preserving edge context without changing crystal geometry.
    margin = max(
        minimum_margin_px,
        int(
            math.ceil(
                object_side
                * margin_fraction
            )
        ),
    )

    x1 = max(
        0,
        int(geometry["x_min"])
        - margin,
    )

    y1 = max(
        0,
        int(geometry["y_min"])
        - margin,
    )

    x2 = min(
        image_width,
        int(
            geometry[
                "x_max_exclusive"
            ]
        )
        + margin,
    )

    y2 = min(
        image_height,
        int(
            geometry[
                "y_max_exclusive"
            ]
        )
        + margin,
    )

    return x1, y1, x2, y2, margin


def square_pad(
    image: Image.Image,
    *,
    fill: tuple[int, int, int],
) -> tuple[Image.Image, int, int]:
    width, height = image.size
    side = max(width, height)

    output = Image.new(
        "RGB",
        (side, side),
        fill,
    )

    offset_x = (
        side - width
    ) // 2

    offset_y = (
        side - height
    ) // 2

    output.paste(
        image,
        (offset_x, offset_y),
    )

    return output, offset_x, offset_y


def save_png(
    image: Image.Image,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".png.partial"
    )

    image.save(
        temporary,
        format="PNG",
        optimize=False,
    )

    temporary.replace(path)


def sanitise_path_component(
    value: str,
) -> str:
    safe = []

    for character in value:
        if (
            character.isalnum()
            or character in "-_."
        ):
            safe.append(character)
        else:
            safe.append("_")

    return "".join(safe)
