"""Create the two raw-scan, segmentation and crystal-example panels."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from .style import BACKGROUND, BLUE, NAVY, ORANGE, use_report_style

SELECTED = "#75D0F3"


def _load_labels(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a binary or integer instance mask and verify scan alignment."""

    with Image.open(path) as image:
        array = np.asarray(image)
    if array.shape[:2] != shape:
        raise ValueError(f"Mask {array.shape[:2]} does not match scan {shape}.")
    if array.ndim == 3:
        colours, inverse, counts = np.unique(array[..., :3].reshape(-1, 3), axis=0, return_inverse=True, return_counts=True)
        background = int(np.argmax(counts))
        mapped = np.zeros_like(inverse, dtype=np.int32)
        next_label = 1
        for index in range(len(colours)):
            if index != background:
                mapped[inverse == index] = next_label
                next_label += 1
        return mapped.reshape(shape)
    values, counts = np.unique(array, return_counts=True)
    background = values[int(np.argmax(counts))]
    if len(values) <= 2:
        return ndimage.label(array != background, structure=np.ones((3, 3)))[0].astype(np.int32)
    labels = np.asarray(array, dtype=np.int32).copy()
    labels[array == background] = 0
    return labels


def _largest_internal(labels: np.ndarray, count: int) -> list[int]:
    """Select the largest instances that do not touch an image edge."""

    edge = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    values, sizes = np.unique(labels, return_counts=True)
    candidates = [(int(size), int(value)) for value, size in zip(values, sizes, strict=True) if value != 0 and value not in edge]
    return [value for _, value in sorted(candidates, reverse=True)[:count]]


def _crop(image: np.ndarray, labels: np.ndarray, label: int, margin: float = 0.1) -> Image.Image:
    """Create a masked, square-padded crop for one instance."""

    rows, columns = np.where(labels == label)
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    pad = int(max(y1 - y0, x1 - x0) * margin)
    y0, y1 = max(0, y0 - pad), min(image.shape[0], y1 + pad)
    x0, x1 = max(0, x0 - pad), min(image.shape[1], x1 + pad)
    crop = image[y0:y1, x0:x1].copy()
    crop[labels[y0:y1, x0:x1] != label] = 0
    return ImageOps.pad(Image.fromarray(crop), (256, 256), color="black", method=Image.Resampling.LANCZOS)


def plot_segmentation_example(
    scan_path: str | Path,
    mask_path: str | Path,
    *,
    labels: Sequence[int] = (),
    examples: int = 6,
) -> Figure:
    """Plot a raw scan, boundary overlay and representative crystal crops."""

    use_report_style()
    with Image.open(scan_path) as image:
        scan = np.asarray(image.convert("RGB"))
    instances = _load_labels(Path(mask_path), scan.shape[:2])
    selected = list(labels) or _largest_internal(instances, examples)
    if len(selected) < examples:
        raise ValueError(f"Only {len(selected)} suitable instances were found.")

    boundaries = instances != ndimage.grey_erosion(instances, size=(3, 3))
    chosen = np.isin(instances, selected)
    chosen_boundary = chosen != ndimage.binary_erosion(chosen)
    overlay = scan.copy()
    overlay[boundaries] = np.array([236, 170, 38], dtype=np.uint8)
    overlay[chosen_boundary] = np.array([117, 208, 243], dtype=np.uint8)

    figure = plt.figure(figsize=(7.09, 4.41), constrained_layout=True)
    grid = figure.add_gridspec(2, 6, height_ratios=(2.7, 1))
    raw_axis = figure.add_subplot(grid[0, :3])
    overlay_axis = figure.add_subplot(grid[0, 3:])
    raw_axis.imshow(scan)
    overlay_axis.imshow(overlay)
    raw_axis.set_title("(a) Raw CCL scan", loc="left")
    overlay_axis.set_title("(b) Segmented regions", loc="left")
    for axis in (raw_axis, overlay_axis):
        axis.axis("off")
    for number, label in enumerate(selected[:examples], 1):
        rows, columns = np.where(instances == label)
        overlay_axis.text(columns.mean(), rows.mean(), str(number), ha="center", va="center", fontsize=7, weight="bold", color="white", bbox={"boxstyle": "circle,pad=0.25", "facecolor": BLUE, "edgecolor": "none"})
        axis = figure.add_subplot(grid[1, number - 1])
        axis.imshow(_crop(scan, instances, label))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(SELECTED)
    overlay_axis.legend(handles=[Line2D([], [], color=ORANGE, label="All segmented boundaries"), Line2D([], [], color=SELECTED, label="Selected examples")], loc="lower left", fontsize=6, facecolor=BACKGROUND, edgecolor=NAVY)
    figure.text(0.01, 0.31, "(c) Individual crystal images", fontsize=9, weight="bold")
    return figure

