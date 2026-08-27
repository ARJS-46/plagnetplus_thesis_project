"""Draw the four self-supervised learning workflows used in the report."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import BACKGROUND, BLUE, BLUE_FILL, BLUE_PALE, ORANGE, ORANGE_FILL, use_report_style


def _box(
    ax: Axes,
    xy: tuple[float, float],
    text: str,
    *,
    width: float = 0.31,
    height: float = 0.12,
    objective: bool = False,
) -> None:
    """Add one labelled network or objective box in axes coordinates."""

    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.03",
        linewidth=1.05,
        edgecolor=ORANGE if objective else BLUE,
        facecolor=ORANGE_FILL if objective else BLUE_FILL,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5, weight="bold", linespacing=0.9)


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    colour: str = "#6B7C8C",
    connection: str = "arc3",
) -> None:
    """Add a directed relationship between two boxes."""

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.9,
            linestyle="--" if dashed else "-",
            color=colour,
            connectionstyle=connection,
        )
    )


def _heading(ax: Axes, title: str, subtitle: str) -> None:
    ax.text(0.5, 1.035, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=12.5, weight="bold")
    ax.text(0.5, 0.995, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.4, color="#6B7C8C")


def _two_branches(ax: Axes, stages: Iterable[tuple[float, str]], *, top: float = 0.78) -> None:
    """Draw paired view branches; stage y-values are supplied bottom-up."""

    xs = (0.25, 0.75)
    stage_list = list(stages)
    for x, view in zip(xs, ("view 1", "view 2"), strict=True):
        _box(ax, (x, 0.12), view, width=0.28, height=0.10)
        previous = 0.17
        for y, label in stage_list:
            _arrow(ax, (x, previous), (x, y - 0.065))
            _box(ax, (x, y), label, width=0.32, height=0.13)
            previous = y + 0.065


def _simclr(ax: Axes) -> None:
    _heading(ax, "SimCLR", "contrastive agreement between augmented views")
    _two_branches(ax, [(0.36, "encoder"), (0.60, "projector")])
    _box(ax, (0.5, 0.84), "contrastive loss\npull positives · push negatives", width=0.44, height=0.13, objective=True)
    _arrow(ax, (0.25, 0.67), (0.43, 0.78), connection="arc3,rad=-0.08")
    _arrow(ax, (0.75, 0.67), (0.57, 0.78), connection="arc3,rad=0.08")
    ax.text(0.5, 0.74, "gradients update both branches", fontsize=5.8, color="#7B8790", ha="center")
    ax.plot([0.33, 0.67], [0.36, 0.36], linestyle="--", color="#9AA5AE", linewidth=1)
    ax.text(0.5, 0.375, "shared weights", ha="center", va="bottom", fontsize=5.7, color="#7B8790")


def _deepcluster(ax: Axes) -> None:
    _heading(ax, "DeepCluster", "alternating clustering and pseudo-label prediction")
    for y, text in [(0.10, "image batch"), (0.31, "encoder"), (0.52, "embeddings"), (0.73, "classifier\npredictions")]:
        _box(ax, (0.58, y), text, width=0.31, height=0.11 if y != 0.31 else 0.14)
    for start, end in [((0.58, 0.155), (0.58, 0.24)), ((0.58, 0.38), (0.58, 0.455)), ((0.58, 0.575), (0.58, 0.66))]:
        _arrow(ax, start, end)
    _box(ax, (0.25, 0.68), "k-means\nclustering", width=0.30, height=0.13, objective=True)
    _box(ax, (0.58, 0.89), "cross-entropy loss", width=0.36, height=0.12, objective=True)
    _arrow(ax, (0.45, 0.55), (0.30, 0.61), dashed=True, colour=BLUE, connection="arc3,rad=-0.15")
    _arrow(ax, (0.31, 0.75), (0.43, 0.86), colour=ORANGE)
    _arrow(ax, (0.58, 0.79), (0.58, 0.83), colour=ORANGE)
    ax.text(0.30, 0.80, "pseudo-labels", fontsize=5.8, color=ORANGE, ha="center")
    ax.text(0.15, 0.47, "re-cluster\nperiodically", fontsize=5.8, color=BLUE, ha="center")
    ax.text(0.43, 0.57, "alternate cluster + optimise", fontsize=5.6, color="#7B8790", ha="center")


def _swav(ax: Axes) -> None:
    _heading(ax, "SwAV", "swapped prediction of online prototype assignments")
    _two_branches(ax, [(0.32, "encoder +\nprojector"), (0.53, "prototype scores")])
    for x in (0.25, 0.75):
        _arrow(ax, (x, 0.60), (x, 0.665), colour=ORANGE)
        _box(ax, (x, 0.73), "Sinkhorn-Knopp\nassignment", width=0.34, height=0.13, objective=True)
    _box(ax, (0.5, 0.89), "swapped\nprediction loss", width=0.34, height=0.13, objective=True)
    _arrow(ax, (0.25, 0.80), (0.43, 0.84), colour=ORANGE, connection="arc3,rad=-0.08")
    _arrow(ax, (0.75, 0.80), (0.57, 0.84), colour=ORANGE, connection="arc3,rad=0.08")
    ax.plot([0.33, 0.67], [0.32, 0.32], linestyle="--", color="#9AA5AE", linewidth=1)
    ax.text(0.5, 0.335, "shared weights", ha="center", va="bottom", fontsize=5.7, color="#7B8790")


def _simsiam(ax: Axes) -> None:
    _heading(ax, "SimSiam", "prediction against stop-gradient targets")
    _two_branches(ax, [(0.28, "encoder"), (0.48, "projector"), (0.68, "predictor")])
    _box(ax, (0.5, 0.90), "negative cosine similarity\np₁ ↔ stop-grad(z₂) · p₂ ↔ stop-grad(z₁)", width=0.47, height=0.13, objective=True)
    _arrow(ax, (0.25, 0.75), (0.44, 0.84), colour=ORANGE, connection="arc3,rad=-0.08")
    _arrow(ax, (0.75, 0.75), (0.56, 0.84), colour=ORANGE, connection="arc3,rad=0.08")
    _arrow(ax, (0.25, 0.55), (0.69, 0.82), dashed=True, connection="arc3,rad=-0.12")
    _arrow(ax, (0.75, 0.55), (0.31, 0.82), dashed=True, connection="arc3,rad=0.12")
    ax.plot([0.33, 0.67], [0.28, 0.28], linestyle="--", color="#9AA5AE", linewidth=1)
    ax.text(0.5, 0.295, "shared weights", ha="center", va="bottom", fontsize=5.7, color="#7B8790")
    ax.text(0.5, 0.56, "targets are detached", ha="center", fontsize=5.8, color="#7B8790")


def plot_ssl_approaches() -> Figure:
    """Return the report diagram comparing SimCLR, DeepCluster, SwAV and SimSiam."""

    use_report_style()
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    for ax, draw in zip(axes.flat, (_simclr, _deepcluster, _swav, _simsiam), strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        draw(ax)
    figure.suptitle("Comparison of self-supervised representation-learning approaches", fontsize=13.5, weight="bold", y=0.965)
    figure.text(
        0.5,
        0.045,
        "Blue: learned components   ·   Yellow: objectives or assignments   ·   Dashed: shared, detached or alternating path",
        ha="center",
        fontsize=6.2,
        color="#6B7C8C",
    )
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.04, right=0.985, top=0.90, bottom=0.10, wspace=0.10, hspace=0.18)
    return figure

