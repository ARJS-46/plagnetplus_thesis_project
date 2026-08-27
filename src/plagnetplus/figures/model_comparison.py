"""Compact v1 baseline model-comparison plots."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .style import BACKGROUND, BLUE, ORANGE, use_report_style


def plot_metric_comparison(metrics: pd.DataFrame) -> Figure:
    """Plot the six reported validation metrics without combining their scales."""

    use_report_style()
    columns = [
        "Median silhouette score",
        "Stability ARI",
        "FCT cross-sample P@10",
        "Location-probe macro-F1",
        "Source-image enrichment@10",
        "Effective rank",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(11.69, 6.8), constrained_layout=True)
    for axis, column in zip(axes.flat, columns, strict=True):
        bars = axis.bar(metrics.Model, metrics[column], color=[BLUE, "#376399", "#698AB3", ORANGE])
        axis.bar_label(bars, fmt="%.3g", fontsize=7, padding=2)
        axis.set_title(column)
        axis.tick_params(axis="x", rotation=25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("V1 baseline comparison", fontsize=14, weight="bold")
    return figure


def cluster_projection(features: np.ndarray, *, clusters: int = 50, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Fit common-k k-means and return deterministic two-dimensional PCA scores."""

    array = np.asarray(features, dtype=np.float32)
    labels = KMeans(n_clusters=clusters, random_state=seed, n_init=10, algorithm="lloyd").fit_predict(array)
    scores = PCA(n_components=2, random_state=seed).fit_transform(array)
    return scores, labels


def plot_cluster_comparison(
    embeddings: Mapping[str, np.ndarray],
    *,
    clusters: int = 50,
    seed: int = 42,
    maximum_points: int = 5000,
) -> Figure:
    """Plot equivalent k-means/PCA views for all four baseline embeddings."""

    use_report_style()
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    for axis, (name, features) in zip(axes.flat, embeddings.items(), strict=True):
        scores, labels = cluster_projection(features, clusters=clusters, seed=seed)
        indices = np.arange(len(scores))
        if len(indices) > maximum_points:
            indices = np.sort(rng.choice(indices, size=maximum_points, replace=False))
        axis.scatter(scores[indices, 0], scores[indices, 1], c=labels[indices], cmap="tab20", s=4, alpha=0.5, linewidth=0)
        axis.set(title=f"{name} (k={clusters})", xlabel="PC1", ylabel="PC2")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("V1 baseline clustering in PCA space", fontsize=14, weight="bold")
    figure.patch.set_facecolor(BACKGROUND)
    return figure


def plot_model_examples(
    embeddings: Mapping[str, np.ndarray],
    image_paths: Mapping[str, list[str]],
    *,
    clusters: int = 50,
    examples: int = 6,
    seed: int = 42,
) -> Figure:
    """Show crystals nearest the centres of the largest clusters per model."""

    use_report_style()
    figure, axes = plt.subplots(len(embeddings), examples, figsize=(11.69, 1.85 * len(embeddings)), squeeze=False)
    for row_index, (name, features) in enumerate(embeddings.items()):
        model = KMeans(n_clusters=clusters, random_state=seed, n_init=10, algorithm="lloyd").fit(features)
        counts = np.bincount(model.labels_, minlength=clusters)
        selected_clusters = np.argsort(counts)[::-1][:examples]
        paths = image_paths[name]
        if len(paths) != len(features):
            raise ValueError(f"{name}: image paths and embeddings have different row counts.")
        for column_index, cluster in enumerate(selected_clusters):
            members = np.flatnonzero(model.labels_ == cluster)
            distances = np.linalg.norm(features[members] - model.cluster_centers_[cluster], axis=1)
            row = int(members[int(np.argmin(distances))])
            with Image.open(paths[row]) as image:
                crystal = ImageOps.pad(image.convert("RGB"), (256, 256), color="black", method=Image.Resampling.LANCZOS)
            axis = axes[row_index, column_index]
            axis.imshow(crystal)
            axis.set_title(f"C{cluster} (n={counts[cluster]:,})", fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(name, rotation=0, ha="right", va="center", weight="bold", color=BLUE)
    figure.suptitle("Representative validation crystals from the six largest clusters", fontsize=13, weight="bold")
    figure.tight_layout(rect=(0.06, 0.01, 1, 0.95), w_pad=0.15, h_pad=0.5)
    return figure
