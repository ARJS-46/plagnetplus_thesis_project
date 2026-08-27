"""Plots for the reported SimSiam and SwAV v1 hyperparameter screen."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

from .style import BACKGROUND, BLUE, NAVY, ORANGE, use_report_style

GRID_COLOURS = ["#F7FBFF", "#D8EFF8", "#A6D9EC", "#70BEDD", "#389BC9", BLUE, NAVY]
GRID_CMAP = LinearSegmentedColormap.from_list("report_blues", GRID_COLOURS, N=256)


def _cluster_matrix(
    frame: pd.DataFrame,
    row: str,
    columns: Sequence[str],
    score: str,
) -> pd.DataFrame:
    """Reshape one complete factorial screen into a labelled matrix."""

    matrix = frame.pivot(index=row, columns=list(columns), values=score).sort_index().sort_index(axis=1)
    if matrix.isna().any().any():
        raise ValueError("The hyperparameter grid contains an empty cell.")
    return matrix


def _annotated_clustermap(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    title: str,
    label: str,
    selected: set[str],
    baselines: set[str],
    annotation: Callable[[pd.Series], str],
) -> Figure:
    """Draw a clustered heatmap and annotate each cell with its trial ID."""

    norm = Normalize(float(matrix.to_numpy().min()), float(matrix.to_numpy().max()))
    grid = sns.clustermap(
        matrix,
        cmap=GRID_CMAP,
        norm=norm,
        linewidths=0.5,
        linecolor=BACKGROUND,
        figsize=(10, 6.2),
    )
    grid.fig.patch.set_facecolor(BACKGROUND)
    for ax in (grid.ax_heatmap, grid.ax_row_dendrogram, grid.ax_col_dendrogram, grid.cax):
        ax.set_facecolor(BACKGROUND)
    for ax in (grid.ax_row_dendrogram, grid.ax_col_dendrogram):
        for collection in ax.collections:
            collection.set_color(BLUE)

    lookup = frame.set_index([matrix.index.name, *matrix.columns.names])
    row_order = grid.dendrogram_row.reordered_ind
    column_order = grid.dendrogram_col.reordered_ind
    for shown_row, source_row in enumerate(row_order):
        for shown_col, source_col in enumerate(column_order):
            index = (matrix.index[source_row], matrix.columns[source_col])
            record = lookup.loc[index]
            trial = str(record["trial_id"])
            value = float(matrix.iloc[source_row, source_col])
            colour = "white" if norm(value) > 0.58 else "black"
            grid.ax_heatmap.text(shown_col + 0.5, shown_row + 0.5, annotation(record), ha="center", va="center", color=colour, fontsize=8)
            if trial in selected:
                grid.ax_heatmap.add_patch(
                    plt.Rectangle((shown_col + 0.02, shown_row + 0.02), 0.96, 0.96, fill=False, edgecolor=ORANGE, linewidth=3)
                )
            if trial in baselines:
                grid.ax_heatmap.add_patch(
                    plt.Rectangle((shown_col + 0.03, shown_row + 0.03), 0.94, 0.94, fill=False, edgecolor="#D60000", linewidth=2.6)
                )
    grid.cax.set_ylabel(label)
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.fig.suptitle(title, y=1.02, weight="bold")
    return grid.fig


def plot_simsiam_grid(frame: pd.DataFrame) -> Figure:
    """Plot the clustered SimSiam grid using negative validation loss."""

    use_report_style()
    data = frame.assign(score=-frame["final_validation_loss"])
    matrix = _cluster_matrix(data, "learning_rate", ["predictor_learning_rate", "weight_decay"], "score")
    matrix.index = [f"LR={value:g}" for value in matrix.index]
    matrix.columns = [f"PLR={plr:g}\nWD={wd:g}" for plr, wd in matrix.columns]
    matrix.index.name = "learning_rate_label"
    matrix.columns.names = ["predictor_learning_rate_label"]

    # Build a matching labelled lookup without altering the frozen table.
    labelled = data.assign(
        learning_rate_label=lambda x: x.learning_rate.map(lambda value: f"LR={value:g}"),
        predictor_learning_rate_label=lambda x: [f"PLR={plr:g}\nWD={wd:g}" for plr, wd in zip(x.predictor_learning_rate, x.weight_decay, strict=True)],
    )
    return _annotated_clustermap(
        labelled,
        matrix,
        title="SimSiam clustered hyperparameter grid",
        label="Score (-validation loss)",
        selected={"s10"},
        baselines={"s08"},
        annotation=lambda row: f"{row['trial_id']}\nscore={row['score']:.4f}",
    )


def plot_swav_grid(frame: pd.DataFrame) -> Figure:
    """Plot the clustered SwAV grid using occupied prototype fraction."""

    use_report_style()
    matrix = _cluster_matrix(frame, "learning_rate", ["prototype_count", "temperature"], "occupied_fraction")
    matrix.index = [f"LR={value:g}" for value in matrix.index]
    matrix.columns = [f"P={int(count)}\nT={temperature:g}" for count, temperature in matrix.columns]
    matrix.index.name = "learning_rate_label"
    matrix.columns.names = ["prototype_temperature_label"]
    labelled = frame.assign(
        learning_rate_label=lambda x: x.learning_rate.map(lambda value: f"LR={value:g}"),
        prototype_temperature_label=lambda x: [f"P={int(count)}\nT={temperature:g}" for count, temperature in zip(x.prototype_count, x.temperature, strict=True)],
    )
    return _annotated_clustermap(
        labelled,
        matrix,
        title="SwAV clustered hyperparameter grid",
        label="Occupied prototype fraction",
        selected={"w05", "w12"},
        baselines={"w11"},
        annotation=lambda row: f"{row['trial_id']}\noccupancy={row['occupied_fraction']:.4f}\nloss={row['final_validation_loss']:.3f}",
    )


def plot_marginal_response(
    frame: pd.DataFrame,
    *,
    score: str,
    parameters: Sequence[tuple[str, str]],
    title: str,
    ylabel: str,
) -> Figure:
    """Plot individual trials and marginal means for each grid factor."""

    use_report_style()
    figure, axes = plt.subplots(1, len(parameters), figsize=(4.1 * len(parameters), 3.8), constrained_layout=True)
    for axis, (parameter, label) in zip(np.atleast_1d(axes), parameters, strict=True):
        means = frame.groupby(parameter, observed=True)[score].mean().sort_index()
        axis.scatter(frame[parameter], frame[score], color="#9BB1CC", alpha=0.55, s=28)
        axis.plot(means.index, means.values, color=BLUE, marker="o", linewidth=2)
        for level, mean in means.items():
            axis.annotate(f"{mean:.4f}", (level, mean), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        axis.set(xlabel=label, ylabel=ylabel)
        if parameter == "weight_decay":
            axis.set_xscale("log")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title, weight="bold")
    return figure
