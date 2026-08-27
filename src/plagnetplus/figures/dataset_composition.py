"""Dataset-summary tables and the unified dataset v2 composition figure."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from PIL import Image, ImageOps

from plagnetplus.data.unified_dataset import load_image_root_map

from .style import BACKGROUND, LOCATION_NAMES, use_report_style

LOCATION_ORDER = ["Aegina", "FCT", "Methana", "MtStHelens", "Rumania"]
LOCATION_COLOURS = {
    "Aegina": "#6BB6DE",
    "FCT": "#4D9CC9",
    "Methana": "#2F80B4",
    "MtStHelens": "#1D628E",
    "Rumania": "#0E456B",
}
SPLIT_SHADE = {"train": "#F2F2F2", "validation": "#D8D8D8", "test": "#BEBEBE"}
SPLIT_LABEL = {"train": "Training", "validation": "Validation", "test": "Test"}
SAMPLE_ORDER = {
    "Aegina": ["Aeg6", "Aeg12", "Aeg14", "Aeg20", "Aeg27", "Aeg28", "Aeg29", "Aeg30"],
    "FCT": ["83b", "6861_FCT", "7191", "7193", "BIG'", "FNGR-2"],
    "Methana": ["M83-7-1-(II)", "M83-7-1-II", "M83-27_IX", "M91-3-10-", "M91-8791-71-1", "Met91-77-8-XI", "Met91-78-9-XII", "Met91-100-3(III)"],
    "MtStHelens": ["18_1a", "71512"],
    "Rumania": ["1b", "5", "10"],
}


def version_table(v1_location_counts: pd.DataFrame, v2_samples: pd.DataFrame) -> pd.DataFrame:
    """Return the location-level v1/v2 count and percentage comparison."""
    v1 = v1_location_counts.rename(columns={v1_location_counts.columns[0]: "Location"}).set_index("Location")["total"]
    v2 = v2_samples.groupby("location", observed=True)["Images"].sum()
    table = pd.DataFrame({"V1 images": v1, "V2 images": v2}).reindex(LOCATION_ORDER)
    table["Change"] = table["V2 images"] - table["V1 images"]
    table["V1 (%)"] = table["V1 images"] / table["V1 images"].sum() * 100
    table["V2 (%)"] = table["V2 images"] / table["V2 images"].sum() * 100
    return table.reset_index(names="Location")


def load_training_examples(
    manifest_path: str | Path,
    root_map_path: str | Path,
    *,
    count: int = 6,
    seed: int = 42,
) -> Mapping[str, list[tuple[Path, str]]]:
    """Select deterministic training examples and resolve their image paths."""
    manifest = pd.read_csv(manifest_path, low_memory=False)
    split = manifest["benchmark_split"].astype(str).str.lower()
    manifest = manifest.loc[split == "train"].copy()
    roots = load_image_root_map(root_map_path, data_root=os.environ.get("PNP_DATA_ROOT"))
    output: dict[str, list[tuple[Path, str]]] = {}
    for location in LOCATION_ORDER:
        rows = manifest.loc[manifest.locality.astype(str) == location]
        selected = rows.sample(n=min(count, len(rows)), random_state=seed)
        output[location] = [
            (roots[str(row.image_root_key)] / str(row.image_relative_path), str(row.acquisition_id))
            for row in selected.itertuples(index=False)
        ]
    return output


def _ordered_rows(sample_summary: pd.DataFrame, location: str) -> pd.DataFrame:
    rows = sample_summary.loc[sample_summary.location == location].copy()
    rank = {name: i for i, name in enumerate(SAMPLE_ORDER[location])}
    rows["_rank"] = rows["sample"].map(rank)
    return rows.sort_values("_rank").drop(columns="_rank")


def _sample_table(axis: plt.Axes, rows: pd.DataFrame) -> None:
    """Draw the compact four-column sample table used in the submitted report."""
    axis.axis("off")
    cells = [
        [str(r["sample"]), f"{int(r['Images']):,}", f"{float(r['Dataset (%)']):.2f}%", SPLIT_LABEL[str(r["split"]).lower()]]
        for _, r in rows.iterrows()
    ]
    table = axis.table(
        cellText=cells,
        colLabels=["Sample", "Images", "Overall (%)", "Split"],
        colWidths=[0.39, 0.18, 0.24, 0.19],
        loc="upper center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.85)
    table.scale(1.0, 1.08)
    for (ri, ci), cell in table.get_celld().items():
        cell.set_linewidth(0.90)
        cell.set_edgecolor("white")
        cell.PAD = 0.038
        if ri == 0:
            cell.set_facecolor("#4B95BC")
            cell.set_text_props(weight="bold", color="white", fontsize=5.70)
        else:
            split = str(rows.iloc[ri - 1]["split"]).lower()
            cell.set_facecolor(SPLIT_SHADE[split])


def _image_grid(figure: Figure, spec, items: list[tuple[Path, str]]) -> None:
    grid = spec.subgridspec(3, 2, hspace=0.055, wspace=0.055)
    for i in range(6):
        ax = figure.add_subplot(grid[i // 2, i % 2])
        ax.axis("off")
        ax.set_facecolor("black")
        if i >= len(items):
            continue
        path, _sample = items[i]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            ax.imshow(ImageOps.pad(image.convert("RGB"), (180, 180), color="black", method=Image.Resampling.LANCZOS))


def _location_panel(figure: Figure, spec, *, number: int, location: str, rows: pd.DataFrame, items: list[tuple[Path, str]]) -> None:
    """Draw one report-style location panel with a separate title row."""
    panel = spec.subgridspec(2, 2, height_ratios=[0.13, 0.87], width_ratios=[1.22, 0.78], hspace=0.015, wspace=0.045)
    title_ax = figure.add_subplot(panel[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0, 0.68,
        f"{number}. {LOCATION_NAMES[location]} — sample composition",
        ha="left", va="center", fontsize=8.90, weight="bold",
        transform=title_ax.transAxes,
    )
    table_ax = figure.add_subplot(panel[1, 0])
    _sample_table(table_ax, rows)
    _image_grid(figure, panel[1, 1], items)


def plot_v2_composition(sample_summary: pd.DataFrame, examples: Mapping[str, list[tuple[Path, str]]]) -> Figure:
    """Reproduce the compact unified-dataset summary used as Figure 2 in the report."""
    use_report_style()
    counts = sample_summary.groupby("location", observed=True)["Images"].sum().reindex(LOCATION_ORDER)
    total = int(counts.sum())

    figure = plt.figure(figsize=(8.55, 9.70))
    figure.patch.set_facecolor(BACKGROUND)
    outer = figure.add_gridspec(
        4, 2,
        height_ratios=[1.02, 0.98, 1.04, 0.80],
        hspace=0.135,
        wspace=0.095,
    )

    # Top summary: pie left, text/keys right.
    top_left = figure.add_subplot(outer[0, 0])
    colours = [LOCATION_COLOURS[k] for k in LOCATION_ORDER]
    values = counts.to_numpy()
    wedges, _ = top_left.pie(
        values,
        colors=colours,
        startangle=90,
        counterclock=False,
        radius=1.25,
        wedgeprops={"linewidth": 1.15, "edgecolor": "white"},
    )
    import numpy as np
    for i, (wedge, value) in enumerate(zip(wedges, values, strict=True)):
        pct = value / total * 100
        angle = (wedge.theta1 + wedge.theta2) / 2
        x, y = 0.78 * np.cos(np.deg2rad(angle)), 0.78 * np.sin(np.deg2rad(angle))
        if i == 3:
            top_left.annotate(
                f"4\n{value:,}\n[{pct:.1f}%]",
                xy=(1.13 * np.cos(np.deg2rad(angle)), 1.13 * np.sin(np.deg2rad(angle))),
                xytext=(-1.55, 0.50),
                arrowprops={"arrowstyle": "->", "lw": 0.65, "color": "#333333", "shrinkA": 1, "shrinkB": 1},
                ha="right", va="center", fontsize=6.85, weight="bold", linespacing=1.02,
            )
        else:
            top_left.text(
                x, y, f"{i + 1}\n{value:,}\n[{pct:.1f}%]",
                ha="center", va="center", fontsize=7.10, linespacing=1.02,
                color="white" if i in (2, 4) else "#18384F",
            )
    top_left.set_aspect("equal")

    top_right = figure.add_subplot(outer[0, 1])
    top_right.axis("off")
    top_right.text(0.02, 0.82, f"{total:,} segmented crystal images\n5 locations  |  27 samples", fontsize=8.65, va="top")
    top_right.text(0.02, 0.54, "Location key", fontsize=9.25, weight="bold")
    for i, location in enumerate(LOCATION_ORDER):
        y = 0.47 - i * 0.074
        top_right.add_patch(plt.Rectangle((0.02, y - 0.020), 0.050, 0.036, transform=top_right.transAxes,
                                          facecolor=LOCATION_COLOURS[location], edgecolor="#333333", linewidth=0.35))
        top_right.text(0.086, y, f"{i+1}. {LOCATION_NAMES[location]}", transform=top_right.transAxes, fontsize=7.65, va="center")
    top_right.text(0.57, 0.54, "Benchmark split shading", fontsize=9.25, weight="bold")
    for j, split in enumerate(("train", "validation", "test")):
        y = 0.47 - j * 0.090
        top_right.add_patch(plt.Rectangle((0.57, y - 0.020), 0.060, 0.036, transform=top_right.transAxes,
                                          facecolor=SPLIT_SHADE[split], edgecolor="#666666", linewidth=0.35))
        top_right.text(0.648, y, SPLIT_LABEL[split], transform=top_right.transAxes, fontsize=7.65, va="center")

    panel_positions = {
        "Aegina": (1, 0), "FCT": (1, 1),
        "Methana": (2, 0), "MtStHelens": (2, 1),
        "Rumania": (3, 0),
    }
    for number, location in enumerate(LOCATION_ORDER, start=1):
        r, c = panel_positions[location]
        rows = _ordered_rows(sample_summary, location)
        _location_panel(
            figure, outer[r, c], number=number, location=location,
            rows=rows, items=examples.get(location, []),
        )

    blank = figure.add_subplot(outer[3, 1])
    blank.axis("off")
    figure.suptitle("Unified Dataset Version 2", fontsize=15.5, weight="bold", y=0.987)
    figure.subplots_adjust(left=0.035, right=0.992, top=0.945, bottom=0.035)
    return figure
