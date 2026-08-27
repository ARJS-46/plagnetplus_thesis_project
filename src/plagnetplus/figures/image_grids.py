"""Image-grid helpers for cluster and PCA-extreme report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .style import BACKGROUND, use_report_style

REGION_COLOURS = {
    1: "#1F77B4", 2: "#FF7F0E", 3: "#2CA02C", 4: "#D62728",
    5: "#9467BD", 6: "#8C564B", 7: "#E377C2", 8: "#7F7F7F",
}
# Fixed colours reproduce the coloured row strips/borders in the submitted A4 panel.
CLUSTER_ROW_COLOURS = {
    5: "#62B5E5", 15: "#D8D93C", 2: "#4D4A9B", 12: "#63CE3D",
    8: "#37C7C8", 16: "#D8B947", 24: "#B32616", 0: "#442458",
    4: "#4B79D8", 23: "#D64A13", 20: "#E16B21", 25: "#9D1C0F",
}


def resolve_recorded_path(value: str | Path, data_root: str | Path) -> Path:
    recorded = Path(value)
    root = Path(data_root).expanduser().resolve()
    parts = recorded.parts
    for marker in ("processed", "interim", "raw"):
        if marker in parts:
            return root.joinpath(*parts[parts.index(marker):])
    return recorded if recorded.is_absolute() else root / recorded


def _open_square(path: Path, size: int = 256) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.pad(image.convert("RGB"), (size, size), color="black", method=Image.Resampling.LANCZOS)


def plot_grouped_crystals(selections: pd.DataFrame, *, group: str, data_root: str | Path, title: str, images_per_group: int = 6) -> Figure:
    """Legacy grouped grid retained for backwards compatibility."""
    use_report_style()
    table = selections.sort_values([group, "Image rank"])
    groups = list(dict.fromkeys(table[group].tolist()))
    figure, axes = plt.subplots(len(groups), images_per_group, figsize=(11.69, max(4.5, 1.05 * len(groups))), squeeze=False)
    for row_index, group_value in enumerate(groups):
        rows = table.loc[table[group] == group_value].head(images_per_group)
        for column_index, axis in enumerate(axes[row_index]):
            axis.set_facecolor("black"); axis.set_xticks([]); axis.set_yticks([])
            if column_index >= len(rows): axis.axis("off"); continue
            record = rows.iloc[column_index]
            path = resolve_recorded_path(record.image_path, data_root)
            if not path.is_file(): raise FileNotFoundError(path)
            axis.imshow(_open_square(path))
        axes[row_index, 0].set_ylabel(str(group_value), rotation=0, ha="right", va="center", fontsize=7)
    figure.suptitle(title, fontsize=13, weight="bold")
    figure.tight_layout(rect=(0.06, 0.01, 1, 0.97), h_pad=0.25, w_pad=0.12)
    return figure


def plot_representative_clusters(selections: pd.DataFrame, *, data_root: str | Path, title: str, images_per_group: int = 6) -> Figure:
    """Reproduce Figure 5a with a dedicated label column and six bordered crystals."""
    use_report_style()
    ordered = (selections[["Cluster", "Cluster size", "Diversity selection rank"]]
               .drop_duplicates().sort_values("Diversity selection rank"))
    clusters = ordered["Cluster"].astype(int).tolist()

    figure = plt.figure(figsize=(8.0, 9.55))
    grid = figure.add_gridspec(
        len(clusters), images_per_group + 1,
        width_ratios=[0.62] + [1.0] * images_per_group,
        hspace=0.075, wspace=0.22,
    )

    for ri, cluster in enumerate(clusters):
        rows = selections.loc[selections["Cluster"] == cluster].sort_values("Image rank").head(images_per_group)
        size = int(rows["Cluster size"].iloc[0])
        colour = CLUSTER_ROW_COLOURS.get(cluster, "#4C78A8")

        label_ax = figure.add_subplot(grid[ri, 0])
        label_ax.axis("off")
        label_ax.add_patch(Rectangle((0.03, 0.04), 0.075, 0.92, transform=label_ax.transAxes,
                                     facecolor=colour, edgecolor="#333333", linewidth=0.45))
        label_ax.text(0.16, 0.62, f"Cluster {cluster}", transform=label_ax.transAxes,
                      ha="left", va="center", fontsize=6.4, weight="bold", color="#111111")
        label_ax.text(0.16, 0.36, f"n={size:,}", transform=label_ax.transAxes,
                      ha="left", va="center", fontsize=6.1, weight="bold", color="#111111")

        for ci in range(images_per_group):
            ax = figure.add_subplot(grid[ri, ci + 1])
            ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color(colour); spine.set_linewidth(1.15)
            if ci >= len(rows):
                ax.axis("off"); continue
            path = resolve_recorded_path(rows.iloc[ci].image_path, data_root)
            if not path.is_file(): raise FileNotFoundError(path)
            ax.imshow(_open_square(path, 220))

    figure.suptitle(title, fontsize=11.15, weight="bold", y=0.992)
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.963, bottom=0.012)
    return figure


def build_validation_pca_frame(features: np.ndarray, crystal_ids: np.ndarray, assignments: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the frozen W05 validation PCA plane and align it to recorded Figure-5 anchors."""
    x = normalize(np.asarray(features, dtype=np.float32), norm="l2")
    raw = PCA(n_components=2, random_state=42).fit_transform(x)
    frame = pd.DataFrame({"crystal_id": np.asarray(crystal_ids).astype(str), "raw1": raw[:, 0], "raw2": raw[:, 1]})
    assign = assignments[["crystal_id", "cluster"]].copy()
    assign["crystal_id"] = assign["crystal_id"].astype(str)
    frame = frame.merge(assign, on="crystal_id", how="inner", validate="one_to_one")

    anchor = anchors[["crystal_id", "PC1", "PC2"]].drop_duplicates("crystal_id").copy()
    anchor["crystal_id"] = anchor["crystal_id"].astype(str)
    joined = frame.merge(anchor, on="crystal_id", how="inner")
    if len(joined) < 8:
        raise ValueError("Not enough recorded PCA anchors were found to align the validation PCA plane.")

    # PCA sign is arbitrary. Try component permutations/signs and retain the highest anchor correlation.
    candidates = []
    for swap in (False, True):
        a, b = ("raw2", "raw1") if swap else ("raw1", "raw2")
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                c1 = np.corrcoef(joined[a] * s1, joined["PC1"])[0, 1]
                c2 = np.corrcoef(joined[b] * s2, joined["PC2"])[0, 1]
                candidates.append((abs(c1) + abs(c2), a, b, s1, s2))
    _, a, b, s1, s2 = max(candidates, key=lambda t: t[0])

    # Linear calibration to the frozen displayed coordinates (preserves point ordering/shape).
    ja = joined[a].to_numpy() * s1
    jb = joined[b].to_numpy() * s2
    coef1 = np.polyfit(ja, joined["PC1"].to_numpy(), 1)
    coef2 = np.polyfit(jb, joined["PC2"].to_numpy(), 1)
    frame["PC1"] = np.polyval(coef1, frame[a].to_numpy() * s1)
    frame["PC2"] = np.polyval(coef2, frame[b].to_numpy() * s2)
    return frame[["crystal_id", "cluster", "PC1", "PC2"]]


def plot_pca_extremes(
    selections: pd.DataFrame,
    region_summary: pd.DataFrame,
    *,
    data_root: str | Path,
    title: str,
    pca_points: pd.DataFrame | None = None,
    images_per_group: int = 4,
) -> Figure:
    """Reproduce Figure 5b with the dense validation PCA cloud and clean region labels."""
    use_report_style()
    summary = region_summary.sort_values("Region number").copy()
    regions = summary["Region number"].astype(int).tolist()
    figure = plt.figure(figsize=(11.2, 7.1))
    outer = figure.add_gridspec(1, 2, width_ratios=[0.96, 1.44], wspace=0.08)

    ax = figure.add_subplot(outer[0, 0])
    ax.set_title("Eight sampled PCA-extreme regions", fontsize=8.7, weight="bold", pad=5)
    if pca_points is not None:
        palette = []
        for name in ("tab20", "tab20b", "tab20c"):
            cmap = plt.get_cmap(name)
            palette.extend([cmap(i) for i in range(cmap.N)])
        colours = np.array([palette[int(c) % len(palette)] for c in pca_points["cluster"].to_numpy()])
        ax.scatter(pca_points.PC1, pca_points.PC2, c=colours, s=2.2, alpha=0.20, linewidths=0, rasterized=True)

    for _, row in summary.iterrows():
        region = int(row["Region number"])
        colour = REGION_COLOURS[region]
        subset = selections.loc[selections["Region number"] == region]
        ax.scatter(subset.PC1, subset.PC2, facecolors="none", edgecolors=colour, s=30, linewidth=0.95, zorder=5)
        ax.scatter([row["Target PC1"]], [row["Target PC2"]], marker="X", s=68, color=colour,
                   edgecolor="#222222", linewidth=0.5, zorder=6)
        ax.annotate(str(region), (row["Target PC1"], row["Target PC2"]), xytext=(5, 5), textcoords="offset points",
                    fontsize=6.3, weight="bold",
                    bbox={"boxstyle": "square,pad=0.11", "fc": BACKGROUND, "ec": colour, "lw": 0.75})

    ax.set_xlabel("Principal component 1", fontsize=7.2)
    ax.set_ylabel("Principal component 2", fontsize=7.2)
    ax.tick_params(labelsize=6.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Right-hand side: a dedicated label column followed by four image columns.
    right = outer[0, 1].subgridspec(
        len(regions), images_per_group + 1,
        width_ratios=[0.63] + [1.0] * images_per_group,
        hspace=0.10, wspace=0.09,
    )
    for ri, region in enumerate(regions):
        colour = REGION_COLOURS[region]
        rows = selections.loc[selections["Region number"] == region].sort_values("Image rank").head(images_per_group)
        region_name = str(summary.loc[summary["Region number"] == region, "Region"].iloc[0])

        label_ax = figure.add_subplot(right[ri, 0])
        label_ax.axis("off")
        label_ax.add_patch(Rectangle((0.04, 0.04), 0.075, 0.92, transform=label_ax.transAxes,
                                     facecolor=colour, edgecolor="#333333", linewidth=0.4))
        label_ax.text(0.18, 0.52, f"{region}. {region_name}", transform=label_ax.transAxes,
                      ha="left", va="center", fontsize=5.8, weight="bold", color="#111111")

        for ci in range(images_per_group):
            iax = figure.add_subplot(right[ri, ci + 1])
            iax.set_xticks([]); iax.set_yticks([]); iax.set_facecolor("black")
            for spine in iax.spines.values():
                spine.set_visible(True); spine.set_color(colour); spine.set_linewidth(1.15)
            if ci >= len(rows):
                iax.axis("off"); continue
            rec = rows.iloc[ci]
            path = resolve_recorded_path(rec.image_path, data_root)
            if not path.is_file(): raise FileNotFoundError(path)
            iax.imshow(_open_square(path, 175))
            iax.set_title(f"C{int(rec['cluster'])}", fontsize=5.8, pad=1.1)

    figure.suptitle(title, fontsize=10.9, weight="bold", y=0.985)
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.935, bottom=0.07)
    return figure
