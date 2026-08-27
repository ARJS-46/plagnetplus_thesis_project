"""Final W05 test-set plots built from frozen sample-composition tables."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd

from .style import BACKGROUND, use_report_style

LOCATION_COLOURS = {"Aegina": "#053C80", "FCT": "#ECAA26", "Methana": "#D60000", "MtStHelens": "#EE8585", "Rumania": "#86ACD2"}
LOCATION_NAMES = {"Aegina": "Aegina, Greece", "FCT": "Fish Canyon Tuff, USA", "Methana": "Methana, Greece", "MtStHelens": "Mount St Helens, USA", "Rumania": "Caraci, Romania"}
LOCATION_LEGEND_ORDER = ["Aegina", "Rumania", "FCT", "Methana", "MtStHelens"]
SPLIT_MARKERS = {"Training": "o", "Validation": "s", "Test": "^"}


def _convex_hull(points: np.ndarray) -> np.ndarray | None:
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) < 3: return None
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p)<=0: lower.pop()
        lower.append(p)
    upper=[]
    for p in pts[::-1]:
        while len(upper)>=2 and cross(upper[-2], upper[-1], p)<=0: upper.pop()
        upper.append(p)
    return np.vstack((lower[:-1], upper[:-1]))


def plot_canopy_biplot(scores: pd.DataFrame, loadings: pd.DataFrame, *, loading_labels: int = 8) -> Figure:
    """Reproduce the report's convex-hull PCA canopy biplot with in-panel legends."""
    use_report_style()
    figure, axis = plt.subplots(figsize=(8.8, 5.75))

    for location in LOCATION_LEGEND_ORDER:
        location_frame = scores.loc[scores.Location == location]
        if location_frame.empty:
            continue
        colour = LOCATION_COLOURS[location]
        points = location_frame[["PC1", "PC2"]].to_numpy()
        if len(points) >= 3:
            hull = _convex_hull(points)
            axis.add_patch(Polygon(hull, closed=True, facecolor=colour, edgecolor=colour,
                                   alpha=0.12, linewidth=1.15, zorder=1))
        elif len(points) == 2:
            axis.plot(points[:, 0], points[:, 1], color=colour, linewidth=1.10,
                      linestyle="--", alpha=0.85, zorder=1)

        for split, sf in location_frame.groupby("Split", observed=True):
            axis.scatter(sf.PC1, sf.PC2, s=45, marker=SPLIT_MARKERS[str(split)], color=colour,
                         edgecolor="black", linewidth=0.52, zorder=3)
        for row in location_frame.itertuples(index=False):
            axis.annotate(str(row.Sample), (row.PC1, row.PC2), xytext=(3, 3),
                          textcoords="offset points", fontsize=5.6)

    leading = loadings.nlargest(loading_labels, "Loading magnitude")
    x_span = max(float(np.ptp(scores.PC1)), 1e-9)
    y_span = max(float(np.ptp(scores.PC2)), 1e-9)
    scale = 0.30 * min(
        x_span / max(leading["PC1 loading"].abs().max(), 1e-9),
        y_span / max(leading["PC2 loading"].abs().max(), 1e-9),
    )
    for _, row in leading.iterrows():
        x = float(row["PC1 loading"]) * scale
        y = float(row["PC2 loading"]) * scale
        axis.annotate("", xy=(x, y), xytext=(0, 0),
                      arrowprops={"arrowstyle": "->", "color": "#D60000", "linewidth": 0.85})
        axis.text(x * 1.04, y * 1.04, f"C{int(row['Cluster'])}", color="#D60000",
                  fontsize=5.8, weight="bold")

    axis.axhline(0, color="#B5B5B5", linewidth=0.7)
    axis.axvline(0, color="#B5B5B5", linewidth=0.7)
    axis.set_xlabel("Principal component 1 (30.7%)", fontsize=7.0)
    axis.set_ylabel("Principal component 2 (22.2%)", fontsize=7.0)
    axis.set_title("SwAV W05 v2: PCA biplot with location canopies (training, validation and test)",
                   fontsize=8.8, weight="bold", pad=7)
    axis.tick_params(labelsize=6.2)
    axis.spines[["top", "right"]].set_visible(False)

    # Match the report: both legends sit inside unused right-hand plot space.
    location_handles = [Patch(facecolor=LOCATION_COLOURS[k], edgecolor="black", label=LOCATION_NAMES[k])
                        for k in LOCATION_LEGEND_ORDER]
    split_handles = [Line2D([], [], marker=m, linestyle="none", markerfacecolor="#D0D0D0",
                            markeredgecolor="black", label=s) for s, m in SPLIT_MARKERS.items()]
    first = axis.legend(handles=location_handles, title="Location", loc="upper right",
                        bbox_to_anchor=(0.985, 0.985), fontsize=5.55, title_fontsize=5.9,
                        frameon=True, borderpad=0.35, labelspacing=0.22, handlelength=1.5)
    axis.add_artist(first)
    axis.legend(handles=split_handles, title="Data split", loc="upper right",
                bbox_to_anchor=(0.985, 0.575), fontsize=5.55, title_fontsize=5.9,
                frameon=True, borderpad=0.35, labelspacing=0.25, handlelength=1.5)

    # Keep the same view as the report while leaving room for the legends.
    axis.set_xlim(scores.PC1.min() - 0.03, scores.PC1.max() + 0.055)
    axis.set_ylim(scores.PC2.min() - 0.025, scores.PC2.max() + 0.025)
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.12)
    return figure

def plot_cluster_composition(assignments: pd.DataFrame) -> Figure:
    use_report_style()
    counts=assignments.groupby(["Location","Sample","Split","Cluster"],observed=True).size().unstack(fill_value=0)
    clusters=sorted(int(v) for v in counts.columns); counts=counts.reindex(columns=clusters,fill_value=0)
    proportions=counts.div(counts.sum(axis=1),axis=0)
    ordered=sorted(counts.index,key=lambda x:(LOCATION_NAMES.get(str(x[0]),str(x[0])),str(x[1]),str(x[2])))
    counts,proportions=counts.reindex(ordered),proportions.reindex(ordered)
    labels=[f"{LOCATION_NAMES.get(str(l),l)} | {s} [{sp}]" for l,s,sp in ordered]
    colours=[]
    for name in ("tab20","tab20b","tab20c"):
        cmap=plt.get_cmap(name); colours.extend(cmap(i) for i in range(cmap.N))
    colours=colours[:len(clusters)]; y=np.arange(len(labels))
    figure,axes=plt.subplots(1,2,figsize=(11.69,8.27),sharey=True,constrained_layout=True)
    for i,cluster in enumerate(clusters):
        axes[0].barh(y,counts[cluster],left=counts[clusters[:i]].sum(axis=1),color=colours[i],edgecolor=BACKGROUND,linewidth=0.18)
        axes[1].barh(y,proportions[cluster],left=proportions[clusters[:i]].sum(axis=1),color=colours[i],edgecolor=BACKGROUND,linewidth=0.18)
    axes[0].set_yticks(y,labels,fontsize=6.6); axes[0].invert_yaxis(); axes[0].set(xlabel="Number of crystal images",title="Cluster counts")
    axes[1].set(xlabel="Within-sample proportion",xlim=(0,1),title="Cluster proportions")
    for axis in axes: axis.spines[["top","right"]].set_visible(False)
    cmap=ListedColormap(colours); norm=BoundaryNorm(np.arange(-0.5,len(clusters)+0.5),len(clusters))
    cb=figure.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,orientation="horizontal",fraction=0.045,pad=0.03)
    cb.set_label("Cluster ID (categorical block colours)"); cb.set_ticks(clusters[::max(1,len(clusters)//12)])
    figure.suptitle(f"SwAV W05 v2: cluster composition by location and sample including final test (k={len(clusters)})",weight="bold")
    return figure
