"""Shared report-figure colours and Matplotlib defaults."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt

BACKGROUND = "#FFFFF0"
BLUE = "#053C80"
BLUE_FILL = "#C3DDE9"
BLUE_PALE = "#E6F0F4"
NAVY = "#00152F"
ORANGE = "#ECAA26"
ORANGE_FILL = "#F8D996"

LOCATION_COLOURS: Mapping[str, str] = {
    "Aegina": "#0072B2",
    "FCT": "#E69F00",
    "Methana": "#009E73",
    "MtStHelens": "#CC79A7",
    "Rumania": "#D55E00",
}

LOCATION_NAMES: Mapping[str, str] = {
    "Aegina": "Aegina, Greece",
    "FCT": "Fish Canyon Tuff, USA",
    "Methana": "Methana, Greece",
    "MtStHelens": "Mount St Helens, USA",
    "Rumania": "Caraci, Romania",
}


def use_report_style() -> None:
    """Apply the restrained colour and typography settings used in the report."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "savefig.transparent": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

