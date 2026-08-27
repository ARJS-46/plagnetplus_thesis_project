#!/usr/bin/env python3
"""Execute report-figure notebooks and save their outputs back into each notebook.

This is intentionally different from running nbconvert with an output directory:
outputs are written to the source .ipynb files so the saved figures are visible
when those notebooks are opened later in Jupyter or VS Code.
"""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "report_figures"
FINAL_FIGURE_DIR = ROOT / "figures" / "final"

NOTEBOOKS = [
    "figure_01_segmentation_examples.ipynb",
    "figure_02_unified_dataset.ipynb",
    "model_comparison.ipynb",
    "figure_04_grid_search.ipynb",
    "figure_05_optimised_models.ipynb",
    "figure_06_final_w05_test.ipynb",
]

REQUIRED_ENV = ["PNP_DATA_ROOT", "PNP_RUNS_ROOT", "PNP_RAW_SCAN_ROOT"]


def remove_figure_pdfs() -> None:
    FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FINAL_FIGURE_DIR.glob("*.pdf"):
        path.unlink()


def main() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing required environment variable(s): " + ", ".join(missing))

    os.environ.setdefault("PNP_PROJECT_ROOT", str(ROOT))
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    src = str(ROOT / "src")
    if src not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = src + (os.pathsep + current_pythonpath if current_pythonpath else "")

    remove_figure_pdfs()

    for name in NOTEBOOKS:
        path = NOTEBOOK_DIR / name
        print(f"\n=== EXECUTING: {name} ===", flush=True)
        notebook = nbformat.read(path, as_version=4)
        executor = ExecutePreprocessor(timeout=-1, allow_errors=False)
        executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
        nbformat.write(notebook, path)
        print(f"PASS: {name}", flush=True)

    remove_figure_pdfs()
    print("\nAll report notebooks executed in place.")
    print(f"Notebook outputs are saved in: {NOTEBOOK_DIR}")
    print(f"Final figures are in: {FINAL_FIGURE_DIR}")


if __name__ == "__main__":
    main()
