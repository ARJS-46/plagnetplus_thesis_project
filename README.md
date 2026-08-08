# PlagNetPlus

PlagNetPlus is the codebase developed for an Individual Research Project as part of an Msc in Geo-Energy and Machine Learning and Data Science investigating self-supervised representation learning for cold cathodoluminescence images of volcanogenic plagioclase crystals.

The project compares four self-supervised learning methods:

- SimCLR
- SimSiam
- DeepCluster
- SwAV

The reusable Python package is located in `src/plagnetplus`.

## Repository structure

- `src/plagnetplus/` - reusable Python package
- `configs/` - dataset, model and experiment configurations
- `tests/` - automated package tests
- `project_code/` - project notebooks and analysis
- `deliverables/` - submitted project documents
- `logbook/` - project logbook
- `title/` - project title information

## Installation

PlagNetPlus requires Python 3.10 or later.

Install the core package from the repository root with:

    python -m pip install -e .

Install the self-supervised learning and development dependencies with:

    python -m pip install -e ".[ssl,dev]"
