# Frozen reference tables

These small tables are copied from the audited development outputs or from the
reported grid-search notebook. They let the report notebooks reproduce tables
and plots without silently recomputing model-selection results.

- `datasets/`: unified dataset v1 and v2 composition.
- `v1_baselines/`: four-model baseline comparison metrics retained in the submitted report.
- `v1_grid_search/`: all 24 SimSiam and SwAV screen results.
- `v1_selected/`: archived S10, W05 and W12 long-run diagnostics.
- `v2_selected/`: S10, W05 and W12 V2 validation metrics.
- `w05_test/`: frozen final W05 test metrics and access records.
- `figure_05/` and `figure_06/`: image selections and cluster-composition data.

The report-facing baseline comparison is retained from V1; V2 baseline run
artifacts and evaluation scripts are provided separately and should not be
silently substituted into the frozen report table.

Do not replace these files with test-set-guided alternatives after model
selection.

