# Manifest contract

Dataset manifests aren't duplicated into this repository on purpose.

 Set:
`PNP_DATA_ROOT` so paths in frozen JSON configurations resolve to:

- `processed/plagnetplus/ssl_unified_v1/` for the grid search; and
- `processed/plagnetplus/ssl_unified_v2/` for v2 baseline and selected runs.

Each benchmark directory must contain train, validation and test CSV manifests,
the image-root map and the recorded transform configuration. Expected row counts
and, where ti is available, manifest fingerprints are stored in the run configs.

More about data access is noted in the main repository README.

