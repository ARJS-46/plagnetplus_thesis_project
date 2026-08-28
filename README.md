# PlagNetPlus

This is the repository for PlagNetPlus. It was built as part of an IRP (Independent Research Project) contributing to a masters in Geo-Energy with Machine Learning and Data Science from Imperial College London. The project focuses on applying self supervised learning (SSL) methods to learning important plagiclase crystal textures from cold-CL scans. This repository contains all the code and resources needed to reproduce any of the results seen within the final project report (ars223-final-report.pdf).


The V2 Dataset (deuplicates removed) contains **123,264 crystal images from 5 locations which have 27 physical samples between them**:

- training: 91,357
- validation: 14,919
- test: 16,988

Note that whilst most of the raw image data is in the for of cold-CL scans. The location Aegina already had the crystals segmented.

## Repository structure

```text
configs/                  grid-search, baseline and selected-model settings
figures/                  Final figures which are also in the report
manifests/                Manifest contract and external-data path guidance
notebooks/report_figures/ analysis and figure notebooks
results/reference/        Frozen small tables and JSON used in the submitted results
scripts/                  Training, evaluation and experiment entry points
src/plagnetplus/          data, model, training, evaluation and figure code
tools/dataset/            Dataset construction and normalization utilities
tools/segmentation/       PlagNet segmentation and output-packaging utilities
environment.yml           Conda environment
deliverables/             deliverables for IRP project
logbook/                  logbook for IRP project
title/                    project title for IRP
```



## External data and large artifacts

To fully reproduce results then the raw data and run artifacts need to be accessed at the zenodo page below.

**Public data/run release:** https://zenodo.org/records/22130319

The Zenodo page holds the following:

- raw non-Aegina cold-CL scans;
- inherited PlagNet checkpoint plus segmentation configuration;
- frozen segmentation manifests;
- pre-segmented Aegina source crops and manifest;
- frozen V1 split-assignment metadata;
- V2 run artifacts such as for the best-performing model w05

The suggested convenient layout for your workspce is:

```text
workspace/
├── <this-repository>/
├── data/
└── runs/
```


## Environment

From the root of the reposiory:

```bash
conda env create -f environment.yml
conda activate pnp

export PYTHONPATH="$PWD/src"
export PNP_PROJECT_ROOT="/path/to/workspace"
export PNP_DATA_ROOT="$PNP_PROJECT_ROOT/data"
export PNP_RUNS_ROOT="$PNP_PROJECT_ROOT/runs"
```

`PNP_PROJECT_ROOT` contains the external `runs/` and `data/` folders. Note: a CUDA-capable GPU should be used for any training being undertaken; other less taxing tasks like configuration dry-runs can be performed on CPU.

## Reproduction overview

workflow:

```text
raw non-Aegina cold-CL scans images
        ↓
segmentation with exisitng PlagNet package
        ↓
mask/output packaging and export of crystals
        ↓
non-Aegina crystal dataset
        +
already segmented Aegina crystal dataset
        ↓
unified dataset V1
        ↓
frozen sample-group benchmark split
        ↓
duplicate-free unified dataset V2
        ↓
SimCLR / DeepCluster / SimSiam / SwAV baselines (These are the SSL models)
        ↓
V1-selected S10 / W05 / W12 retrained on V2 (These are optimised configrations of SwAV (w05 and w12) and SimSiam (s10))
        ↓
common validation evaluation
        ↓
frozen W05 held-out test evaluation
        ↓
figures and tables from/for the final report
```

## 1. Segmentation and dataset construction

### Non-Aegina segmentation

The entry point for segmentation is is:

```bash
python tools/segmentation/run_plagnet_segmentation.py --help
```

It requires the inherited PlagNet configuration, checkpoint, a per-sample input manifest, the raw-scan root and an output root. The production thresholds used in this project are the script defaults:

```text
score threshold          0.30
duplicate IoU            0.50
duplicate containment    0.80
```

After all sample tasks are completed then continue and package the non-core mask outputs:

```bash
python tools/segmentation/pack_plagnet_noncore.py
```

This packer recreates project archive convention and leaves core reports and overlays unarchived.

### Build the crystal datasets

Once you have the segmentation outputs and source metadata, run:

```bash
python tools/dataset/extract_clean_label_maps.py
python tools/dataset/build_non_aegina_candidate_manifest.py
python tools/dataset/export_plagnet_ssl_production.py
python tools/dataset/export_aegina_ssl_compatibility.py
python tools/dataset/build_unified_ssl_manifest.py
python tools/dataset/freeze_pooled_broad_split.py
```

`freeze_pooled_broad_split.py` applies the frozen sample-group assignments used for reported benchmark. This does not repeat the earlier exploratory split search. The required `candidate_split_assignments_v1.tsv` therefore is a frozen external metadata input.

The V1 benchmark that results from this should have these or similar characteristics (depending on whether you alter things like training split)

```text
pooled_broad_p80_10_10_v1
training     101,682
validation    14,919
test          16,988
total        133,589
```

You can then calculate V1 normalization from the training split:

```bash
V1_BENCH="$PNP_DATA_ROOT/processed/plagnetplus/ssl_unified_v1/benchmarks/pooled_broad_p80_10_10_v1"

python tools/dataset/calculate_training_channel_statistics.py \
    --irp-root "$PNP_PROJECT_ROOT" \
    --output-root "$V1_BENCH/metadata/normalization_v1"
```

### Build corrected V2

V2 is a much better choice to run models on as it removes the duplicate acquisition variants identified after V1 and preserves frozen validation. However, v1 will still produce sensible reults not dissimilar from V2 results depsite the duplicate issue. 

```bash
python tools/dataset/build_unified_ssl_v2_from_v1.py
```

V2 expected benchmark is:

```text
pooled_broad_p80_10_10_v2
training      91,357
validation    14,919
test          16,988
total        123,264
```

calculate V2 normalization from corrected training split:

```bash
V2_BENCH="$PNP_DATA_ROOT/processed/plagnetplus/ssl_unified_v2/benchmarks/pooled_broad_p80_10_10_v2"

python tools/dataset/calculate_training_channel_statistics_v2.py \
    --irp-root "$PNP_PROJECT_ROOT" \
    --output-root "$V2_BENCH/metadata/normalization_v2"
```

You do not need to use use validation or test images to calculate normalization.

## 2. Grid search and model training

### The SimSiam/SwAV grid search

This is a 24-run screen which contains 12 SimSiam and 12 SwAV configurations:

```bash
python scripts/run_grid.py --trial s10 --device cuda
python scripts/run_grid.py --trial w05 --device cuda
```

You can omit/remove `--trial` to execute the complete grid sequentially.

The phrase **grid-search design V2** refers to the second grid design. The grid itself was conducted using **unified dataset V1**, not unified dataset V2.

### V2 baselines

The four final V2 baseline training runs are SimCLR, DeepCluster, SimSiam and SwAV. The frozen four-model baseline comparison retained in the submitted report is the V1 comparison; the V2 baseline runs and evaluation workflow are provided separately for reproducibility.

Verify the frozen configurations without starting training:

```bash
python scripts/run_v2.py --cohort baselines --device cpu --dry-run
```

Run cohort on GPU:

```bash
python scripts/run_v2.py --cohort baselines --device cuda
```

### V2 selected configurations

S10 (SimSiam), W05 and W12 (SwAV) were decided by using V1 grid search and then retrained on the corrected V2 dataset. They couldn't be **re-optimised on V2**. due to time constraints and hpc queues. However, this was justified as the observed difference between V1 and V2 model runs results was often very minor.

You can try a Dry-run:

```bash
python scripts/run_v2.py --cohort selected --device cpu --dry-run
```

Then move on to full training:

```bash
python scripts/run_v2.py --cohort selected --device cuda
```

Note: The run wrapper refuses silent overwriting so use empty or appropriately archived `PNP_RUNS_ROOT` to reproduce training from scratch.

## 3. V2 evaluation

There were some issues with the completed V2 baseline entry points originally exported validation embeddings only. Export the missing SimCLR and DeepCluster training embeddings before the common comparison:

```bash
python scripts/export_missing_v2_baseline_train_embeddings.py \
    --data-root "$PNP_DATA_ROOT" \
    --runs-root "$PNP_RUNS_ROOT" \
    --device cuda
```

Evaluate all four V2 baselines:

```bash
python scripts/evaluate_v2_baselines.py \
    --benchmark-root "$V2_BENCH" \
    --run-root "$PNP_RUNS_ROOT/hpc/v2/baselines/simclr/simclr_resnet18_pooled_broad_natural_seed42_v2" \
    --run-root "$PNP_RUNS_ROOT/hpc/v2/baselines/deepcluster/deepcluster_resnet18_pooled_broad_natural_seed42_v2" \
    --run-root "$PNP_RUNS_ROOT/hpc/v2/baselines/simsiam/simsiam_resnet18_pooled_broad_natural_seed42_v2" \
    --run-root "$PNP_RUNS_ROOT/hpc/v2/baselines/swav/swav_resnet18_pooled_broad_natural_seed42_v2" \
    --output-root results/reproduced/v2_baselines
```

Now evaluate w05, w12, s10 configurations:

```bash
python scripts/evaluate_v2_selected.py \
    --benchmark-root "$V2_BENCH" \
    --run-root "$PNP_RUNS_ROOT/hpc/v2/selected/simsiam_s10_lr0p1_plr0p025_wd0p0001_e100_seed42_v2" \
    --run-root "$PNP_RUNS_ROOT/hpc/v2/selected/swav_w05_lr0p3_p250_t0p1_eps0p05_e100_seed42_v2" \
    --run-root "$PNP_RUNS_ROOT/hpc/v2/selected/swav_w12_lr0p6_p500_t0p1_eps0p05_e100_seed42_v2" \
    --output-root results/reproduced/v2_selected
```

The shared evaluation metrics for assessing SSL model performance comparitive to other SSL models:

- median silhouette score;
- adjusted Rand index (ARI) across repeated k-means seeds;
- Fish Canyon Tuff cross-sample Precision@10;
- location-probe macro-F1;
- source-image enrichment@10;
- effective rank.

metric comparison here uses **k = 50** with 42, 43 and 44 as clustering seeds

## 4. W05 held-out test

W05 was frozen before the unseen test data was opened. The test set was not used for model or hyperparameter selection up until this point.

Check test set without opening test image pixels:

```bash
python scripts/evaluate_w05_final_test.py \
    --device cpu \
    --dry-run
```

Begin with a dry run. If successful, you should see:

```text
W05 checkpoint SHA-256: PASS
W05 configuration SHA-256: PASS
W05 benchmark fingerprint: PASS
Training performed: NO
Model/hyperparameter selection changed: NO
TEST IMAGE PIXELS ACCESSED: NO
W05 FINAL TEST DRY RUN: PASS
```

Now to reproduce final-test evaluation from the frozen W05 checkpoint and test data:

```bash
python scripts/evaluate_w05_final_test.py --device cuda
```

The script reuses a verified existing test embedding when available plus it compares reproduced results against frozen reference results.

The final-test/common-metric clustering uses **k = 50**. The separate W05 cluster-composition visualisation in Figure 6 uses **k = 27**; these two cluster counts serve different purposes. k=27 is the optimised k for that specific w05 run and used to produce the best figures but do not confuse it with the k bvalue here.

## 5. Reproduce report figures

Report-facing notebooks are within this folder `notebooks/report_figures/`:

```text
figure_01_segmentation_examples.ipynb
figure_02_unified_dataset.ipynb
model_comparison.ipynb
figure_04_grid_search.ipynb
figure_05_optimised_models.ipynb
figure_06_final_w05_test.ipynb
```

Run Jupyter from the repository root after setting the environment variables above:

```bash
jupyter lab
```

Alternatively, final figure files are likely already written to `figures/final/`. Tables in `results/reference/` allow report outputs to be checked without repeating every expensive training run (although this is possible/reproducible too if needed).

