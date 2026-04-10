# Benchmarking With memscore

`memscore` exposes a stable benchmark CLI on top of the released `ResMem` baseline and simpler frozen-embedding challengers.

Use the benchmark layer when you want one of three things:

- compare frozen CLIP challengers against the released `ResMem` checkpoint
- run a controlled LaMem or custom-manifest benchmark
- reproduce the repeated FIGRIM robustness study used in this repo

## Install

From a clone of this repository:

```bash
python -m pip install ".[benchmark]"
```

## Which Command Should You Run?

- `memscore benchmark-manifest`: compare `ResMem` to one or more CLIP challengers on your own train/val/test split file
- `memscore study-manifest`: run repeated train/val/test studies from a scored-image CSV with optional stratification labels
- `memscore benchmark-lamem`: reproduce the same comparison on official LaMem split files
- `memscore study-lamem`: run a reduced-cost LaMem sweep with repeated official-split subsamples, fusion, and ensembles
- `memscore study-figrim`: run repeated stratified FIGRIM splits with ridge, PCA, PLS, and ensemble variants
- `memscore study-memcat`: run repeated stratified MemCat splits with the same evaluation pipeline

## Manifest Schema

The manifest CSV must contain:

- `path`
- `score`
- `split`

It may also contain:

- `id`

The `split` column should use `train`, `val`, and `test`. Relative paths are resolved against `--root` when provided.

## LaMem Benchmark

```bash
memscore benchmark-lamem \
  --lamem-root /path/to/LaMem \
  --splits-dir /path/to/LaMem/splits \
  --fold 1 \
  --clip-models ViT-B-32 RN50 ViT-B-16 \
  --regressors ridge \
  --clip-tta fivecrop \
  --clip-tta-resize 256 \
  --output-json results/lamem_fold1.json \
  --output-predictions results/lamem_fold1_predictions.csv
```

## LaMem Study

Use this when you want to search a wider frozen-model space on LaMem without paying the full cost of 45K / 3.7K / 10K every run.

```bash
memscore study-lamem \
  --lamem-root /path/to/LaMem \
  --splits-dir /path/to/LaMem/splits \
  --fold 1 \
  --clip-models ViT-B-32 RN50 ViT-B-16 \
  --clip-tta fivecrop \
  --clip-tta-resize 256 \
  --pca-dims none 256 128 \
  --num-splits 3 \
  --limit-train 12000 \
  --limit-val 3000 \
  --limit-test 5000 \
  --output-json results/lamem_study.json \
  --output-csv results/lamem_study_summary.csv
```

This study keeps the official LaMem train/val/test split structure, repeatedly draws smaller subsets within each split, and evaluates:

- single-backbone ridge / PCA / optional PLS models
- fused-feature ridge models
- CLIP-only ensembles
- optional hybrid `ResMem + CLIP` ensembles chosen on validation performance

## Custom Manifest Benchmark

```bash
memscore benchmark-manifest \
  --manifest /path/to/manifest.csv \
  --root /path/to/images \
  --clip-models ViT-B-32 RN50 \
  --regressors ridge \
  --output-json results/manifest_results.json \
  --output-predictions results/manifest_predictions.csv
```

## Generic Manifest Study

Use this when you have a single CSV of scored images rather than predefined train/val/test splits. This is the easiest route for adding new memorability datasets such as the original Isola scenes, THINGS memorability, or artwork memorability once they are normalized to a common manifest.

```bash
memscore study-manifest \
  --manifest /path/to/dataset.csv \
  --root /path/to/images \
  --dataset-name things \
  --score-column score \
  --label-column category \
  --clip-models ViT-B-32 RN50 ViT-B-16 \
  --clip-tta fivecrop \
  --clip-tta-resize 256 \
  --pca-dims none 128 64 \
  --num-splits 10 \
  --output-json results/things_study.json \
  --output-csv results/things_study_summary.csv
```

The manifest must contain:

- `path`
- `score`

It may also contain:

- `id`
- a label column such as `category`, `concept`, or `artist` for stratified repeated splits

## FIGRIM Study

```bash
memscore study-figrim \
  --figrim-root /tmp/figrim \
  --protocols across within \
  --score-types corrected_hit_rate \
  --clip-models ViT-B-32 RN50 ViT-B-16 \
  --clip-tta fivecrop \
  --clip-tta-resize 256 \
  --pca-dims none 128 64 32 \
  --num-splits 10 \
  --output-json results/figrim_study.json \
  --output-csv results/figrim_study_summary.csv
```

When multiple backbones are provided, the study also evaluates:

- fused-feature ridge models
- optional PLS models
- mean ensembles
- validation-weighted ensembles
- rank-based ensembles
- stacked ridge ensembles

The main outputs are:

- a JSON study object with split-level and summary metrics
- a CSV summary table for downstream plotting or paper tables

## MemCat Study

```bash
memscore study-memcat \
  --memcat-root /path/to/MemCat \
  --clip-models ViT-B-32 RN50 ViT-B-16 \
  --clip-tta fivecrop \
  --clip-tta-resize 256 \
  --pca-dims none 128 64 32 \
  --num-splits 10 \
  --output-json results/memcat_study.json \
  --output-csv results/memcat_study_summary.csv
```

MemCat contains ~10,000 images across five top-level categories (animal, food, landscape, sports, vehicle) with memorability scores based on a visual memory game with false-alarm correction. The study generates repeated stratified splits (stratified by category) and evaluates the same CLIP + ridge / PCA / ensemble pipeline used for FIGRIM.

When multiple backbones are provided, the study also evaluates:

- fused-feature ridge models
- optional PLS models
- mean ensembles
- validation-weighted ensembles
- rank-based ensembles
- stacked ridge ensembles

The `--score-column` flag (default `memorability_w_fa_correction`) selects the target column from the MemCat CSV. The `--csv-path` flag overrides the default CSV location (`<memcat-root>/data/memcat_image_data.csv`).

## Current Results

### Recommended uniform out-of-sample model

For general-purpose prediction, use one fixed frozen recipe rather than selecting a different model per dataset:

- `ViT-B-32 + RN50 + ViT-B-16`
- `fivecrop` TTA (resize 256)
- separate ridge head per backbone
- final prediction = plain mean of the three backbone predictions

Locked-test results for this uniform recipe are in `results/uniform_3clip_locked_ood.json`.

| Dataset | ResMem | Uniform 3-CLIP mean | Delta |
|---|---:|---:|---:|
| FIGRIM | 0.441 | **0.755** | +0.315 |
| Isola | 0.567 | **0.760** | +0.193 |
| MemCat | 0.796 | **0.818** | +0.021 |
| THINGS 500x10 locked pilot | 0.208 | **0.393** | +0.184 |

Mean Spearman across the four locked OOD tests is **0.681**. The all-dataset supervised multihead run (`results/multihead_vitb32_all5.json`) reached **0.642** on the same four OOD tests, so the frozen uniform ensemble remains the stronger default. Full LaMem `RN50` and `ViT-B-16` fivecrop embeddings are not yet cached, so LaMem is still reported separately below.

### Cross-dataset summary

| Dataset | N images | ResMem | memscore (best) | Delta | Trained on? |
|---|---|---|---|---|---|
| **Isola** | 2,222 | 0.645 | **0.789** | +0.144 | No |
| **FIGRIM** | 630 | 0.524 | **0.749** | +0.225 | No |
| **MemCat** | 10,000 | 0.787 | **0.823** | +0.036 | No |
| **THINGS** (balanced pilot, 500x10) | 5,000 | 0.243 | **0.409** | +0.166 | No |
| **LaMem** (fold 1) | 58,741 | **0.815** | 0.695 | -0.120 | Yes |

ResMem was fine-tuned on LaMem and wins on its home turf. On out-of-distribution datasets (Isola, FIGRIM, MemCat, and balanced THINGS pilots) where generalization matters, frozen CLIP challengers outperform ResMem consistently. The gap is especially large on classic scene memorability datasets like Isola and FIGRIM. Absolute Spearman on THINGS is lower for both model families, suggesting a harder fine-grained object-ranking benchmark, but the relative advantage still favors memscore even as the balanced pilot scales up.

### Isola (original scene memorability dataset)

5 repeated random splits, `fivecrop` TTA (resize 256). The full study used two CLIP backbones (`ViT-B-32`, `RN50`) plus fusion and ensemble variants on the original 2,222-scene Isola release.

| Model | Mean Spearman | Win Rate vs ResMem |
|---|---|---|
| ResMem | 0.645 | -- |
| ViT-B-32 (ridge, no PCA) | 0.774 | 100% |
| RN50 (ridge, no PCA) | 0.765 | 100% |
| Fused ViT-B-32+RN50 (ridge) | 0.782 | 100% |
| Stacked ridge ensemble | 0.788 | 100% |
| Rank-mean ensemble | 0.789 | 100% |
| **Mean ensemble (2 CLIP)** | **0.789** | **100%** |

**Best delta: +0.144 Spearman** (mean ensemble vs ResMem). Isola is a strong scene-level transfer benchmark and lines up with the broader pattern: frozen CLIP features plus a simple linear head generalize better than ResMem once evaluation moves off LaMem.

### FIGRIM (`across_category`, corrected hit rate)

10 repeated stratified splits, `fivecrop` TTA (resize 256).

| Model | Mean Spearman | Win Rate vs ResMem |
|---|---|---|
| ResMem | 0.524 | -- |
| ViT-B-32 (ridge, no PCA) | 0.700 | 100% |
| RN50 (ridge, no PCA) | 0.718 | 100% |
| ViT-B-16 (ridge, no PCA) | 0.689 | 100% |
| ViT-L-14 (ridge, no PCA) | 0.685 | 100% |
| DINOv2 ViT-L/14 (ridge, no PCA) | 0.491 | 40% |
| CLIP ViT-B-32 + DINOv2 fused (ridge) | 0.676 | 100% |
| Fused ViT-B-32+RN50+ViT-B-16 (ridge) | 0.746 | 100% |
| **Mean ensemble (3 CLIP)** | **0.749** | **100%** |
| Stacked ridge ensemble | 0.744 | 100% |

**Best delta: +0.225 Spearman** (mean ensemble vs ResMem). On this small dataset (630 images) the 3-model CLIP ensemble is the strongest; DINOv2 fusion hurts due to the high dimensionality relative to the small sample size.

### MemCat (`memorability_w_fa_correction`)

10 repeated stratified splits (by category), `fivecrop` TTA (resize 256).

| Model | Mean Spearman | Win Rate vs ResMem |
|---|---|---|
| ResMem | 0.787 | -- |
| ViT-B-32 (ridge, no PCA) | 0.803 | 90% |
| RN50 (ridge, no PCA) | 0.804 | 100% |
| ViT-B-16 (ridge, no PCA) | 0.805 | 100% |
| ViT-L-14 (ridge, no PCA) | 0.811 | 100% |
| DINOv2 ViT-L/14 (ridge, no PCA) | 0.767 | 0% |
| CLIP ViT-B-32 + DINOv2 fused (ridge) | 0.812 | 100% |
| Fused ViT-B-32+RN50+ViT-B-16 (ridge) | 0.823 | 100% |
| **Mean ensemble (3 CLIP)** | **0.823** | **100%** |
| Stacked ridge ensemble | 0.823 | 100% |

**Best delta: +0.036 Spearman** (mean ensemble vs ResMem). On this larger dataset, CLIP + DINOv2 fusion (0.812) nearly matches the 3-model CLIP ensemble (0.823) with a simpler two-backbone setup.

### THINGS (balanced concept pilot)

Balanced pilot with 500 concepts x 10 images each (5,000 total), 5 repeated stratified splits by concept, `fivecrop` TTA (resize 256). The full study below evaluates single backbones plus fused, mean, weighted, rank, and stacked ensembles with no PCA reduction.

| Model | Mean Spearman | Win Rate vs ResMem |
|---|---|---|
| ResMem | 0.243 | -- |
| ViT-B-32 (ridge, no PCA) | 0.389 | 100% |
| RN50 (ridge, no PCA) | 0.370 | 100% |
| ViT-B-16 (ridge, no PCA) | 0.397 | 100% |
| Fused ViT-B-32+RN50+ViT-B-16 (ridge) | 0.405 | 100% |
| Mean ensemble (3 CLIP) | 0.409 | 100% |
| **Rank-mean ensemble (3 CLIP)** | **0.409** | **100%** |
| Validation-weighted ensemble | 0.407 | 100% |
| Stacked ridge ensemble | 0.404 | 100% |

**Best delta: +0.166 Spearman** (rank-mean ensemble vs ResMem). THINGS is a harder object-centric benchmark than FIGRIM or MemCat, so absolute Spearman is lower for both approaches. The relative result remains stable as the balanced pilot grows from 150x10 to 300x10 to 500x10: the 3-model CLIP family stays clearly ahead of ResMem on every split. The simple raw-score mean ensemble is nearly tied with the rank-mean variant and preserves the MSE advantage; rank ensembles optimize Spearman at the expense of calibration.

### LaMem (fold 1, official splits)

Single train/val/test split (45K / 3.7K / 10K), `fivecrop` TTA (resize 256), ridge regressor.

| Model | Test Spearman | Delta vs ResMem |
|---|---|---|
| **ResMem** | **0.815** | -- |
| Best CLIP (ViT-B-32, ridge) | 0.695 | -0.120 |

ResMem was trained on LaMem and outperforms frozen challengers on this dataset. This is expected: end-to-end fine-tuning on the exact training distribution yields a strong in-domain model. The frozen CLIP approach trades in-domain performance for better generalization to unseen datasets.
