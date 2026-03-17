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
- `memscore benchmark-lamem`: reproduce the same comparison on official LaMem split files
- `memscore study-figrim`: run repeated stratified FIGRIM splits with ridge, PCA, PLS, and ensemble variants

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

## Current FIGRIM Result

The strongest practical setting in this repo is still:

- frozen `ViT-B/32 + RN50 + ViT-B/16`
- `fivecrop` test-time augmentation
- plain mean ensemble across the three ridge heads

That outperforms released `ResMem` on FIGRIM:

- `across_category`: `0.7413` vs `0.4510` Spearman
- `within_category`: `0.6393` vs `0.2366` Spearman
