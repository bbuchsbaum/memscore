# memscore

`memscore` is a mixed Python + R project for image memorability scoring and reproducible benchmarking.

It has two user-facing layers:

- a Python CLI and API for scoring images and running benchmarks
- a native R wrapper package that can use either `reticulate` or the CLI backend

The flagship scorer is the standard `memscore` ensemble:

- frozen `ViT-B-32 + RN50 + ViT-B-16`
- five-crop CLIP embeddings at resize 256
- one ridge head per backbone
- final score = plain mean of the three ridge predictions

`ResMem` remains in the package as a reference baseline for direct comparison.

## Install

### Python scoring

Install the Python package from the repo root:

```bash
python -m pip install .
```

This includes the standard `memscore` scorer runtime: CLIP embedding, ridge
regression, and image preprocessing.

### Python benchmark extras

The benchmark extra is kept for compatibility and currently resolves to the
same scientific stack used by the standard scorer:

```bash
python -m pip install ".[benchmark]"
```

The equivalent focused extra is:

```bash
python -m pip install ".[standard]"
```

### Python docs

Install the doc build dependencies when working on the Sphinx site:

```bash
python -m pip install ".[docs]"
```

### R package

Install the R wrapper from a clean clone of this repository:

```r
install.packages("pak")
pak::pak("local::.")
```

If this working tree contains large local datasets, caches, or virtualenvs, use the curated installer instead. It stages only the R package files before installation:

```r
source("scripts/install_r_package.R")
install_memscore_r()
```

The R package does not bundle Python itself. It expects either:

- a `memscore` executable on `PATH`, or
- a Python interpreter that can run `python -m memscore`

If you want to avoid the CLI path entirely, configure the R wrapper to use
`reticulate` instead:

```r
memscore::memscore_configure(
  python = "/path/to/python",
  backend = "reticulate",
  workdir = "/path/to/repo"
)
```

## Get Started In Python

### CLI

Train a standard `memscore` scorer artifact from a locked labeled manifest:

```bash
memscore train-standard-scorer \
  --manifest /path/to/manifest.csv \
  --root /path/to/images \
  --output-model artifacts/memscore_standard.pkl
```

By default this trains on `train` and `val` rows only. It does not train on
`test` rows unless you explicitly pass `--include-test`.

Score a new image or directory with the standard `memscore` scorer:

```bash
memscore predict-standard \
  --model artifacts/memscore_standard.pkl \
  ./images \
  --recursive \
  --output results/predictions.csv
```

That writes a CSV with:

- `id`
- `image_path`
- `memorability`

Input images can be different sizes and aspect ratios. The Python backend opens
each file with Pillow, converts it to RGB, and applies the same CLIP
preprocessing used during training. The standard scorer uses five crops after
resizing to 256 pixels, so users should pass the original images rather than
manually forcing them to square thumbnails.

Score images with the released `ResMem` reference baseline:

```bash
memscore predict-resmem ./images --recursive --output results/resmem_predictions.csv
```

The older `memscore predict` command is kept as an alias for the ResMem
reference path.

Benchmark frozen CLIP challengers against `ResMem` from a manifest:

```bash
memscore benchmark-manifest \
  --manifest /path/to/manifest.csv \
  --root /path/to/images \
  --clip-models ViT-B-32 RN50 \
  --regressors ridge
```

Run the standardized out-of-sample model on a locked train/val/test manifest:

```bash
memscore benchmark-standard \
  --manifest /path/to/manifest.csv \
  --root /path/to/images \
  --output-json results/standard_benchmark.json
```

Run the repeated FIGRIM study:

```bash
memscore study-figrim \
  --figrim-root /path/to/figrim \
  --clip-models ViT-B-32 RN50 ViT-B-16 \
  --clip-tta fivecrop
```

### Python API

The public Python API is intentionally small:

```python
from memscore import predict_standard_paths, predict_paths, benchmark_standard_manifest

predictions = predict_standard_paths(
    ["images/example.jpg"],
    model_path="artifacts/memscore_standard.pkl",
)
print(predictions[0].score)

resmem_reference = predict_paths(["images/example.jpg"])

result = benchmark_standard_manifest("manifest.csv", root="images")
print(result["standard_model"]["summary"]["mean_spearman"])
```

## Get Started In R

Configure the backend once per session:

```r
library(memscore)

memscore_configure(
  python = "/path/to/python",
  workdir = "/path/to/repo"
)

memscore_cli_info()
```

Train a standard scorer artifact and score images from R:

```r
manifest <- system.file("extdata/tiny_manifest.csv", package = "memscore")

memscore_train_standard_scorer(
  manifest = manifest,
  root = dirname(manifest),
  output_model = "memscore_standard.pkl"
)

scores <- memscore_predict_standard(
  c("image1.jpg", "image2.jpg"),
  model = "memscore_standard.pkl"
)
scores[, c("id", "memorability")]
```

Use `memscore_predict()` when you specifically want the ResMem reference
baseline.

Run the standard benchmark from R:

```r
manifest <- system.file("extdata/tiny_manifest.csv", package = "memscore")

benchmark <- memscore_benchmark_standard(
  manifest = manifest,
  root = dirname(manifest)
)

benchmark$summary$standard_model$comparison_vs_resmem
```

## Documentation

- Python docs: [python-docs/index.rst](python-docs/index.rst)
- R getting-started vignette: [vignettes/memscore.Rmd](vignettes/memscore.Rmd)
- R benchmarking vignette: [vignettes/benchmarking.Rmd](vignettes/benchmarking.Rmd)
- Benchmark guide: [BENCHMARKS.md](BENCHMARKS.md)

## Repository Layout

- `memscore/`: public Python API and CLI
- `membench/`: Python benchmark engine
- `resmem_legacy/`: internal packaged copy of the released ResMem baseline code
- `R/`: native R wrapper package
- `vignettes/`: R package articles
- `python-docs/`: Sphinx documentation source for the Python and mixed-language workflows
- `docs/`: pkgdown site output
- `pytests/`: Python tests
- `tests/testthat/`: R tests

## Benchmark Notes

The recommended uniform out-of-sample model in this repo is:

- frozen `ViT-B/32 + RN50 + ViT-B/16`
- `fivecrop` CLIP embeddings at resize 256
- separate ridge head per backbone
- plain mean ensemble across the ridge predictions

See [BENCHMARKS.md](BENCHMARKS.md) for commands, manifest format, and study workflows.

<!-- albersdown:theme-note:start -->
## Albers theme
This package uses the albersdown theme. Existing vignette theme hooks are replaced so `albers.css` and local `albers.js` render consistently on CRAN and GitHub Pages. The defaults are configured via `params$family` and `params$preset` (family = 'red', preset = 'homage'). The pkgdown site uses `template: { package: albersdown }`.
<!-- albersdown:theme-note:end -->
