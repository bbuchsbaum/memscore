# memscore

`memscore` is a mixed Python + R project for image memorability scoring and reproducible benchmarking.

It has two user-facing layers:

- a Python CLI and API for scoring images and running benchmarks
- a native R wrapper package that can use either `reticulate` or the CLI backend

The package currently ships the released `ResMem` model as the default scorer and includes benchmark workflows for simpler frozen CLIP challengers.

## Install

### Python scoring only

Install the core scorer from the repo root:

```bash
python -m pip install .
```

### Python scoring plus benchmark extras

Install the benchmark dependencies when you want frozen CLIP challengers, LaMem benchmarking, or FIGRIM studies:

```bash
python -m pip install ".[benchmark]"
```

### Python docs

Install the doc build dependencies when working on the Sphinx site:

```bash
python -m pip install ".[docs]"
```

### R package

Install the R wrapper from a clone of this repository:

```r
install.packages("pak")
pak::pak("local::.")
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

Score a directory of images with the released `ResMem` checkpoint:

```bash
memscore predict ./images --recursive --output results/predictions.csv
```

That writes a CSV with:

- `id`
- `image_path`
- `memorability`

Benchmark frozen CLIP challengers against `ResMem` from a manifest:

```bash
memscore benchmark-manifest \
  --manifest /path/to/manifest.csv \
  --root /path/to/images \
  --clip-models ViT-B-32 RN50 \
  --regressors ridge
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
from memscore import predict_paths, benchmark_manifest, benchmark_lamem, study_figrim

predictions = predict_paths(["images/example.jpg"])
print(predictions[0].score)
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

Score images from R:

```r
scores <- memscore_predict(c("image1.jpg", "image2.jpg"))
scores[, c("id", "memorability")]
```

Run a benchmark from R:

```r
benchmark <- memscore_benchmark_manifest(
  manifest = "manifest.csv",
  clip_models = c("ViT-B-32", "RN50"),
  regressors = "ridge"
)

benchmark$summary$delta_vs_resmem
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

The current best practical FIGRIM challenger in this repo is:

- frozen `ViT-B/32 + RN50 + ViT-B/16`
- `fivecrop` CLIP embeddings
- mean ensemble across the ridge heads

See [BENCHMARKS.md](BENCHMARKS.md) for commands, manifest format, and study workflows.

<!-- albersdown:theme-note:start -->
## Albers theme
This package uses the albersdown theme. Existing vignette theme hooks are replaced so `albers.css` and local `albers.js` render consistently on CRAN and GitHub Pages. The defaults are configured via `params$family` and `params$preset` (family = 'red', preset = 'homage'). The pkgdown site uses `template: { package: albersdown }`.
<!-- albersdown:theme-note:end -->
