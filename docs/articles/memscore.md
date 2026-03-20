# Getting Started with memscore

``` r
library(memscore)
```

## What does memscore do?

`memscore` gives R users access to the Python `memscore` package from
ordinary R functions. You can run it in-process through `reticulate`, or
fall back to a CLI bridge when that is more convenient for deployment.

This vignette uses a mocked backend so the examples run during package
checks. The public R API and return values are the same as in a real
session.

## How do you configure the backend?

The recommended path is to point R at a Python interpreter and set
`backend = "reticulate"`.

``` r
memscore_configure(
  python = "/usr/bin/python3",
  backend = "reticulate",
  workdir = tempdir()
)

getOption("memscore.backend")
#> [1] "reticulate"
stopifnot(identical(getOption("memscore.backend"), "reticulate"))
```

If you need a pure command-line fallback instead, you can force that
explicitly.

``` r
memscore_configure(
  cli = "/path/to/memscore",
  backend = "cli",
  workdir = "/path/to/memscore"
)
```

## How do you score images?

The first workflow is simple scoring. The return value is a data frame
with one row per input image.

``` r
scores <- memscore_predict(c("beach.jpg", "forest.jpg"))
scores
#>       id      image_path memorability
#> 1  beach  /tmp/beach.jpg         0.72
#> 2 forest /tmp/forest.jpg         0.54
stopifnot(all(c("id", "image_path", "memorability") %in% names(scores)))
stopifnot(all(is.finite(scores$memorability)))
stopifnot(all(scores$memorability >= 0 & scores$memorability <= 1))
```

## How do you benchmark a custom manifest?

The manifest workflow compares the released `ResMem` checkpoint against
frozen CLIP challengers on your own train/val/test split file.

``` r
benchmark <- memscore_benchmark_manifest(
  manifest = manifest_path,
  clip_models = c("ViT-B-32", "RN50"),
  regressors = "ridge"
)

benchmark$summary$delta_vs_resmem
#> $spearman
#> [1] 0.18
#> 
#> $mse
#> [1] -0.012
stopifnot(benchmark$summary$delta_vs_resmem$spearman > 0)
file.exists(benchmark$output_json)
#> [1] TRUE
file.exists(benchmark$output_predictions)
#> [1] TRUE
```

## How do you launch the FIGRIM study?

For repeated FIGRIM comparisons,
[`memscore_study_figrim()`](https://bbuchsbaum.github.io/memscore/reference/memscore_study_figrim.md)
exposes the same workflow regardless of whether you route through
`reticulate` or the CLI.

``` r
study <- memscore_study_figrim(
  figrim_root = tempdir(),
  clip_models = c("ViT-B-32", "RN50", "ViT-B-16"),
  clip_tta = "fivecrop"
)

study$study$summary_rows
#>          protocol model_mean_spearman
#> 1 across_category                0.74
#> 2 within_category                0.64
stopifnot(nrow(study$study$summary_rows) >= 1)
```

## Where next?

- Use
  [`?memscore_predict`](https://bbuchsbaum.github.io/memscore/reference/memscore_predict.md)
  for the scoring interface.
- Use
  [`?memscore_benchmark_manifest`](https://bbuchsbaum.github.io/memscore/reference/memscore_benchmark_manifest.md)
  and
  [`?memscore_study_figrim`](https://bbuchsbaum.github.io/memscore/reference/memscore_study_figrim.md)
  for the two main evaluation workflows.
- See the top-level `README.md` and `BENCHMARKS.md` files in the
  repository for the matching Python commands.
