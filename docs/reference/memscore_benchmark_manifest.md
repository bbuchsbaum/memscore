# Benchmark a custom manifest

Runs `memscore benchmark-manifest` and returns the parsed summary plus
output file locations.

## Usage

``` r
memscore_benchmark_manifest(
  manifest,
  root = NULL,
  clip_models = c("ViT-B-32"),
  clip_pretrained = "openai",
  regressors = c("ridge", "mlp"),
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output_json = NULL,
  output_predictions = NULL,
  random_state = 0L,
  limit_train = NULL,
  limit_val = NULL,
  limit_test = NULL,
  clip_tta = "none",
  clip_tta_resize = NULL,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
)
```

## Arguments

- manifest:

  CSV containing `path`, `score`, `split`, and optional `id`.

- root:

  Optional base directory for relative image paths in the manifest.

- clip_models:

  Character vector of frozen CLIP backbones to evaluate.

- clip_pretrained:

  Pretrained tag passed through to the backend.

- regressors:

  Character vector of lightweight regressors to compare.

- batch_size:

  Batch size for inference.

- device:

  Torch device string passed through to the backend.

- cache_dir:

  Optional cache directory for downloaded checkpoints.

- resmem_checkpoint:

  Optional explicit path to a ResMem checkpoint.

- output_json:

  Optional summary JSON path. A temporary file is created if omitted.

- output_predictions:

  Optional predictions CSV path. A temporary file is created if omitted.

- random_state:

  Random seed for challenger selection.

- limit_train:

  Optional cap on the number of training records.

- limit_val:

  Optional cap on the number of validation records.

- limit_test:

  Optional cap on the number of test records.

- clip_tta:

  CLIP test-time augmentation: `none`, `fivecrop`, or `tencrop`.

- clip_tta_resize:

  Optional resize target for CLIP test-time augmentation.

- cli:

  Optional explicit path to a `memscore` executable.

- python:

  Optional explicit path to a Python interpreter.

- backend:

  Backend mode. Use `"cli"` to force command-line execution or
  `"reticulate"` to call the Python package in-process.

- workdir:

  Optional working directory for backend execution.

## Value

A list with `summary`, `output_json`, `output_predictions`, and
`stdout`.

## Examples

``` r
if (FALSE) { # \dontrun{
memscore_benchmark_manifest(
  manifest = "manifest.csv",
  root = "images",
  clip_models = c("ViT-B-32", "RN50"),
  regressors = "ridge"
)
} # }
```
