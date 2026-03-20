# Benchmark LaMem split files

Runs `memscore benchmark-lamem`.

## Usage

``` r
memscore_benchmark_lamem(
  lamem_root,
  splits_dir,
  fold = 1L,
  train_file = NULL,
  val_file = NULL,
  test_file = NULL,
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

- lamem_root:

  Path to the LaMem dataset root.

- splits_dir:

  Directory containing official LaMem split files.

- fold:

  Fold number when using `train_<fold>.txt` style split files.

- train_file:

  Optional explicit training split file.

- val_file:

  Optional explicit validation split file.

- test_file:

  Optional explicit test split file.

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
memscore_benchmark_lamem(
  lamem_root = "/path/to/LaMem",
  splits_dir = "/path/to/LaMem/splits",
  clip_models = c("ViT-B-32", "RN50"),
  regressors = "ridge"
)
} # }
```
