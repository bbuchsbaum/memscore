# Run the repeated FIGRIM study

Runs `memscore study-figrim`.

## Usage

``` r
memscore_study_figrim(
  figrim_root,
  protocols = c("across", "within"),
  score_types = c("corrected_hit_rate"),
  clip_models = c("ViT-B-32", "RN50"),
  clip_pretrained = "openai",
  clip_tta = "none",
  clip_tta_resize = NULL,
  pca_dims = c("none", "128", "64", "32"),
  pls_dims = character(),
  num_splits = 10L,
  train_size = 0.7,
  val_size = 0.1,
  test_size = 0.2,
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output_json = NULL,
  output_csv = NULL,
  random_state = 0L,
  ensemble_weight_step = 0.05,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
)
```

## Arguments

- figrim_root:

  Path to the FIGRIM download directory.

- protocols:

  FIGRIM protocols to evaluate.

- score_types:

  FIGRIM target definitions to evaluate.

- clip_models:

  Character vector of frozen CLIP backbones to evaluate.

- clip_pretrained:

  Pretrained tag passed through to the backend.

- clip_tta:

  CLIP test-time augmentation: `none`, `fivecrop`, or `tencrop`.

- clip_tta_resize:

  Optional resize target for CLIP test-time augmentation.

- pca_dims:

  PCA dimensions to evaluate. Use `"none"` for full embeddings.

- pls_dims:

  Optional PLS component counts.

- num_splits:

  Number of repeated stratified splits.

- train_size:

  Fraction assigned to the training split.

- val_size:

  Fraction assigned to the validation split.

- test_size:

  Fraction assigned to the test split.

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

- output_csv:

  Optional summary CSV path. A temporary file is created if omitted.

- random_state:

  Random seed for challenger selection.

- ensemble_weight_step:

  Weight step used for validation-weighted ensembling.

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

A list with `study`, `output_json`, `output_csv`, and `stdout`.

## Examples

``` r
if (FALSE) { # \dontrun{
memscore_study_figrim(
  figrim_root = "/path/to/figrim",
  clip_models = c("ViT-B-32", "RN50", "ViT-B-16"),
  clip_tta = "fivecrop"
)
} # }
```
