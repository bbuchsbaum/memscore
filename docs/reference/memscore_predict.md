# Score image memorability

Scores one or more image files or directories using either the
command-line backend or the optional `reticulate` bridge.

## Usage

``` r
memscore_predict(
  paths,
  recursive = FALSE,
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output = NULL,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
)
```

## Arguments

- paths:

  Character vector of image files or directories.

- recursive:

  Recurse into subdirectories when a path is a directory.

- batch_size:

  Batch size for inference.

- device:

  Torch device string passed through to the backend.

- cache_dir:

  Optional cache directory for downloaded checkpoints.

- resmem_checkpoint:

  Optional explicit path to a ResMem checkpoint.

- output:

  Optional CSV path for backend-generated predictions.

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

A data frame with `id`, `image_path`, and `memorability`.

## Examples

``` r
if (FALSE) { # \dontrun{
memscore_configure(
  python = "/usr/bin/python3",
  workdir = "/path/to/memscore"
)

memscore_predict(c("image1.jpg", "image2.jpg"))
} # }
```
