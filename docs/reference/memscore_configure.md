# Configure how R finds the memscore backend

`memscore` can use either:

## Usage

``` r
memscore_configure(
  cli = NULL,
  python = NULL,
  backend = NULL,
  module = "memscore",
  workdir = NULL
)
```

## Arguments

- cli:

  Optional explicit path to a `memscore` executable.

- python:

  Optional explicit path to a Python interpreter.

- backend:

  Backend mode. Use `"cli"` to force command-line execution or
  `"reticulate"` to call the Python package through `reticulate`.

- module:

  Python module name to run when using a Python backend.

- workdir:

  Optional working directory for backend execution. This is useful
  during local development when running `python -m memscore` from a
  source checkout.

## Value

Invisibly returns the resolved backend configuration.

## Details

- a `memscore` executable on `PATH`, or

- a Python interpreter that can run `python -m memscore`

- a `reticulate` bridge to the Python package itself

These settings are stored as R options and are used by the other package
functions.

## Examples

``` r
info <- memscore_cli_info(cli = "/usr/local/bin/memscore")
info$source
#> [1] "explicit-cli"
```
