# Inspect the resolved memscore backend

Inspect the resolved memscore backend

## Usage

``` r
memscore_cli_info(
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  module = getOption("memscore.module", "memscore"),
  workdir = getOption("memscore.workdir")
)
```

## Arguments

- cli:

  Optional explicit path to a `memscore` executable.

- python:

  Optional explicit path to a Python interpreter.

- module:

  Python module name to run when using a Python backend.

- workdir:

  Optional working directory for backend execution.

## Value

A list with `command`, `args`, `workdir`, and `source`.

## Examples

``` r
memscore_cli_info(cli = "/usr/local/bin/memscore")
#> $command
#> [1] "/usr/local/bin/memscore"
#> 
#> $args
#> character(0)
#> 
#> $workdir
#> NULL
#> 
#> $source
#> [1] "explicit-cli"
#> 
memscore_cli_info(python = "/usr/bin/python3", workdir = tempdir())
#> $command
#>                     memscore 
#> "/opt/homebrew/bin/memscore" 
#> 
#> $args
#> character(0)
#> 
#> $workdir
#> [1] "/var/folders/9h/nkjq6vss7mqdl4ck7q1hd8ph0000gp/T//Rtmp5egoxn"
#> 
#> $source
#> [1] "path-cli"
#> 
```
