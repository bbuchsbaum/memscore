#' Configure how R finds the memscore backend
#'
#' `memscore` can use either:
#'
#' - a `memscore` executable on `PATH`, or
#' - a Python interpreter that can run `python -m memscore`
#' - a `reticulate` bridge to the Python package itself
#'
#' These settings are stored as R options and are used by the other package
#' functions.
#'
#' @param cli Optional explicit path to a `memscore` executable.
#' @param python Optional explicit path to a Python interpreter.
#' @param backend Backend mode. Use `"cli"` to force command-line execution or
#'   `"reticulate"` to call the Python package through `reticulate`.
#' @param module Python module name to run when using a Python backend.
#' @param workdir Optional working directory for backend execution. This is
#'   useful during local development when running `python -m memscore` from a
#'   source checkout.
#'
#' @return Invisibly returns the resolved backend configuration.
#'
#' @examples
#' info <- memscore_cli_info(cli = "/usr/local/bin/memscore")
#' info$source
#' @export
memscore_configure <- function(cli = NULL, python = NULL, backend = NULL, module = "memscore", workdir = NULL) {
  if (!is.null(cli)) {
    options(memscore.cli = cli)
  }
  if (!is.null(python)) {
    options(memscore.python = python)
  }
  if (!is.null(backend)) {
    options(memscore.backend = .normalize_backend(backend))
  }
  if (!is.null(module)) {
    options(memscore.module = module)
  }
  if (!is.null(workdir)) {
    options(memscore.workdir = workdir)
  }

  invisible(
    .resolve_memscore_backend(
      backend = getOption("memscore.backend", "auto"),
      cli = getOption("memscore.cli"),
      python = getOption("memscore.python"),
      module = getOption("memscore.module", "memscore"),
      workdir = getOption("memscore.workdir")
    )
  )
}

#' Inspect the resolved memscore backend
#'
#' @param cli Optional explicit path to a `memscore` executable.
#' @param python Optional explicit path to a Python interpreter.
#' @param module Python module name to run when using a Python backend.
#' @param workdir Optional working directory for backend execution.
#'
#' @return A list with `command`, `args`, `workdir`, and `source`.
#'
#' @examples
#' memscore_cli_info(cli = "/usr/local/bin/memscore")
#' memscore_cli_info(python = "/usr/bin/python3", workdir = tempdir())
#' @export
memscore_cli_info <- function(
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  module = getOption("memscore.module", "memscore"),
  workdir = getOption("memscore.workdir")
) {
  cli <- cli %||% Sys.getenv("MEMSCORE_CLI", unset = "")
  if (nzchar(cli)) {
    return(list(
      command = .normalize_optional_path(cli),
      args = character(),
      workdir = workdir %||% NULL,
      source = "explicit-cli"
    ))
  }

  path_cli <- Sys.which("memscore")
  if (nzchar(path_cli)) {
    return(list(
      command = path_cli,
      args = character(),
      workdir = workdir %||% NULL,
      source = "path-cli"
    ))
  }

  python <- python %||% Sys.getenv("MEMSCORE_PYTHON", unset = "")
  if (!nzchar(python)) {
    python3 <- Sys.which("python3")
    python <- if (nzchar(python3)) python3 else Sys.which("python")
  }

  if (!nzchar(python)) {
    stop(
      paste(
        "Could not find a memscore backend.",
        "Set `options(memscore.cli = ...)`, put `memscore` on PATH, or",
        "configure a Python interpreter with `memscore_configure(python = ...)`."
      ),
      call. = FALSE
    )
  }

  list(
    command = python,
    args = c("-m", module),
    workdir = workdir %||% NULL,
    source = "python-module"
  )
}
