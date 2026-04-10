#!/usr/bin/env Rscript

copy_path <- function(source, destination) {
  if (!file.exists(source)) {
    return(invisible(FALSE))
  }
  ok <- file.copy(source, destination, recursive = TRUE, copy.date = TRUE)
  if (!ok) {
    stop(sprintf("Failed to copy %s", source), call. = FALSE)
  }
  invisible(TRUE)
}

install_memscore_r <- function(repo = getwd(), lib = NULL) {
  repo <- normalizePath(repo, mustWork = TRUE)
  required <- c("DESCRIPTION", "NAMESPACE", "R")
  missing <- required[!file.exists(file.path(repo, required))]
  if (length(missing) > 0L) {
    stop(
      sprintf("Not a memscore R package root; missing: %s", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }

  staging_root <- tempfile("memscore-rpkg-")
  dir.create(staging_root, recursive = TRUE)
  on.exit(unlink(staging_root, recursive = TRUE, force = TRUE), add = TRUE)
  package_root <- file.path(staging_root, "memscore")
  dir.create(package_root)

  for (path in c("DESCRIPTION", "LICENSE", "NAMESPACE", "README.md", ".Rbuildignore", "_pkgdown.yml")) {
    copy_path(file.path(repo, path), package_root)
  }
  for (path in c("R", "inst", "man", "tests", "vignettes")) {
    copy_path(file.path(repo, path), package_root)
  }

  args <- c("CMD", "INSTALL", "--no-multiarch")
  if (!is.null(lib)) {
    lib <- normalizePath(lib, mustWork = TRUE)
    args <- c(args, "-l", lib)
  }
  args <- c(args, package_root)
  status <- system2(file.path(R.home("bin"), "R"), args)
  if (!identical(status, 0L)) {
    stop("R package installation failed.", call. = FALSE)
  }
  invisible(package_root)
}

if (sys.nframe() == 0L) {
  install_memscore_r()
}
