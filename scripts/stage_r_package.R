args <- commandArgs(trailingOnly = TRUE)

source_root <- normalizePath(".", winslash = "/", mustWork = TRUE)
dest_root <- if (length(args) >= 1L) {
  normalizePath(args[[1]], winslash = "/", mustWork = FALSE)
} else {
  tempfile("memscore-rpkg-")
}

dir.create(dest_root, recursive = TRUE, showWarnings = FALSE)

copy_entry <- function(path) {
  source <- file.path(source_root, path)
  target <- file.path(dest_root, path)
  if (!file.exists(source)) {
    stop(sprintf("Missing source path: %s", source), call. = FALSE)
  }
  dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  if (dir.exists(source)) {
    dir.create(target, recursive = TRUE, showWarnings = FALSE)
    files <- list.files(source, recursive = TRUE, all.files = TRUE, no.. = TRUE)
    for (entry in files) {
      from <- file.path(source, entry)
      to <- file.path(target, entry)
      dir.create(dirname(to), recursive = TRUE, showWarnings = FALSE)
      file.copy(from, to, overwrite = TRUE, recursive = FALSE, copy.date = TRUE)
    }
  } else {
    file.copy(source, target, overwrite = TRUE, recursive = FALSE, copy.date = TRUE)
  }
}

entries <- c(
  "DESCRIPTION",
  "LICENSE",
  "NAMESPACE",
  "README.md",
  ".Rbuildignore",
  "R",
  "man",
  "tests"
)

invisible(lapply(entries, copy_entry))
cat(dest_root, "\n")
