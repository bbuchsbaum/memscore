test_that("memscore_cli_info prefers explicit CLI configuration", {
  old <- options(memscore.cli = "/tmp/memscore-cli")
  on.exit(options(old), add = TRUE)

  info <- memscore_cli_info()

  expect_equal(info$command, normalizePath("/tmp/memscore-cli", winslash = "/", mustWork = FALSE))
  expect_equal(info$source, "explicit-cli")
  expect_length(info$args, 0L)
})

test_that("memscore_cli_info uses memscore from PATH when available", {
  tmp <- tempfile("memscore-path-")
  dir.create(tmp)
  cli <- file.path(tmp, "memscore")
  writeLines(c("#!/bin/sh", "exit 0"), cli)
  Sys.chmod(cli, mode = "755")

  old_opts <- options(memscore.cli = "", memscore.python = "/usr/bin/python3", memscore.workdir = "/tmp")
  old_path <- Sys.getenv("PATH", unset = "")
  on.exit(options(old_opts), add = TRUE)
  on.exit(Sys.setenv(PATH = old_path), add = TRUE)

  Sys.setenv(PATH = paste(tmp, old_path, sep = .Platform$path.sep))

  info <- memscore_cli_info()
  resolved <- normalizePath(unname(info$command), winslash = "/", mustWork = TRUE)

  expect_equal(resolved, normalizePath(cli, winslash = "/", mustWork = TRUE))
  expect_equal(info$args, character())
  expect_equal(info$workdir, "/tmp")
  expect_equal(info$source, "path-cli")
})

test_that("memscore_cli_info falls back to python module", {
  old <- options(memscore.cli = "", memscore.python = "/usr/bin/python3", memscore.workdir = "/tmp")
  old_path <- Sys.getenv("PATH", unset = "")
  on.exit(options(old), add = TRUE)
  on.exit(Sys.setenv(PATH = old_path), add = TRUE)

  Sys.setenv(PATH = "")

  info <- memscore_cli_info()

  expect_equal(info$command, "/usr/bin/python3")
  expect_equal(info$args, c("-m", "memscore"))
  expect_equal(info$workdir, "/tmp")
  expect_equal(info$source, "python-module")
})
