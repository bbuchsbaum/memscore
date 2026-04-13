test_that("memscore_predict parses JSON output", {
  calls <- list()
  image_path <- system.file("extdata/images/red_square.png", package = "memscore", mustWork = TRUE)

  testthat::local_mocked_bindings(
    .system2 = function(command, args, stdout = TRUE, stderr = TRUE) {
      calls <<- list(command = command, args = args, stdout = stdout, stderr = stderr)
      expect_true(file.exists(args[[2]]))
      c(
        "Wrote predictions to /tmp/preds.csv",
        sprintf('[{"id":"red_square.png","image_path":"%s","memorability":0.42}]', image_path)
      )
    },
    .package = "memscore"
  )

  old <- options(memscore.cli = "/tmp/memscore")
  on.exit(options(old), add = TRUE)

  result <- memscore_predict(
    paths = c(image_path),
    recursive = TRUE,
    batch_size = 8L,
    device = "cpu",
    output = "preds.csv"
  )

  expect_s3_class(result, "data.frame")
  expect_equal(result$id, "red_square.png")
  expect_equal(result$memorability, 0.42)
  expect_equal(calls$command, normalizePath("/tmp/memscore", winslash = "/", mustWork = FALSE))
  expect_true("predict" %in% calls$args)
  expect_true(normalizePath(image_path, winslash = "/", mustWork = TRUE) %in% calls$args)
  expect_true("--recursive" %in% calls$args)
  expect_true("--output" %in% calls$args)
})

test_that("memscore_predict_standard parses JSON output", {
  calls <- list()
  image_path <- system.file("extdata/images/red_square.png", package = "memscore", mustWork = TRUE)
  model_path <- tempfile(fileext = ".pkl")
  writeBin(charToRaw("fixture"), model_path)

  testthat::local_mocked_bindings(
    .system2 = function(command, args, stdout = TRUE, stderr = TRUE) {
      calls <<- list(command = command, args = args, stdout = stdout, stderr = stderr)
      c(
        "Wrote predictions to /tmp/standard-preds.csv",
        sprintf('[{"id":"red_square.png","image_path":"%s","memorability":0.77}]', image_path)
      )
    },
    .package = "memscore"
  )

  old <- options(memscore.cli = "/tmp/memscore")
  on.exit(options(old), add = TRUE)

  result <- memscore_predict_standard(
    paths = c(image_path),
    model = model_path,
    recursive = TRUE,
    batch_size = 4L,
    device = "cpu",
    output = "standard-preds.csv",
    clip_scores = FALSE
  )

  expect_s3_class(result, "data.frame")
  expect_equal(result$id, "red_square.png")
  expect_equal(result$memorability, 0.77)
  expect_equal(calls$command, normalizePath("/tmp/memscore", winslash = "/", mustWork = FALSE))
  expect_equal(calls$args[[1]], "predict-standard")
  expect_true(normalizePath(image_path, winslash = "/", mustWork = TRUE) %in% calls$args)
  expect_equal(unname(calls$args[match("--model", calls$args) + 1L]), normalizePath(model_path, winslash = "/", mustWork = TRUE))
  expect_true("--recursive" %in% calls$args)
  expect_true("--no-clip-scores" %in% calls$args)
  expect_true("--output" %in% calls$args)
})
