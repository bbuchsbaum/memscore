test_that("memscore_predict parses JSON output", {
  calls <- list()

  testthat::local_mocked_bindings(
    .system2 = function(command, args, stdout = TRUE, stderr = TRUE) {
      calls <<- list(command = command, args = args, stdout = stdout, stderr = stderr)
      c(
        "Wrote predictions to /tmp/preds.csv",
        '[{"id":"img1","image_path":"/tmp/img1.jpg","memorability":0.42}]'
      )
    },
    .package = "memscore"
  )

  old <- options(memscore.cli = "/tmp/memscore")
  on.exit(options(old), add = TRUE)

  result <- memscore_predict(
    paths = c("img1.jpg"),
    recursive = TRUE,
    batch_size = 8L,
    device = "cpu",
    output = "preds.csv"
  )

  expect_s3_class(result, "data.frame")
  expect_equal(result$id, "img1")
  expect_equal(result$memorability, 0.42)
  expect_equal(calls$command, normalizePath("/tmp/memscore", winslash = "/", mustWork = FALSE))
  expect_true("predict" %in% calls$args)
  expect_true("--recursive" %in% calls$args)
  expect_true("--output" %in% calls$args)
})
