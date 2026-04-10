test_that("memscore_predict can dispatch through the reticulate backend", {
  testthat::local_mocked_bindings(
    .resolve_memscore_backend = function(...) {
      list(
        kind = "reticulate",
        source = "reticulate",
        python = "/usr/bin/python3",
        module = "memscore",
        workdir = "/tmp"
      )
    },
    .predict_via_reticulate = function(...) {
      data.frame(
        id = "img1",
        image_path = "/tmp/img1.jpg",
        memorability = 0.42,
        stringsAsFactors = FALSE
      )
    },
    .package = "memscore"
  )

  result <- memscore_predict(paths = "img1.jpg", backend = "reticulate")

  expect_s3_class(result, "data.frame")
  expect_equal(result$id, "img1")
  expect_equal(result$memorability, 0.42)
})

test_that("memscore_benchmark_manifest can dispatch through the reticulate backend", {
  manifest <- tempfile(fileext = ".csv")
  writeLines("path,score,split\nimg1.jpg,0.5,test", manifest)

  testthat::local_mocked_bindings(
    .resolve_memscore_backend = function(...) {
      list(
        kind = "reticulate",
        source = "reticulate",
        python = "/usr/bin/python3",
        module = "memscore",
        workdir = "/tmp"
      )
    },
    .benchmark_manifest_via_reticulate = function(...) {
      list(
        summary = list(dataset = "manifest", delta_vs_resmem = list(spearman = 0.2, mse = -0.01)),
        output_json = "/tmp/summary.json",
        output_predictions = "/tmp/predictions.csv",
        stdout = character()
      )
    },
    .package = "memscore"
  )

  result <- memscore_benchmark_manifest(manifest = manifest, backend = "reticulate")

  expect_equal(result$summary$dataset, "manifest")
  expect_equal(result$summary$delta_vs_resmem$spearman, 0.2)
  expect_identical(result$stdout, character())
})

test_that("memscore_benchmark_standard can dispatch through the reticulate backend", {
  manifest <- tempfile(fileext = ".csv")
  writeLines("path,score,split\nimg1.jpg,0.5,test", manifest)

  testthat::local_mocked_bindings(
    .resolve_memscore_backend = function(...) {
      list(
        kind = "reticulate",
        source = "reticulate",
        python = "/usr/bin/python3",
        module = "memscore",
        workdir = "/tmp"
      )
    },
    .benchmark_standard_via_reticulate = function(...) {
      list(
        summary = list(standard_model = list(summary = list(mean_spearman = 0.75))),
        output_json = "/tmp/standard.json",
        stdout = character()
      )
    },
    .package = "memscore"
  )

  result <- memscore_benchmark_standard(manifest = manifest, backend = "reticulate")

  expect_equal(result$summary$standard_model$summary$mean_spearman, 0.75)
  expect_identical(result$stdout, character())
})

test_that("memscore_study_figrim can dispatch through the reticulate backend", {
  testthat::local_mocked_bindings(
    .resolve_memscore_backend = function(...) {
      list(
        kind = "reticulate",
        source = "reticulate",
        python = "/usr/bin/python3",
        module = "memscore",
        workdir = "/tmp"
      )
    },
    .study_figrim_via_reticulate = function(...) {
      list(
        study = list(summary_rows = data.frame(protocol = "across_category", model_mean_spearman = 0.74)),
        output_json = "/tmp/study.json",
        output_csv = "/tmp/study.csv",
        stdout = character()
      )
    },
    .package = "memscore"
  )

  result <- memscore_study_figrim(figrim_root = tempdir(), backend = "reticulate")

  expect_equal(result$study$summary_rows$protocol[[1]], "across_category")
  expect_identical(result$stdout, character())
})
