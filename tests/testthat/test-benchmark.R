test_that("memscore_benchmark_manifest reads generated output files", {
  manifest <- system.file("extdata/tiny_manifest.csv", package = "memscore", mustWork = TRUE)
  root <- dirname(manifest)

  testthat::local_mocked_bindings(
    .system2 = function(command, args, stdout = TRUE, stderr = TRUE) {
      manifest_arg <- args[match("--manifest", args) + 1L]
      root_arg <- args[match("--root", args) + 1L]
      expect_true(file.exists(manifest_arg))
      expect_true(dir.exists(root_arg))
      json_path <- args[match("--output-json", args) + 1L]
      pred_path <- args[match("--output-predictions", args) + 1L]
      writeLines(
        c(
          "{",
          '  "dataset": "manifest",',
          '  "delta_vs_resmem": {',
          '    "spearman": 0.2,',
          '    "mse": -0.01',
          "  }",
          "}"
        ),
        json_path
      )
      writeLines(
        "id,image_path,ground_truth,resmem_prediction,clip_prediction\nimg1,/tmp/img1.jpg,0.5,0.4,0.6",
        pred_path
      )
      c(
        sprintf("Wrote summary to %s", json_path),
        sprintf("Wrote test predictions to %s", pred_path)
      )
    },
    .package = "memscore"
  )

  old <- options(memscore.cli = "/tmp/memscore")
  on.exit(options(old), add = TRUE)

  result <- memscore_benchmark_manifest(
    manifest = manifest,
    root = root,
    clip_models = c("ViT-B-32", "RN50"),
    regressors = "ridge"
  )

  expect_equal(result$summary$dataset, "manifest")
  expect_equal(result$summary$delta_vs_resmem$spearman, 0.2)
  expect_true(file.exists(result$output_json))
  expect_true(file.exists(result$output_predictions))
})

test_that("memscore_benchmark_standard reads generated output file", {
  manifest <- system.file("extdata/tiny_manifest.csv", package = "memscore", mustWork = TRUE)
  root <- dirname(manifest)

  testthat::local_mocked_bindings(
    .system2 = function(command, args, stdout = TRUE, stderr = TRUE) {
      expect_equal(args[[1]], "benchmark-standard")
      expect_false("--random-state" %in% args)
      expect_true(all(c("ViT-B-32", "RN50", "ViT-B-16") %in% args))
      manifest_arg <- args[match("--manifest", args) + 1L]
      root_arg <- args[match("--root", args) + 1L]
      expect_true(file.exists(manifest_arg))
      expect_true(dir.exists(root_arg))
      json_path <- args[match("--output-json", args) + 1L]
      writeLines(
        c(
          "{",
          '  "metadata": {"standard_ensemble": "ensemble_mean"},',
          '  "standard_model": {',
          '    "summary": {"mean_spearman": 0.75},',
          '    "comparison_vs_resmem": {"mean_delta_spearman": 0.2}',
          "  }",
          "}"
        ),
        json_path
      )
      sprintf("Wrote standard benchmark summary to %s", json_path)
    },
    .package = "memscore"
  )

  old <- options(memscore.cli = "/tmp/memscore")
  on.exit(options(old), add = TRUE)

  result <- memscore_benchmark_standard(manifest = manifest, root = root)

  expect_equal(result$summary$standard_model$summary$mean_spearman, 0.75)
  expect_equal(result$summary$metadata$standard_ensemble, "ensemble_mean")
  expect_true(file.exists(result$output_json))
})

test_that("memscore_train_standard_scorer parses generated artifact metadata", {
  manifest <- system.file("extdata/tiny_manifest.csv", package = "memscore", mustWork = TRUE)
  root <- dirname(manifest)
  output_model <- tempfile(fileext = ".pkl")
  calls <- list()

  testthat::local_mocked_bindings(
    .system2 = function(command, args, stdout = TRUE, stderr = TRUE) {
      calls <<- list(command = command, args = args, stdout = stdout, stderr = stderr)
      expect_equal(args[[1]], "train-standard-scorer")
      expect_false("--include-test" %in% args)
      expect_true(all(c("ViT-B-32", "RN50", "ViT-B-16") %in% args))
      expect_equal(args[match("--manifest", args) + 1L], normalizePath(manifest, winslash = "/", mustWork = TRUE))
      expect_equal(args[match("--root", args) + 1L], normalizePath(root, winslash = "/", mustWork = TRUE))
      expect_equal(args[match("--output-model", args) + 1L], normalizePath(output_model, winslash = "/", mustWork = FALSE))
      c(
        "{",
        sprintf('  "artifact_path": "%s",', normalizePath(output_model, winslash = "/", mustWork = FALSE)),
        '  "model_type": "memscore-standard-ridge-mean",',
        '  "clip_models": ["ViT-B-32", "RN50", "ViT-B-16"]',
        "}",
        sprintf("Wrote standard scorer artifact to %s", output_model)
      )
    },
    .package = "memscore"
  )

  old <- options(memscore.cli = "/tmp/memscore")
  on.exit(options(old), add = TRUE)

  result <- memscore_train_standard_scorer(
    manifest = manifest,
    root = root,
    output_model = output_model
  )

  expect_equal(result$model_type, "memscore-standard-ridge-mean")
  expect_equal(result$clip_models, c("ViT-B-32", "RN50", "ViT-B-16"))
  expect_equal(calls$command, normalizePath("/tmp/memscore", winslash = "/", mustWork = FALSE))
})

test_that("memscore_study_memcat reads generated study output", {
  memcat_root <- tempdir()

  testthat::local_mocked_bindings(
    .system2 = function(command, args, stdout = TRUE, stderr = TRUE) {
      json_path <- args[match("--output-json", args) + 1L]
      csv_path <- args[match("--output-csv", args) + 1L]
      writeLines(
        c(
          "{",
          '  "summary_rows": [',
          '    {"protocol": "memcat", "model_mean_spearman": 0.68}',
          "  ]",
          "}"
        ),
        json_path
      )
      writeLines(
        "protocol,model_mean_spearman\nmemcat,0.68",
        csv_path
      )
      c(
        sprintf("Wrote study JSON to %s", json_path),
        sprintf("Wrote study CSV to %s", csv_path)
      )
    },
    .package = "memscore"
  )

  old <- options(memscore.cli = "/tmp/memscore")
  on.exit(options(old), add = TRUE)

  result <- memscore_study_memcat(
    memcat_root = memcat_root,
    clip_models = c("ViT-B-32", "RN50")
  )

  expect_equal(result$study$summary_rows$protocol[[1]], "memcat")
  expect_true(file.exists(result$output_json))
  expect_true(file.exists(result$output_csv))
})

test_that("memscore_study_figrim reads generated study output", {
  figrim_root <- tempdir()

  testthat::local_mocked_bindings(
    .system2 = function(command, args, stdout = TRUE, stderr = TRUE) {
      json_path <- args[match("--output-json", args) + 1L]
      csv_path <- args[match("--output-csv", args) + 1L]
      writeLines(
        c(
          "{",
          '  "summary_rows": [',
          '    {"protocol": "across_category", "model_mean_spearman": 0.74}',
          "  ]",
          "}"
        ),
        json_path
      )
      writeLines(
        "protocol,model_mean_spearman\nacross_category,0.74",
        csv_path
      )
      c(
        sprintf("Wrote study JSON to %s", json_path),
        sprintf("Wrote study CSV to %s", csv_path)
      )
    },
    .package = "memscore"
  )

  old <- options(memscore.cli = "/tmp/memscore")
  on.exit(options(old), add = TRUE)

  result <- memscore_study_figrim(
    figrim_root = figrim_root,
    clip_models = c("ViT-B-32", "RN50")
  )

  expect_equal(result$study$summary_rows$protocol[[1]], "across_category")
  expect_true(file.exists(result$output_json))
  expect_true(file.exists(result$output_csv))
})
