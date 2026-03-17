test_that("memscore_benchmark_manifest reads generated output files", {
  manifest <- tempfile(fileext = ".csv")
  writeLines("path,score,split\nimg1.jpg,0.5,test", manifest)

  testthat::local_mocked_bindings(
    .system2 = function(command, args, stdout = TRUE, stderr = TRUE) {
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
    clip_models = c("ViT-B-32", "RN50"),
    regressors = "ridge"
  )

  expect_equal(result$summary$dataset, "manifest")
  expect_equal(result$summary$delta_vs_resmem$spearman, 0.2)
  expect_true(file.exists(result$output_json))
  expect_true(file.exists(result$output_predictions))
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
