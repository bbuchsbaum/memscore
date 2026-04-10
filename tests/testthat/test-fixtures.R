test_that("installed image fixtures are real files", {
  manifest <- system.file("extdata/tiny_manifest.csv", package = "memscore", mustWork = TRUE)
  image_root <- dirname(manifest)
  rows <- utils::read.csv(manifest, stringsAsFactors = FALSE)

  expect_equal(names(rows), c("path", "score", "split", "id"))
  expect_equal(sort(rows$split), c("test", "train", "val"))
  expect_true(all(file.exists(file.path(image_root, rows$path))))
  expect_true(all(file.info(file.path(image_root, rows$path))$size > 0))
})

