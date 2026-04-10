#' Benchmark a custom manifest
#'
#' Runs `memscore benchmark-manifest` and returns the parsed summary plus output
#' file locations.
#'
#' @inheritParams memscore_predict
#' @param manifest CSV containing `path`, `score`, `split`, and optional `id`.
#' @param root Optional base directory for relative image paths in the manifest.
#' @param clip_models Character vector of frozen CLIP backbones to evaluate.
#' @param clip_pretrained Pretrained tag passed through to the backend.
#' @param regressors Character vector of lightweight regressors to compare.
#' @param output_json Optional summary JSON path. A temporary file is created if
#'   omitted.
#' @param output_predictions Optional predictions CSV path. A temporary file is
#'   created if omitted.
#' @param random_state Random seed for challenger selection.
#' @param limit_train Optional cap on the number of training records.
#' @param limit_val Optional cap on the number of validation records.
#' @param limit_test Optional cap on the number of test records.
#' @param clip_tta CLIP test-time augmentation: `none`, `fivecrop`, or
#'   `tencrop`.
#' @param clip_tta_resize Optional resize target for CLIP test-time
#'   augmentation.
#'
#' @return A list with `summary`, `output_json`, `output_predictions`, and
#'   `stdout`.
#'
#' @examples
#' \dontrun{
#' memscore_benchmark_manifest(
#'   manifest = "manifest.csv",
#'   root = "images",
#'   clip_models = c("ViT-B-32", "RN50"),
#'   regressors = "ridge"
#' )
#' }
#' @export
memscore_benchmark_manifest <- function(
  manifest,
  root = NULL,
  clip_models = c("ViT-B-32"),
  clip_pretrained = "openai",
  regressors = c("ridge", "mlp"),
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output_json = NULL,
  output_predictions = NULL,
  random_state = 0L,
  limit_train = NULL,
  limit_val = NULL,
  limit_test = NULL,
  clip_tta = "none",
  clip_tta_resize = NULL,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
) {
  backend_info <- .resolve_memscore_backend(
    backend = backend,
    cli = cli,
    python = python,
    module = getOption("memscore.module", "memscore"),
    workdir = workdir
  )

  if (identical(backend_info$kind, "reticulate")) {
    return(
      .benchmark_manifest_via_reticulate(
        manifest = manifest,
        root = root,
        clip_models = clip_models,
        clip_pretrained = clip_pretrained,
        regressors = regressors,
        batch_size = batch_size,
        device = device,
        cache_dir = cache_dir,
        resmem_checkpoint = resmem_checkpoint,
        output_json = output_json,
        output_predictions = output_predictions,
        random_state = random_state,
        limit_train = limit_train,
        limit_val = limit_val,
        limit_test = limit_test,
        clip_tta = clip_tta,
        clip_tta_resize = clip_tta_resize,
        python = backend_info$python,
        module = backend_info$module,
        workdir = backend_info$workdir
      )
    )
  }

  output_json <- output_json %||% tempfile("memscore-benchmark-", fileext = ".json")
  output_predictions <- output_predictions %||% tempfile("memscore-predictions-", fileext = ".csv")

  args <- c("benchmark-manifest", "--manifest", .normalize_optional_path(manifest))
  args <- .append_flag(args, "--root", root)
  args <- c(args, "--clip-models", clip_models)
  args <- .append_flag(args, "--clip-pretrained", clip_pretrained)
  args <- c(args, "--regressors", regressors)
  args <- .append_flag(args, "--batch-size", as.integer(batch_size))
  args <- .append_flag(args, "--device", device)
  args <- .append_flag(args, "--cache-dir", cache_dir)
  args <- .append_flag(args, "--resmem-checkpoint", resmem_checkpoint)
  args <- .append_flag(args, "--output-json", .normalize_optional_path(output_json))
  args <- .append_flag(args, "--output-predictions", .normalize_optional_path(output_predictions))
  args <- .append_flag(args, "--random-state", as.integer(random_state))
  args <- .append_flag(args, "--limit-train", limit_train)
  args <- .append_flag(args, "--limit-val", limit_val)
  args <- .append_flag(args, "--limit-test", limit_test)
  args <- .append_flag(args, "--clip-tta", clip_tta)
  args <- .append_flag(args, "--clip-tta-resize", clip_tta_resize)

  stdout <- .run_memscore_cli(
    backend_info,
    args = args
  )

  list(
    summary = jsonlite::read_json(output_json, simplifyVector = TRUE),
    output_json = .normalize_optional_path(output_json),
    output_predictions = .normalize_optional_path(output_predictions),
    stdout = stdout
  )
}

#' Benchmark the standard memscore model on a locked manifest
#'
#' Runs the standard out-of-sample recipe: frozen `ViT-B-32`, `RN50`, and
#' `ViT-B-16` CLIP embeddings; five-crop TTA at resize 256; one ridge head per
#' backbone; and a plain mean ensemble of the three ridge predictions.
#'
#' @inheritParams memscore_benchmark_manifest
#' @param dataset_name Optional label used for caches and output metadata.
#' @param ensemble_weight_step Step size for the optional validation-weighted
#'   ensemble report.
#'
#' @return A list with `summary`, `output_json`, and `stdout`.
#'
#' @examples
#' \dontrun{
#' memscore_benchmark_standard(
#'   manifest = "manifest.csv",
#'   root = "images"
#' )
#' }
#' @export
memscore_benchmark_standard <- function(
  manifest,
  root = NULL,
  dataset_name = NULL,
  clip_models = c("ViT-B-32", "RN50", "ViT-B-16"),
  clip_pretrained = "openai",
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output_json = NULL,
  clip_tta = "fivecrop",
  clip_tta_resize = 256L,
  ensemble_weight_step = 0.05,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
) {
  backend_info <- .resolve_memscore_backend(
    backend = backend,
    cli = cli,
    python = python,
    module = getOption("memscore.module", "memscore"),
    workdir = workdir
  )

  if (identical(backend_info$kind, "reticulate")) {
    return(
      .benchmark_standard_via_reticulate(
        manifest = manifest,
        root = root,
        dataset_name = dataset_name,
        clip_models = clip_models,
        clip_pretrained = clip_pretrained,
        batch_size = batch_size,
        device = device,
        cache_dir = cache_dir,
        resmem_checkpoint = resmem_checkpoint,
        output_json = output_json,
        clip_tta = clip_tta,
        clip_tta_resize = clip_tta_resize,
        ensemble_weight_step = ensemble_weight_step,
        python = backend_info$python,
        module = backend_info$module,
        workdir = backend_info$workdir
      )
    )
  }

  output_json <- output_json %||% tempfile("memscore-standard-", fileext = ".json")

  args <- c("benchmark-standard", "--manifest", .normalize_optional_path(manifest))
  args <- .append_flag(args, "--root", root)
  args <- .append_flag(args, "--dataset-name", dataset_name)
  args <- c(args, "--clip-models", clip_models)
  args <- .append_flag(args, "--clip-pretrained", clip_pretrained)
  args <- .append_flag(args, "--batch-size", as.integer(batch_size))
  args <- .append_flag(args, "--device", device)
  args <- .append_flag(args, "--cache-dir", cache_dir)
  args <- .append_flag(args, "--resmem-checkpoint", resmem_checkpoint)
  args <- .append_flag(args, "--output-json", .normalize_optional_path(output_json))
  args <- .append_flag(args, "--clip-tta", clip_tta)
  args <- .append_flag(args, "--clip-tta-resize", clip_tta_resize)
  args <- .append_flag(args, "--ensemble-weight-step", ensemble_weight_step)

  stdout <- .run_memscore_cli(
    backend_info,
    args = args
  )

  list(
    summary = jsonlite::read_json(output_json, simplifyVector = TRUE),
    output_json = .normalize_optional_path(output_json),
    stdout = stdout
  )
}

#' Benchmark LaMem split files
#'
#' Runs `memscore benchmark-lamem`.
#'
#' @inheritParams memscore_benchmark_manifest
#' @param lamem_root Path to the LaMem dataset root.
#' @param splits_dir Directory containing official LaMem split files.
#' @param fold Fold number when using `train_<fold>.txt` style split files.
#' @param train_file Optional explicit training split file.
#' @param val_file Optional explicit validation split file.
#' @param test_file Optional explicit test split file.
#'
#' @return A list with `summary`, `output_json`, `output_predictions`, and
#'   `stdout`.
#'
#' @examples
#' \dontrun{
#' memscore_benchmark_lamem(
#'   lamem_root = "/path/to/LaMem",
#'   splits_dir = "/path/to/LaMem/splits",
#'   clip_models = c("ViT-B-32", "RN50"),
#'   regressors = "ridge"
#' )
#' }
#' @export
memscore_benchmark_lamem <- function(
  lamem_root,
  splits_dir,
  fold = 1L,
  train_file = NULL,
  val_file = NULL,
  test_file = NULL,
  clip_models = c("ViT-B-32"),
  clip_pretrained = "openai",
  regressors = c("ridge", "mlp"),
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output_json = NULL,
  output_predictions = NULL,
  random_state = 0L,
  limit_train = NULL,
  limit_val = NULL,
  limit_test = NULL,
  clip_tta = "none",
  clip_tta_resize = NULL,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
) {
  backend_info <- .resolve_memscore_backend(
    backend = backend,
    cli = cli,
    python = python,
    module = getOption("memscore.module", "memscore"),
    workdir = workdir
  )

  if (identical(backend_info$kind, "reticulate")) {
    return(
      .benchmark_lamem_via_reticulate(
        lamem_root = lamem_root,
        splits_dir = splits_dir,
        fold = fold,
        train_file = train_file,
        val_file = val_file,
        test_file = test_file,
        clip_models = clip_models,
        clip_pretrained = clip_pretrained,
        regressors = regressors,
        batch_size = batch_size,
        device = device,
        cache_dir = cache_dir,
        resmem_checkpoint = resmem_checkpoint,
        output_json = output_json,
        output_predictions = output_predictions,
        random_state = random_state,
        limit_train = limit_train,
        limit_val = limit_val,
        limit_test = limit_test,
        clip_tta = clip_tta,
        clip_tta_resize = clip_tta_resize,
        python = backend_info$python,
        module = backend_info$module,
        workdir = backend_info$workdir
      )
    )
  }

  output_json <- output_json %||% tempfile("memscore-lamem-", fileext = ".json")
  output_predictions <- output_predictions %||% tempfile("memscore-lamem-predictions-", fileext = ".csv")

  args <- c(
    "benchmark-lamem",
    "--lamem-root", .normalize_optional_path(lamem_root),
    "--splits-dir", .normalize_optional_path(splits_dir),
    "--fold", as.integer(fold)
  )
  args <- .append_flag(args, "--train-file", train_file)
  args <- .append_flag(args, "--val-file", val_file)
  args <- .append_flag(args, "--test-file", test_file)
  args <- c(args, "--clip-models", clip_models)
  args <- .append_flag(args, "--clip-pretrained", clip_pretrained)
  args <- c(args, "--regressors", regressors)
  args <- .append_flag(args, "--batch-size", as.integer(batch_size))
  args <- .append_flag(args, "--device", device)
  args <- .append_flag(args, "--cache-dir", cache_dir)
  args <- .append_flag(args, "--resmem-checkpoint", resmem_checkpoint)
  args <- .append_flag(args, "--output-json", .normalize_optional_path(output_json))
  args <- .append_flag(args, "--output-predictions", .normalize_optional_path(output_predictions))
  args <- .append_flag(args, "--random-state", as.integer(random_state))
  args <- .append_flag(args, "--limit-train", limit_train)
  args <- .append_flag(args, "--limit-val", limit_val)
  args <- .append_flag(args, "--limit-test", limit_test)
  args <- .append_flag(args, "--clip-tta", clip_tta)
  args <- .append_flag(args, "--clip-tta-resize", clip_tta_resize)

  stdout <- .run_memscore_cli(
    backend_info,
    args = args
  )

  list(
    summary = jsonlite::read_json(output_json, simplifyVector = TRUE),
    output_json = .normalize_optional_path(output_json),
    output_predictions = .normalize_optional_path(output_predictions),
    stdout = stdout
  )
}

#' Run the repeated FIGRIM study
#'
#' Runs `memscore study-figrim`.
#'
#' @inheritParams memscore_benchmark_manifest
#' @param figrim_root Path to the FIGRIM download directory.
#' @param protocols FIGRIM protocols to evaluate.
#' @param score_types FIGRIM target definitions to evaluate.
#' @param pca_dims PCA dimensions to evaluate. Use `"none"` for full
#'   embeddings.
#' @param pls_dims Optional PLS component counts.
#' @param num_splits Number of repeated stratified splits.
#' @param train_size Fraction assigned to the training split.
#' @param val_size Fraction assigned to the validation split.
#' @param test_size Fraction assigned to the test split.
#' @param output_csv Optional summary CSV path. A temporary file is created if
#'   omitted.
#' @param ensemble_weight_step Weight step used for validation-weighted
#'   ensembling.
#'
#' @return A list with `study`, `output_json`, `output_csv`, and `stdout`.
#'
#' @examples
#' \dontrun{
#' memscore_study_figrim(
#'   figrim_root = "/path/to/figrim",
#'   clip_models = c("ViT-B-32", "RN50", "ViT-B-16"),
#'   clip_tta = "fivecrop"
#' )
#' }
#' @export
memscore_study_figrim <- function(
  figrim_root,
  protocols = c("across", "within"),
  score_types = c("corrected_hit_rate"),
  clip_models = c("ViT-B-32", "RN50"),
  clip_pretrained = "openai",
  clip_tta = "none",
  clip_tta_resize = NULL,
  pca_dims = c("none", "128", "64", "32"),
  pls_dims = character(),
  num_splits = 10L,
  train_size = 0.7,
  val_size = 0.1,
  test_size = 0.2,
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output_json = NULL,
  output_csv = NULL,
  random_state = 0L,
  ensemble_weight_step = 0.05,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
) {
  backend_info <- .resolve_memscore_backend(
    backend = backend,
    cli = cli,
    python = python,
    module = getOption("memscore.module", "memscore"),
    workdir = workdir
  )

  if (identical(backend_info$kind, "reticulate")) {
    return(
      .study_figrim_via_reticulate(
        figrim_root = figrim_root,
        protocols = protocols,
        score_types = score_types,
        clip_models = clip_models,
        clip_pretrained = clip_pretrained,
        clip_tta = clip_tta,
        clip_tta_resize = clip_tta_resize,
        pca_dims = pca_dims,
        pls_dims = pls_dims,
        num_splits = num_splits,
        train_size = train_size,
        val_size = val_size,
        test_size = test_size,
        batch_size = batch_size,
        device = device,
        cache_dir = cache_dir,
        resmem_checkpoint = resmem_checkpoint,
        output_json = output_json,
        output_csv = output_csv,
        random_state = random_state,
        ensemble_weight_step = ensemble_weight_step,
        python = backend_info$python,
        module = backend_info$module,
        workdir = backend_info$workdir
      )
    )
  }

  output_json <- output_json %||% tempfile("memscore-figrim-", fileext = ".json")
  output_csv <- output_csv %||% tempfile("memscore-figrim-", fileext = ".csv")

  args <- c(
    "study-figrim",
    "--figrim-root", .normalize_optional_path(figrim_root),
    "--protocols", protocols,
    "--score-types", score_types,
    "--clip-models", clip_models,
    "--clip-pretrained", clip_pretrained,
    "--clip-tta", clip_tta,
    "--pca-dims", pca_dims,
    "--num-splits", as.integer(num_splits),
    "--train-size", train_size,
    "--val-size", val_size,
    "--test-size", test_size,
    "--batch-size", as.integer(batch_size),
    "--device", device,
    "--output-json", .normalize_optional_path(output_json),
    "--output-csv", .normalize_optional_path(output_csv),
    "--random-state", as.integer(random_state),
    "--ensemble-weight-step", ensemble_weight_step
  )
  args <- .append_flag(args, "--clip-tta-resize", clip_tta_resize)
  args <- .append_flag(args, "--cache-dir", cache_dir)
  args <- .append_flag(args, "--resmem-checkpoint", resmem_checkpoint)
  if (length(pls_dims)) {
    args <- c(args, "--pls-dims", pls_dims)
  }

  stdout <- .run_memscore_cli(
    backend_info,
    args = args
  )

  list(
    study = jsonlite::read_json(output_json, simplifyVector = TRUE),
    output_json = .normalize_optional_path(output_json),
    output_csv = .normalize_optional_path(output_csv),
    stdout = stdout
  )
}

#' Run the repeated MemCat study
#'
#' Runs `memscore study-memcat`.
#'
#' @inheritParams memscore_benchmark_manifest
#' @param memcat_root Path to the MemCat dataset root directory.
#' @param csv_path Optional explicit path to the MemCat CSV metadata file.
#' @param score_column Column name for memorability scores in the CSV.
#' @param pca_dims PCA dimensions to evaluate. Use `"none"` for full
#'   embeddings.
#' @param pls_dims Optional PLS component counts.
#' @param num_splits Number of repeated stratified splits.
#' @param train_size Fraction assigned to the training split.
#' @param val_size Fraction assigned to the validation split.
#' @param test_size Fraction assigned to the test split.
#' @param output_csv Optional summary CSV path. A temporary file is created if
#'   omitted.
#' @param ensemble_weight_step Weight step used for validation-weighted
#'   ensembling.
#'
#' @return A list with `study`, `output_json`, `output_csv`, and `stdout`.
#'
#' @examples
#' \dontrun{
#' memscore_study_memcat(
#'   memcat_root = "/path/to/memcat",
#'   clip_models = c("ViT-B-32", "RN50", "ViT-B-16"),
#'   clip_tta = "fivecrop"
#' )
#' }
#' @export
memscore_study_memcat <- function(
  memcat_root,
  csv_path = NULL,
  score_column = "memorability_w_fa_correction",
  clip_models = c("ViT-B-32", "RN50"),
  clip_pretrained = "openai",
  clip_tta = "none",
  clip_tta_resize = NULL,
  pca_dims = c("none", "128", "64", "32"),
  pls_dims = character(),
  num_splits = 10L,
  train_size = 0.7,
  val_size = 0.1,
  test_size = 0.2,
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output_json = NULL,
  output_csv = NULL,
  random_state = 0L,
  ensemble_weight_step = 0.05,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
) {
  backend_info <- .resolve_memscore_backend(
    backend = backend,
    cli = cli,
    python = python,
    module = getOption("memscore.module", "memscore"),
    workdir = workdir
  )

  if (identical(backend_info$kind, "reticulate")) {
    return(
      .study_memcat_via_reticulate(
        memcat_root = memcat_root,
        csv_path = csv_path,
        score_column = score_column,
        clip_models = clip_models,
        clip_pretrained = clip_pretrained,
        clip_tta = clip_tta,
        clip_tta_resize = clip_tta_resize,
        pca_dims = pca_dims,
        pls_dims = pls_dims,
        num_splits = num_splits,
        train_size = train_size,
        val_size = val_size,
        test_size = test_size,
        batch_size = batch_size,
        device = device,
        cache_dir = cache_dir,
        resmem_checkpoint = resmem_checkpoint,
        output_json = output_json,
        output_csv = output_csv,
        random_state = random_state,
        ensemble_weight_step = ensemble_weight_step,
        python = backend_info$python,
        module = backend_info$module,
        workdir = backend_info$workdir
      )
    )
  }

  output_json <- output_json %||% tempfile("memscore-memcat-", fileext = ".json")
  output_csv <- output_csv %||% tempfile("memscore-memcat-", fileext = ".csv")

  args <- c(
    "study-memcat",
    "--memcat-root", .normalize_optional_path(memcat_root),
    "--score-column", score_column,
    "--clip-models", clip_models,
    "--clip-pretrained", clip_pretrained,
    "--clip-tta", clip_tta,
    "--pca-dims", pca_dims,
    "--num-splits", as.integer(num_splits),
    "--train-size", train_size,
    "--val-size", val_size,
    "--test-size", test_size,
    "--batch-size", as.integer(batch_size),
    "--device", device,
    "--output-json", .normalize_optional_path(output_json),
    "--output-csv", .normalize_optional_path(output_csv),
    "--random-state", as.integer(random_state),
    "--ensemble-weight-step", ensemble_weight_step
  )
  args <- .append_flag(args, "--csv-path", csv_path)
  args <- .append_flag(args, "--clip-tta-resize", clip_tta_resize)
  args <- .append_flag(args, "--cache-dir", cache_dir)
  args <- .append_flag(args, "--resmem-checkpoint", resmem_checkpoint)
  if (length(pls_dims)) {
    args <- c(args, "--pls-dims", pls_dims)
  }

  stdout <- .run_memscore_cli(
    backend_info,
    args = args
  )

  list(
    study = jsonlite::read_json(output_json, simplifyVector = TRUE),
    output_json = .normalize_optional_path(output_json),
    output_csv = .normalize_optional_path(output_csv),
    stdout = stdout
  )
}
