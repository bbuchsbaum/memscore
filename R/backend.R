`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) {
    return(y)
  }
  if (is.character(x) && length(x) == 1L && !nzchar(x)) {
    return(y)
  }
  x
}

.normalize_backend <- function(backend = getOption("memscore.backend", "auto")) {
  backend <- tolower(backend %||% "auto")
  if (!backend %in% c("auto", "cli", "reticulate")) {
    stop(
      sprintf("Unsupported memscore backend %s. Use 'auto', 'cli', or 'reticulate'.", shQuote(backend)),
      call. = FALSE
    )
  }
  backend
}

.resolve_memscore_backend <- function(
  backend = getOption("memscore.backend", "auto"),
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  module = getOption("memscore.module", "memscore"),
  workdir = getOption("memscore.workdir")
) {
  backend <- .normalize_backend(backend)

  if (identical(backend, "reticulate")) {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
      stop(
        paste(
          "reticulate backend requested but the 'reticulate' package is not installed.",
          "Install it or choose backend = 'cli'."
        ),
        call. = FALSE
      )
    }

    python <- python %||% Sys.getenv("MEMSCORE_PYTHON", unset = "")
    if (is.character(python) && length(python) == 1L && nzchar(python)) {
      python <- .normalize_optional_path(python)
    } else {
      python <- NULL
    }

    return(list(
      kind = "reticulate",
      source = "reticulate",
      python = python,
      module = module,
      workdir = workdir %||% NULL
    ))
  }

  info <- memscore_cli_info(
    cli = cli,
    python = python,
    module = module,
    workdir = workdir
  )
  info$kind <- "cli"
  info
}

.write_json_file <- function(value, output_path) {
  jsonlite::write_json(
    value,
    path = .normalize_optional_path(output_path),
    pretty = TRUE,
    auto_unbox = TRUE,
    null = "null"
  )
}

.import_memscore_module <- function(python = NULL, module = "memscore", workdir = NULL) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("reticulate is required for the reticulate backend.", call. = FALSE)
  }

  if (is.character(python) && length(python) == 1L && nzchar(python)) {
    reticulate::use_python(.normalize_optional_path(python), required = TRUE)
  }

  if (!is.null(workdir) && nzchar(workdir)) {
    return(
      reticulate::import_from_path(
        module,
        path = .normalize_optional_path(workdir),
        delay_load = FALSE
      )
    )
  }

  reticulate::import(module, delay_load = FALSE)
}

.predict_via_reticulate <- function(
  paths,
  recursive = FALSE,
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output = NULL,
  python = NULL,
  module = "memscore",
  workdir = NULL
) {
  api <- .import_memscore_module(python = python, module = module, workdir = workdir)
  output_path <- output %||% tempfile("memscore-predict-", fileext = ".csv")
  if (is.null(output)) {
    on.exit(unlink(output_path), add = TRUE)
  }

  predictions <- api$predict_paths(
    as.list(vapply(paths, .normalize_optional_path, character(1))),
    recursive = recursive,
    checkpoint_path = resmem_checkpoint %||% NULL,
    batch_size = as.integer(batch_size),
    device = device,
    cache_dir = cache_dir %||% NULL
  )
  api$write_prediction_csv(predictions, .normalize_optional_path(output_path))
  utils::read.csv(output_path, stringsAsFactors = FALSE)
}

.benchmark_manifest_via_reticulate <- function(
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
  python = NULL,
  module = "memscore",
  workdir = NULL
) {
  api <- .import_memscore_module(python = python, module = module, workdir = workdir)
  output_json <- output_json %||% tempfile("memscore-benchmark-", fileext = ".json")
  output_predictions <- output_predictions %||% tempfile("memscore-predictions-", fileext = ".csv")

  run <- api$benchmark_manifest(
    .normalize_optional_path(manifest),
    root = root %||% NULL,
    clip_models = as.list(clip_models),
    clip_pretrained = clip_pretrained,
    regressors = as.list(regressors),
    batch_size = as.integer(batch_size),
    device = device,
    cache_dir = cache_dir %||% NULL,
    checkpoint_path = resmem_checkpoint %||% NULL,
    random_state = as.integer(random_state),
    limit_train = limit_train %||% NULL,
    limit_val = limit_val %||% NULL,
    limit_test = limit_test %||% NULL,
    tta_mode = clip_tta,
    tta_resize = clip_tta_resize %||% NULL
  )

  summary <- reticulate::py_to_r(run$summary)
  .write_json_file(summary, output_json)
  api$write_benchmark_predictions(run, .normalize_optional_path(output_predictions))

  list(
    summary = summary,
    output_json = .normalize_optional_path(output_json),
    output_predictions = .normalize_optional_path(output_predictions),
    stdout = character()
  )
}

.benchmark_lamem_via_reticulate <- function(
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
  python = NULL,
  module = "memscore",
  workdir = NULL
) {
  api <- .import_memscore_module(python = python, module = module, workdir = workdir)
  output_json <- output_json %||% tempfile("memscore-lamem-", fileext = ".json")
  output_predictions <- output_predictions %||% tempfile("memscore-lamem-predictions-", fileext = ".csv")

  run <- api$benchmark_lamem(
    .normalize_optional_path(lamem_root),
    splits_dir = .normalize_optional_path(splits_dir),
    fold = as.integer(fold),
    train_file = train_file %||% NULL,
    val_file = val_file %||% NULL,
    test_file = test_file %||% NULL,
    clip_models = as.list(clip_models),
    clip_pretrained = clip_pretrained,
    regressors = as.list(regressors),
    batch_size = as.integer(batch_size),
    device = device,
    cache_dir = cache_dir %||% NULL,
    checkpoint_path = resmem_checkpoint %||% NULL,
    random_state = as.integer(random_state),
    limit_train = limit_train %||% NULL,
    limit_val = limit_val %||% NULL,
    limit_test = limit_test %||% NULL,
    tta_mode = clip_tta,
    tta_resize = clip_tta_resize %||% NULL
  )

  summary <- reticulate::py_to_r(run$summary)
  .write_json_file(summary, output_json)
  api$write_benchmark_predictions(run, .normalize_optional_path(output_predictions))

  list(
    summary = summary,
    output_json = .normalize_optional_path(output_json),
    output_predictions = .normalize_optional_path(output_predictions),
    stdout = character()
  )
}

.benchmark_standard_via_reticulate <- function(
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
  python = NULL,
  module = "memscore",
  workdir = NULL
) {
  api <- .import_memscore_module(python = python, module = module, workdir = workdir)
  output_json <- output_json %||% tempfile("memscore-standard-", fileext = ".json")

  result <- api$benchmark_standard_manifest(
    .normalize_optional_path(manifest),
    root = root %||% NULL,
    dataset_name = dataset_name %||% NULL,
    clip_models = as.list(clip_models),
    clip_pretrained = clip_pretrained,
    batch_size = as.integer(batch_size),
    device = device,
    cache_dir = cache_dir %||% NULL,
    checkpoint_path = resmem_checkpoint %||% NULL,
    tta_mode = clip_tta,
    tta_resize = clip_tta_resize %||% NULL,
    ensemble_weight_step = ensemble_weight_step
  )

  summary <- reticulate::py_to_r(result)
  .write_json_file(summary, output_json)

  list(
    summary = summary,
    output_json = .normalize_optional_path(output_json),
    stdout = character()
  )
}

.study_figrim_via_reticulate <- function(
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
  python = NULL,
  module = "memscore",
  workdir = NULL
) {
  api <- .import_memscore_module(python = python, module = module, workdir = workdir)
  output_json <- output_json %||% tempfile("memscore-figrim-", fileext = ".json")
  output_csv <- output_csv %||% tempfile("memscore-figrim-", fileext = ".csv")

  study <- api$study_figrim(
    .normalize_optional_path(figrim_root),
    protocols = as.list(protocols),
    score_types = as.list(score_types),
    clip_models = as.list(clip_models),
    clip_pretrained = clip_pretrained,
    clip_tta_mode = clip_tta,
    clip_tta_resize = clip_tta_resize %||% NULL,
    pca_dims = as.list(pca_dims),
    pls_dims = as.list(pls_dims),
    num_splits = as.integer(num_splits),
    batch_size = as.integer(batch_size),
    device = device,
    cache_dir = cache_dir %||% NULL,
    random_state = as.integer(random_state),
    train_size = train_size,
    val_size = val_size,
    test_size = test_size,
    ensemble_weight_step = ensemble_weight_step,
    checkpoint_path = resmem_checkpoint %||% NULL,
    output_json = .normalize_optional_path(output_json),
    output_csv = .normalize_optional_path(output_csv)
  )

  list(
    study = reticulate::py_to_r(study),
    output_json = .normalize_optional_path(output_json),
    output_csv = .normalize_optional_path(output_csv),
    stdout = character()
  )
}

.study_memcat_via_reticulate <- function(
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
  python = NULL,
  module = "memscore",
  workdir = NULL
) {
  api <- .import_memscore_module(python = python, module = module, workdir = workdir)
  output_json <- output_json %||% tempfile("memscore-memcat-", fileext = ".json")
  output_csv <- output_csv %||% tempfile("memscore-memcat-", fileext = ".csv")

  study <- api$study_memcat(
    .normalize_optional_path(memcat_root),
    csv_path = if (!is.null(csv_path)) .normalize_optional_path(csv_path) else NULL,
    score_column = score_column,
    clip_models = as.list(clip_models),
    clip_pretrained = clip_pretrained,
    clip_tta_mode = clip_tta,
    clip_tta_resize = clip_tta_resize %||% NULL,
    pca_dims = as.list(pca_dims),
    pls_dims = as.list(pls_dims),
    num_splits = as.integer(num_splits),
    batch_size = as.integer(batch_size),
    device = device,
    cache_dir = cache_dir %||% NULL,
    random_state = as.integer(random_state),
    train_size = train_size,
    val_size = val_size,
    test_size = test_size,
    ensemble_weight_step = ensemble_weight_step,
    checkpoint_path = resmem_checkpoint %||% NULL,
    output_json = .normalize_optional_path(output_json),
    output_csv = .normalize_optional_path(output_csv)
  )

  list(
    study = reticulate::py_to_r(study),
    output_json = .normalize_optional_path(output_json),
    output_csv = .normalize_optional_path(output_csv),
    stdout = character()
  )
}
