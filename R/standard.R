#' Score image memorability with the memscore standard model
#'
#' Scores images with a saved `memscore` standard scorer artifact: frozen CLIP
#' backbones, ridge heads, and a mean ensemble. Use [memscore_train_standard_scorer()]
#' to create the artifact from labeled manifests. ResMem remains available via
#' [memscore_predict()].
#'
#' @inheritParams memscore_predict
#' @param model Path to a saved standard scorer artifact. If omitted, the Python
#'   backend uses a packaged artifact when one is installed.
#' @param clip_scores Clamp predicted scores to the `[0, 1]` memorability range.
#'
#' @return A data frame with `id`, `image_path`, and `memorability`.
#'
#' @examples
#' \dontrun{
#' memscore_predict_standard("image.jpg", model = "memscore_standard.pkl")
#' }
#' @export
memscore_predict_standard <- function(
  paths,
  model = NULL,
  recursive = FALSE,
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  output = NULL,
  clip_scores = TRUE,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
) {
  stopifnot(is.character(paths), length(paths) >= 1L)

  backend_info <- .resolve_memscore_backend(
    backend = backend,
    cli = cli,
    python = python,
    module = getOption("memscore.module", "memscore"),
    workdir = workdir
  )

  if (identical(backend_info$kind, "reticulate")) {
    return(
      .predict_standard_via_reticulate(
        paths = paths,
        model = model,
        recursive = recursive,
        batch_size = batch_size,
        device = device,
        cache_dir = cache_dir,
        output = output,
        clip_scores = clip_scores,
        python = backend_info$python,
        module = backend_info$module,
        workdir = backend_info$workdir
      )
    )
  }

  args <- c("predict-standard")
  args <- .append_flag(args, "--model", if (!is.null(model)) .normalize_optional_path(model) else NULL)
  args <- c(args, vapply(paths, .normalize_optional_path, character(1)))
  args <- .append_flag(args, "--recursive", recursive)
  args <- .append_flag(args, "--batch-size", as.integer(batch_size))
  args <- .append_flag(args, "--device", device)
  args <- .append_flag(args, "--cache-dir", cache_dir)
  args <- .append_flag(args, "--output", output)
  args <- .append_flag(args, "--no-clip-scores", !clip_scores)

  output_lines <- .run_memscore_cli(
    backend_info,
    args = args
  )

  parsed <- .parse_json_output(output_lines)
  as.data.frame(parsed, stringsAsFactors = FALSE)
}

#' Train the memscore standard scorer artifact
#'
#' Trains the flagship `memscore` scorer from labeled locked manifests. By
#' default, only train and validation splits are used so test splits remain
#' unbiased references.
#'
#' @inheritParams memscore_benchmark_standard
#' @param output_model Path where the fitted scorer artifact should be written.
#' @param dataset_config Optional CSV with `dataset`, `manifest`, and optional
#'   `root` columns. Use either `manifest` or `dataset_config`.
#' @param include_test Include manifest test splits in training. Defaults to
#'   `FALSE` to preserve held-out evaluation.
#'
#' @return A list describing the saved artifact.
#'
#' @examples
#' \dontrun{
#' manifest <- system.file("extdata/tiny_manifest.csv", package = "memscore")
#' memscore_train_standard_scorer(
#'   manifest = manifest,
#'   root = dirname(manifest),
#'   output_model = "memscore_standard.pkl"
#' )
#' }
#' @export
memscore_train_standard_scorer <- function(
  output_model,
  manifest = NULL,
  root = NULL,
  dataset_config = NULL,
  clip_models = c("ViT-B-32", "RN50", "ViT-B-16"),
  clip_pretrained = "openai",
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  clip_tta = "fivecrop",
  clip_tta_resize = 256L,
  include_test = FALSE,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  backend = getOption("memscore.backend", "auto"),
  workdir = getOption("memscore.workdir")
) {
  if (is.null(manifest) == is.null(dataset_config)) {
    stop("Pass exactly one of `manifest` or `dataset_config`.", call. = FALSE)
  }

  backend_info <- .resolve_memscore_backend(
    backend = backend,
    cli = cli,
    python = python,
    module = getOption("memscore.module", "memscore"),
    workdir = workdir
  )

  if (identical(backend_info$kind, "reticulate")) {
    return(
      .train_standard_scorer_via_reticulate(
        output_model = output_model,
        manifest = manifest,
        root = root,
        dataset_config = dataset_config,
        clip_models = clip_models,
        clip_pretrained = clip_pretrained,
        batch_size = batch_size,
        device = device,
        cache_dir = cache_dir,
        clip_tta = clip_tta,
        clip_tta_resize = clip_tta_resize,
        include_test = include_test,
        python = backend_info$python,
        module = backend_info$module,
        workdir = backend_info$workdir
      )
    )
  }

  args <- c("train-standard-scorer")
  args <- .append_flag(args, "--manifest", manifest)
  args <- .append_flag(args, "--dataset-config", dataset_config)
  args <- .append_flag(args, "--root", root)
  args <- .append_flag(args, "--output-model", .normalize_optional_path(output_model))
  args <- c(args, "--clip-models", clip_models)
  args <- .append_flag(args, "--clip-pretrained", clip_pretrained)
  args <- .append_flag(args, "--batch-size", as.integer(batch_size))
  args <- .append_flag(args, "--device", device)
  args <- .append_flag(args, "--cache-dir", cache_dir)
  args <- .append_flag(args, "--clip-tta", clip_tta)
  args <- .append_flag(args, "--clip-tta-resize", clip_tta_resize)
  args <- .append_flag(args, "--include-test", include_test)

  output_lines <- .run_memscore_cli(
    backend_info,
    args = args
  )

  .parse_json_output(output_lines)
}
