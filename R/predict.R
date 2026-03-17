#' Score image memorability
#'
#' Scores one or more image files or directories using the external `memscore`
#' backend.
#'
#' @param paths Character vector of image files or directories.
#' @param recursive Recurse into subdirectories when a path is a directory.
#' @param batch_size Batch size for inference.
#' @param device Torch device string passed through to the backend.
#' @param cache_dir Optional cache directory for downloaded checkpoints.
#' @param resmem_checkpoint Optional explicit path to a ResMem checkpoint.
#' @param output Optional CSV path for backend-generated predictions.
#' @param cli Optional explicit path to a `memscore` executable.
#' @param python Optional explicit path to a Python interpreter.
#' @param workdir Optional working directory for backend execution.
#'
#' @return A data frame with `id`, `image_path`, and `memorability`.
#'
#' @examples
#' \dontrun{
#' memscore_configure(
#'   python = "/usr/bin/python3",
#'   workdir = "/path/to/memscore"
#' )
#'
#' memscore_predict(c("image1.jpg", "image2.jpg"))
#' }
#' @export
memscore_predict <- function(
  paths,
  recursive = FALSE,
  batch_size = 32L,
  device = "cpu",
  cache_dir = NULL,
  resmem_checkpoint = NULL,
  output = NULL,
  cli = getOption("memscore.cli"),
  python = getOption("memscore.python"),
  workdir = getOption("memscore.workdir")
) {
  stopifnot(is.character(paths), length(paths) >= 1L)

  args <- c("predict", vapply(paths, .normalize_optional_path, character(1)))
  args <- .append_flag(args, "--recursive", recursive)
  args <- .append_flag(args, "--batch-size", as.integer(batch_size))
  args <- .append_flag(args, "--device", device)
  args <- .append_flag(args, "--cache-dir", cache_dir)
  args <- .append_flag(args, "--resmem-checkpoint", resmem_checkpoint)
  args <- .append_flag(args, "--output", output)

  output_lines <- .run_memscore_cli(
    memscore_cli_info(cli = cli, python = python, workdir = workdir),
    args = args
  )

  parsed <- .parse_json_output(output_lines)
  as.data.frame(parsed, stringsAsFactors = FALSE)
}
