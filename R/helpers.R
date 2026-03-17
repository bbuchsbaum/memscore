`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) {
    return(y)
  }
  if (is.character(x) && length(x) == 1L && !nzchar(x)) {
    return(y)
  }
  x
}

.system2 <- function(command, args, stdout = TRUE, stderr = TRUE) {
  system2(command = command, args = args, stdout = stdout, stderr = stderr)
}

.with_workdir <- function(wd, expr) {
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(wd)
  force(expr)
}

.normalize_existing_path <- function(path) {
  normalizePath(path, winslash = "/", mustWork = TRUE)
}

.normalize_optional_path <- function(path) {
  normalizePath(path, winslash = "/", mustWork = FALSE)
}

.append_flag <- function(args, flag, value = TRUE) {
  if (is.null(value) || identical(value, FALSE)) {
    return(args)
  }
  if (identical(value, TRUE)) {
    return(c(args, flag))
  }
  c(args, flag, as.character(value))
}

.parse_json_output <- function(output) {
  text <- paste(output, collapse = "\n")
  start_candidates <- c(
    gregexpr("\\{", text, perl = TRUE)[[1]],
    gregexpr("\\[", text, perl = TRUE)[[1]]
  )
  start_candidates <- start_candidates[start_candidates > 0L]
  end_candidates <- c(
    gregexpr("\\}", text, perl = TRUE)[[1]],
    gregexpr("\\]", text, perl = TRUE)[[1]]
  )
  end_candidates <- end_candidates[end_candidates > 0L]

  if (!length(start_candidates) || !length(end_candidates)) {
    stop("Could not find JSON in memscore output.", call. = FALSE)
  }

  start <- min(start_candidates)
  end <- max(end_candidates)
  if (end < start) {
    stop("Could not parse JSON from memscore output.", call. = FALSE)
  }

  jsonlite::fromJSON(substr(text, start, end), simplifyVector = TRUE)
}

.run_memscore_cli <- function(command_info, args) {
  runner <- function() {
    .system2(
      command = command_info$command,
      args = c(command_info$args, args),
      stdout = TRUE,
      stderr = TRUE
    )
  }

  output <- if (is.null(command_info$workdir)) {
    runner()
  } else {
    .with_workdir(command_info$workdir, runner())
  }

  status <- attr(output, "status") %||% 0L
  if (!identical(status, 0L)) {
    stop(
      sprintf(
        "memscore command failed with status %s\n%s",
        status,
        paste(output, collapse = "\n")
      ),
      call. = FALSE
    )
  }

  output
}
