args <- commandArgs(trailingOnly = TRUE)
input_file  <- args[1]
csv_file    <- args[2]

orig_input <- input_file
orig_csv   <- csv_file

rm(list = setdiff(ls(), c("orig_input", "orig_csv")))

success <- tryCatch({
  load(orig_input)
  loaded_objs <- ls()
  message("Loaded objects: ", paste(loaded_objs, collapse = ", "))

  mat <- NULL
  for (nm in loaded_objs) {
    obj <- get(nm)
    cls <- class(obj)
    message("Object '", nm, "' has class: ", paste(cls, collapse = "/"))

    if (inherits(obj, "list")) {
      for (subnm in names(obj)) {
        candidate <- obj[[subnm]]
        if (is.matrix(candidate) || is.data.frame(candidate) ||
            inherits(candidate, "Matrix") ||
            inherits(candidate, "igraph")) {

          if (inherits(candidate, "igraph")) {
            if (!"igraph" %in% loadedNamespaces()) library(igraph)
            candidate <- as_adjacency_matrix(candidate, attr = "weight", sparse = FALSE)
          }

          if (inherits(candidate, "Matrix")) {
            if (!"Matrix" %in% loadedNamespaces()) library(Matrix)
            candidate <- as.matrix(candidate)
          }

          mat <- as.matrix(candidate)
          message("Using list element '", subnm, "' from '", nm, "' as adjacency")
          break
        }
      }
      if (!is.null(mat)) break
    }

    if (is.matrix(obj) || is.data.frame(obj)) {
      mat <- as.matrix(obj)
      message("Using top-level object '", nm, "' as matrix/data.frame")
      break
    }

    if (inherits(obj, "Matrix")) {
      if (!"Matrix" %in% loadedNamespaces()) library(Matrix)
      mat <- as.matrix(obj)
      message("Coerced sparse '", nm, "' to dense matrix")
      break
    }

    if (inherits(obj, "igraph")) {
      if (!"igraph" %in% loadedNamespaces()) library(igraph)
      mat <- as.matrix(as_adjacency_matrix(obj, attr = "weight", sparse = FALSE))
      message("Converted igraph '", nm, "' to adjacency matrix")
      break
    }
  }

  if (is.null(mat)) {
    stop("Error: No suitable matrix/data.frame/Matrix/igraph object found in RData")
  }

  out_dir <- dirname(orig_csv)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  message("Writing CSV to: ", orig_csv)
  write.csv(mat, file = orig_csv, row.names = TRUE)
  message("CSV written successfully to: ", orig_csv)

  TRUE
}, error = function(e) {
  message("Skipping network conversion for file: ", orig_input)
  message("Error: ", e$message)
  FALSE
})

quit(save = "no", status = 0)
