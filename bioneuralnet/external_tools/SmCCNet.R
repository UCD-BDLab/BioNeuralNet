#!/usr/bin/env Rscript

library("SmCCNet")
library("WGCNA")
library("jsonlite")
library("dplyr")

options(stringsAsFactors = FALSE)
allowWGCNAThreads(nThreads = 4)

#allowWGCNAThreads(nThreads = 1)
#library(future); plan(sequential)
#options(future.globals.maxSize = 8000e6)

library(future)
plan(multisession, workers = 8)
options(future.globals.maxSize = 8000e6)

json_input <- readLines(con = "stdin")
if (length(json_input) == 0) {
  stop("No input data received.")
}
input_data <- fromJSON(paste(json_input, collapse = "\n"))

if (!("phenotype" %in% names(input_data))) {
  stop("Phenotype data not found in input.")
}

phenotype_df <- read.csv(text = input_data$phenotype, stringsAsFactors = FALSE)
rownames(phenotype_df) <- phenotype_df$SampleID

omics_keys <- grep("^omics_", names(input_data), value = TRUE)
if (length(omics_keys) < 1) {
  stop("No omics data found in input.")
}

omics_list <- list()
for (key in omics_keys) {
  omics_df <- read.csv(text = input_data[[key]], stringsAsFactors = FALSE)
  rownames(omics_df) <- omics_df$SampleID
  omics_values <- as.matrix(omics_df[, -1])
  omics_list[[length(omics_list) + 1]] <- omics_values
}

clean_matrix <- function(mat){
  storage.mode(mat) <- "numeric"
  mat[is.infinite(mat)] <- NA

  for (j in seq_len(ncol(mat))) {
    if (anyNA(mat[, j])) {
      med <- median(mat[, j], na.rm = TRUE)
      mat[is.na(mat[, j]), j] <- med
    }
  }

  ok1 <- apply(mat, 2, function(col) all(is.finite(col)))
  mat  <- mat[, ok1, drop = FALSE]

  vars <- apply(mat, 2, var)
  mat  <- mat[, vars > 0, drop = FALSE]

  mat
}
omics_list <- lapply(omics_list, clean_matrix)

ids <- Reduce(intersect, lapply(omics_list, rownames))
ids <- intersect(ids, rownames(phenotype_df))
omics_list   <- lapply(omics_list, function(m) m[ids, , drop = FALSE])
phenotype_df <- phenotype_df[ids, , drop = FALSE]

Y <- as.numeric(phenotype_df$phenotype)
Yfactor <- factor(Y)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 10) {
  stop("Expected 10 arguments: data_types, kfold, summarization, seed, eval_method, ncomp_pls, subSampNum, between_shrinkage, CutHeight, preprocess")
}

data_types <- strsplit(args[1], ",")[[1]]
kfold <- as.numeric(args[2])
summarization <- args[3]
seed <- as.numeric(args[4])
eval_method <- args[5]
ncomp_pls_arg <- as.numeric(args[6])
subSampNum <- as.numeric(args[7])
bShrink <- as.numeric(args[8])
CutHeight <- as.numeric(args[9])
preprocess_int <- as.numeric(args[10])

# cmd = [
#     rscript_path,
#     r_script,
#     ",".join(self.data_types),
#     str(self.kfold),
#     self.summarization,
#     str(self.seed),
#     self.eval_method,
#     str(self.ncomp_pls),
#     str(self.subSampNum),
#     str(self.between_shrinkage),
#     str(self.cut_height),
#     str(self.preprocess),
# ]

if (preprocess_int == 1) {
  preprocess <- TRUE
}else {
  preprocess <- FALSE
}

if (is.na(ncomp_pls_arg) || ncomp_pls_arg <= 0) {
  ncomp_pls <- NULL
} else {
  ncomp_pls <- ncomp_pls_arg
}
set.seed(seed)

if (length(data_types) != length(omics_list)) {
  stop("data_types length doesn't match number of omics datasets.")
}

if (any(is.na(Y))) {
  stop("Phenotype contains NA.")
}

message("DEBUG: We have ", length(Y), " samples in Y. Range: [", min(Y), ", ", max(Y), "]")
message("DEBUG: Omics data shapes:")
for (i in seq_along(omics_list)) {
  message("  Omics ", i, ": ", nrow(omics_list[[i]]), " x ", ncol(omics_list[[i]]))
}

for (i in seq_along(omics_list)) {
  mat <- omics_list[[i]]
  cat(sprintf("Post-clean Omics %d: dim = [%d x %d], NAs = %d, Infs = %d\n",
      i, nrow(mat), ncol(mat),
      sum(is.na(mat)), sum(is.infinite(mat))))
  if (nrow(mat) == 0 || ncol(mat) == 0) {
    stop(paste("ERROR: Omics", i, "is empty!"))
  }
  if (any(!is.finite(mat))) {
    stop(paste("ERROR: Non-finite values remain in Omics", i))
  }
}

if (any(is.na(Y)) || any(is.infinite(Y))) {
  stop("ERROR: Phenotype vector Y contains NA or Inf")
}

#Y_binary <- ifelse(Y > median(Y), 1, 0)

if (length(data_types) == 1 && !is.null(ncomp_pls)) {
  message("Single-omics PLS scenario")
  result <- fastAutoSmCCNet(
    X = omics_list,
    Y = Yfactor,
    DataType = data_types,
    EvalMethod = "auc",
    Kfold = kfold,
    subSampNum = subSampNum,
    summarization = summarization,
    CutHeight = CutHeight,
    ncomp_pls = ncomp_pls,
    preprocess = preprocess,
    seed = seed
  )

} else if (length(data_types) == 1) {
  message("Single-omics CCA scenario")
  result <- fastAutoSmCCNet(
    X = omics_list,
    Y = Y,
    DataType = data_types,
    Kfold = kfold,
    subSampNum = subSampNum,
    summarization = summarization,
    CutHeight = CutHeight,
    preprocess = preprocess,
    seed = seed
  )

} else if (length(data_types) > 1 && !is.null(ncomp_pls)) {
  message("Multi-omics PLS scenario")
  result <- fastAutoSmCCNet(
    X = omics_list,
    Y = Yfactor,
    DataType = data_types,
    EvalMethod = "auc",
    Kfold = kfold,
    subSampNum = subSampNum,
    summarization = summarization,
    CutHeight = CutHeight,
    BetweenShrinkage = bShrink,
    ncomp_pls = ncomp_pls,
    preprocess = preprocess,
    seed = seed
  )

} else {
  message("Multi-omics CCA scenario")
  result <- fastAutoSmCCNet(
    X = omics_list,
    Y = Y,
    DataType = data_types,
    Kfold = kfold,
    subSampNum = subSampNum,
    summarization = summarization,
    CutHeight = CutHeight,
    BetweenShrinkage = bShrink,
    preprocess = preprocess,
    seed = seed
  )
}

write.csv(result$AdjacencyMatrix, file = "GlobalNetwork.csv",  row.names = TRUE)

current_dir <- getwd()
message("Current working directory: ", current_dir)
pattern <- "^size_.*\\.Rdata$"
rdata_files <- list.files(path = current_dir, pattern = pattern, full.names = TRUE)
message("Found files: ", paste(rdata_files, collapse = ", "))

if (length(rdata_files) == 0) {
  message("No RData files found in the current directory.\n")
} else {
  for (file in rdata_files) {
    message("Processing file: ", file, "\n")

    temp_env <- new.env()
    loaded_names <- load(file, envir = temp_env)

    if ("M" %in% loaded_names && exists("M", envir = temp_env)) {
      sub_net <- get("M", envir = temp_env)

      file_base   <- tools::file_path_sans_ext(basename(file))
      csv_filename <- paste0(file_base, ".csv")

      write.csv(sub_net, file = csv_filename, row.names = TRUE)
      message("Subnetwork matrix from ", file, " written to ", csv_filename, "\n\n")
    } else {
      message("Warning: Object 'M' was not found in ", file, "\n\n")
    }
  }
}

quit(status = 0)
