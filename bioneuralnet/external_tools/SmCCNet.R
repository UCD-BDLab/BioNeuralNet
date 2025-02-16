#!/usr/bin/env Rscript

library("SmCCNet")
library("WGCNA")
library("jsonlite")
library("dplyr") 

options(stringsAsFactors = FALSE)
allowWGCNAThreads()

json_input <- readLines(con = "stdin")
if (length(json_input) == 0) {
  stop("No input data received.")
}
input_data <- fromJSON(paste(json_input, collapse = "\n"))

if (!("phenotype" %in% names(input_data))) {
  stop("Phenotype data not found in input.")
}

phenotype_df <- read.csv(text = input_data$phenotype, stringsAsFactors = FALSE)
if (!("SampleID" %in% colnames(phenotype_df))) {
  stop("SampleID column not found in phenotype data.")
}
rownames(phenotype_df) <- phenotype_df$SampleID

omics_keys <- grep("^omics_", names(input_data), value = TRUE)
if (length(omics_keys) < 1) {
  stop("No omics data found in input.")
}

omics_list <- list()
for (key in omics_keys) {
  omics_df <- read.csv(text = input_data[[key]], stringsAsFactors = FALSE)
  if (!("SampleID" %in% colnames(omics_df))) {
    stop(paste("SampleID column not found in", key))
  }
  rownames(omics_df) <- omics_df$SampleID
  
  omics_values <- as.matrix(omics_df[, -1])

  common_samples <- intersect(rownames(omics_values), rownames(phenotype_df))
  if (length(common_samples) == 0) {
    stop(paste("No matching sample IDs between", key, "and phenotype data."))
  }
  
  omics_values <- omics_values[common_samples, , drop=FALSE]
  omics_list[[length(omics_list)+1]] <- omics_values
}
rownames(omics_df) <- omics_df$SampleID
omics_values <- as.matrix(omics_df[, -1])

common_samples_all <- rownames(phenotype_df)
for (mat in omics_list) {
  common_samples_all <- intersect(common_samples_all, rownames(mat))
}
if (length(common_samples_all) == 0) {
  stop("No common samples across all omics datasets and phenotype.")
}
phenotype_df <- phenotype_df[common_samples_all, , drop=FALSE]
omics_list   <- lapply(omics_list, function(m) m[common_samples_all, , drop=FALSE])


args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop("Expected 8 arguments: data_types, kfold, summarization, seed, eval_method, ncomp_pls, subSampNum, between_shrinkage")
}

data_types    <- strsplit(args[1], ",")[[1]]
kfold         <- as.numeric(args[2])
summarization <- args[3]
seed          <- as.numeric(args[4])
eval_method   <- args[5]
ncomp_pls_arg <- args[6]
subSampNum    <- as.numeric(args[7])
bShrink       <- as.numeric(args[8])

if (ncomp_pls_arg == "") {
  ncomp_pls <- NULL
} else {
  ncomp_pls <- as.numeric(ncomp_pls_arg)
}
set.seed(seed)

if (length(data_types) != length(omics_list)) {
  stop("data_types length doesn't match number of omics datasets.")
}

Y <- as.numeric(phenotype_df[[2]])
if (any(is.na(Y))) {
  stop("Phenotype contains NA.")
}

message("DEBUG: We have ", length(Y), " samples in Y. Range: [", min(Y), ", ", max(Y), "]")
message("DEBUG: Omics data shapes:")
for (i in seq_along(omics_list)) {
  message("  Omics ", i, ": ", nrow(omics_list[[i]]), " x ", ncol(omics_list[[i]]))
}

foldIndices <- split.default(seq_len(nrow(omics_list[[1]])), sample(seq_len(nrow(omics_list[[1]])), kfold))
for (f in seq_along(foldIndices)) {
  idx <- foldIndices[[f]]
  message("Debug fold #", f, " => size: ", length(idx))
  for (om in seq_along(omics_list)) {
    subset_omics <- omics_list[[om]][idx, , drop=FALSE]
    var_ <- apply(subset_omics, 2, var, na.rm=TRUE)

    bad_cols <- which(is.na(var_) | is.nan(var_) | var_ == 0)
    if (length(bad_cols) > 0) {
      subset_omics <- subset_omics[, -bad_cols, drop=FALSE]
      message("Fold ", f, ", omics #", om, " => removing ", length(bad_cols), 
              " zero/NA-variance cols.")
    }
  }
}

Y_binary <- ifelse(Y > median(Y), 1, 0) 

if (length(data_types) == 1 && !is.null(ncomp_pls)) {
  message("Single-omics PLS scenario")
  result <- fastAutoSmCCNet(
    X = omics_list,
    Y = as.factor(Y_binary),
    DataType = data_types,
    Kfold = kfold,
    subSampNum = subSampNum,
    CutHeight = 1 - 0.1^10,
    EvalMethod = "auc",
    summarization = summarization,
    ncomp_pls = ncomp_pls,
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
    CutHeight = 1 - 0.1^10,
    seed = seed
  )
  
} else if (length(data_types) > 1 && !is.null(ncomp_pls)) {
  message("Multi-omics PLS scenario")
  result <- fastAutoSmCCNet(
    X = omics_list,
    Y = as.factor(Y),
    DataType = data_types,
    EvalMethod = "auc",
    Kfold = kfold,
    subSampNum = subSampNum,
    summarization = summarization,
    CutHeight = 1 - 0.1^10,
    ncomp_pls = ncomp_pls,
    seed = seed,
    BetweenShrinkage = bShrink
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
    CutHeight = 1 - 0.1^10,
    seed = seed,
    BetweenShrinkage = bShrink
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