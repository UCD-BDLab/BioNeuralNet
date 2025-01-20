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

omics_list <- list()
omics_keys <- grep("^omics_", names(input_data), value = TRUE)
if (length(omics_keys) == 0) {
  stop("No omics data found in input.")
}
for (omics_key in omics_keys) {
  omics_df <- read.csv(text = input_data[[omics_key]], stringsAsFactors = FALSE)
  omics_values <- omics_df[, -1]
  rownames(omics_values) <- omics_df[[1]]
  omics_list[[omics_key]] <- as.matrix(omics_values)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Exactly 4 arguments must be supplied: data_types, kfold, summarization, seed")
}

data_types <- strsplit(args[1], ",")[[1]]
kfold <- as.numeric(args[2])
summarization <- args[3]
seed <- as.numeric(args[4])

if (is.na(kfold) || is.na(seed)) {
  stop("kfold and seed must be numeric values.")
}

if (!(summarization %in% c("PCA", "SVD", "NetSHy"))) {
  stop("Summarization method must be 'PCA', 'SVD', or 'NetSHy'.")
}

set.seed(seed)

omics_values <- omics_list[[1]]
common_samples <- intersect(rownames(omics_values), phenotype_df[[1]])
if (length(common_samples) == 0) {
  stop("No matching sample IDs between omics data and phenotype.")
}
omics_values <- omics_values[common_samples, , drop = FALSE]
phenotype_df <- phenotype_df[match(common_samples, phenotype_df[[1]]), , drop = FALSE]

message("dim(omics_values) = ", paste(dim(omics_values), collapse = " x "))
message("dim(phenotype_df) = ", paste(dim(phenotype_df), collapse = " x "))

Y <- as.numeric(phenotype_df[[2]])

if (length(data_types) == 1) {
  message("Single-omics mode detected.")
  if (summarization != "NetSHy") {
    message("Overriding summarization method to 'NetSHy' for single-omics analysis.")
    summarization <- "NetSHy"
  }
  if (length(unique(Y)) == 2) {
    message("Binary phenotype detected. Setting ncomp_pls to 5 for single-omics PLS.")
    ncomp_pls <- 5
  } else {
    ncomp_pls <- NULL
  }
} else {
  message("Multi-omics mode detected.")
  ncomp_pls <- NULL
}

if (!is.null(ncomp_pls)) {
  result <- fastAutoSmCCNet(
    X = list(omics_values),
    Y = Y,
    DataType = data_types,
    Kfold = kfold,
    subSampNum = 50,
    EvalMethod = "accuracy",
    summarization = summarization,
    ncomp_pls = ncomp_pls,
    seed = seed,
  )
} else {
  result <- fastAutoSmCCNet(
    X = list(omics_values),
    Y = Y,
    DataType = data_types,
    Kfold = kfold,
    subSampNum = 50,
    EvalMethod = "accuracy",
    summarization = summarization,
    seed = seed,
  )
}

result_json <- toJSON(result, pretty = TRUE, auto_unbox = TRUE)
cat(result_json)

quit(status = 0)
