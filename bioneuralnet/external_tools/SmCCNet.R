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
if (length(omics_keys) == 0) {
  stop("No omics data found in input.")
}
omics_df <- read.csv(text = input_data[[omics_keys[1]]], stringsAsFactors = FALSE)
if (!("SampleID" %in% colnames(omics_df))) {
  stop("SampleID column not found in omics data.")
}
rownames(omics_df) <- omics_df$SampleID
omics_values <- as.matrix(omics_df[, -1])

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


common_samples <- intersect(rownames(omics_values), rownames(phenotype_df))
if (length(common_samples) == 0) {
  stop("No matching sample IDs between omics data and phenotype.")
}
omics_values <- omics_values[common_samples, , drop = FALSE]
phenotype_df <- phenotype_df[common_samples, , drop = FALSE]

Y <- as.numeric(phenotype_df[[2]])
if (any(is.na(Y))) {
  stop("Phenotype vector contains NA values.")
}

tryCatch(
  {
    if (length(data_types) == 1) {
      message("Single-omics mode detected.")
      if (summarization != "NetSHy") {
        message("Overriding summarization method to 'NetSHy' for single-omics analysis.")
        summarization <- "NetSHy"
      }
      if (length(unique(Y)) == 2) {
        message("Binary phenotype detected. Setting ncomp_pls = 5 for single-omics PLS.")
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
        seed = seed
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
        seed = seed
      )
    }
  },
  error = function(e) {
    message("Error during fastAutoSmCCNet execution: ", conditionMessage(e))
    quit(status = 1)
  }
)

write.csv(result$AdjacencyMatrix, file = "AdjacencyMatrix.csv", row.names = TRUE)


quit(status = 0)
