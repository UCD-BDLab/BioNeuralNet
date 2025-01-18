#!/usr/bin/env Rscript

library("SmCCNet")
library("jsonlite")
library("dplyr")
#library("WGCNA")

options(stringsAsFactors = FALSE)
#allowWGCNAThreads()
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

if (!(summarization %in% c("PCA", "SVD"))) {
  stop("Summarization method must be either 'PCA' or 'SVD'.")
}

set.seed(seed)
omics_df <- read.csv(text = input_data[["omics_1"]], stringsAsFactors = FALSE)
tryCatch({
  result <- fastAutoSmCCNet(
    X = list(omics_df),
    Y = phenotype_df[[2]], 
    DataType = data_types,
    Kfold = kfold,
    Summarization = summarization,
    seed = seed,
    verbose = FALSE
  )
  
  adjacency_matrix <- result$globalNetwork$AdjacencyMatrix
  adjacency_json <- toJSON(adjacency_matrix, dataframe = "columns")
  
  cat(adjacency_json)
}, error = function(e) {
  # Serialize error message to JSON and output
  error_json <- toJSON(list(error = e$message))
  cat(error_json)
  quit(status = 1)
})
