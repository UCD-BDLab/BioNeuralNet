#!/usr/bin/env Rscript

# Load necessary libraries

library("SmCCNet")
library("jsonlite")
library("dplyr")


# Increase the maximum size of connections
options(stringsAsFactors = FALSE)
allowWGCNAThreads()

# Reading JSON data from stdin
json_input <- readLines(con = "stdin")

if (length(json_input) == 0) {
  stop("No input data received.")
}

# Parse JSON data
input_data <- fromJSON(paste(json_input, collapse = "\n"))

# Extract phenotype and omics data
if (!("phenotype" %in% names(input_data))) {
  stop("Phenotype data not found in input.")
}

phenotype_df <- read.csv(text = input_data$phenotype, stringsAsFactors = FALSE)

# Initialize list to hold omics data
omics_list <- list()

# Extract omics data
omics_keys <- grep("^omics_", names(input_data), value = TRUE)
if (length(omics_keys) == 0) {
  stop("No omics data found in input.")
}

for (omics_key in omics_keys) {
  omics_df <- read.csv(text = input_data[[omics_key]], stringsAsFactors = FALSE)
  omics_values <- omics_df[, -1]  # Remove the sample IDs column
  rownames(omics_values) <- omics_df[[1]]
  omics_list[[omics_key]] <- as.matrix(omics_values)
}

# Retrieve command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 4) {
  stop("Exactly 4 arguments must be supplied: data_types, kfold, summarization, seed")
}

# Extract the arguments
data_types <- strsplit(args[1], ",")[[1]]
kfold <- as.numeric(args[2])
summarization <- args[3]
seed <- as.numeric(args[4])

# Check if kfold and seed are numeric
if (is.na(kfold) || is.na(seed)) {
  stop("kfold and seed must be numeric values.")
}

# Check summarization method
if (!(summarization %in% c("PCA", "SVD"))) {
  stop("Summarization method must be either 'PCA' or 'SVD'.")
}

# Set seed for reproducibility
set.seed(seed)

# Perform SmCCNet analysis
tryCatch({
  # Execute SmCCNet
  result <- fastAutoSmCCNet(
    X = omics_list,
    Y = phenotype_df[[2]],  # Assuming the second column is the phenotype
    DataType = data_types,
    Kfold = kfold,
    Summarization = summarization,
    seed = seed,
    verbose = FALSE
  )
  
  # Extract the global adjacency matrix
  adjacency_matrix <- result$globalNetwork$AdjacencyMatrix
  
  # Serialize adjacency matrix to JSON
  adjacency_json <- toJSON(adjacency_matrix, dataframe = "columns")
  
  # Output adjacency matrix JSON to stdout
  cat(adjacency_json)
}, error = function(e) {
  # Serialize error message to JSON and output
  error_json <- toJSON(list(error = e$message))
  cat(error_json)
  quit(status = 1)
})
