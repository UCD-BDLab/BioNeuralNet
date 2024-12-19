#!/usr/bin/env Rscript


library("WGCNA")
library("jsonlite")

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
  omics_values <- as.data.frame(omics_df[, -1], stringsAsFactors = FALSE)
  rownames(omics_values) <- omics_df[[1]]
  omics_list[[omics_key]] <- as.matrix(omics_values)
}

# Parameters from command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Exactly 3 arguments must be supplied: soft_power, min_module_size, merge_cut_height")
}

soft_power <- as.numeric(args[1])
min_module_size <- as.numeric(args[2])
merge_cut_height <- as.numeric(args[3])

# Check if soft_power is numeric
if (is.na(soft_power)) {
  stop("soft_power must be a numeric value.")
}

# Assuming we're working with the first omics dataset for simplicity
# Modify as needed to handle multiple omics datasets
if (length(omics_list) < 1) {
  stop("At least one omics dataset is required.")
}

omics_matrix <- omics_list[[1]]

# Construct WGCNA network
tryCatch({
  # Calculate adjacency matrix
  adjacency_matrix <- adjacency(omics_matrix, power = soft_power, type = "unsigned")
  
  # Calculate Topological Overlap Matrix (TOM)
  TOM <- TOMsimilarity(adjacency_matrix)
  dissTOM <- 1 - TOM
  
  # Hierarchical clustering
  gene_tree <- hclust(as.dist(dissTOM), method = "average")
  
  # Dynamic tree cut for module detection
  dynamic_modules <- cutreeDynamic(dendro = gene_tree, distM = dissTOM,
                                   deepSplit = 2, pamRespectsDendro = FALSE,
                                   minClusterSize = min_module_size)
  
  # Convert numeric labels to colors
  module_colors <- labels2colors(dynamic_modules)
  
  # Merge modules with similar eigengenes
  ME_list <- moduleEigengenes(omics_matrix, colors = module_colors)
  MEs <- ME_list$eigengenes
  
  merged <- mergeCloseModules(omics_matrix, module_colors, cutHeight = merge_cut_height, verbose = 3)
  merged_colors <- merged$colors
  merged_MEs <- merged$newMEs
  
  # Prepare adjacency matrix for output
  adjacency_output <- as.data.frame(adjacency_matrix)
  adjacency_output$Gene <- rownames(adjacency_matrix)
  adjacency_output <- adjacency_output[, c(ncol(adjacency_output), 1:(ncol(adjacency_output)-1))]
  
  # Serialize adjacency matrix to JSON
  adjacency_json <- toJSON(adjacency_output, dataframe = "columns")
  
  # Output adjacency matrix JSON to stdout
  cat(adjacency_json)
}, error = function(e) {
  # Serialize error message to JSON and output
  error_json <- toJSON(list(error = e$message))
  cat(error_json)
  quit(status = 1)
})
