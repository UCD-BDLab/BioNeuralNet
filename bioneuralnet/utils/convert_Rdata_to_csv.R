#!/usr/bin/env Rscript

# check for arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
    stop("Usage: convert_Rdata_to_csv.R input.Rdata output.csv", call. = FALSE)
}

input_file <- args[1]
output_file <- args[2]

# load the Rdata file to the workspace
load(input_file)

if (exists("AdjacencyMatrix")) {
    mat <- AdjacencyMatrix
} else if (exists("M")) {
    mat <- M
} else {
    stop("Neither 'AdjacencyMatrix' nor 'M' was found in the Rdata file.", call. = FALSE)
}

# write the matrix or data.frame to a csv file.
write.csv(mat, file = output_file, row.names = TRUE, quote = FALSE)
