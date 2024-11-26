library(dplyr)
library(SmCCNet)

# Retrieve command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Validate the number of arguments
if (length(args) != 7) {
  stop("Exactly 7 arguments must be supplied: phenotype_file, omics_files, data_types, kfold, summarization, seed, output_dir")
}

# Extract the arguments
phenotype_file <- args[1]
omics_files <- strsplit(args[2], ",")[[1]]
data_types <- strsplit(args[3], ",")[[1]]
kfold <- as.numeric(args[4])
summarization <- args[5]
seed <- as.numeric(args[6])
output_dir <- args[7]

# Ensure the output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Print arguments for verification
print(paste("Phenotype file:", phenotype_file))
print(paste("Omics files:", paste(omics_files, collapse = ", ")))
print(paste("Data types:", paste(data_types, collapse = ", ")))
print(paste("Kfold:", kfold))
print(paste("Summarization method:", summarization))
print(paste("Seed:", seed))
print(paste("Saving directory:", output_dir))

# Load the phenotype data
print(paste("Loading phenotype file:", phenotype_file))
phenotype_data <- read.csv(phenotype_file, header = TRUE, stringsAsFactors = FALSE)

# Assuming the first column is sample IDs and the second column is phenotype labels
sample_ids_pheno <- phenotype_data[[1]]
phenotype_vector <- as.factor(phenotype_data[[2]])

print(paste("Phenotype data length:", length(phenotype_vector)))
print(paste("Phenotype factor levels:", paste(levels(phenotype_vector), collapse = ", ")))

# Initialize the list to hold aligned omics data
omics_list <- list()

# Data loading and alignment
for (i in seq_along(omics_files)) {
  file_path <- omics_files[i]
  print(paste("Loading omics file:", file_path))

  # Read the CSV file
  omics_data <- read.csv(file_path, header = TRUE, stringsAsFactors = FALSE)

  # Assuming the first column is sample IDs
  sample_ids_omics <- omics_data[[1]]
  omics_values <- omics_data[, -1] # Remove the sample IDs column

  # Align the omics data with the phenotype data using sample IDs
  aligned_indices <- match(sample_ids_pheno, sample_ids_omics)

  # Check for NA indices (samples not found)
  if (any(is.na(aligned_indices))) {
    print("Warning: Some samples in phenotype data were not found in omics data.")
    # Remove samples not found in omics data
    valid_indices <- which(!is.na(aligned_indices))
    phenotype_vector <- phenotype_vector[valid_indices]
    aligned_indices <- aligned_indices[valid_indices]
  }

  aligned_data <- omics_values[aligned_indices, ]

  # Convert to numeric matrix
  omics_data_numeric <- data.matrix(aligned_data)

  # Add to omics_list
  omics_list[[i]] <- omics_data_numeric
  print(paste("Omics data dimensions:", nrow(omics_data_numeric), "samples x", ncol(omics_data_numeric), "features"))
}

# Now, phenotype_vector and each element in omics_list should have the same number of samples
num_samples <- length(phenotype_vector)
print(paste("Number of samples after alignment:", num_samples))

# Set seed for reproducibility
set.seed(seed)

# Execute SmCCNet
result <- fastAutoSmCCNet(
  X = omics_list,
  Y = phenotype_vector,
  DataType = data_types,
  Kfold = kfold,
  summarization = summarization,
  seed = seed,
  saving_dir = output_dir
)

# Load and write the global network to CSV
load(file.path(output_dir, "globalNetwork.Rdata"))
write.csv(globalNetwork$AdjacencyMatrix, file = file.path(output_dir, "global_network.csv"))

# Process subnetworks
subnetworks <- list.files(path = output_dir, pattern = "^size_.*\\.Rdata$", full.names = TRUE)

for (subnet in subnetworks) {
  load(subnet)

  # Extract the file name without the directory and extension
  file_name <- basename(subnet)
  file_name_clean <- sub("\\.Rdata$", "", file_name)

  # Retrieve the adjacency matrix
  adjacency_matrix <- M

  # Write to CSV
  write.csv(adjacency_matrix, file = file.path(output_dir, paste0(file_name_clean, ".csv")))
}
