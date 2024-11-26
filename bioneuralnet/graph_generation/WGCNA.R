library(WGCNA)

# Reading arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 6) {
  stop("Exactly 6 arguments must be supplied: phenotype_file, omics_files, soft_power, min_module_size, merge_cut_height, output_dir")
}

phenotype_file <- args[1]
omics_files <- strsplit(args[2], ",")[[1]]
soft_power <- as.numeric(args[3])
min_module_size <- as.numeric(args[4])
merge_cut_height <- as.numeric(args[5])
saving_dir <- args[6]

# Print arguments for verification
print(paste("Phenotype file:", phenotype_file))
print(paste("Omics files:", paste(omics_files, collapse = ", ")))
print(paste("Soft Power:", soft_power))
print(paste("Minimum Module Size:", min_module_size))
print(paste("Merge Cut Height:", merge_cut_height))
print(paste("Saving directory:", saving_dir))

# Load the phenotype data
phenotype_data <- read.csv(phenotype_file, header = TRUE, stringsAsFactors = FALSE)

# Assuming the first column is sample IDs and the second column is phenotype labels
sample_ids_pheno <- phenotype_data[[1]]
phenotype_vector <- as.factor(phenotype_data[[2]])

print(paste("Phenotype data length:", length(phenotype_vector)))
print(paste("Phenotype factor levels:", paste(levels(phenotype_vector), collapse = ", ")))

# Initialize the list to hold aligned omics data
omics_list <- list()

# Data loading
for (i in seq_along(omics_files)) {
  file_path <- omics_files[i]
  print(paste("Loading omics file:", file_path))

  # Read the CSV file
  omics_data <- read.csv(file_path, header = TRUE, stringsAsFactors = FALSE)

  # Assuming the first column is sample IDs
  sample_ids_omics <- omics_data[[1]]
  omics_values <- omics_data[, -1] # Remove the sample IDs column

  # Since data is preprocessed in Python, assume alignment is correct
  omics_data_numeric <- data.matrix(omics_values)
  rownames(omics_data_numeric) <- sample_ids_pheno

  omics_list[[i]] <- omics_data_numeric
  print(paste("Omics data dimensions:", nrow(omics_data_numeric), "samples x", ncol(omics_data_numeric), "features"))
}

# Set seed for reproducibility
# Note: If a separate seed parameter is provided, replace 'soft_power' with 'seed'
set.seed(soft_power) 

# Construct WGCNA network for the first omics dataset
# Modify as needed to handle multiple omics datasets
adjacency_matrix <- adjacency(omics_list[[1]], power = soft_power, type = "unsigned")
TOM <- TOMsimilarity(adjacency_matrix)
dissimilarity <- 1 - TOM

gene_tree <- hclust(as.dist(dissimilarity), method = "average")
dynamic_modules <- cutreeDynamic(dendro = gene_tree, distM = dissimilarity,
                                 deepSplit = 2, pamRespectsDendro = FALSE,
                                 minClusterSize = min_module_size)

# Convert numeric labels to colors
module_colors <- labels2colors(dynamic_modules)

# Merge modules
ME_list <- moduleEigengenes(omics_list[[1]], colors = module_colors)
MEs <- ME_list$eigengenes

merge <- mergeCloseModules(omics_list[[1]], module_colors, cutHeight = merge_cut_height, verbose = 3)
merged_colors <- merge$colors
merged_MEs <- merge$newMEs

# Save adjacency matrix
write.csv(adjacency_matrix, file = file.path(saving_dir, "global_network.csv"), row.names = TRUE)

# Save module assignments
module_assignments <- data.frame(Gene = colnames(omics_list[[1]]), Module = merged_colors)
write.csv(module_assignments, file = file.path(saving_dir, "module_assignments.csv"), row.names = FALSE)

# Save workspace
save(list = ls(), file = file.path(saving_dir, "wgcna_results.RData"))
