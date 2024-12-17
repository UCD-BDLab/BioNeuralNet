#!/bin/bash

# Automated R Dependencies Setup Script

set -e 

echo "Setting up R dependencies for BioNeuralNet..."

# Install system libraries required for R and WGCNA
echo "Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
    libxml2-dev libcurl4-openssl-dev libssl-dev libpng-dev

# Install R (if not installed)
if ! command -v R &> /dev/null; then
    echo "R is not installed. Installing R..."
    sudo apt-get update && sudo apt-get install -y r-base
else
    echo "R is already installed."
fi

# Install required R packages, including Bioconductor packages
echo "Installing R packages: dplyr, SmCCNet, WGCNA, and dependencies..."

# Install CRAN and Bioconductor packages
Rscript -e "options(repos = c(CRAN = 'https://cran.r-project.org')); install.packages(c('dplyr', 'SmCCNet'))"

Rscript -e "if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager')"
Rscript -e "BiocManager::install(c('impute', 'preprocessCore', 'GO.db', 'AnnotationDbi'), update=FALSE, ask=FALSE)"

# Install WGCNA explicitly
Rscript -e "options(repos = c(CRAN = 'https://cran.r-project.org')); install.packages('WGCNA')"

echo "R dependencies setup completed!"
