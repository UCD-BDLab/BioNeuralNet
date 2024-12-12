#!/bin/bash

# Automated R Dependencies Setup Script

set -e 

echo "Setting up R dependencies for BioNeuralNet..."

# Install R (if not installed)
if ! command -v R &> /dev/null; then
    echo "R is not installed. Installing R..."
    if [[ "$(uname)" == "Linux" ]]; then
        sudo apt-get update && sudo apt-get install -y r-base
    elif [[ "$(uname)" == "Darwin" ]]; then
        brew install r
    fi
else
    echo "R is already installed."
fi

# Install required R packages, including Bioconductor packages used by WGCNA
echo "Installing R packages: dplyr, SmCCNet, WGCNA, and dependencies..."

# Install CRAN packages with explicit CRAN mirror
Rscript -e "options(repos = c(CRAN = 'https://cran.r-project.org')); install.packages(c('dplyr', 'SmCCNet'))"

# Install Bioconductor packages needed by WGCNA
Rscript -e "options(repos = c(CRAN = 'https://cran.r-project.org')); if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager')"
Rscript -e "options(repos = c(CRAN = 'https://cran.r-project.org')); BiocManager::install(c('impute', 'preprocessCore', 'GO.db', 'AnnotationDbi'))"

# Install WGCNA
Rscript -e "options(repos = c(CRAN = 'https://cran.r-project.org')); install.packages('WGCNA')"

echo "R dependencies setup completed!"
