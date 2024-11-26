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

# Install required R packages
echo "Installing R packages: dplyr, SmCCNet, WGCNA..."
Rscript -e "install.packages(c('dplyr', 'SmCCNet', 'WGCNA'), repos='http://cran.r-project.org')"

echo "R dependencies setup completed!"
