#!/bin/bash

set -e 
echo "Starting BioNeuralNet setup..."
cd "$(dirname "$0")/.."

if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python 3.7 or higher."
    exit
fi

echo "Creating a virtual environment at the root (./.venv)..."
python3 -m venv .venv
echo "Activating the virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing base dependencies from requirements.txt at the root..."
pip install -r requirements.txt

if [ -z "$INSTALL_TYPE" ]; then
    echo "Select installation type:"
    echo "1. CPU-only"
    echo "2. CUDA-enabled (CUDA 11.7)"
    read -rp "Enter choice [1/2]: " choice
else
    choice=$INSTALL_TYPE
    echo "Using INSTALL_TYPE from environment: $choice"
fi

if [ -z "$INSTALL_DEV" ]; then
    echo "Do you want to install development dependencies? [y/N]"
    read -rp "Enter choice [y/N]: " dev_choice
else
    dev_choice=$INSTALL_DEV
    echo "Using INSTALL_DEV from environment: $dev_choice"
fi

if [[ "$dev_choice" =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install -r scripts/requirements-dev.txt
fi



echo "BioNeuralNet enviroment setup complete."