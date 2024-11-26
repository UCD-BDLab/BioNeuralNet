#!/bin/bash

# Automated setup script for BioNeuralNet

set -e 

echo "Starting BioNeuralNet setup..."

# Navigate to the project root
cd "$(dirname "$0")/.."

# Checking if Python 3.7+ is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python 3.7 or higher."
    exit
fi

# Step 2: Creating a virtual environment in the root directory
echo "Creating a virtual environment at the root (./.venv)..."
python3 -m venv .venv

# Step 3: Activate the virtual environment
echo "Activating the virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# Step 4: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 5: Installing base dependencies
echo "Installing base dependencies from requirements.txt at the root..."
pip install -r requirements.txt

# Step 6: Install system-specific dependencies
echo "Select installation type:"
echo "1. CPU-only"
echo "2. CUDA-enabled (e.g., CUDA 11.7)"
read -rp "Enter choice [1/2]: " choice

if [ "$choice" -eq 1 ]; then
    echo "Installing CPU-specific dependencies..."
    pip install -r scripts/requirements-cpu.txt
elif [ "$choice" -eq 2 ]; then
    echo "Installing CUDA-specific dependencies..."
    pip install -r scripts/requirements-cuda.txt
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Step 7: Installing development dependencies
echo "Do you want to install development dependencies? [y/N]"
read -rp "Enter choice [y/N]: " dev_choice

if [[ "$dev_choice" =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install -r scripts/requirements-dev.txt
fi

# Initialize the project using quick_start
echo "Initializing the project with quick_start..."
python -c "from bioneuralnet.utils.quick_start import quick_start; quick_start()"

echo "BioNeuralNet setup completed successfully!"
echo "To activate the virtual environment in the future, run:"
echo "source .venv/bin/activate"
