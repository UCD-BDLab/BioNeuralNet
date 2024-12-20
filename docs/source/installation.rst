Installation
============

**Requirements:** Python 3.7+ and optional R for SmCCNet/WGCNA.

1. **Python Installation via pip:**

   ```bash
   pip install bioneuralnet
   ```

   *This installs the latest stable release from PyPI, including all base dependencies.*

2. **Optional R Dependencies (For WGCNA/SmCCNet):**

   If you need R-based algorithms (WGCNA, SmCCNet), install R and relevant packages:

   ```r
   install.packages(c("dplyr", "SmCCNet", "WGCNA"))
   ```

3. **GPU Acceleration (Optional):**

   For GPU-enabled installations, ensure a compatible NVIDIA GPU and CUDA version are installed. 
   Refer to PyTorch and PyG documentation for details.

4. **Fast Installation with `fast-install.py`:**

   For an automated setup including R-based and development dependencies, run:

   ```bash
   git clone https://github.com/UCD-BDLab/BioNeuralNet.git
   cd BioNeuralNet/scripts
   python3 fast-install.py
   ```

   This script:
   - Creates and activates a virtual environment
   - Installs base, development, and optional dependencies (including R, if desired)
   - Sets up pre-commit hooks

After installation, proceed to the [Usage](usage) section for examples and workflows.
