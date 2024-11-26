Installation
============

To install BioNeuralNet, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/https://github.com/UCD-BDLab/BioNeuralNet.git
   cd bioneuralnet
   ```

2. **Run the Setup Script:**

   ```bash
   ./setup.sh
   ```

   The script will:
   - Create and activate a Python virtual environment.
   - Install base dependencies.
   - Prompt you to select between CPU-only or CUDA-enabled installations.
   - Optionally install development dependencies.
   - Initialize the project directories and configuration files.

3. **Activate the Virtual Environment (If Not Automatically Activated):**

   ```bash
   source venv/bin/activate
   ```

4. **Verify Installation:**

   ```bash
   python -c "import bioneuralnet; print(bioneuralnet.__version__)"
   ```

   This should print the version of BioNeuralNet installed.

**Note:** Ensure that you have Python 3.7 or higher installed on your system. For CUDA-enabled installations, make sure that the appropriate CUDA version is installed and compatible with your system's GPU drivers.

---