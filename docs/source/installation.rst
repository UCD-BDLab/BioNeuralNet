Installation
============

**Requirements:** Python 3.10 and 3.11 have been tested with this beta release.

1. **Python Installation via pip:**

   .. code-block:: bash

      pip install bioneuralnet==0.1.0b1

2. **Dependencies Installation with `fast-install.py`:**

   BioNeuralNet uses a variety of Python and R packages for data processing and graph generation.
   We have provided a quick way to install these dependencies using a script.

   Download options:

     - `Download zip content <https://github.com/UCD-BDLab/BioNeuralNet/archive/refs/heads/main.zip>`_
     - Clone or download from the `BioNeuralNet GitHub repository <https://github.com/UCD-BDLab/BioNeuralNet>`_

   .. figure:: _static/repo.png
      :align: center
      :alt: Repository Screenshot

   **Continue from .zip download:**

   - Extract the contents of the .zip file to a folder.
     - In the example below, the folder is named 'tutorial'.
   - Open a terminal window and navigate to the folder.
   - Run the following command from the root directory:

   .. code-block:: bash

      cd tutorial/BioNeuralNet-main
      python3 scripts/fast-install.py

   .. figure:: _static/terminal.png
      :align: center
      :alt: Terminal window during installation

   **fast-install.py Options:**

   The script offers a few options depending on your system and requirements:

   - **GPU-acceleration:**
      - BioNeuralNet supports GPU-accelerated training with CUDA.
      - We support all stable CUDA versions (11.8, 12.1, 12.4) as detailed in the `PyTorch documentation <https://pytorch.org/get-started/locally/>`_.
      - First, check your installed CUDA version with:

        .. code-block:: bash

           nvcc --version

      - Pass the version number when using the script:

        .. code-block:: bash

           python3 scripts/fast-install.py --cuda --cuda-version 12.1

   - **Development dependencies:**
     - For testing, documentation, and contributing, you can install development dependencies:

       .. code-block:: bash

          python3 scripts/fast-install.py --dev

   - **Combined Installation:**
     - Install both CUDA and development dependencies with:

       .. code-block:: bash

          python3 scripts/fast-install.py --cuda --cuda-version 11.8 --dev

   **Command Examples:**

   - Default Installation (CPU-only, no dev dependencies):

     .. code-block:: bash

        python fast-install.py

   - Install Development Dependencies:

     .. code-block:: bash

        python fast-install.py --dev

   - Install CUDA-enabled PyTorch (Default CUDA Version Detection):

     .. code-block:: bash

        python fast-install.py --cuda

   - Install Specific CUDA Version with Development Dependencies:

     .. code-block:: bash

        python fast-install.py --cuda --cuda-version 12.1 --dev

   - Install Both CUDA and Development Dependencies with a Supported CUDA Version:

     .. code-block:: bash

        python fast-install.py --cuda --cuda-version 11.8 --dev

   If you cloned the repository:

   .. code-block:: bash

      git clone https://github.com/UCD-BDLab/BioNeuralNet.git
      cd BioNeuralNet
      python3 scripts/fast-install.py

   *Follow the same steps as above.*

3. **Success Message:**

   If you see the following message, the installation was successful:

   .. figure:: _static/terminal2.png
      :align: center
      :alt: Terminal window prompts
   
   You can now activate your virtual environment and start using BioNeuralNet:

   .. code-block:: bash

      source ./bioneuralnet-env/bin/activate

   See the :doc:`tutorials <tutorials/index>` or :doc:`tools <tools/index>` sections for examples.

4. **Running tests:**
   
   If you install development dependencies, you can run tests with the following command from the root directory:

   .. code-block:: bash

      pytest tests/

   - You should see the following output if installation was successful:

   .. figure:: _static/tests.png
      :align: center
      :alt: Pytest running tests

   This script:
   - Creates and activates a virtual environment.
   - Installs base, development, and R dependencies for graph generation.

After installation, proceed to the :doc:`tutorials <tutorials/index>` or :doc:`tools <tools/index>` sections for code examples.
