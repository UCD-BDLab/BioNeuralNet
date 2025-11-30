Installation
============

BioNeuralNet is fully compatible with Python 3.10, 3.11 and 3.12 and supports Windows, macOS, and Linux platforms. Follow these steps to install BioNeuralNet along with necessary dependencies.

Recommended System Requirements
-------------------------------

.. list-table:: Recommended system requirements for running BioNeuralNet.
   :header-rows: 1

   * - Component
     - Recommendation
   * - Operating System
     - Linux, macOS, or Windows
   * - Python Version
     - 3.10 - 3.12
   * - Processor (CPU)
     - Modern multi-core CPU (Intel i5/i7, AMD Ryzen 5/7); for large datasets/models: high-core CPUs like Intel Xeon or AMD EPYC
   * - Memory (RAM)
     - Minimum 8GB; 16GB recommended for standard workflows; 32GB+ for large-scale models
   * - Storage
     - Minimum 64GB for system and software; SSD recommended; 500GB+ for large datasets
   * - GPU (Optional but recommended)
     - NVIDIA GPU with `>=` 6GB VRAM or Intel Arc GPU for faster training; ensure drivers and CUDA/cuDNN versions match PyTorch
   * - Notes
     - GPU acceleration significantly improves training speed; hardware requirements increase with model and dataset size

1. **Install BioNeuralNet via pip**

   The core modules, including graph embeddings, disease prediction (DPMON), and clustering, can be installed directly:

   .. code-block:: bash

      pip install bioneuralnet

   This will also install the required Ray components for hyperparameter tuning (Ray Tune and Ray Train) if your environment honors package extras.

2. **Install PyTorch and PyTorch Geometric (Required for GNNs and DPMON)**

   BioNeuralNet utilizes PyTorch and PyTorch Geometric for graph neural network computations (e.g., GNN embeddings and the DPMON model). These are **not** pinned automatically because GPU/CPU builds and CUDA versions vary by system, so you should install them separately.

   Basic installation (CPU-only or default build):

   .. code-block:: bash

      pip install torch
      pip install torch_geometric

   We recommend the following tested versions for DPMON and GNN-based workflows:

   .. code-block:: bash

      pip install "torch==2.7.0+cu118"
      pip install "torch_geometric==2.6.1"

   For GPU-enabled installations or advanced configurations, please refer to the official documentation and choose the appropriate build based on your system and GPU availability:

   - `PyTorch Installation Guide <https://pytorch.org/get-started/locally/>`_
   - `PyTorch Geometric Installation Guide <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_

   .. figure:: _static/pytorch.png
      :align: center
      :alt: PyTorch Installation Guide Example

   .. figure:: _static/geometric.png
      :align: center
      :alt: PyTorch Geometric Installation Guide Example

3. **Ray for Hyperparameter Tuning (DPMON and Other HPO Workflows)**

   BioNeuralNet uses `Ray Tune` and `Ray Train` for hyperparameter optimization and checkpointing in components such as DPMON.

   If you installed BioNeuralNet via pip in a standard environment, Ray with the necessary extras should be installed automatically. For reproducibility or if your environment strips extras, you can explicitly install the tested Ray version:

   .. code-block:: bash

      pip install "ray[tune,train]==2.46.0"

   This ensures compatibility with the hyperparameter tuning utilities used throughout the library.

4. **Core Python Dependencies**

   BioNeuralNet is built on a standard scientific Python stack. While the installer handles these automatically, we are transparent about the specific versions used in our testing environment:

   .. code-block:: text

      pandas = 2.2.3
      numpy = 1.26.4
      scipy = 1.13.1
      matplotlib = 3.10.3
      scikit-learn = 1.6.1
      statsmodels = 0.14.4
      networkx = 3.4.2
      python-louvain = 0.16
      pydantic >= 2.5
      ray[tune,train] = 2.46.0

   *Note: The installer is flexible. It will attempt to install these tested versions (or compatible newer releases) to ensure stability, but will not strictly force them if your environment requires slight deviations.*

5. **Optional: Install R and External Tools (e.g., SmCCNet)**

   For advanced network construction using external R tools like **SmCCNet**, follow these additional steps:

   - Install R (version 4.4.2 or newer recommended) from the `R Project <https://www.r-project.org/>`_.
   - Within R, install the required packages:

     .. code-block:: r

        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")
        
        install.packages(c("dplyr", "jsonlite"))
        BiocManager::install(c("impute", "preprocessCore", "GO.db", "AnnotationDbi"))
        install.packages("SmCCNet")
        install.packages("WGCNA")

   See :doc:`external_tools/index` for further details on external tools.


Next Steps
----------

After installation, explore our step-by-step tutorials:

- :doc:`notebooks/index`
- :doc:`Quick_Start`
