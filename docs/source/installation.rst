Installation
============

BioNeuralNet is fully compatible with Python 3.10 or higher and supports Windows, macOS, and Linux platforms. Follow these steps to install BioNeuralNet along with necessary dependencies.

1. **Install BioNeuralNet via pip**

   The core modules, including graph embeddings, disease prediction (DPMON), and clustering, can be installed directly:

   .. code-block:: bash

      pip install bioneuralnet

2. **Install PyTorch and PyTorch Geometric (Required)**

   BioNeuralNet utilizes PyTorch and PyTorch Geometric for graph neural network computations. Install these separately:

   .. code-block:: bash

      pip install torch
      pip install torch_geometric

   For GPU-enabled installations or advanced configurations, please refer to the official documentation:

   - `PyTorch Installation Guide <https://pytorch.org/get-started/locally/>`_
   - `PyTorch Geometric Installation Guide <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_

   Choose the appropriate build based on your system and GPU availability.

   .. figure:: _static/pytorch.png
      :align: center
      :alt: PyTorch Installation Guide Example

   .. figure:: _static/geometric.png
      :align: center
      :alt: PyTorch Geometric Installation Guide Example

3. **Optional: Install R and External Tools (e.g., SmCCNet)**

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

- :doc:`tutorials/index`
- :doc:`Quick_Start`
- :doc:`TCGA-BRCA_Dataset`
