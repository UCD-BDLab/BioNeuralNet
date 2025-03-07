Installation
============

BioNeuralNet supports Python 3.10 and 3.11 in this beta release. Follow the steps below to set up BioNeuralNet and its dependencies.

1. **Install BioNeuralNet via pip**:

   .. code-block:: bash

      pip install bioneuralnet

   This installs the core BioNeuralNet modules for GNN embeddings, subject representation,
   disease prediction (DPMON), and clustering.

2. **Install PyTorch and PyTorch Geometric** (Separately):

   BioNeuralNet relies on PyTorch and PyTorch Geometric for GNN operations:

   .. code-block:: bash

      pip install torch torchvision torchaudio
      pip install torch_geometric

   For GPU-accelerated builds or other configurations visit the official sites:

   - `PyTorch Installation Guide <https://pytorch.org/get-started/locally/>`_
   - `PyTorch Geometric Installation Guide <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_

   Select the appropriate build for your system (e.g., Stable, Linux, pip, Python, CPU).

   .. figure:: _static/pytorch.png
      :align: center
      :alt: PyTorch Installation

   .. figure:: _static/geometric.png
      :align: center
      :alt: PyTorch Geometric Installation

3. **(Optional) Install R and External Tools**:

   If you plan to use **WGCNA** or **SmCCNet** for network construction:

   - Install R from `The R Project <https://www.r-project.org/>`_.
      - Version 4.4.2 or higher is recommended.
   - Install the required R packages. Open R and run:

     .. code-block:: r

        if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
        install.packages(c("dplyr", "jsonlite"))
        BiocManager::install(c("impute", "preprocessCore", "GO.db", "AnnotationDbi"))
        install.packages("SmCCNet")
        install.packages("WGCNA")

4. **Additional Notes for External Tools**:

   Refer to the :doc:`external_tools/index`.

5. **Next Steps**:

   - Explore :doc:`tutorials/index` and :doc:`BioNeuralNet` for end-to-end workflows and examples.
