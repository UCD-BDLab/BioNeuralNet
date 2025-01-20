Installation
============

BioNeuralNet supports Python 3.10 and 3.11 for this beta release.

1. **Install BioNeuralNet via pip**:

   .. code-block:: bash

      pip install bioneuralnet==0.1.0b1

   This installs BioNeuralNet’s Python modules for GNN embeddings, subject representation,
   disease prediction (DPMON), and clustering.

2. **Install PyTorch and Pytorch Geometric** (Separately):

   BioNeuralNet relies on PyTorch for GNN operations. Visit `the official PyTorch site <https://pytorch.org/get-started/locally/>`_
   and `the official PyTorch Geometric site <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ to install CPU or GPU builds. For example:

   .. code-block:: bash

      pip install torch torchvision torchaudio
      pip install torch_geometric

   or refer to the site for GPU-accelerated builds matching your CUDA version.

3. **(Optional) R and External Tools**:

   - If you plan to use **WGCNA** or **SmCCNet** for network construction:
     1. Install R from `The R Project <https://www.r-project.org/>`_
     2. In R, install packages:

        .. code-block:: r

           install.packages("WGCNA")

   - For Node2Vec, feature selection modules, or visualization, see :doc:`external_tools/index`.

4. **Verification**:

   After installation, verify that `import bioneuralnet` and `import torch` both run
   without errors in a Python shell. Optionally, run tests if you’ve cloned the repository:

   .. code-block:: bash

      git clone https://github.com/UCD-BDLab/BioNeuralNet.git
      cd BioNeuralNet
      pip install -r requirements-dev.txt
      pytest

   This ensures all dependencies are properly set up.

5. **Next Steps**:

   - Explore :doc:`tutorials/index` for end-to-end workflows
   - See :doc:`tools/index` for embedding, clustering, and subject representation methods
   - Check :doc:`external_tools/index` if you want to use WGCNA, SmCCNet, Node2Vec, or
     other external utilities
