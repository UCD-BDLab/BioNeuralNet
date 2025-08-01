BioNeuralNet: Graph Neural Networks for Multi-Omics Network Analysis
====================================================================

.. image:: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg
   :target: https://creativecommons.org/licenses/by-nc-nd/4.0/

.. image:: https://img.shields.io/pypi/v/bioneuralnet
   :target: https://pypi.org/project/bioneuralnet/

.. image:: https://static.pepy.tech/badge/bioneuralnet
   :target: https://pepy.tech/project/bioneuralnet

.. image:: https://img.shields.io/badge/GitHub-View%20Code-blue
   :target: https://github.com/UCD-BDLab/BioNeuralNet

.. figure:: _static/LOGO_TB.png
   :align: center
   :alt: BioNeuralNet Logo


Installation
------------

BioNeuralNet is available as a Python package on PyPI:

.. code-block:: bash

   pip install bioneuralnet

For additional installation details and troubleshooting, see :doc:`installation`.

Quick Start Examples
--------------------

Get started quickly with these end-to-end examples demonstrating the BioNeuralNet workflow:

- :doc:`Quick_Start`
- :doc:`TCGA-BRCA_Dataset`

**BioNeuralNet Workflow Overview**
----------------------------------

.. figure:: _static/BioNeuralNet.png
   :align: center
   :alt: BioNeuralNet Workflow Overview

   `View BioNeuralNet Workflow. <https://bioneuralnet.readthedocs.io/en/latest/_images/BioNeuralNet.png>`_

What is BioNeuralNet?
---------------------

BioNeuralNet is a flexible, modular Python framework developed to facilitate end-to-end network-based multi-omics analysis using **Graph Neural Networks (GNNs)**. It addresses the complexities associated with multi-omics data, such as high dimensionality, sparsity, and intricate molecular interactions, by converting biological networks into meaningful, low-dimensional embeddings suitable for downstream tasks.

BioNeuralNet provides:

- **Network Construction**: Easily build informative networks from multi-omics datasets to capture biologically relevant molecular interactions.
- **GNN Embeddings**: Transform complex biological networks into versatile embeddings, capturing both structural relationships and molecular interactions.
- **Phenotype-Aware Analysis**: Integrate phenotype or clinical variables to enhance the biological relevance of the embeddings.
- **Disease Prediction**: Utilize network-derived embeddings for accurate and scalable predictive modeling of diseases and phenotypes.
- **Interoperability**: Outputs structured as **Pandas DataFrames**, ensuring compatibility with common Python tools and seamless integration into existing bioinformatics pipelines.

BioNeuralNet emphasizes usability, reproducibility, and adaptability, making advanced network-based multi-omics analyses accessible to researchers working in precision medicine and systems biology.

Why Graph Neural Networks for Multi-Omics?
------------------------------------------

Traditional machine learning methods often struggle with the complexity and high dimensionality of multi-omics data, particularly their inability to effectively capture intricate molecular interactions and dependencies. BioNeuralNet overcomes these limitations by using **graph neural networks (GNNs)**, which naturally encode biological structures and relationships.

BioNeuralNet supports several state-of-the-art GNN architectures optimized for biological applications:

- **Graph Convolutional Networks (GCN)**: Aggregate biological signals from neighboring molecules, effectively modeling local interactions such as gene co-expression or regulatory relationships.
- **Graph Attention Networks (GAT)**: Use attention mechanisms to dynamically prioritize important molecular interactions, highlighting the most biologically relevant connections.
- **GraphSAGE**: Facilitate inductive learning, enabling the model to generalize embeddings to previously unseen molecular data, thereby enhancing predictive power and scalability.
- **Graph Isomorphism Networks (GIN)**: Provide powerful and expressive graph embeddings, accurately distinguishing subtle differences in molecular interaction patterns.

For detailed explanations of BioNeuralNet's supported GNN architectures and their biological relevance, see :doc:`gnns`.

Example: Network-Based Multi-Omics Analysis for Disease Prediction
------------------------------------------------------------------

`View full-size image: Network-Based Multi-Omics Analysis for Disease Prediction <https://bioneuralnet.readthedocs.io/en/latest/_images/Overview.png>`_

.. figure:: _static/Overview.png
   :align: center
   :alt: BioNeuralNet's workflow for network-based multi-omics analysis

   **BioNeuralNet Workflow**: Network-Based Multi-Omics Analysis for Disease Prediction

Below is a concise example demonstrating the following key steps:

1. **Data Preparation**:
   
   - Load your multi-omics data (e.g., transcriptomics, proteomics) along with phenotype and clinical covariates.

2. **Network Construction**:
   
   - Here, we construct the multi-omics network using an external R package, **SmCCNet** [1]_.
   - BioNeuralNet provides convenient wrappers for external tools (like SmCCNet) through `bioneuralnet.external_tools`. Note: R must be installed for these wrappers.

3. **Disease Prediction with DPMON**:
   
   - **DPMON** [2]_ integrates omics data and network structures to predict disease phenotypes.
   - It provides an end-to-end pipeline, complete with built-in hyperparameter tuning, and outputs predictions directly as pandas DataFrames for easy interoperability.

**Example Usage**:

.. code-block:: python

   import pandas as pd
   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.downstream_task import DPMON
   from bioneuralnet.datasets import DatasetLoader

   # Step 1: Load the dataset and access individual omics modalities
   example = DatasetLoader("example1")
   omics_genes = example.data["X1"]
   omics_proteins = example.data["X2"]
   phenotype = example.data["Y"]
   clinical = example.data["clinical"]

   # Step 2: Network Construction with SmCCNet
   smccnet = SmCCNet(
       phenotype_df=phenotype,
       omics_dfs=[omics_genes, omics_proteins],
       data_types=["Genes", "Proteins"],
       kfold=5,
       summarization="PCA",
   )
   global_network, clusters = smccnet.run()
   print("Adjacency matrix generated.")

    # Step 3: Disease Prediction using DPMON
   dpmon = DPMON(
       adjacency_matrix=global_network,
       omics_list=[omics_genes, omics_proteins],
       phenotype_data=phenotype,
       clinical_data=clinical,
       model="GCN",
       repeat_num=5,
       tune=True,
       gpu=True, 
       cuda=0,
       output_dir="./output"
   )
   predictions, avg_accuracy = dpmon.run()
   print("Disease phenotype predictions:\n", predictions)

Explore BioNeuralNet's Documentation
------------------------------------

For detailed examples and tutorials, visit:

- :doc:`Quick_Start`: A quick walkthrough demonstrating the BioNeuralNet workflow from start to finish.
- :doc:`TCGA-BRCA_Dataset`: A detailed real-world example applying BioNeuralNet to breast cancer subtype prediction.

**Documentation Sections:**

- :doc:`gnns`: Overview of supported GNN architectures (GCN, GAT, GraphSAGE, GIN) and embedding generation.
- :doc:`clustering`: How to identify biologically relevant functional modules using correlated clustering methods.
- :doc:`downstream_tasks`: Performing downstream analyses such as subject representation and phenotype prediction (DPMON).
- :doc:`metrics`: Methods for visualization, quality evaluation, and performance benchmarking.
- :doc:`utils`: Tools for preprocessing, feature selection, network construction, and data summarization.
- :doc:`external_tools/index`: Integration of external methods, such as SmCCNet, for advanced network construction.
- :doc:`user_api`: Detailed API documentation for developers and advanced users.

Contributing to BioNeuralNet
----------------------------

We welcome contributions to BioNeuralNet! If you have ideas for new features, improvements, or bug fixes, please follow these steps:

- **Ways to contribute**:
   
   - Report issues or bugs on our `GitHub Issues page <https://github.com/UCD-BDLab/BioNeuralNet/issues>`_.
   - Suggest new features or improvements.
   - Share your experiences or use cases with the community.

- **Implementing new features**:

   - Fork the repo and create a feature branch `UCD-BDLab/BioNeuralNet <https://github.com/UCD-BDLab/BioNeuralNet>`_.
   - Add tests and documentation for new features.
   - Run the test suite and and pre-commit hooks before opening a Pull Request(PR).
   - A new PR should pass all tests and adhere to the project's coding standards.

.. code-block:: bash
   
   git clone https://github.com/UCD-BDLab/BioNeuralNet.git
   cd BioNeuralNet
   pip install -r requirements-dev.txt
   pre-commit install
   pytest --cov=bioneuralnet


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   gnns
   clustering
   metrics
   utils
   downstream_tasks
   datasets.ipynb
   Quick_Start.ipynb
   TCGA-BRCA_Dataset.ipynb
   tutorials/index
   external_tools/index
   user_api
   faq


Indices and References
======================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [1] Liu, W., Vu, T., Konigsberg, I. R., Pratte, K. A., Zhuang, Y., & Kechris, K. J. (2023). "Network-Based Integration of Multi-Omics Data for Biomarker Discovery and Phenotype Prediction." *Bioinformatics*, 39(5), btat204. DOI: `10.1093/bioinformatics/btat204 <https://doi.org/10.1093/bioinformatics/btat204>`_.
.. [2] Hussein, S., Ramos, V., et al. "Learning from Multi-Omics Networks to Enhance Disease Prediction: An Optimized Network Embedding and Fusion Approach." In *2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, Lisbon, Portugal, 2024, pp. 4371-4378. DOI: `10.1109/BIBM62325.2024.10822233 <https://doi.org/10.1109/BIBM62325.2024.10822233>`_.
