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

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.17503083.svg
   :target: https://doi.org/10.5281/zenodo.17503083

.. figure:: _static/logo_update.png
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

- :doc:`quick_start/index`
- :doc:`notebooks/index`

**BioNeuralNet Workflow Overview**
----------------------------------

.. figure:: _static/BioNeuralNet.png
   :align: center
   :alt: BioNeuralNet Workflow Overview

   `View BioNeuralNet Workflow. <https://bioneuralnet.readthedocs.io/en/latest/_images/BioNeuralNet.png>`_


What is BioNeuralNet?
---------------------

BioNeuralNet is a flexible and modular Python framework tailored for **end-to-end network-based multi-omics data analysis**. It leverages Graph Neural Networks (GNNs) to learn biologically meaningful low-dimensional representations from multi-omics networks, converting complex molecular interactions into versatile embeddings.

**Core Analytical Modules:**

- **Network Construction**: Build informative networks from raw tabular data using strategies like **Similarity**, **Correlation**, **Neighborhood-based**, or **Phenotype-driven** (e.g., SmCCNet) approaches. [1]_
- **Biomarker Discovery**: Identify biological modules and key molecular interactions that drive disease phenotypes.
- **Disease Prediction**: Implement end-to-end supervised disease classification using the **DPMON** (Disease Prediction using Multi-Omics Networks) module. [2]_
- **Subject Representation**: Generate enhanced subject-level embeddings for stratification and clustering.

**Visualizing Multi-Omics Networks**

BioNeuralNet allows you to inspect the topology of your constructed networks. The visualization below, from our **TCGA Lower Grade Glioma (LGG)** analysis, highlights a survival-associated module of highly correlated omics features identified by HybridLouvain.

.. figure:: _static/net_lgg.png
   :align: center
   :alt: Multi-Omics Network Visualization
   :width: 100%

   *Network visualization of a highly connected gene module identified in the TCGA-LGG dataset.*
   `See Network Full Size <https://bioneuralnet.readthedocs.io/en/latest/_images/net_lgg.png>`_

**Top Identified Biomarkers (Hub Omics)**

The table below lists the top hub features identified in the network above, ranked by their degree centrality.

.. list-table:: Omics with high degree
   :widths: 40 10 10 10
   :header-rows: 1
   :align: center

   * - Feature Name (Omic)
     - Index
     - Degree
     - Source
   * - HIVEP3
     - 20
     - 7
     - RNA
   * - DBH
     - 19
     - 7
     - RNA
   * - ERMP1
     - 8
     - 7
     - RNA
   * - LFNG
     - 12
     - 6
     - RNA
   * - MIR23A
     - 21
     - 6
     - miRNA
   * - THADA
     - 4
     - 6
     - RNA


Why Graph Neural Networks for Multi-Omics?
------------------------------------------

Traditional statistical methods typically represent multi-omics data as high-dimensional tabular matrices, often overlooking the intricate relationships and interactions between biomolecular entities. BioNeuralNet overcomes these limitations by using **Graph Neural Networks (GNNs)** to explicitly model multi-omics data as biological networks.

BioNeuralNet supports several GNN architectures suited to different biological contexts:

* **GCN**: Effective for uniformly connected graphs.
* **GAT**: Uses attention mechanisms to highlight key biological relationships.
* **GraphSAGE**: Designed for large or dynamic datasets.
* **GIN**: Sensitive to subtle feature variations in molecular structures.

**Network Embeddings**

By projecting high-dimensional omics networks into latent spaces, BioNeuralNet distills complex, nonlinear molecular relationships into compact vectorized representations. The t-SNE projection below reveals distinct clusters corresponding to different omics modalities (e.g., DNA Methylation, RNA, miRNA).

.. figure:: _static/emb_lgg.png
   :align: center
   :alt: t-SNE visualization of Network Embeddings
   :width: 100%

   *2D projection of Network Embeddings showing distinct separation between omics modalities.*
   `See Embeddings Full Size <https://bioneuralnet.readthedocs.io/en/latest/_images/emb_lgg.png>`_

For detailed explanations of BioNeuralNet's supported GNN architectures, see :doc:`gnns`.

Key Considerations for Robust Analysis
---------------------------------------

Multi-omics pipelines involve sequential decisions across data alignment, feature selection, network construction, and downstream modeling. Each stage shapes the one that follows. BioNeuralNet provides a structured **Data Decision Framework** to guide these choices with concrete parameter recommendations grounded in empirical results from TCGA and COPD workflows.

.. figure:: _static/UpdatedFlowChart.png
   :align: center
   :alt: BioNeuralNet Data-Driven Decision Flowchart
   :width: 100%

   *Data-driven decision flowchart for navigating the BioNeuralNet pipeline.*
   `See Flow Chart Full Size <https://bioneuralnet.readthedocs.io/en/latest/_images/UpdatedFlowChart.png>`_

For the full stage-by-stage parameter reference, see :doc:`quick_start/data_framework`.
For preprocessing utilities, see `Preprocessing Utilities <https://bioneuralnet.readthedocs.io/en/latest/utils.html#preprocessing-utilities>`_.
Per-cohort feature implementation details are available in the :doc:`notebooks/index`.

Explore BioNeuralNet's Documentation
------------------------------------

For detailed examples and tutorials, visit:

- :doc:`quick_start/index`: A series of walkthroughs demonstrating the BioNeuralNet workflow from start to finish.
- :doc:`notebooks/index`: A collection of demonstration notebooks showcasing end-to-end analyses on TCGA datasets.

**Documentation Sections:**

- :doc:`quick_start/index`: End-to-end notebook walkthrough using a synthetic demo dataset, covering network construction, subgraph detection, and disease prediction.
- :doc:`quick_start/data_framework`: Comprehensive stage-by-stage parameter reference and decision guide grounded in empirical results from TCGA and COPD workflows.
- :doc:`notebooks/index`: Demonstration notebooks showcasing end-to-end analyses on TCGA-BRCA, TCGA-LGG, TCGA-KIPAN, and ROSMAP datasets.
- :doc:`gnns`: Overview of supported GNN architectures (GCN, GAT, GraphSAGE, GIN) and embedding generation.
- :doc:`subgraph`: Phenotype-aware subgraph detection using CorrelatedLouvain, CorrelatedPageRank, and HybridLouvain, with TCGA-LGG and ROSMAP case studies.
- :doc:`network`: Tools for network construction, topology analysis, and automated network search.
- :doc:`downstream_tasks`: Downstream analysis pipelines including DPMON for phenotype prediction and SubjectRepresentation for patient-level profiling.
- :doc:`metrics`: Visualization, quality evaluation, and performance benchmarking utilities.
- :doc:`utils`: Data preprocessing, feature selection, imputation, normalization, and network pruning.
- :doc:`datasets`: Built-in multi-omics benchmark datasets (BRCA, LGG, KIPAN, ROSMAP) with cohort summaries and feature selection details.
- :doc:`external_tools/index`: Utility functions for interoperability with R-based tools including SmCCNet cross-validation fold export.
- :doc:`user_api`: Full API reference for developers and advanced users.

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

Citation
--------

If you use BioNeuralNet in your research, we kindly ask that you cite our paper:

   Ramos, V., Hussein, S., et al. (2025).
   `BioNeuralNet: A Graph Neural Network based Multi-Omics Network Data Analysis Tool <https://arxiv.org/abs/2507.20440>`_.
   *arXiv preprint arXiv:2507.20440* | `DOI: 10.48550/arXiv.2507.20440 <https://doi.org/10.48550/arXiv.2507.20440>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quick_start/index
   notebooks/index
   gnns
   subgraph
   metrics
   utils
   network
   downstream_tasks
   datasets
   examples/index
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