BioNeuralNet - Multi-Omics Integration with Graph Neural Networks
=================================================================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE

.. image:: https://img.shields.io/pypi/v/bioneuralnet
   :target: https://pypi.org/project/bioneuralnet/

.. image:: https://static.pepy.tech/badge/bioneuralnet
   :target: https://pepy.tech/project/bioneuralnet

.. image:: https://img.shields.io/badge/GitHub-View%20Code-blue
   :target: https://github.com/UCD-BDLab/BioNeuralNet

.. figure:: _static/LOGO_WB.png
   :align: center
   :alt: BioNeuralNet Logo


:doc:`installation`
-------------------

To install BioNeuralNet via pip:

.. code-block:: bash

   pip install bioneuralnet

For additional installation details, see :doc:`installation`.

For end-to-end examples of `BioNeuralNet`: 

   - :doc:`Quick_Start`.
   - :doc:`TCGA-BRCA_Dataset`.

**BioNeuralNet Overview**
-------------------------
.. figure:: _static/BioNeuralNet.png
   :align: center
   :alt: BioNeuralNet Logo

   Embeddings form the core of BioNeuralNet, enabling a number of downstream applications.


What is BioNeuralNet?
---------------------
BioNeuralNet is a **Python-based** framework designed to bridge the gap between **multi-omics data analysis** and **Graph Neural Networks (GNNs)**. By leveraging advanced techniques, it enables:

- **Graph Clustering**: Identifies biologically meaningful communities within omics networks.  
- **GNN Embeddings**: Learns network-based feature representations from biological graphs, capturing both **biological structure** and **feature correlations** for enhanced analysis.  
- **Subject Representation**: Generates high-quality embeddings for individuals based on multi-omics profiles.  
- **Disease Prediction**: Builds predictive models using integrated multi-layer biological networks.
- **Interoperability**: Component outputs are structured as **pandas DataFrames**, ensuring easy integration with existing workflows and tools.

Why GNNs?
---------
Traditional methods often struggle to model complex multi-omics relationships due to their inability to capture **biological interactions and dependencies**. BioNeuralNet addresses this challenge by utilizing **GNN-powered embeddings**, incorporating models such as:

- **Graph Convolutional Networks (GCN)**: Aggregates features from neighboring nodes to capture local structure.  
- **Graph Attention Networks (GAT)**: Applies attention mechanisms to prioritize important interactions between biomolecules.  
- **GraphSAGE**: Enables inductive learning, making it applicable to unseen omics data.  
- **Graph Isomorphism Networks (GIN)**: Improves expressiveness in graph-based learning tasks.  

By integrating omics features within a **network-aware framework**, BioNeuralNet preserves biological interactions, leading to **more accurate and interpretable predictions**.
For a deeper dive into how BioNeuralNet applies GNN embeddings, see :doc:`gnns`.

Seamless Data Integration
-------------------------
One of BioNeuralNet's core strengths is **interoperability**:

- Outputs are structured as **pandas DataFrames**, ensuring easy downstream analysis.  
- Supports integration with **external tools and machine learning frameworks**, making it adaptable to various research workflows.  
- Works seamlessly with network-based and graph-learning pipelines.
- Our :doc:`user_api` provides detailed information on how to use BioNeuralNet's modules and functions.


**Example: Transforming Multi-Omics for Enhanced Disease Prediction**
---------------------------------------------------------------------

`View full-size image: Transforming Multi-Omics for Enhanced Disease Prediction <https://bioneuralnet.readthedocs.io/en/latest/_images/Overview.png>`_

.. figure:: _static/Overview.png
   :align: center
   :alt: Overview of BioNeuralNet's multi-omics integration process

   **BioNeuralNet**: Transforming Multi-Omics for Enhanced Disease Prediction

Below is a quick example demonstrating the following steps:

1. **Data Preparation**:

   - Input your multi-omics data (e.g., proteomics, metabolomics) along with phenotype and clinical data.

2. **Network Construction**:

   - In this example we generate the network using a external R package (SmCCNet[3]_).
   - Lightweight wrappers (SmCCNet) are available in `bioneuralnet.external_tools` for convenience, R is required for their usage.

3. **Disease Prediction**:

   - Use **DPMON** to predict disease phenotypes by integrating the network information with omics data.
   - DPMON supports an end-to-end pipeline with hyperparameter tuning that can return predictions as pandas DataFrames, enabling seamless integration with existing workflows.

**Code Example**:

.. code-block:: python

   import pandas as pd
   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.downstream_task import DPMON
   from bioneuralnet.datasets import DatasetLoader

   # Step 1: Load your data or use one of the provided datasets
   Example = DatasetLoader("example1")
   omics_genes = Example.data["X1"]
   omics_proteins = Example.data["X2"]
   phenotype = Example.data["Y"]
   clinical = Example.data["clinical_data"]

   # Step 2: Network Construction
   smccnet = SmCCNet(
       phenotype_df=phenotype,
       omics_dfs=[omics_genes, omics_proteins],
       data_types=["Genes", "Proteins"],
       kfold=5,
       summarization="PCA",
   )
   global_network, clusters = smccnet.run()
   print("Adjacency matrix generated.")

   # Step 3: Disease Prediction (DPMON)
   dpmon = DPMON(
       adjacency_matrix=global_network,
       omics_list=[omics_genes, omics_proteins],
       phenotype_data=phenotype,
       clinical_data=clinical,
       model="GCN",
   )
   predictions, avg_accuracy = dpmon.run()
   print("Disease phenotype predictions:\n", predictions)

**BioNeuralNet Core Features**
------------------------------

For an End-to-End example example of BioNeuralNet, see :doc:`Quick_Start` and :doc:`TCGA-BRCA_Dataset`.

:doc:`gnns`:
   - Given a multi-omics network as input, BioNeuralNet can generate embeddings using Graph Neural Networks (GNNs).
   - Generate embeddings using methods such as **GCN**, **GAT**, **GraphSAGE**, and **GIN**.
   - Outputs can be obtained as native tensors or converted to pandas DataFrames for easy analysis and visualization.
   - Embeddings unlock numerous downstream applications, including disease prediction, enhanced subject representation, clustering, and more.

:doc:`clustering`:
   - Identify functional modules or communities using **correlated clustering methods** (e.g., CorrelatedPageRank, CorrelatedLouvain, HybridLouvain) that integrate phenotype correlation to extract biologically relevant modules [1]_.
   - Clustering methods can be applied to any network represented allowing flexible analysis across different domains.
   - All clustering components return either raw partitions dictionaries or induced subnetwork adjacency matrices (as DataFrames) for visualization.
   - Use cases include, feature selection, biomarker discovery, and network-based analysis.

:doc:`downstream_tasks`:
   - **Subject Representation**:
      - Integrate node embeddings back into omics data to enrich subject-level profiles by weighting features with learned embedding.
      - This embedding-enriched data can be used for downstream tasks such as disease prediction or biomarker discovery.
      - The result can be returned as a DataFrame or a PyTorch tensor, fitting naturally into downstream analyses.

   - **Disease Prediction for Multi-Omics Network DPMON** [2]_:
      - Classification End-to-End pipeline for disease prediction using Graph Neural Network embeddings.
      - DPMON supports hyperparameter tuning-when enabled, it finds the best for the given data.
      - This approach, along with the native pandas integration across modules, ensures that BioNeuralNet can be easily incorporated into your analysis workflows.

:doc:`metrics`:
   - Visualize embeddings, feature variance, clustering comparison, and network structure in 2D.  
   - Evaluate embedding quality and clustering relevance using correlation with phenotype.  
   - Performance benchmarking tools for classification tasks using various models.  
   - Useful for assessing feature importance, validating network structure, and comparing cluster outputs.

:doc:`utils`:
   - Build graphs using k-NN similarity, Pearson/Spearman correlation, RBF kernels, mutual information, or soft-thresholding.  
   - Filter and preprocess omics or clinical data by variance, correlation, random forest importance, or ANOVA F-test.  
   - Tools for network pruning, feature selection, and data cleaning.  
   - Quickly summarize datasets with variance, zero-fraction, expression level, or correlation overviews.  
   - Includes conversion tools for RData and integrated logging.

:doc:`external_tools/index`:
   - **Graph Construction**:
      - BioNeuralNet provides additional tools in the `bioneuralnet.external_tools` module.
      - Includes support for **SmCCNet** (Sparse Multiple Canonical Correlation Network), an R-based tool for constructing phenotype-informed correlation networks [3]_.
      - These tools are optional but enhance BioNeuralNet's graph construction capabilities and are recommended for more integrative or exploratory workflows.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   gnns
   clustering
   utils
   metrics
   downstream_tasks
   Quick_Start.ipynb
   TCGA-BRCA_Dataset.ipynb
   TOPMED.ipynb
   tutorials/index
   external_tools/index
   user_api
   faq


Indices and References
======================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [1] Abdel-Hafiz, M., Najafi, M., et al. "Significant Subgraph Detection in Multi-omics Networks for Disease Pathway Identification." *Frontiers in Big Data*, 5 (2022). DOI: `10.3389/fdata.2022.894632 <https://doi.org/10.3389/fdata.2022.894632>`_.
.. [2] Hussein, S., Ramos, V., et al. "Learning from Multi-Omics Networks to Enhance Disease Prediction: An Optimized Network Embedding and Fusion Approach." In *2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, Lisbon, Portugal, 2024, pp. 4371-4378. DOI: `10.1109/BIBM62325.2024.10822233 <https://doi.org/10.1109/BIBM62325.2024.10822233>`_.
.. [3] Liu, W., Vu, T., Konigsberg, I. R., Pratte, K. A., Zhuang, Y., & Kechris, K. J. (2023). "Network-Based Integration of Multi-Omics Data for Biomarker Discovery and Phenotype Prediction." *Bioinformatics*, 39(5), btat204. DOI: `10.1093/bioinformatics/btat204 <https://doi.org/10.1093/bioinformatics/btat204>`_.
