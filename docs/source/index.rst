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


Installation
------------

To install BioNeuralNet, simply run:

.. code-block:: bash

   pip install bioneuralnet

For additional installation details, see :doc:`installation`.


**BioNeuralNet Overview**
-------------------------
.. figure:: _static/BioNeuralNet.png
   :align: center
   :alt: BioNeuralNet Logo

   Embeddings form the core of BioNeuralNet, enabling a number of downstream applications.


**BioNeuralNet Core Features**
------------------------------

For an End-to-End example example of BioNeuralNet, see :doc:`BioNeuralNet`.

**Network Embedding**: :doc:`gnns`
   - Given a multi-omics network as input, BioNeuralNet can generate embeddings using Graph Neural Networks (GNNs).
   - Generate embeddings using methods such as **GCN**, **GAT**, **GraphSAGE**, and **GIN**.
   - Outputs can be obtained as native tensors or converted to pandas DataFrames for easy analysis and visualization.
   - Embeddings unlock numerous downstream applications, including disease prediction, enhanced subject representation, clustering, and more.

**Graph Clustering**: :doc:`clustering`
   - Identify functional modules or communities using **correlated clustering methods** (e.g., CorrelatedPageRank, CorrelatedLouvain, HybridLouvain) that integrate phenotype correlation to extract biologically relevant modules [1]_.
   - Clustering methods can be applied to any network represented allowing flexible analysis across different domains.
   - All clustering components return either raw partitions dictionaries or induced subnetwork adjacency matrices (as DataFrames) for visualization.
   - Use cases include, feature selection, biomarker discovery, and network-based analysis.

**Downstream Tasks**: :doc:`downstream_tasks`
   - **Subject Representation**:
      - Integrate node embeddings back into omics data to enrich subject-level profiles by weighting features with learned embedding.
      - This embedding-enriched data can be used for downstream tasks such as disease prediction or biomarker discovery.
      - The result can be returned as a DataFrame or a PyTorch tensor, fitting naturally into downstream analyses.

   - **Disease Prediction for Multi-Omics Network DPMON** [2]_:
      - Classification End-to-End pipeline for disease prediction using Graph Neural Network embeddings.
      - DPMON supports hyperparameter tuning-when enabled, it finds the best for the given data.
      - This approach, along with the native pandas integration across modules, ensures that BioNeuralNet can be easily incorporated into your analysis workflows.

**Metrics**: :doc:`metrics`
   - Several plotting funcctions to visualize networks, emebddings, variance distribution, cluster comparison, and more.
   - Correlation based functions to compare clustersand omics data with the phenotype.

**Utilities**: :doc:`utils`
   - **Filtering Functions**:
      - Network filtering allows users to select variance or zero-fraction filtering to an omics network.
      - Reducing noise, and removing outliers.
   
   - **Data Conversion**:
      - Convert RData files both CSV and to Pandas DataFrame. For ease of integration for R-based workflows.

**External Tools**: :doc:`external_tools/index`
   - **Graph Construction**:
      - BioNeuralNet provides additional tools in the `bioneuralnet.external_tools` module.
      - Allowing users to generate networks using R-based tools like WGCNA and SmCCNet.
      - While optional, these tools enhance BioNeuralNet's capabilities and are recommended for comprehensive analysis.

What is BioNeuralNet?
---------------------
BioNeuralNet is a **Python-based** framework designed to bridge the gap between **multi-omics data analysis** and **Graph Neural Networks (GNNs)**. By leveraging advanced techniques, it enables:

- **Graph Clustering**: Identifies biologically meaningful communities within omics networks.  
- **GNN Embeddings**: Learns network-based feature representations from biological graphs, capturing both **biological structure** and **feature correlations** for enhanced analysis.  
- **Subject Representation**: Generates high-quality embeddings for individuals based on multi-omics profiles.  
- **Disease Prediction**: Builds predictive models using integrated multi-layer biological networks.

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

   - **Not performed internally**: Generate the network adjacency matrix externally (e.g., using SmCCNet).
   - Lightweight wrappers (e.g., WGCNA, SmCCNet) are available in `bioneuralnet.external_tools` for convenience.

3. **Disease Prediction**:

   - Use **DPMON** to predict disease phenotypes by integrating the network information with omics data.
   - DPMON supports an end-to-end pipeline with hyperparameter tuning that can return predictions as pandas DataFrames, enabling seamless integration with existing workflows.

**Code Example**:

.. code-block:: python

   import pandas as pd
   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.downstream_task import DPMON

   # Step 1: Data Preparation
   phenotype_data = pd.read_csv('phenotype_data.csv')
   omics_proteins = pd.read_csv('omics_proteins.csv')
   omics_metabolites = pd.read_csv('omics_metabolites.csv')
   clinical_dt = pd.read_csv('clinical_data.csv')

   # Step 2: Network Construction
   smccnet = SmCCNet(
       phenotype_df=phenotype_data,
       omics_dfs=[omics_proteins, omics_metabolites],
       data_types=["protein", "metabolite"],
       kfold=5,
       summarization="PCA",
   )
   global_network, clusters = smccnet.run()
   print("Adjacency matrix generated.")

   # Step 3: Disease Prediction (DPMON)
   dpmon = DPMON(
       adjacency_matrix=global_network,
       omics_list=[omics_proteins, omics_metabolites],
       phenotype_data=phenotype_data,
       clinical_data=clinical_dt,
       model="GCN",
   )
   predictions = dpmon.run()
   print("Disease phenotype predictions:\n", predictions)


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   gnns
   clustering
   metrics
   utils
   downstream_tasks
   BioNeuralNet.ipynb
   TOPMED.ipynb
   tutorials/index
   external_tools/index
   user_api
   faq
   future


Indices and References
======================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [1] Abdel-Hafiz, M., Najafi, M., et al. "Significant Subgraph Detection in Multi-omics Networks for Disease Pathway Identification." *Frontiers in Big Data*, 5 (2022). DOI: `10.3389/fdata.2022.894632 <https://doi.org/10.3389/fdata.2022.894632>`_.
.. [2] Hussein, S., Ramos, V., et al. "Learning from Multi-Omics Networks to Enhance Disease Prediction: An Optimized Network Embedding and Fusion Approach." In *2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, Lisbon, Portugal, 2024, pp. 4371-4378. DOI: `10.1109/BIBM62325.2024.10822233 <https://doi.org/10.1109/BIBM62325.2024.10822233>`_.
