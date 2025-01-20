.. BioNeuralNet documentation master file

Welcome to BioNeuralNet Beta 0.1
================================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/UCD-BDLab/BioNeuralNet/blob/main/LICENSE
.. image:: https://img.shields.io/pypi/v/bioneuralnet
   :target: https://pypi.org/project/bioneuralnet/
.. image:: https://img.shields.io/pypi/pyversions/bioneuralnet
   :target: https://pypi.org/project/bioneuralnet/

.. figure:: _static/LOGO_WB.png
   :align: center
   :alt: bioneuralnet logo

**Note:** This is a **beta version** of BioNeuralNet. It is under active development, and certain features
may be incomplete or subject to change. Feedback and bug reports are highly encouraged to help us
improve the tool.

BioNeuralNet is a Python-based software tool designed to streamline the integration of multi-omics
data with **Graph Neural Network (GNN)** embeddings. It supports **graph clustering**, **subject representation**,
and **disease prediction**, enabling advanced analyses of complex multi-omics networks.

**Python Installation via pip**:

   .. code-block:: bash

      pip install bioneuralnet==0.1.0b1

For additional installation details, including GPU usage for GNNs, see :doc:`installation`.

.. note::

   **External Tools**:

   - We offer a number of external tools available through the `bioneuralnet.external_tools` module.
   - These tools were implemented to facilitate testing, and should not be considered part of the package's core functionality.
   - The classes inside the `external_tools` module are lightweight wrappers around existing tools and libraries offering minimal functionality.
   - We highly encourage users to explore these tools outside of BioNeuralNet to fully leverage their capabilities.

**Example: Transforming Multi-Omics for Enhanced Disease Prediction**
---------------------------------------------------------------------

`View full-size image: Transforming Multi-Omics for Enhanced Disease Prediction <https://ramosv.github.io/_images/Overview.png>`_

.. figure:: _static/Overview.png
   :align: center
   :alt: Overview of BioNeuralNet's multi-omics integration process

   **BioNeuralNet**: Transforming Multi-Omics for Enhanced Disease Prediction

Below is a quick example demonstrating the following steps:

1. **Building or Importing a Network Adjacency Matrix**:

   - For instance, using external tools like **SmCCNet**.

2. **Using DPMON for Disease Prediction**:

   - A detailed explanation follows.

**Steps:**

1. **Data Preparation**:

   - Input your multi-omics data (e.g., proteomics, metabolomics) along with phenotype and clinical data.

2. **Network Construction**:

   - **Not performed internally**: You need to generate the adjacency matrix externally, using tools like **SmCCNet**.
   - Lightweight wrappers are available in `bioneuralnet.external_tools` (e.g., WGCNA, SmCCNet) for convenience.

3. **Disease Prediction**:
   - Use **DPMON** to predict disease phenotypes using the network information and omics data.


**Code Example**:

.. code-block:: python

   import pandas as pd
   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.downstream_task import DPMON

   # Step 1: Data Preparation
   phenotype_data = pd.read_csv('phenotype_data.csv', index_col=0)
   omics_proteins = pd.read_csv('omics_proteins.csv', index_col=0)
   omics_metabolites = pd.read_csv('omics_metabolites.csv', index_col=0)
   clinical_data = pd.read_csv('clinical_data.csv', index_col=0)

   # Step 2: Network Construction
   smccnet = SmCCNet(
       phenotype_df=phenotype_data,
       omics_dfs=[omics_proteins, omics_metabolites],
       data_types=["protein", "metabolite"],
       kfold=5,
       summarization="PCA",
   )
   adjacency_matrix = smccnet.run()
   print("Adjacency matrix generated.")

   # Step 3: Disease Prediction
   dpmon = DPMON(
       adjacency_matrix=adjacency_matrix,
       omics_list=[omics_proteins, omics_metabolites],
       phenotype_data=phenotype_data,
       clinical_data=clinical_data,
       model="GCN",
   )
   predictions = dpmon.run()
   print("Disease phenotype predictions:\n", predictions)


**Output**:
  - **Adjacency Matrix**: The multi-omics network representation.
  - **Predictions**: Disease phenotype predictions for each sample.

**BioNeuralNet Overview: Multi-Omics Integration with Graph Neural Networks**
-----------------------------------------------------------------------------

BioNeuralNet offers five core steps in a typical workflow:

1. **Graph Construction**:

   - **Not** performed internally. You provide or build adjacency matrices externally (e.g., via WGCNA, SmCCNet, or your own scripts).
   - Lightweight wrappers are available in `bioneuralnet.external_tools` (e.g., WGCNA, SmCCNet) for convenience, but they are **not** mandatory for BioNeuralNet’s pipeline.

2. **Graph Clustering**:

   - Identify functional modules or communities using `PageRank`.
   - The `PageRank` module enables finding subnetwork clusters through personalized sweep cuts, capturing local neighborhoods influenced by seed nodes.

3. **Network Embedding**:

   - Generate embeddings with methods like **GCN**, **GAT**, **GraphSAGE**, and **GIN**.
   - You can attach numeric labels to nodes or remain "unsupervised," relying solely on graph structure and node features (e.g., correlation with clinical data).

4. **Subject Representation**:

   - Integrate node embeddings back into omics data to enrich each subject’s feature vector by weighting columns with the learned embedding scalars.

5. **Downstream Tasks**:

   - Perform advanced analyses such as disease prediction via **DPMON**, which trains a GNN end-to-end alongside a classifier, incorporating both local and global network information.

`View full-size image: BioNeuralNet Overview <https://ramosv.github.io/_images/BioNeuralNet.png>`_

.. figure:: _static/BioNeuralNet.png
   :align: center
   :alt: BioNeuralNet

   BioNeuralNet Overview

**Subject Representation**

Subject representation integrates node embeddings into omics data, enriching phenotypic and clinical context. Below is an example workflow:

`View full-size image: Subject Representation <https://ramosv.github.io/_images/SubjectRepresentation.png>`_

.. figure:: _static/SubjectRepresentation.png
   :align: center
   :alt: Subject Representation Workflow

   Subject-level embeddings provide richer phenotypic and clinical context.

**Disease Prediction**

`View full-size image: Disease Prediction (DPMON) <https://ramosv.github.io/_images/DPMON.png>`_

.. figure:: _static/DPMON.png
   :align: center
   :alt: Disease Prediction (DPMON)

   Embedding-enhanced subject data using DPMON for improved disease prediction.

Documentation Overview
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Documentation Contents:

   installation
   tutorials/index
   tools/index
   external_tools/index
   gnns
   user_api
   faq


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
