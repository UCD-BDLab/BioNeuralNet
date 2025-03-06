GNN Embeddings
==============

BioNeuralNet leverages Graph Neural Networks (GNNs) to generate rich, low-dimensional embeddings that capture the complex relationships inherent in multi-omics data. These embeddings not only preserve the network topology but also integrate biological signals, providing a robust foundation for downstream tasks such as disease prediction.

Key Contributions:
------------------
- **Enhanced Representation:** By training models such as GCN, GAT, GraphSAGE, and GIN, BioNeuralNet generates node embeddings that reflect both local connectivity and supervised signals each node (omics feature) is associated with a numeric label (e.g., Pearson correlation with phenotype) that guides learning.

- **Modularity and Interoperability:** The framework is designed in a modular, Python-based fashion. Its outputs are returned as pandas DataFrames, allowing seamless integration with existing data analysis pipelines and facilitating further exploration with external tools.

- **End-to-End Workflow:** Whether you start with raw multi-omics data or supply your own network, the pipeline proceeds through network construction, embedding generation, and ultimately disease prediction. Ensuring a streamlined workflow from data to actionable insights.

GNN Model Overviews
-------------------
**Graph Convolutional Network (GCN)**: GCN layers aggregate information from neighboring nodes via a spectral-based convolution:

.. math::

   X^{(l+1)} \;=\; \mathrm{ReLU}\!\Bigl(\widehat{D}^{-\tfrac{1}{2}}\,\widehat{A}\,\widehat{D}^{-\tfrac{1}{2}}\,
   X^{(l)}\,W^{(l)}\Bigr),

where :math:`\widehat{A}` adds self-loops to the adjacency matrix, ensuring that each node also considers its own features.

**Graph Attention Network (GAT)**: GAT layers learn attention weights to prioritize the most informative neighbors:

.. math::

   h_{i}^{(l+1)} \;=\; \mathrm{ELU}\!\Bigl(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)}\,W^{(l)}\,h_{j}^{(l)}\Bigr),

with :math:`\alpha_{ij}^{(l)}` representing the attention coefficient for node :math:`j`'s contribution to node :math:`i`.

**GraphSAGE**: GraphSAGE computes embeddings by concatenating a node's own features with an aggregated summary of its neighbors:

.. math::

   h_{i}^{(l+1)} \;=\; \sigma\!\Bigl(W^{(l)}\Bigl(
   h_{i}^{(l)} \,\|\, \mathrm{mean}_{j \,\in\, \mathcal{N}(i)}(h_{j}^{(l)})
   \Bigr)\Bigr),

where the mean aggregator provides a simple yet effective way to summarize local neighborhood information.

**Graph Isomorphism Network (GIN)**: GIN uses a sum-aggregator combined with a learnable parameter and an MLP to capture subtle differences in network structure:

.. math::

   h_i^{(l+1)} \;=\; \mathrm{MLP}^{(l)}\!\Bigl(\,\bigl(1 + \epsilon^{(l)}\bigr)
   h_{i}^{(l)} + \sum_{j \in \mathcal{N}(i)} h_{j}^{(l)}\Bigr),

where :math:`\epsilon^{(l)}` is either learnable or fixed.

Dimensionality Reduction and Downstream Integration
---------------------------------------------------

After obtaining high-dimensional node embeddings from the penultimate GNN layer, BioNeuralNet applies dimensionality reduction (using PCA or autoencoders) to summarize each node with a single value. These reduced embeddings are then integrated into subject-level omics data, yielding enhanced feature representations that boost the performance of predictive models (e.g., via DPMON for disease prediction).

By using GNNs to capture both structural and biological signals, BioNeuralNet delivers embeddings that truly reflect the complexity of multi-omics networks.

Task-Driven (Supervised/Semi-Supervised) GNNs
---------------------------------------------
In our work, the GNNs are primarily **task-driven**:

- **Node Labeling via Phenotype Correlation:** For each node, we compute the Pearson correlation between the omics data and phenotype (or clinical) data. This correlation serves as the target label during training.

- **Supervised Training Objective:** The GNN is trained to predict these correlation values using a Mean Squared Error (MSE) loss. This strategy aligns node embeddings with biological signals relevant to the disease phenotype.

- **Downstream Integration:** The learned node embeddings can be integrated into patient-level datasets for sample-level classification tasks. For example, **DPMON** (Disease Prediction using Multi-Omics Networks) leverages these embeddings in an end-to-end pipeline where the final objective is to classify disease outcomes.

Generating Low-Dimensional Embeddings for Multi-Omics
-----------------------------------------------------
The following figure illustrates an end-to-end workflow, from raw omics data to correlation-based node labeling, GNN-driven embedding generation, dimensionality reduction, and final integration into subject-level features:

.. figure:: _static/SubjectRepresentation.png
   :align: center
   :alt: Subject Representation Workflow

   A high-level overview of BioNeuralNet's process for creating enhanced subject-level representations. Nodes represent omics features, labeled by correlation to a phenotype; GNNs learn embeddings that are reduced (PCA/autoencoder) and then reintegrated into the original omics data for improved predictive performance.

`View full-size image: Subject Representation <https://bioneuralnet.readthedocs.io/en/latest/_images/SubjectRepresentation.png>`_

Key Insights into GNN Parameters and Outputs
--------------------------------------------
1. **Input Parameters:**

   - **Node Features Matrix:** Built by correlating omics data with clinical variables.
   
   - **Edge Index:** Derived from the network's adjacency matrix.
   
   - **Target Labels:** Numeric values representing the correlation between omics features and phenotype data.

2. **Output Embeddings:**

   - The penultimate layer of the GNN produces dense node embeddings that capture both local connectivity and supervised signals.
   
   - These embeddings can be further reduced (e.g., via PCA or an Autoencoder) for visualization or integrated into subject-level data.

Dimensionality Reduction: PCA vs. Autoencoders
----------------------------------------------

After training a GNN, the resulting node embeddings are typically high-dimensional. To integrate these embeddings into the original omics data-by reweighting each feature-a further reduction step is performed to obtain a single summary value per feature. BioNeuralNet supports two primary approaches for this reduction:

**Principal Component Analysis (PCA):**

PCA is a linear dimensionality reduction technique that computes orthogonal components capturing the maximum variance in the data. The first principal component (PC1) is often used as a concise summary of each feature's variation. PCA is:

- **Deterministic and Fast:** A closed-form solution is computed from the covariance matrix.

- **Simple and Interpretable:** The linear combination of the original variables is straightforward to understand.

- **Limited to Linear Relationships:** It may not capture more complex, nonlinear structures in the data.

**Autoencoders (AE):**  

Autoencoders are neural network models designed to learn a compressed representation (latent code) through a bottleneck architecture. They use nonlinear activations (e.g., ReLU) to model complex relationships:

- **Nonlinear Transformation:** The encoder learns to capture intricate patterns that a linear method might miss.

- **Learned Representations:** The latent code is obtained by minimizing a reconstruction loss, making it adaptive to the data.

- **Flexible and Tunable:** Being neural network-based, autoencoders allow tuning of architecture parameters (e.g., number of layers, hidden dimensions, epochs, learning rate) to better capture the signal. In our framework, we highly recommend using autoencoders (i.e., setting `tune=True`) to leverage their enhanced expressivity for complex multi-omics data.

In practice, PCA offers simplicity and interpretability, whereas autoencoders may yield superior performance by capturing more nuanced nonlinear relationships. The choice depends on the complexity of your data and the computational resources available. Our recommendation is to enable tuning (using `tune=True`) to optimize the autoencoder parameters for your specific dataset.

How DPMON Uses GNNs Differently
-------------------------------
**DPMON** (Disease Prediction using Multi-Omics Networks) reuses the same GNN architectures but with a different objective:

- Instead of node-level MSE regression, DPMON aggregates node embeddings with patient-level omics data.

- A downstream classification head (e.g., softmax layer with CrossEntropyLoss) is applied for sample-level disease prediction.

- This end-to-end approach leverages both local (node-level) and global (patient-level) network information.

.. figure:: _static/DPMON.png
   :align: center
   :alt: Disease Prediction (DPMON)

   Embedding-enhanced subject data using DPMON for improved disease prediction.

`View full-size image: Disease Prediction (DPMON) <https://bioneuralnet.readthedocs.io/en/latest/_images/DPMON.png>`_

Example Usage
-------------
Below is a simplified example that demonstrates the task-driven approach-where node labels are derived from phenotype correlations and used to train the GNN:

.. code-block:: python

   from bioneuralnet.network_embedding import GNNEmbedding
   import pandas as pd

   gnn = GNNEmbedding(
       adjacency_matrix=adjacency_matrix,
       omics_data=omics_data,
       phenotype_data=phenotype_data,
       clinical_data=clinical_data,
       phenotype_col='finalgold_visit',
       model_type='GAT',
       hidden_dim=64
   )
   gnn.fit()
   node_embeds = gnn.embed()

Return to :doc:`../index`
