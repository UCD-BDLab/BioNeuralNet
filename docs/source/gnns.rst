GNN Embeddings for Multi-Omics
==============================

This section provides an overview of **Graph Neural Networks (GNNs)** in BioNeuralNet for multi-omics data—focusing on task-driven, supervised or semi-supervised approaches. In our framework, each node (representing an omics feature such as a gene or protein) is assigned a numeric label computed as the Pearson correlation between its expression values and the phenotype (or clinical variable) of interest. This supervision guides the training of our GNNs (GCN, GAT, GraphSAGE, and GIN) to learn embeddings that capture biologically meaningful signals for downstream tasks such as disease prediction.

.. contents::
   :local:
   :depth: 2

Graph Convolutional Network (GCN)
---------------------------------
GCN layers apply a spectral-based convolution operator to aggregate information from a node’s neighbors. The core update equation for a single GCN layer is:

.. math::

   X^{(l+1)} \;=\; \mathrm{ReLU}\!\Bigl(\widehat{D}^{-\tfrac{1}{2}}\,\widehat{A}\,\widehat{D}^{-\tfrac{1}{2}}\,
   X^{(l)}\,W^{(l)}\Bigr),

where

- :math:`X^{(l)}` is the node feature matrix at layer :math:`l`.
- :math:`\widehat{A} = A + I` is the adjacency matrix with self-loops.
- :math:`\widehat{D}` is the diagonal degree matrix of :math:`\widehat{A}`.
- :math:`W^{(l)}` denotes the trainable parameters at layer :math:`l`.
- :math:`\mathrm{ReLU}` is the activation function.

Graph Attention Network (GAT)
-----------------------------
GAT layers incorporate learned attention coefficients to weight the contributions of neighboring nodes:

.. math::

   h_{i}^{(l+1)} \;=\; \mathrm{ELU}\!\Bigl(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)}\,W^{(l)}\,h_{j}^{(l)}\Bigr),

where

- :math:`h_{i}^{(l)}` is the embedding of node :math:`i` at layer :math:`l`.
- :math:`\alpha_{ij}^{(l)}` is the learned attention coefficient for the edge between nodes :math:`i` and :math:`j`.
- :math:`W^{(l)}` is a trainable linear transformation.
- :math:`\mathrm{ELU}` is the exponential linear unit activation.
- :math:`\mathcal{N}(i)` denotes the neighbors of node :math:`i`.

GraphSAGE
---------
GraphSAGE (SAmple and aggreGatE) computes node embeddings by concatenating a node’s own features with an aggregated summary (e.g., mean) of its neighbors’ features:

.. math::

   h_{i}^{(l+1)} \;=\; \sigma\!\Bigl(
       W^{(l)}
       \bigl(\,
         h_{i}^{(l)} \,\|\, \mathrm{mean}_{j \,\in\, \mathcal{N}(i)}(h_{j}^{(l)})
       \bigr)\Bigr),

where

- :math:`\|` denotes vector concatenation.
- :math:`W^{(l)}` is a trainable weight matrix.
- :math:`\sigma` is a nonlinear activation function (e.g., ReLU).

Graph Isomorphism Network (GIN)
-------------------------------
GIN employs a sum-aggregator combined with a learnable :math:`\epsilon` parameter and a multi-layer perceptron (MLP) to update node representations:

.. math::

   h_i^{(l+1)} \;=\; \mathrm{MLP}^{(l)}\!\Bigl(\,\bigl(1 + \epsilon^{(l)}\bigr)\,
   h_{i}^{(l)} \;+\; \sum_{j \in \mathcal{N}(i)} h_{j}^{(l)}\Bigr),

where

- :math:`\epsilon^{(l)}` is a learnable (or fixed) parameter.
- :math:`\mathrm{MLP}^{(l)}` denotes a multi-layer perceptron.

Task-Driven (Supervised/Semi-Supervised) GNNs
---------------------------------------------
In our work, the GNNs are primarily **task-driven**:

- **Node Labeling via Phenotype Correlation:**  
  For each node, we compute the Pearson correlation between the omics data and phenotype (or clinical) data. This correlation serves as the target label during training.

- **Supervised Training Objective:**  
  The GNN is trained to predict these correlation values using a Mean Squared Error (MSE) loss. This strategy aligns node embeddings with biological signals relevant to the disease phenotype.

- **Downstream Integration:**  
  The learned node embeddings can be integrated into patient-level datasets for sample-level classification tasks. For example, **DPMON** (Disease Prediction using Multi-Omics Networks) leverages these embeddings in an end-to-end pipeline where the final objective is to classify disease outcomes.

Key Insights into GNN Parameters and Outputs
--------------------------------------------
1. **Input Parameters:**

   - **Node Features Matrix:** Built by correlating omics data with clinical variables.
   - **Edge Index:** Derived from the network’s adjacency matrix.
   - **Target Labels:** Numeric values representing the correlation between omics features and phenotype data.

2. **Output Embeddings:**

   - The penultimate layer of the GNN produces dense node embeddings that capture both local connectivity and supervised signals.
   - These embeddings can be further reduced (e.g., via PCA or an Autoencoder) for visualization or integrated into subject-level data.

Dimensionality Reduction: PCA vs. Autoencoders
------------------------------------------------
After training a GNN, the resulting node embeddings are typically high-dimensional. To integrate these embeddings into the original omics data—by reweighting each feature—a further reduction step is performed to obtain a single summary value per feature. BioNeuralNet supports two primary approaches for this reduction:

**Principal Component Analysis (PCA)**  
PCA is a linear dimensionality reduction technique that computes orthogonal components capturing the maximum variance in the data. The first principal component (PC1) is often used as a concise summary of each feature's variation. PCA is:

- **Deterministic and Fast:** A closed-form solution is computed from the covariance matrix.
- **Simple and Interpretable:** The linear combination of the original variables is straightforward to understand.
- **Limited to Linear Relationships:** It may not capture more complex, nonlinear structures in the data.

**Autoencoders (AE)**  
Autoencoders are neural network models designed to learn a compressed representation (latent code) through a bottleneck architecture. They use nonlinear activations (e.g., ReLU) to model complex relationships:

- **Nonlinear Transformation:** The encoder learns to capture intricate patterns that a linear method might miss.
- **Learned Representations:** The latent code is obtained by minimizing a reconstruction loss, making it adaptive to the data.
- **Flexible and Tunable:** Being neural network–based, autoencoders allow tuning of architecture parameters (e.g., number of layers, hidden dimensions, epochs, learning rate) to better capture the signal. In our framework, we highly recommend using autoencoders (i.e., setting `tune=True`) to leverage their enhanced expressivity for complex multi-omics data.

In practice, PCA offers simplicity and interpretability, whereas autoencoders may yield superior performance by capturing more nuanced nonlinear relationships. The choice depends on the complexity of your data and the computational resources available. Our recommendation is to enable tuning (using `tune=True`) to optimize the autoencoder parameters for your specific dataset.

How DPMON Uses GNNs Differently
-------------------------------
**DPMON** (Disease Prediction using Multi-Omics Networks) reuses the same GNN architectures but with a different objective:

- Instead of node-level MSE regression, DPMON aggregates node embeddings with patient-level omics data.
- A downstream classification head (e.g., softmax layer with CrossEntropyLoss) is applied for sample-level disease prediction.
- This end-to-end approach leverages both local (node-level) and global (patient-level) network information.

Example Usage
-------------
Below is a simplified example that demonstrates the task-driven approach—where node labels are derived from phenotype correlations and used to train the GNN:

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
