.. _gnns:

GNNs for Multi-Omics
====================

This section provides an overview of **Graph Neural Networks (GNNs)** in BioNeuralNet for multi-omics data. We support four GNN architectures (GCN, GAT, GraphSAGE, GIN), each of which can be **task-driven** (using numeric values or other labels per node) or **unsupervised** (using intrinsic objectives) to produce node embeddings.

.. contents::
   :local:
   :depth: 2

Graph Convolutional Network (GCN)
---------------------------------

GCN layers apply a spectral-based convolution operator to aggregate neighbor information. The core update equation for a single GCN layer is:

.. math::

   X^{(l+1)} \;=\; \mathrm{ReLU}\!\Bigl(\widehat{D}^{-\tfrac{1}{2}}\,\widehat{A}\,\widehat{D}^{-\tfrac{1}{2}}\,
   X^{(l)}\,W^{(l)}\Bigr),

where

- :math:`X^{(l)}` is the node feature matrix at layer :math:`l`.
- :math:`\widehat{A} = A + I` is the adjacency matrix with inserted self-loops.
- :math:`\widehat{D}` is the diagonal degree matrix of :math:`\widehat{A}`.
- :math:`W^{(l)}` denotes the trainable parameters at layer :math:`l`.
- :math:`\mathrm{ReLU}` is the rectified linear unit activation function.


Graph Attention Network (GAT)
-----------------------------

GAT layers incorporate attention coefficients to weight the contribution of neighbors:

.. math::

   h_{i}^{(l+1)} \;=\; \mathrm{ELU}\!\Bigl(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)}\,W^{(l)}\,h_{j}^{(l)}\Bigr),

where

- :math:`h_{i}^{(l)}` is the embedding of node :math:`i` at layer :math:`l`.
- :math:`\alpha_{ij}^{(l)}` is the learned attention coefficient for the edge between nodes :math:`i` and :math:`j`.
- :math:`W^{(l)}` is a trainable linear transformation.
- :math:`\mathrm{ELU}` is the exponential linear unit activation.
- :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`.


GraphSAGE
---------

GraphSAGE (SAmple and aggreGatE) updates each node by concatenating its own features with an aggregated summary of its neighbors. In a mean-aggregator setting:

.. math::

   h_{i}^{(l+1)} \;=\; \sigma\!\Bigl(
       W^{(l)}
       \bigl(\,
         h_{i}^{(l)} \,\|\, \mathrm{mean}_{j \,\in\, \mathcal{N}(i)}(h_{j}^{(l)})
       \bigr)\Bigr),

where

- :math:`\mathcal{N}(i)` are neighbors of node :math:`i`.
- :math:`\|` denotes vector concatenation.
- :math:`W^{(l)}` is a trainable weight matrix at layer :math:`l`.
- :math:`\sigma` is a nonlinear activation function, e.g. ReLU.


Graph Isomorphism Network (GIN)
-------------------------------

GIN uses a sum-aggregator combined with a learnable :math:`\epsilon` parameter
and an MLP to update node representations:

.. math::

   h_i^{(l+1)} \;=\; \mathrm{MLP}^{(l)}\!\Bigl(\,\bigl(1 + \epsilon^{(l)}\bigr)\,
   h_{i}^{(l)} \;+\; \sum_{j \in \mathcal{N}(i)} h_{j}^{(l)}\Bigr),

where

- :math:`\epsilon^{(l)}` is a (learnable or fixed) parameter for each layer :math:`l`.
- :math:`\mathrm{MLP}^{(l)}` is a multi-layer perceptron at layer :math:`l`.


Task-Driven vs. Unsupervised GNNs
---------------------------------

1. **Task-Driven**:

   If each node (e.g., a gene or protein) has a **numeric value** (e.g., correlation with a disease phenotype), you can train a GNN to predict this value (with MSE loss). This aligns node embeddings with the target measure, grouping nodes that have similar relationships.

   **Pre-Computed or Random Node Features**:

      - You can correlate each node with clinical variables to build a rich feature vector, **or** simply initialize node features randomly.
      - Even with random features, the GNN can still learn **the label** drives training via MSE on each node’s predicted value vs. its real label.

2. **Unsupervised**:

   If no explicit node label is provided, the GNN can learn from the **graph structure itself**:

      - **Intrinsic Objectives**:

         e.g., graph autoencoders or contrastive losses.

      - **Random Features**:

         Possibly leaving node features random or minimal.

      - **Structure as Signal**:

         Even in the absence of external labels, adjacency patterns guide the GNN to produce embeddings capturing local/global relationships.

      - **Practical Implementation**:

         - In some quick explorations, you might skip external labeling entirely.
         - The GNN produces a representation that captures the topology.

Overall, both approaches yield node embeddings that can be integrated into **subject-level** datasets or used for clustering or further analysis.

Key Insights into GNN Parameters and Outputs
--------------------------------------------

1. **Input Parameters**:

   - **Node Features Matrix**: :math:`(N \times F)` shape. You can use pre-computed correlations or random initialization.
   - **Edge Index**: :math:`(2 \times E)` shape, specifying edges.
   - **Edge Attributes (Optional)**: Weighted edges, etc.
   - **Target Labels (Optional)**: If **task-driven** (MSE regression or classification), each node has a numeric or categorical label.

2. **Output Embeddings**:

   - Each node is represented as a dense vector in the hidden layers.
   - Typically, a final linear may reduce to 1D if you do node-level MSE, but the **penultimate** layer is multi-dimensional.
   - After training, you retrieve the multi-dimensional “node embeddings” if needed.

3. **Dimensionality Reduction**:

   - After generating embeddings, techniques like PCA can reduce dimensionality for simpler usage and improved interpretability.

How DPMON Uses GNNs Differently
-------------------------------

**DPMON** (Disease Prediction using Multi-Omics Networks) is an **end-to-end** pipeline
that uses GNN layers for **sample-level classification**, rather than node-level MSE:

1. **Local + Global Structure**:
   - DPMON’s GNN still aggregates information per node. However, it doesn’t use numeric per-node labels in MSE.
   - Instead, it extracts node embeddings to feed into an autoencoder/dimension-averaging pipeline.

2. **Classification Head**:
   - Ultimately, DPMON’s GNN embeddings are combined with patient-level omics data.
   - A downstream neural network (e.g., a softmax layer) predicts the patient’s disease phenotype.
   - This is a **sample-level** classification objective (e.g., CrossEntropyLoss on patient classes).

Hence, DPMON’s approach reuses the **same** GNN architectures (GCN, GAT, etc.) but with a **different final objective** (classification at sample level, instead of node-level MSE).

Example Usage
-------------

Below is a **simplified** snippet showing a **task-driven** node-level approach, where each node is assigned a numeric correlation with disease severity. The GNN (choose GCN, GAT, SAGE, or GIN) tries to predict that correlation, producing a trained embedding:

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

If **no** numeric label is used, you can rely on an **unsupervised** or random feature initialization. Even when we skip correlation-based features and MSE training, the GNN still provides embeddings reflecting the graph structure.

For sample-level classification, see :ref:`dpmon`. DPMON integrates node embeddings into an end-to-end disease prediction pipeline, focusing on **patient** labels rather than node-level numeric regression.

Return to :doc:`../index`
