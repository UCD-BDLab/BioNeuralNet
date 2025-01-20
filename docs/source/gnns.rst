.. _gnns:

GNNs for Multi-Omics
====================

This section provides an overview of **Graph Neural Networks (GNNs)** in BioNeuralNet
for multi-omics data. We support four GNN architectures (GCN, GAT, GraphSAGE, GIN), each
of which can be **task-driven** (using numeric values or other labels per node) or
**unsupervised** (using intrinsic objectives) to produce node embeddings.

.. contents::
   :local:
   :depth: 2

Graph Convolutional Network (GCN)
---------------------------------

GCN layers apply a spectral-based convolution operator to aggregate neighbor
information. The core update equation for a single GCN layer is:

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

GraphSAGE (SAmple and aggreGatE) updates each node by concatenating its own features
with an aggregated summary of its neighbors. In a mean-aggregator setting:

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

   If each node (e.g., a gene or protein) has a **numeric value**
   (e.g., correlation with a disease phenotype), you can train a GNN to predict
   this value with MSE or another loss. This aligns node embeddings with the
   target measure, grouping nodes that have similar values or relationships.

2. **Unsupervised**:
   If no explicit node label is provided, the GNN can learn from the **graph structure itself**:

   - **Intrinsic Objectives**: Methods like graph autoencoders or contrastive losses let GNNs reconstruct edges, distinguish real vs. random neighbors, etc.
   - **Structure as Signal**: Even in the absence of external labels, adjacency patterns guide the GNN to produce meaningful embeddings (local/global relationships).
   - **Practical Implementation**: In some workflows (e.g., a quick exploration), you might skip external labeling entirely. The GNN could run with random weights or minimal self-supervision, still generating a representation that captures the topology to some extent.

Overall, both approaches yield node embeddings that can be integrated into **subject-level**
datasets or used for clustering or further analysis.


Example Usage
-------------

Below is a **simplified** snippet showing a **task-driven** approach, where each node is
assigned a numeric correlation with disease severity. The GNN (GCN, GAT, SAGE, GIN) tries
to predict that correlation, producing a **trained** embedding:

.. code-block:: python

   from bioneuralnet.network_embeddings import GNNEmbedding
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
   out_dict = gnn.run()
   node_embeds = out_dict['graph']


If **no** numeric label is specified, you can rely on an **unsupervised** approach—either
a self-supervised pipeline or even a random initialization—for a simpler dimensional
embedding of the graph structure.


How DPMON Uses GNNs Differently
-------------------------------

**DPMON** (Disease Prediction using Multi-Omics Networks) is an **end-to-end** pipeline
that also employs GNNs, but with a distinct focus:

- **Local + Global Structure**: It captures how nodes relate across the entire multi-omics network without using correlation-based node labels.
- **Joint Optimization**: The GNN is trained jointly with a downstream neural net for disease phenotype prediction, maximizing predictive power.
- **Feature Fusion**: The resulting embeddings are integrated into the patient-level multi-omics data, enhancing classification accuracy.

By focusing on the **intrinsic connectivity** of the multi-omics graph, DPMON avoids
overfitting to single-sample signals and achieves more robust predictions.


Return to :doc:`../index`
