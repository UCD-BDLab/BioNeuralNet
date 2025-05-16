Graph Embeddings
================

BioNeuralNet multiple graph embedding approaches:

**GNNEmbedding**:

GCN - Graph Convolutional Network
GAT - Graph Attention Network
GraphSAGE - Graph Sample and Aggregation
GIN - Graph Isomorphism Network

**GNN Embedding Example**:

.. literalinclude:: ../examples/gnn_embedding_example.py
   :language: python
   :caption: Generating GNN-based Embeddings with correlation-based node features (optional).

The resulting embeddings can be used for:
- Clustering
- Subject-level integration
- Visualization
- Disease prediction
