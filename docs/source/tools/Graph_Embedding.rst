Graph Embeddings
================

BioNeuralNet supports multiple embedding approaches:

1. **GNNEmbedding** (GCN, GAT, GraphSAGE, GIN)
2. **Node2Vec** (classic random-walk embedding)

**GNN Embedding Example**:

.. literalinclude:: ../examples/gnn_embedding_example.py
   :language: python
   :caption: Generating GNN-based Embeddings with correlation-based node features (optional).

**Node2Vec Embedding Example**:

.. literalinclude:: ../examples/node2vec_example.py
   :language: python
   :caption: Using Node2Vec to produce node embeddings from an adjacency matrix.

The resulting embeddings can be used for:
- Clustering
- Subject-level integration
- Visualization
- Disease prediction
