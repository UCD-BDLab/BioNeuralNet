Tutorials
=========

These tutorials illustrate how to combine BioNeuralNet components for a cohesive multi-omics analysis.

.. toctree::
   :maxdepth: 2

   example_1
   example_2

**Example 1** demonstrates:
   - Generating a **network adjacency** using SmCCNet (external tool).
   - Building **GNN embeddings** from the adjacency.
   - Integrating embeddings into subject data for further analysis.

**Example 2** demonstrates:
   - Constructing a network (SmCCNet).
   - Leveraging **DPMON** for end-to-end **disease prediction**, combining adjacency and omics data.


BioNeuralNet offers a variety of **tools** for graph-based analyses of multi-omics data, including:

- **Graph Embedding**: Generate GNN or Node2Vec embeddings.
- **Subject Representation**: Integrate embeddings into omics data.
- **Disease Prediction**: DPMON for end-to-end classification.
- **Graph Clustering**: PageRank or hierarchical clustering for subnetwork identification.

.. toctree::
   :maxdepth: 2
   :caption: Usage Examples

   Graph_Embedding
   Subject_Representation
   Disease_Prediction
   Graph_Clustering

These examples illustrate typical usage patterns for each module.
