Example 1: SmCCNet + GNN Embeddings + Subject Representation
============================================================

This tutorial shows how to:

1. **Construct** a multi-omics network adjacency using SmCCNet (an external R-based tool).
2. **Generate** node embeddings with a Graph Neural Network (GNN).
3. **Integrate** those embeddings into subject-level omics data for enhanced representation.

**Step-by-Step**:

1. **Data Setup**:
   - Omics data, phenotype data, clinical data as Pandas DataFrames or Series.

2. **Network Construction** (SmCCNet):
   - We call `SmCCNet.run()` to produce an adjacency matrix from the multi-omics data.

3. **GNN Embedding**:
   - We pass the adjacency, omics data, and (optionally) clinical data to `GNNEmbedding`.
   - GNNEmbedding’s `.run()` yields node embeddings.

4. **Subject Representation**:
   - We can integrate these embeddings back into omics data via `GraphEmbedding`.

.. note::
   For a **complete** script, see `examples/example_1.py` in the repository.

Below is a **partial** snippet:

.. code-block:: python

   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.network_embedding import GNNEmbedding
   from bioneuralnet.subject_representation import GraphEmbedding

   # 1) Prepare data
   omics_data = ...
   phenotype_data = ...
   clinical_data = ...

   # 2) Run SmCCNet to get adjacency
   smcc = SmCCNet(phenotype_data=phenotype_data, omics_data=omics_data)
   adjacency_matrix = smcc.run()

   # 3) Generate embeddings
   gnn = GNNEmbedding(
       adjacency_matrix=adjacency_matrix,
       omics_data=omics_data,
       phenotype_data=phenotype_data,
       clinical_data=clinical_data,
       model_type='GCN'
   )
   embedding_dict = gnn.run()
   node_embeddings = embedding_dict["graph"]

   # 4) Subject-level representation
   graph_embed = GraphEmbedding(
       adjacency_matrix=adjacency_matrix,
       omics_data=omics_data,
       phenotype_data=phenotype_data,
       clinical_data=clinical_data
   )
   enhanced_data = graph_embed.run()

   print("Enhanced omics data shape:", enhanced_data.shape)

**Results**:
- **Adjacency Matrix** from SmCCNet
- **Node Embeddings** from GNN
- **Enhanced Omics Data**, integrating node embeddings for subject-level analysis
