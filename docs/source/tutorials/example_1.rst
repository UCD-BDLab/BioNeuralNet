Example 1: SmCCNet + GNN Embeddings + Subject Representation
============================================================
This tutorial illustrates how to:

1. **Build:** an adjacency matrix with SmCCNet (external).
2. **Enhance Representation:** Enhance the representation of subjects using GNN embeddings.

**Workflow**:

1. **Construct**:
   - A multi-omics network adjacency using SmCCNet (an external R-based tool).
2. **Generate**:
   - Node embeddings with a Graph Neural Network (GNN).
3. **Integrate** :
   - Those embeddings into subject-level omics data for enhanced representation.
4. **Diagram of the workflow**:
   - The figure below illustrates the process.

`View full-size image: Subject Representation <https://bioneuralnet.readthedocs.io/en/latest/_images/SubjectRepresentation.png>`_

.. figure:: _static/SubjectRepresentation.png
   :align: center
   :alt: Subject Representation Workflow

   Subject-level embeddings provide richer phenotypic and clinical context.


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

Below is a **complete** snippet:

.. code-block:: python

   import pandas as pd
   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.network_embedding import GNNEmbedding
   from bioneuralnet.subject_representation import GraphEmbedding

   # 1) Prepare data
   omics_data = pd.read_csv("data/omics_data.csv")
   phenotype_data = pd.read_csv("data/phenotype_data.csv")
   clinical_data = pd.read_csv("data/clinical_data.csv")

   # 2) Run SmCCNet to get adjacency
   smccnet = SmCCNet(
      phenotype_df=phenotype_data,
      omics_df=omics_data,
      data_types=["genes, proteins"]
      kfolds=5,
      summarization = "NetSHy",
      seed: 127,
      )
   adjacency_matrix = smccnet.run()

   # 3) Generate embeddings
   gnn_embedding = GNNEmbedding(
       adjacency_matrix=adjacency_matrix,
       omics_data=omics_data,
       phenotype_data=phenotype_data,
       clinical_data=clinical_data,
       phenotype_col = "phenotype",
       model_type='GAT',
       hidden_dim=32,
       layer_num = 4,
       dropout=True,
       num_epochs=20,
       lr=1e-3,
       weight_decay=1e-4,
   )
   gnn_embedding.fit()
   embeddings_tensor = gnn_embedding.embed()

   node_names = adjacency_matrix.columns.tolist()
   embeddings_df = pd.DataFrame(embeddings_tensor.numpy(), index=node_names)
   print("GNN embeddings generated. Shape:", embeddings_df.shape)

   # 4) Subject-level representation:
   graph_embed = GraphEmbedding(
       adjacency_matrix=adjacency_matrix,
       omics_data=omics_data,
       phenotype_data=phenotype_data,
       clinical_data=clinical_data,
       embeddings = node_embeddings_df,
       reduce_method = "PCA"
   )
   #Other options to reduce embeddings inlcude "PCA", "avg" and "max".
   enhanced_data = graph_embed.run()
   print("Enhanced omics data shape:", enhanced_data.shape)
   print("Enhanced omics data columns:", enhanced_data.head())

   #saving your enhanced data to a csv file
   enhanced_data.to_csv("enhanced_omics_data.csv")

**Results**:

   - **Adjacency Matrix** from SmCCNet
   - **Node Embeddings** from GNN
   - **Enhanced Omics Data**, integrating node embeddings for subject-level analysis
