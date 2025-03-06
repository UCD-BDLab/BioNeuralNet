Example 1: SmCCNet + GNN Embeddings + Subject Representation
============================================================
This tutorial illustrates how to:

1. **Build**: an adjacency matrix with SmCCNet.
2. **Enhance Representation**: Generate node embeddings using GNNEmbedding.
3. **Integrate**: Incorporate these embeddings into subject-level omics data using GraphEmbedding.

**Workflow**:

1. **Construct**:
   - A multi-omics network adjacency using SmCCNet.
2. **Generate**:
   - Node embeddings with a Graph Neural Network (GNN).
3. **Integrate**:
   - These embeddings into subject-level omics data for enhanced representation.

.. figure:: ../_static/SubjectRepresentation.png
   :align: center
   :alt: Subject Representation Workflow

   Subject-level embeddings provide richer phenotypic and clinical context.

`View full-size image: Subject Representation <https://bioneuralnet.readthedocs.io/en/latest/_images/SubjectRepresentation.png>`_

**Step-by-Step Instructions**:

1. **Data Setup**:
   - Load omics data, phenotype data, and clinical data using DatasetLoader.

2. **Network Construction (SmCCNet)**:
   - Call `SmCCNet.run()` to produce an adjacency matrix from multi-omics data.

3. **GNN Embedding**:
   - Pass the adjacency, omics data, and (optionally) clinical data to `GNNEmbedding`.
   - Use `.fit()` and `.embed()` to generate node embeddings.

4. **Subject Representation**:
   - Integrate these embeddings into omics data via `GraphEmbedding`.


Below is a **complete** Python implementation:

.. code-block:: python

   import pandas as pd
   from bioneuralnet.datasets import DatasetLoader
   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.network_embedding import GNNEmbedding
   from bioneuralnet.subject_representation import GraphEmbedding

   # 1) Load dataset
   loader = DatasetLoader("example1")
   omics1, omics2, phenotype, clinical = loader.load_data()

   # 2) Merge omics data
   merged_omics = pd.concat([omics1, omics2], axis=1)

   # 3) Generate adjacency matrix using SmCCNet
   smccnet = SmCCNet(
       phenotype_df=phenotype,
       omics_dfs=[omics1, omics2],
       data_types=["genes", "proteins"],
       kfold=3,
       subSampNum=500,
   )
   global_network, smccnet_clusters = smccnet.run()

   # 4) Generate embeddings using GNNEmbedding
   gnn_embedding = GNNEmbedding(
       adjacency_matrix=global_network,
       omics_data=merged_omics,
       phenotype_data=phenotype,
       clinical_data=clinical,
       tune=True,
   )
   gnn_embedding.fit()
   embeddings_output = gnn_embedding.embed(as_df=True)

   print(f"GNN embeddings generated. Shape: {embeddings_output.shape}")

   # 5) Perform subject representation using GraphEmbedding
   graph_embedding = GraphEmbedding(
       omics_data=merged_omics,
       embeddings=embeddings_output,
       phenotype_data=phenotype,
       tune=True,
   )

   enhanced_data = graph_embedding.run()
   print(f"Enhanced omics data shape: {enhanced_data.shape}")

   # Save enhanced omics data
   enhanced_data.to_csv("enhanced_omics_data.csv")

**Results**:

- **Adjacency Matrix** generated using SmCCNet.
- **Node Embeddings** from GNN.
- **Enhanced Omics Data**, integrating node embeddings for subject-level analysis.
