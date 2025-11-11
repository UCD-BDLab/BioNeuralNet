Example 2: SmCCNet + GNN Embeddings + Subject Representation
============================================================
This tutorial illustrates how to:

1. **Build**: an adjacency matrix with SmCCNet.
2. **Enhance Representation**: Generate node embeddings using GNNEmbedding.
3. **Integrate**: Incorporate these embeddings into subject-level omics data using SubjectRepresentation.

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

3. **Generate GNN Embeddings**:
   - Pass the adjacency, omics data, and (optionally) clinical data to `GNNEmbedding`.
   - Use `.fit()` and `.embed()` to generate node embeddings.

4. **Subject Representation**:
   - Integrate these embeddings into omics data via `SubjectRepresentation`.


Below is a **complete** Python implementation:

.. code-block:: python

   import pandas as pd
   from bioneuralnet.datasets import DatasetLoader
   from bioneuralnet.external_tools import SmCCNet
   from bioneuralnet.network_embedding import GNNEmbedding
   from bioneuralnet.downstream_task import SubjectRepresentation

   # 1) Data Setup
   Example = DatasetLoader("example1")
   omics_genes = Example.data["X1"]
   omics_proteins = Example.data["X2"]
   phenotype = Example.data["Y"]
   clinical = Example.data["clinical_data"]

   # 2) Network Construction
   smccnet = SmCCNet(
       phenotype_df=phenotype,
       omics_dfs=[omics_genes, omics_proteins],
       data_types=["Genes", "Proteins"],
       kfold=5,
       subSampNum=500,
   )
   global_network, clusters = smccnet.run()

   # 3) Generate embeddings using GNNEmbedding
   merged_omics = pd.concat([omics_genes, omics_proteins], axis=1)

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

   # 4) Enhance subject profiles using with the embeddings from GNNs with SubjectRepresentation
   graph_embedding = SubjectRepresentation(
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
