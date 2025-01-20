User API
========

The **User API** lists BioNeuralNet’s key classes, methods, and utilities.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   bioneuralnet.external_tools
   bioneuralnet.network_embedding
   bioneuralnet.downstream_task
   bioneuralnet.subject_representation
   bioneuralnet.utils

Executables
-----------

Certain classes expose a high-level ``run()`` method to perform end-to-end workflows:

- **SmCCNet** or **WGCNA** (from `bioneuralnet.external_tools`) for adjacency generation
- **GNNEmbedding** or **Node2Vec** for embeddings
- **GraphEmbedding** for integrating embeddings into subject-level data
- **DPMON** for disease prediction

**Usage Pattern**:

1. **Instantiate** the class with the relevant data (omics, adjacency, phenotype, etc.).
2. **Call** the `run()` method to perform the pipeline.

Example:

.. code-block:: python

   from bioneuralnet.downstream_task import DPMON

   dpmon_obj = DPMON(
       adjacency_matrix=adjacency_matrix,
       omics_list=omics_list,
       phenotype_data=phenotype_data,
       clinical_data=clinical_data,
       model='GAT'
   )

   predictions = dpmon_obj.run()
   print(predictions.head())

**Methods**:

Below are references to the ``run()`` methods:

.. automethod:: bioneuralnet.external_tools.smccnet.SmCCNet.run
   :no-index:

.. automethod:: bioneuralnet.network_embedding.gnn_embedding.GNNEmbedding.run
   :no-index:

.. automethod:: bioneuralnet.subject_representation.GraphEmbedding.run
   :no-index:

.. automethod:: bioneuralnet.downstream_task.DPMON.run
   :no-index:
