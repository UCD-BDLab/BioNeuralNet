User API
========

The **User API** lists BioNeuralNet's key classes, methods, and utilities.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   bioneuralnet.network_embedding
   bioneuralnet.clustering
   bioneuralnet.downstream_task
   bioneuralnet.subject_representation
   bioneuralnet.utils
   bioneuralnet.datasets
   bioneuralnet.metrics
   bioneuralnet.external_tools

Executables
-----------

Certain classes expose a high-level ``run()`` method to perform end-to-end workflows:

- **SmCCNet** or **WGCNA** (from `bioneuralnet.external_tools`) for adjacency generation
- **GraphEmbedding** for integrating embeddings into subject-level data
- **CorrelatedLouvain** or **HybridLouvain** for clustering
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

**Methods**:

Below are references to the ``run()`` methods:

.. automethod:: bioneuralnet.external_tools.smccnet.SmCCNet.run
   :no-index:

.. automethod:: bioneuralnet.downstream_task.GraphEmbedding.run
   :no-index:

.. automethod:: bioneuralnet.clustering.CorrelatedLouvain.run
   :no-index:

.. automethod:: bioneuralnet.downstream_task.DPMON.run
   :no-index:
