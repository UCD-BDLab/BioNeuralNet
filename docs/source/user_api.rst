User API
========

The User API provides detailed documentation for BioNeuralNet's modules, classes, and functions.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   bioneuralnet.graph_generation
   bioneuralnet.network_embedding
   bioneuralnet.downstream_task
   bioneuralnet.analysis
   bioneuralnet.utils

Executables
-----------

The following `run()` methods allow you to execute core workflows directly after creating an instance of the respective class and passing the necessary components or parameters. These methods encapsulate the end-to-end pipeline for specific tasks, such as graph generation, embedding creation, or disease prediction.

**Usage Example**:

1. **Instantiate a Class**: Create an instance of the desired class, passing in the required data and configurations.
2. **Execute the Workflow**: Call the `run()` method on the instance to execute the workflow.

**Example**:

.. code-block:: python

   from bioneuralnet.downstream_task import DPMON

   # Required inputs
   adjacency_matrix = adjacency_df  # Adjacency matrix from a prior graph-generation step
   omics_list = [omics_df1, omics_df2]  # List of omics data DataFrames
   phenotype_data = phenotype_df  # DataFrame of phenotype information
   clinical_data = clinical_df  # DataFrame of clinical data
   model = 'GCN'  # Specify the model type (e.g., GCN, GAT, SAGE)

   # Create an instance of DPMON
   dpmon_instance = DPMON(
       adjacency_matrix=adjacency_matrix,
       omics_list=omics_list,
       phenotype_data=phenotype_data,
       clinical_data=clinical_data,
       model=model
   )

   # Execute the disease prediction workflow
   predictions = dpmon_instance.run()
   print(predictions)

Below are direct references to the `run()` methods for quick access to their workflow details:

.. automethod:: bioneuralnet.graph_generation.SmCCNet.run
   :no-index:

.. automethod:: bioneuralnet.graph_generation.WGCNA.run
   :no-index:

.. automethod:: bioneuralnet.downstream_task.dpmon.DPMON.run
   :no-index:

.. automethod:: bioneuralnet.network_embedding.gnn_embedding.GNNEmbedding.run
   :no-index:

.. automethod:: bioneuralnet.network_embedding.node2vec.Node2VecEmbedding.run
   :no-index:

.. automethod:: bioneuralnet.subject_representation.GraphEmbedding.run
   :no-index:
