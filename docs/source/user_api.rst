User API
========

The **User API** lists BioNeuralNet's key classes, methods, and utilities and summarizes the main entry points exposed at the top-level ``bioneuralnet`` namespace.

Top-level Imports
-----------------

After installation, the most common pattern is:

.. code-block:: python

   import bioneuralnet as bnn

   print(bnn.__version__)

   # Core entry points
   from bioneuralnet import (
       GNNEmbedding,
       SubjectRepresentation,
       DPMON,
       SmCCNet,
       DatasetLoader,
       CorrelatedPageRank,
       CorrelatedLouvain,
       HybridLouvain,
       load_example,
       load_monet,
       load_brca,
       load_lgg,
       load_kipan,
       load_paad,
       set_seed,
       get_logger,
   )


Architecture Overview
---------------------

The figure below summarizes the main BioNeuralNet modules and public objects.

.. figure:: _static/bioneuralnet_api.png
   :align: center
   :alt: BioNeuralNet module and API overview

   BioNeuralNet 1.2.1: core modules (utils, network\_embedding, clustering, downstream\_task, datasets, metrics, external\_tools) and their key user-facing functions and classes.
   `View BioNeuralNet API. <https://bioneuralnet.readthedocs.io/en/latest/_images/bioneuralnet_api.png>`_

Module Reference
----------------

The following submodules are documented via autosummary:

.. autosummary::
   :toctree: _autosummary
   :recursive:

   bioneuralnet .

Executables
-----------

Several classes expose a high-level ``run()`` method to perform end-to-end workflows:

- :class:`bioneuralnet.downstream_task.SubjectRepresentation` for integrating embeddings into subject-level representations.
- :class:`bioneuralnet.clustering.CorrelatedLouvain` and :class:`bioneuralnet.clustering.HybridLouvain` for phenotype-aware clustering and subgraph detection.
- :class:`bioneuralnet.downstream_task.DPMON` for disease prediction using multi-omics networks.
- :class:`bioneuralnet.external_tools.SmCCNet` as a wrapper around external network-inference tools.

Usage pattern:

1. **Instantiate** the class with the relevant data (omics, adjacency, phenotype, etc.).
2. **Call** the :py:meth:`run()` method to execute the pipeline.

Example
-------

.. code-block:: python

   from bioneuralnet.downstream_task import DPMON

   dpmon_obj = DPMON(
       adjacency_matrix=adjacency_matrix,
       omics_list=omics_list,
       phenotype_data=phenotype_data,
       clinical_data=clinical_data,
       model="GAT",
   )
   predictions, avg_accuracy = dpmon.run()
   print("Disease phenotype predictions:\n", predictions)
   print("Average accuracy:", avg_accuracy)

Run Methods
-----------

Direct links to the main ``run()`` methods:

.. automethod:: bioneuralnet.external_tools.SmCCNet.run
   :no-index:

.. automethod:: bioneuralnet.downstream_task.SubjectRepresentation.run
   :no-index:

.. automethod:: bioneuralnet.downstream_task.DPMON.run
   :no-index:

.. automethod:: bioneuralnet.clustering.CorrelatedLouvain.run
   :no-index:

.. automethod:: bioneuralnet.clustering.HybridLouvain.run
   :no-index:
