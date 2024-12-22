API Reference
=============

The API Reference provides detailed documentation for BioNeuralNet’s modules, classes, and functions.

.. autosummary::
    :toctree: _autosummary
    :recursive:

    bioneuralnet.graph_generation.SmCCNet
    bioneuralnet.graph_generation.WGCNA
    bioneuralnet.network_embedding.GNNEmbedding
    bioneuralnet.network_embedding.Node2VecEmbedding
    bioneuralnet.subject_representation.GraphEmbedding
    bioneuralnet.downstream_task.DPMON
    bioneuralnet.utils.data_utils.combine_omics_data
    bioneuralnet.utils.file_helpers.find_files
    bioneuralnet.utils.path_utils.validate_paths
    bioneuralnet.analysis.feature_selector.FeatureSelector
    bioneuralnet.analysis.static_visualization.StaticVisualizer
    bioneuralnet.analysis.dynamic_visualization.DynamicVisualizer

Detailed Run Methods
--------------------

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
