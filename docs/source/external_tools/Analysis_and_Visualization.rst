Analysis and Visualization
==========================

BioNeuralNet includes several classes in `external_tools` that can complement the main pipeline:

**FeatureSelector**:
  - Provides quick methods (LassoCV, RandomForestClassifier) to select top features from a dataset.

.. literalinclude:: ../examples/feature_selector_example.py
   :language: python
   :caption: Using FeatureSelector for simple feature ranking.

**StaticVisualizer**:
  - Uses NetworkX + Matplotlib to produce static images of your network.

.. literalinclude:: ../examples/static_visualization_example.py
   :language: python
   :caption: Creating a static network image.

**DynamicVisualizer**:
  - Employs PyVis or other interactive libs to build HTML-based network visualizations.

.. literalinclude:: ../examples/dynamic_visualization_example.py
   :language: python
   :caption: Generating an interactive HTML network visualization.

**HierarchicalClustering**:
  - Performs agglomerative clustering, silhouette scoring, etc., on adjacency or embedded data.

.. note::
   For deeper clustering examples (like PageRank-based sweeps), check :doc:`../tools/index`
   or the `examples/` folder.
