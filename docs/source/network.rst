Network Construction & Analysis
================================

The ``network`` module provides tools for constructing multi-omics networks from
raw tabular data and analyzing their topology.

.. code-block:: python

   from bioneuralnet.network import (
       similarity_network,
       correlation_network,
       threshold_network,
       gaussian_knn_network,
       NetworkAnalyzer,
       network_search,
       auto_pysmccnet,
   )

All construction functions accept a :class:`pandas.DataFrame` of shape
``(n_samples, n_features)`` and return a weighted adjacency matrix of shape
``(n_features, n_features)``. All methods are GPU-accelerated via PyTorch when
CUDA is available.

Network Construction
--------------------

- :func:`bioneuralnet.network.similarity_network`: Builds a k-NN similarity graph using cosine similarity or a Gaussian kernel on Euclidean distances.

  .. code-block:: python

      from bioneuralnet.network import similarity_network
      net = similarity_network(X, k=15, metric="cosine", mutual=False, normalize=True)

  Parameters: ``k`` (neighbors per node), ``metric`` (``"cosine"`` or ``"euclidean"``), ``mutual`` (restrict to mutual kNN edges), ``per_node`` (per-node vs global cutoff), ``self_loops``, ``normalize`` (row-normalize).

- :func:`bioneuralnet.network.correlation_network`: Builds a correlation-based graph using Pearson or Spearman correlation mapped to [0,1] via ``(C + 1) / 2`` (signed) or ``|C|`` (unsigned).

  .. code-block:: python

      from bioneuralnet.network import correlation_network
      net = correlation_network(X, k=15, method="spearman", signed=True, normalize=True)

  Parameters: ``k``, ``method`` (``"pearson"`` or ``"spearman"``), ``signed``, ``mutual``, ``per_node``, ``threshold`` (overrides ``k`` in global cutoff mode), ``self_loops``, ``normalize``.

- :func:`bioneuralnet.network.threshold_network`: Builds a soft-threshold co-expression graph by raising absolute Pearson correlations to power ``b``, then applying a kNN mask.

  .. code-block:: python

      from bioneuralnet.network import threshold_network
      net = threshold_network(X, b=6.3, k=22, mutual=False, normalize=True)

  Parameters: ``b`` (soft-threshold exponent), ``k``, ``mutual``, ``self_loops``, ``normalize``.

- :func:`bioneuralnet.network.gaussian_knn_network`: Builds a Gaussian RBF k-NN graph from pairwise Euclidean distances. When ``sigma=None``, a median squared distance heuristic is used. Self-loops are enabled by default.

  .. code-block:: python

      from bioneuralnet.network import gaussian_knn_network
      net = gaussian_knn_network(X, k=15, sigma=None, mutual=False, normalize=True)

  Parameters: ``k``, ``sigma`` (RBF bandwidth; ``None`` for median heuristic), ``mutual``, ``self_loops``, ``normalize``.

Phenotype-Driven Construction
------------------------------

- :func:`bioneuralnet.network.auto_pysmccnet`: Constructs a phenotype-specific multi-omics network using Sparse Multiple Canonical Correlation Analysis (SmCCNet 2.0). Automatically selects CCA mode for continuous phenotypes and PLS mode for binary phenotypes. R and the SmCCNet CRAN package are required.

  .. code-block:: python

      from bioneuralnet.network import auto_pysmccnet

      result = auto_pysmccnet(
          X=[omics1, omics2],
          Y=phenotype,
          DataType=["genes", "mirna"],
          subSampNum=1000,
          Kfold=3,
          BetweenShrinkage=5,
          CutHeight=1 - 0.1**10,
          summarization="NetSHy",
      )
      global_network = result["AdjacencyMatrix"]
      subnetworks    = result["Subnetworks"]

  Returns a dict with keys: ``"AdjacencyMatrix"``, ``"Data"``, ``"CVResult"``, ``"Subnetworks"``.

  Key parameters: ``X`` (list of omics DataFrames), ``Y`` (phenotype), ``DataType`` (layer names), ``subSampNum`` (subsampling iterations), ``Kfold``, ``BetweenShrinkage`` (between-omics scaling shrinkage), ``tuneRangeCCA``, ``tuneRangePLS``, ``EvalMethod``, ``CutHeight``, ``min_size``, ``max_size``, ``summarization``.

  For full parameter reference see the `SmCCNet documentation <https://kechrislab.github.io/SmCCNet/>`_.

Network Quality Assessment
---------------------------

- :class:`bioneuralnet.network.NetworkAnalyzer`: GPU-accelerated network topology analysis class. Auto-detects omics types from feature name prefixes (e.g., ``genes_Gene_7`` resolves to ``genes``). Pass ``source_omics`` to override with original DataFrames.

  .. code-block:: python

      from bioneuralnet.network import NetworkAnalyzer

      analyzer = NetworkAnalyzer(adjacency_matrix=net)
      analyzer.edge_weight_analysis()
      analyzer.basic_statistics(threshold=0.1)
      analyzer.hub_analysis(threshold=0.1, top_n=10)
      analyzer.cross_omics_analysis(threshold=0.1)
      _ = analyzer.find_strongest_edges(top_n=10)

  Methods:

  - ``edge_weight_analysis()``: Returns an array of all non-zero edge weights and logs distribution statistics and percentiles.
  - ``basic_statistics(threshold)``: Returns a dict with node count, edge count, density, average/max/min degree, and isolated node count.
  - ``hub_analysis(threshold, top_n)``: Returns a DataFrame of the top-N nodes by degree with omics type assignments.
  - ``cross_omics_analysis(threshold)``: Returns a dict of within-layer and between-layer edge counts and density for each omics pair. Requires ``dtype=torch.long`` for index tensors.
  - ``find_strongest_edges(top_n)``: Returns a DataFrame of the top-N highest-weight feature pairs with omics assignments.
  - ``degree_distribution(threshold)``: Returns a DataFrame of degree, count, and percentage across all nodes.
  - ``clustering_coefficient_gpu(threshold, sample_size)``: Computes local clustering coefficients using GPU matrix operations; samples up to 5,000 nodes on large graphs.
  - ``connected_components(threshold)``: Returns component count, node label assignments, and component size distribution via scipy sparse BFS.

Network Search
--------------

- :func:`bioneuralnet.network.network_search`: Searches over graph construction hyperparameters. Scores each candidate configuration using a centrality-weighted Ridge classifier proxy blended with a topological quality term (node connectivity and largest connected component ratio). Returns the best graph, best parameters, and a full results DataFrame.

  .. code-block:: python

      from bioneuralnet.network import network_search

      best_graph, best_params, results_df = network_search(
          omics_data=feature_matrix,
          y_labels=phenotype,
          methods=["correlation", "threshold", "similarity", "gaussian"],
          trials=50,
          topology_weight=0.15,
          scoring="f1_macro",
          seed=123,
      )

  Parameters: ``methods`` (construction methods to evaluate), ``trials`` (cap on configurations; ``None`` evaluates the full grid), ``topology_weight`` (blending factor in [0,1] between classifier F1 and topological quality), ``centrality_mode`` (``"eigenvector"`` or ``"degree"``), ``scoring`` (scikit-learn scoring string).