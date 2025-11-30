Utils
=====

The ``utils`` module provides a collection of supporting functions for data preparation, logging, graph construction, and exploratory analysis. These utilities streamline common preprocessing workflows and enable efficient manipulation of omics and clinical datasets.

Limitations and Best Practices
------------------------------

To ensure robust results, consider the following guidelines when using BioNeuralNet's utilities:

* **Suitable Use Cases:** These tools are optimized for tabular multi-omics data (rows as samples, columns as features). They work best when samples are consistently labeled across different omics layers.
* **Feature Selection:** For high-dimensional data (e.g., >20,000 features), we strongly recommend applying feature selection (e.g., ``select_top_k_variance`` or ``select_top_randomforest``) prior to network construction to reduce noise and computational load.
* **Network Construction:** Different graph construction methods capture different biological signals.

   * Use **Correlation** (Pearson/Spearman) for capturing linear co-expression patterns (e.g., WGCNA-style analysis).
   * Use **Similarity** (Cosine/RBF) for capturing broader geometric relationships in the data space.
   * Use **KNN** (k-Nearest Neighbors) to ensure sparse, connected topologies suitable for GNNs.

* **Sparse Data:** For datasets with high dropout rates (e.g., single-cell data or metabolomics), use ``clean_inf_nan`` and imputation tools (``impute_omics``) before running sensitive metrics like correlation.

RData Conversion
----------------

- :func:`bioneuralnet.utils.rdata_convert.rdata_to_df` converts an RData file to a CSV and loads it into a pandas DataFrame, facilitating interoperability with R-based bioinformatics pipelines.

Logging
-------

- :func:`bioneuralnet.utils.logger.get_logger` configures and returns a standardized logger writing to ``bioneuralnet.log`` at the project root, ensuring reproducible tracking of analysis steps.

Graph Generation
----------------

This section details utility functions for generating networks from omics data matrices. All methods are based on established literature.

.. rubric:: Methods and Examples

1. **k-NN Cosine/RBF Similarity Graph**

   Computes either cosine similarity:

   .. math::
      S_{ij} = \frac{x_i^\top x_j}{\|x_i\|\,\|x_j\|}

   or Gaussian RBF kernel:

   .. math::
      S_{ij} = \exp\bigl(-\|x_i - x_j\|^2 /(2\sigma^2)\bigr)

   Sparsifies by keeping the top-\(k\) neighbors per node (optionally mutual).

   .. code-block:: python

      from bioneuralnet.utils import gen_similarity_graph
      A = gen_similarity_graph(X, k=15, metric='cosine', mutual=True)

   Reference: Hastie et al., 2009 [Hastie2009_]

2. **Pearson/Spearman Co-expression Graph**

   Computes correlation:

   .. math::
      C_{ij} = \mathrm{corr}(x_i, x_j)

   Sparsifies by keeping the top-\(k\) correlations or applying a hard threshold.

   .. code-block:: python

      from bioneuralnet.utils import gen_correlation_graph
      A = gen_correlation_graph(X, k=15, method='pearson')

   Reference: Langfelder & Horvath, 2008 [Langfelder2008_]

3. **Soft-Threshold Graph**

   Applies a power function to absolute correlations, similar to WGCNA, to emphasize strong connections:

   .. math::
      W_{ij} = |C_{ij}|^\beta

   Followed by optional top-\(k\) selection.

   .. code-block:: python

      from bioneuralnet.utils import gen_threshold_graph
      A = gen_threshold_graph(X, b=6.0, k=15)

   Reference: Langfelder & Horvath, 2008 [Langfelder2008_]

4. **Gaussian k-NN Graph**

   Constructs a Gaussian kernel graph sparsified by k-nearest neighbors:

   .. math::
          S_{ij} = \exp\bigl(-\|x_i - x_j\|^2 /(2\sigma^2)\bigr),\quad W = \text{Top}_k(S)

   .. code-block:: python

      from bioneuralnet.utils import gen_gaussian_knn_graph
      A = gen_gaussian_knn_graph(X, k=15, sigma=None)

   Credit: Adapts common practice from spectral clustering (Ng et al., 2002).

5. **LASSO Graph (Graphical Lasso)**

   Estimates the sparse inverse covariance matrix (precision matrix) to infer conditional independence:

   .. code-block:: python

      from bioneuralnet.utils import gen_lasso_graph
      A = gen_lasso_graph(X, alpha=0.01)

   Reference: Friedman et al., 2008 [Friedman2008_]

6. **Shared Nearest Neighbor (SNN) Graph**

   Constructs a graph based on the overlap of neighborhoods between nodes, robust to varying densities:

   .. code-block:: python

      from bioneuralnet.utils import gen_snn_graph
      A = gen_snn_graph(X, k=15)

7. **Minimum Spanning Tree (MST) Graph**

   Constructs a backbone structure of the data using a Minimum Spanning Tree:

   .. code-block:: python

      from bioneuralnet.utils import gen_mst_graph
      A = gen_mst_graph(X, metric='euclidean')

Preprocessing Utilities
-----------------------

A collection of data-cleaning, imputation, and feature-selection functions for clinical and omics datasets.

**Clinical Preprocessing**

- :func:`bioneuralnet.utils.preprocess.preprocess_clinical` splits numeric and categorical features; replaces Inf/NaN; optionally scales numeric data (RobustScaler); encodes categoricals; drops zero-variance columns; and selects top-k features by RandomForest importance.

  **Example**:

  .. code-block:: python

      from bioneuralnet.utils import preprocess_clinical
      df_top = preprocess_clinical(X, y, top_k=10, scale=True)

- :func:`bioneuralnet.utils.preprocess.clean_inf_nan` replaces Inf with NaN, imputes missing values (median), drops zero-variance columns, and logs data statistics.

  **Example**:

  .. code-block:: python

      from bioneuralnet.utils import clean_inf_nan
      df_clean = clean_inf_nan(df)

**Imputation & Normalization**

- :func:`bioneuralnet.utils.preprocess.impute_omics`: Simple imputation (mean/median/zero) for tabular data.
- :func:`bioneuralnet.utils.preprocess.impute_omics_knn`: Advanced KNN-based imputation for estimating missing values based on similar samples.
- :func:`bioneuralnet.utils.preprocess.normalize_omics`: Standardizes omics data (z-score or min-max scaling).
- :func:`bioneuralnet.utils.preprocess.beta_to_m`: Converts Beta values (DNA methylation) to M-values for better statistical properties.

**Variance-Based Selection**

- :func:`bioneuralnet.utils.preprocess.select_top_k_variance` cleans data, then retains the top-k numeric features with the highest variance.

  **Example**:

  .. code-block:: python

      from bioneuralnet.utils import select_top_k_variance
      df_var = select_top_k_variance(df, k=500)

**Correlation-Based Selection**

- :func:`bioneuralnet.utils.preprocess.select_top_k_correlation`:
    * **Supervised:** If ``y`` is provided, selects features with the highest absolute Pearson correlation to the target.
    * **Unsupervised:** If ``y=None``, selects features with the *lowest* average inter-feature correlation (redundancy reduction).

  **Example**:

  .. code-block:: python

      from bioneuralnet.utils import select_top_k_correlation
      df_sup = select_top_k_correlation(X, y, top_k=100) # supervised
      df_unsup = select_top_k_correlation(X, top_k=100) # unsupervised

**RandomForest Feature Importance**

- :func:`bioneuralnet.utils.preprocess.select_top_randomforest` fits a RandomForest model (classification or regression) and returns the top-k features ranked by importance.

  **Example**:

  .. code-block:: python

      from bioneuralnet.utils import select_top_randomforest
      df_rf = select_top_randomforest(X, y, top_k=200)

**ANOVA F-Test Selection**

- :func:`bioneuralnet.utils.preprocess.top_anova_f_features` runs an ANOVA F-test, applies FDR correction, selects significant features, and optionally pads the selection to reach ``max_features``.

  **Example**:

  .. code-block:: python

      from bioneuralnet.utils import top_anova_f_features
      df_anova = top_anova_f_features(X, y, max_features=100, alpha=0.05)

**Network Pruning**

- :func:`bioneuralnet.utils.preprocess.prune_network` prunes edges below a weight threshold, removes isolated nodes, and logs network statistics.

  **Example**:

  .. code-block:: python

      from bioneuralnet.utils import prune_network
      pruned = prune_network(adj_df, weight_threshold=0.1)

- :func:`bioneuralnet.utils.preprocess.prune_network_by_quantile` uses a quantile cutoff on edge weights to prune the network.

  **Example**:

  .. code-block:: python

      from bioneuralnet.utils import prune_network_by_quantile
      pruned_q = prune_network_by_quantile(adj_df, quantile=0.75)

- :func:`bioneuralnet.utils.preprocess.network_remove_low_variance` drops nodes (rows/cols) from the adjacency matrix whose variance falls below a threshold.
- :func:`bioneuralnet.utils.preprocess.network_remove_high_zero_fraction` drops nodes where the fraction of zero-weight edges exceeds a threshold (e.g., disconnected nodes).


Graph Analysis Tools
--------------------

Functions for analyzing and repairing graph structures.

- :func:`bioneuralnet.utils.graph_tools.graph_analysis`: Computes comprehensive graph metrics (density, sparsity, connectivity, connected components) and logs a summary.
- :func:`bioneuralnet.utils.graph_tools.repair_graph_connectivity`: Analyzes graph connectivity and, if fragmented, repairs it by adding minimum spanning tree (MST) edges or connecting components to the giant component.
- :func:`bioneuralnet.utils.graph_tools.find_optimal_graph`: Iteratively tests different graph construction parameters (e.g., k-NN `k` values) to maximize a target metric like connectivity or community structure.


Data Summary Utilities
----------------------

- :func:`bioneuralnet.utils.data.variance_summary` computes summary statistics for column variances.
- :func:`bioneuralnet.utils.data.zero_fraction_summary` computes statistics for the fraction of zeros per column (sparsity).
- :func:`bioneuralnet.utils.data.expression_summary` computes summary of mean expression values across features.
- :func:`bioneuralnet.utils.data.correlation_summary` computes statistics of each feature's maximum pairwise correlation.
- :func:`bioneuralnet.utils.data.explore_data_stats` prints a comprehensive summary (variance, sparsity, expression, correlation) to standard output.

  **Example**:

  .. code-block:: python
      
      from bioneuralnet.utils import explore_data_stats
      explore_data_stats(df, name="MyOmicsData")

References
----------

.. [Langfelder2008] Langfelder, P., & Horvath, S. (2008). WGCNA: an R package for weighted correlation network analysis. *BMC Bioinformatics*, 9, 559.

.. [Margolin2006] Margolin, A. A., Nemenman, I., Basso, K., Wiggins, C., Stolovitzky, G., Dalla Favera, R., & Califano, A. (2006). ARACNE: an algorithm for the reconstruction of gene regulatory networks in a mammalian cellular context. *BMC Bioinformatics*, 7(Suppl 1), S7.

.. [Hastie2009] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

.. [Friedman2008] Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432-441.
