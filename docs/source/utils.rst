Utils
=====

The `utils` module provides a collection of supporting functions for data preparation, logging, graph construction, and exploratory analysis. 
These utilities streamline common preprocessing workflows and enable efficient manipulation of omics and clinical datasets.

RData Conversion
----------------

- :func:`bioneuralnet.utils.rdata_to_df` converts an RData file to a CSV and loads it into a pandas DataFrame.

Logging
-------

- :func:`bioneuralnet.utils.get_logger` configures and returns a logger writing to ``bioneuralnet.log`` at the project root.

Graph Generation
----------------

This section details utility functions for generating networks from omics data matrices. All methods are industry-standard and open in the literature.

.. rubric:: Methods and Examples

1. **k-NN Cosine/RBF Similarity Graph**

   Computes either cosine similarity:

   .. math::
      S_{ij} = \frac{x_i^\top x_j}{\|x_i\|\,\|x_j\|}

   or Gaussian RBF kernel:

   .. math::
      S_{ij} = \exp\bigl(-\|x_i - x_j\|^2 /(2\sigma^2)\bigr)

   Sparsify by keeping top-\(k\) neighbors per node, optionally mutual.

   .. code-block:: python

      from bioneuralnet.utils.graphs import gen_similarity_graph
      A = gen_similarity_graph(X, k=15, metric='cosine', mutual=True)

    Reference: Hastie et al., 2009 [Hastie2009_]

2. **Pearson/Spearman Co-expression Graph**

   Compute correlation:

   .. math::
      C_{ij} = \mathrm{corr}(x_i, x_j)

   Then sparsify by top-\(k\) or threshold.

   .. code-block:: python

      from bioneuralnet.utils.graphs import gen_correlation_graph
      A = gen_correlation_graph(X, k=15, method='pearson')

    Reference: Langfelder & Horvath, 2008 [Langfelder2008_]

3. **Soft-Threshold Graph**

   Soft-threshold absolute correlation, similar to WGCNA:

   .. math::
      W_{ij} = |C_{ij}|^\beta

   Then top-\(k\) selection and normalization.

   .. code-block:: python

      from bioneuralnet.utils.graphs import gen_threshold_graph
      A = gen_threshold_graph(X, b=6.0, k=15)

    Reference: Langfelder & Horvath, 2008 [Langfelder2008_]

4. **Gaussian k-NN Graph**

   Gaussian kernel sparsified by k-nearest neighbors:

   .. math::
         S_{ij} = \exp\bigl(-\|x_i - x_j\|^2 /(2\sigma^2)\bigr),\quad W = \text{Top}_k(S)

   .. code-block:: python

      from bioneuralnet.utils.graphs import gen_gaussian_knn_graph
      A = gen_gaussian_knn_graph(X, k=15, sigma=None)

   Credit: adapts common practice from spectral clustering (Ng et al., 2002).

5. **Mutual Information Graph**

   Estimate pairwise mutual information:

   .. math::
      \mathrm{MI}_{ij} = I(x_i; x_j)

   .. code-block:: python

      from bioneuralnet.utils.graphs import gen_mutual_info_graph
      A = gen_mutual_info_graph(X, k=15)

    Reference: Margolin et al., 2006 [Margolin2006_]

Preprocessing Utilities
-----------------------

A collection of data-cleaning and feature-selection functions for clinical and omics datasets.

**Clinical Preprocessing**

- :func:`bioneuralnet.utils.preprocess.preprocess_clinical` splits numeric and categorical features; replaces Inf/NaN; optionally scales numeric data (RobustScaler); encodes categoricals; drops zero-variance; and selects top-k features by RandomForest importance.  

  **Example**:

  .. code-block:: python

      from bioneuralnet.utils.preprocess import preprocess_clinical
      df_top = preprocess_clinical(X, y, top_k=10, scale=True)

- :func:`bioneuralnet.utils.preprocess.clean_inf_nan` replaces Inf with NaN, imputes medians, drops zero-variance columns, and logs counts.  

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.preprocess import clean_inf_nan
      df_clean = clean_inf_nan(df)

**Variance-Based Selection**

- :func:`bioneuralnet.utils.preprocess.select_top_k_variance` cleans data, then picks the top-k numeric features by variance.  

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.preprocess import select_top_k_variance
      df_var = select_top_k_variance(df, k=500)

**Correlation-Based Selection**

- :func:`bioneuralnet.utils.preprocess.select_top_k_correlation`:

   - **Supervised:** if you pass `y`, selects features by absolute Pearson correlation with `y`.  
   - **Unsupervised:** if `y=None`, picks features with the lowest average inter-feature correlation (redundancy reduction).  

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.preprocess import select_top_k_correlation
      df_sup = select_top_k_correlation(X, y, top_k=100) # supervised  
      df_unsup = select_top_k_correlation(X, top_k=100) # unsupervised

**RandomForest Feature Importance**

- :func:`bioneuralnet.utils.preprocess.select_top_randomforest` fits RandomForest (classification or regression), and returns the top-k features by importance.  

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.preprocess import select_top_randomforest
      df_rf = select_top_randomforest(X, y, top_k=200)

**ANOVA F-Test Selection**

- :func:`bioneuralnet.utils.preprocess.top_anova_f_features` runs ANOVA F-test (classification or regression), applies FDR correction, selects all significant features, and pads with next-best to reach `max_features`.  

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.preprocess import top_anova_f_features
      df_anova = top_anova_f_features(X, y, max_features=100, alpha=0.05)

**Network Pruning**

- :func:`bioneuralnet.utils.preprocess.prune_network` prunes edges below a weight threshold, removes isolates, and logs before/after stats.  

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.preprocess import prune_network
      pruned = prune_network(adj_df, weight_threshold=0.1)

- :func:`bioneuralnet.utils.preprocess.prune_network_by_quantile` uses a quantile cutoff on edge weights, prunes accordingly, removes isolates, and logs stats.  

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.preprocess import prune_network_by_quantile
      pruned_q = prune_network_by_quantile(adj_df, quantile=0.75)

- :func:`bioneuralnet.utils.preprocess.network_remove_low_variance` drops rows/columns in the adjacency matrix whose variance falls below a threshold.  

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.preprocess import network_remove_low_variance
      filtered = network_remove_low_variance(adj_df, threshold=1e-5)

- :func:`bioneuralnet.utils.preprocess.network_remove_high_zero_fraction` removes rows/columns where the fraction of zero weights exceeds a threshold (default 0.95).  

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.preprocess import network_remove_high_zero_fraction
      filtered_z = network_remove_high_zero_fraction(adj_df, threshold=0.95)


Data Summary Utilities
----------------------

- :func:`bioneuralnet.utils.data.variance_summary` computes summary statistics for column variances.

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.data import variance_summary
      stats = variance_summary(df, low_var_threshold=1e-4)

- :func:`bioneuralnet.utils.data.zero_fraction_summary` computes statistics for the fraction of zeros per column.

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.data import zero_fraction_summary
      stats = zero_fraction_summary(df, high_zero_threshold=0.5)

- :func:`bioneuralnet.utils.data.expression_summary` computes summary of mean expression values across features.

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.data import expression_summary
      stats = expression_summary(df)

- :func:`bioneuralnet.utils.data.correlation_summary` computes statistics of each feature's maximum pairwise correlation (excluding self).

  **Example**:

   .. code-block:: python

      from bioneuralnet.utils.data import correlation_summary
      stats = correlation_summary(df)

- :func:`bioneuralnet.utils.data.explore_data_stats` prints an overall summary (variance, zero fraction, expression, correlation) to stdout.

  **Example**:

   .. code-block:: python
      
      from bioneuralnet.utils.data import explore_data_stats
      explore_data_stats(df, name="MyOmicsData")

References
----------

.. [Langfelder2008_] Langfelder, P., & Horvath, S. (2008). WGCNA: an R package for weighted correlation network analysis. *BMC Bioinformatics*, 9, 559.

.. [Margolin2006_] Margolin, A. A., Nemenman, I., Basso, K., Wiggins, C., Stolovitzky, G., Dalla Favera, R., & Califano, A. (2006). ARACNE: an algorithm for the reconstruction of gene regulatory networks in a mammalian cellular context. *BMC Bioinformatics*, 7(Suppl 1), S7.

.. [Faith2007_] Faith, J. J., Hayete, B., Thaden, J. T., Mogno, I., Wierzbowski, J., Cottarel, G., ... & Gardner, T. S. (2007). Large-scale mapping and validation of *Escherichia coli* transcriptional regulation from a compendium of expression profiles. *PLoS Biology*, 5(1), e8.

.. [Hastie2009_] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

.. [Friedman2008_] Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432-441.

