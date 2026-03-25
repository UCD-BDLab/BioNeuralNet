Utils
=====

The ``utils`` module provides helper functions for data preprocessing, feature selection,
statistical data exploration, network pruning, and reproducibility.

.. code-block:: python

   from bioneuralnet.utils import (
       set_seed, data_stats, sparse_filter,
       laplacian_score, variance_threshold, mad_filter,
       pca_loadings, correlation_filter, importance_rf, top_anova_f_features,
       m_transform, impute_simple, impute_knn, normalize,
       clean_inf_nan, clean_internal, preprocess_clinical,
       prune_network, prune_network_by_quantile,
       network_remove_low_variance, network_remove_high_zero_fraction,
   )

Reproducibility
---------------

- :func:`bioneuralnet.utils.reproducibility.set_seed`: Sets global random seeds for Python, NumPy, and PyTorch (including all CUDA GPU operations). Configures ``torch.backends.cudnn`` for deterministic algorithms.

  Parameters: ``seed_value`` (int).

  .. code-block:: python

      from bioneuralnet.utils import set_seed
      set_seed(123)

Data Diagnostics
----------------

- :func:`bioneuralnet.utils.data.data_stats`: Combines variance, zero fraction, expression, and NaN summaries for an omics DataFrame and emits actionable recommendations. Correlation summary is skipped by default.

  Parameters: ``df``, ``name`` (str label for logging), ``compute_correlation`` (bool, default ``False``).

  .. code-block:: python

      from bioneuralnet.utils import data_stats
      data_stats(X_mirna, "miRNA")
      data_stats(X_meth,  "Methylation", compute_correlation=False)

- :func:`bioneuralnet.utils.data.sparse_filter`: Drops columns (features) whose missing fraction exceeds ``missing_fraction``, then drops rows (samples) whose missing fraction exceeds the same threshold.

  Parameters: ``df``, ``missing_fraction`` (float, default ``0.20``).

  .. code-block:: python

      from bioneuralnet.utils import sparse_filter
      X_mirna = sparse_filter(X_mirna, missing_fraction=0.20)

- :func:`bioneuralnet.utils.data.nan_summary`: Logs global and per-feature/per-sample NaN rates. Returns the global missing percentage as a float.

  Parameters: ``df``, ``name``, ``missing_threshold`` (float, default ``20.0``).

- :func:`bioneuralnet.utils.data.variance_summary`: Returns a dict of mean, median, min, max, and std of column variances. Optionally counts features below ``var_threshold``.

  Parameters: ``df``, ``var_threshold`` (Optional[float]).

- :func:`bioneuralnet.utils.data.zero_summary`: Returns a dict of statistics for the fraction of zeros per column. Optionally counts features above ``zero_threshold``.

  Parameters: ``df``, ``zero_threshold`` (Optional[float]).

- :func:`bioneuralnet.utils.data.expression_summary`: Returns a dict of mean, median, min, max, and std of feature means.

  Parameters: ``df``.

- :func:`bioneuralnet.utils.data.correlation_summary`: Returns a dict of statistics for each feature's maximum pairwise absolute correlation. Fills diagonal with 0 before computing max.

  Parameters: ``df``.

.. _feature-selection-utilities:

Feature Selection
-----------------

- :func:`bioneuralnet.utils.feature_selection.laplacian_score`: Selects the top ``n_keep`` features by Laplacian Score computed on a symmetric k-NN affinity graph built from standardized data. Lower scores indicate higher importance.

  Parameters: ``df``, ``n_keep`` (int), ``k_neighbors`` (int, default ``5``).

  .. code-block:: python

      from bioneuralnet.utils import laplacian_score
      X_selected = laplacian_score(X, n_keep=200, k_neighbors=5)

- :func:`bioneuralnet.utils.feature_selection.variance_threshold`: Selects the top-k features by variance after applying ``clean_inf_nan``.

  Parameters: ``df``, ``k`` (int, default ``1000``), ``ddof`` (int, default ``0``).

  .. code-block:: python

      from bioneuralnet.utils import variance_threshold
      X_prefiltered = variance_threshold(X, k=2000)

- :func:`bioneuralnet.utils.feature_selection.mad_filter`: Selects the top ``n_keep`` features by Median Absolute Deviation computed across samples.

  Parameters: ``df``, ``n_keep`` (int).

  .. code-block:: python

      from bioneuralnet.utils import mad_filter
      X_selected = mad_filter(X, n_keep=200)

- :func:`bioneuralnet.utils.feature_selection.pca_loadings`: Selects features with the highest absolute PCA loading magnitudes, weighted by explained variance ratio. Scales data with ``StandardScaler`` before PCA.

  Parameters: ``df``, ``n_keep`` (int), ``n_components`` (int, default ``50``), ``seed`` (int, default ``1883``).

  .. code-block:: python

      from bioneuralnet.utils import pca_loadings
      X_selected = pca_loadings(X, n_keep=200, n_components=50)

- :func:`bioneuralnet.utils.feature_selection.correlation_filter`: In unsupervised mode (``y=None``), ranks features by mean absolute inter-feature correlation and selects the top ``top_k``. In supervised mode (``y`` provided), ranks by absolute Pearson correlation with the target.

  Parameters: ``X``, ``y`` (pd.Series or None), ``top_k`` (int, default ``1000``).

  .. code-block:: python

      from bioneuralnet.utils import correlation_filter
      X_selected = correlation_filter(X, top_k=500)
      X_supervised = correlation_filter(X, y=y, top_k=500)

- :func:`bioneuralnet.utils.feature_selection.importance_rf`: Fits a ``RandomForestClassifier`` (when ``y.nunique() <= 10``) or ``RandomForestRegressor`` and selects the top ``top_k`` features by ``feature_importances_``. Drops zero-variance columns before fitting.

  Parameters: ``X``, ``y`` (pd.Series), ``top_k`` (int, default ``1000``), ``seed`` (int, default ``119``).

  .. code-block:: python

      from bioneuralnet.utils import importance_rf
      X_selected = importance_rf(X, y, top_k=200)

- :func:`bioneuralnet.utils.feature_selection.top_anova_f_features`: Computes ANOVA F-statistics (``f_classif`` or ``f_regression``) with Benjamini-Hochberg FDR correction. Returns up to ``max_features`` features ordered by F-statistic, significant features first. Pads with the strongest non-significant features if needed.

  Parameters: ``X``, ``y`` (pd.Series), ``max_features`` (int), ``alpha`` (float, default ``0.05``), ``task`` (``"classification"`` or ``"regression"``).

  .. code-block:: python

      from bioneuralnet.utils import top_anova_f_features
      X_selected = top_anova_f_features(X, y, max_features=200, alpha=0.05, task="classification")

Preprocessing Utilities
-----------------------

- :func:`bioneuralnet.utils.preprocess.m_transform`: Converts Beta values to M-values via ``log2(clip(B, eps, 1-eps) / (1 - clip(B, eps, 1-eps)))``. Non-numeric columns are coerced to numeric before transformation.

  Parameters: ``df``, ``eps`` (float, default ``1e-6``).

  .. code-block:: python

      from bioneuralnet.utils import m_transform
      X_meth = m_transform(X_meth, eps=1e-7)

- :func:`bioneuralnet.utils.preprocess.impute_simple`: Fills NaN values using ``mean``, ``median``, or ``zero`` strategy via ``DataFrame.fillna``.

  Parameters: ``df``, ``method`` (str, default ``"mean"``).

  .. code-block:: python

      from bioneuralnet.utils import impute_simple
      X_imputed = impute_simple(X, method="mean")

- :func:`bioneuralnet.utils.preprocess.impute_knn`: KNN imputation via ``sklearn.impute.KNNImputer``. Raises ``ValueError`` on non-numeric columns. Returns ``df`` unchanged if no NaNs are present.

  Parameters: ``df``, ``n_neighbors`` (int, default ``5``).

  .. code-block:: python

      from bioneuralnet.utils import impute_knn
      X_imputed = impute_knn(X, n_neighbors=5)

- :func:`bioneuralnet.utils.preprocess.normalize`: Scales data using ``StandardScaler`` (``"standard"``), ``MinMaxScaler`` (``"minmax"``), or ``log2(x + 1)`` (``"log2"``).

  Parameters: ``df``, ``method`` (str, default ``"standard"``).

  .. code-block:: python

      from bioneuralnet.utils import normalize
      X_norm = normalize(X, method="standard")

- :func:`bioneuralnet.utils.preprocess.clean_inf_nan`: Replaces ``inf``/``-inf`` with NaN, imputes NaNs with column median, and drops zero-variance columns.

  Parameters: ``df``.

  .. code-block:: python

      from bioneuralnet.utils import clean_inf_nan
      X_clean = clean_inf_nan(X)

- :func:`bioneuralnet.utils.preprocess.clean_internal`: Drops columns with NaN fraction above ``nan_threshold``, removes zero-variance columns, and imputes remaining NaNs with column median via ``SimpleImputer``.

  Parameters: ``df``, ``nan_threshold`` (float, default ``0.5``).

  .. code-block:: python

      from bioneuralnet.utils import clean_internal
      X_clean = clean_internal(X, nan_threshold=0.5)

- :func:`bioneuralnet.utils.preprocess.preprocess_clinical`: Drops specified columns, maps ordinal variables to numeric ranks, coerces continuous columns, one-hot encodes nominal categoricals (with ``dummy_na=True``), optionally scales numeric columns with ``RobustScaler``, and removes zero-variance columns. Returns ``float32`` DataFrame.

  Parameters: ``X``, ``scale`` (bool, default ``False``), ``drop_columns`` (list or None), ``ordinal_mappings`` (dict or None), ``continuous_columns`` (list or None), ``impute`` (bool, default ``False``).

  ..