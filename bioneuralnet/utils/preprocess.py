from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional
from sklearn.preprocessing import RobustScaler,StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression
from sklearn.impute import KNNImputer, SimpleImputer
from statsmodels.stats.multitest import multipletests

from .logger import get_logger
logger = get_logger(__name__)

def beta_to_m(df, eps=1e-6):
    """
    Converts methylation Beta-values (ratio of methylated intensity to total intensity) to M-values using log2 transformation.

    M-values follow a normal distribution, improving statistical analysis, especially for differential methylation studies, by transforming the constrained [0, 1] Beta scale to an unbounded log-transformed scale (-inf to +inf).

    Args:

        df (pd.DataFrame): The input DataFrame containing Beta-values (0 to 1).
        eps (float): A small epsilon value used to clip Beta-values (B) away from 0 and 1, preventing logarithm errors (log(0) or division by zero).

    Returns:

        pd.DataFrame: A new DataFrame containing the log2-transformed M-values, calculated as log2(B / (1 - B)).
    """
    logger.info(f"Starting Beta-to-M value conversion (shape: {df.shape}). Epsilon: {eps}")

    has_non_numeric = False
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            has_non_numeric = True
            break

    if has_non_numeric:
        logger.warning("Coercing non-numeric Beta-values to numeric (NaNs will be introduced)")

    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    B = np.clip(df_numeric.values, eps, 1.0 - eps)
    M = np.log2(B / (1.0 - B))

    logger.info("Beta-to-M conversion complete.")

    return pd.DataFrame(M, index=df_numeric.index, columns=df_numeric.columns)

def impute_omics(omics_df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """
    Imputes missing values (NaNs) in the omics DataFrame using a specified strategy.

    Args:

        omics_df (pd.DataFrame): The input DataFrame containing missing values.
        method (str): The imputation strategy to use. Must be 'mean', 'median', or 'zero'.

    Returns:

        pd.DataFrame: The DataFrame with missing values filled.

    Raises:

        ValueError: If the specified imputation method is not recognized.
    """
    if method == "mean":
        return omics_df.fillna(omics_df.mean())
    elif method == "median":
        return omics_df.fillna(omics_df.median())
    elif method == "zero":
        return omics_df.fillna(0)
    else:
        raise ValueError(f"Imputation method '{method}' not recognized.")

def impute_omics_knn(omics_df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Imputes missing values (NaNs) using the K-Nearest Neighbors (KNN) approach.

    KNN imputation replaces missing values with the average value from the 'n_neighbors' most similar samples/features. This is often more accurate than simple mean imputation.

    Args:

        omics_df (pd.DataFrame): The input DataFrame containing missing values (NaNs).
        n_neighbors (int): The number of nearest neighbors to consider for imputation.

    Returns:

        pd.DataFrame: The DataFrame with missing values filled using KNN.
    """

    has_non_numeric = False
    for col in omics_df.columns:
        if not pd.api.types.is_numeric_dtype(omics_df[col]):
            has_non_numeric = True
            break

    if has_non_numeric:
        logger.error("KNNImputer requires numeric data. Non-numeric columns found.")

    # missing values before imputation
    n_missing_before = omics_df.isna().sum().sum()

    logger.info(
        f"Starting KNN imputation (k={n_neighbors}) on DataFrame "
        f"with shape {omics_df.shape} and {n_missing_before} NaNs."
    )

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(omics_df.values)
    imputed_df = pd.DataFrame(imputed_data, index=omics_df.index, columns=omics_df.columns)

    n_missing_after = imputed_df.isna().sum().sum()

    logger.info(f"New shape after imputation: {imputed_df.shape}")
    logger.info(
        f"KNN imputation complete. Imputed {n_missing_before} values; "
        f"remaining NaNs: {n_missing_after}."
    )

    return imputed_df

def normalize_omics(omics_df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """
    Scales or transforms feature data using common normalization techniques.

    Args:

        omics_df (pd.DataFrame): The input omics DataFrame.
        method (str): The scaling strategy. Must be 'standard' (Z-score), 'minmax', or 'log2'.


    Returns:

        pd.DataFrame: The DataFrame with features normalized according to the specified method.

    Raises:

        ValueError: If the specified normalization method is not recognized.
    """
    logger.info(f"Starting normalization on DataFrame (shape: {omics_df.shape}) using method: '{method}'.")
    data = omics_df.values

    if method == "standard":
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
    elif method == "minmax":
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
    elif method == "log2":
        if np.any(data < 0):
            logger.warning("Log2 transformation applied to data containing negative values. This can lead to unpredictable results")
        scaled_data = np.log2(data + 1)
    else:
        logger.error(f"Normalization method '{method}' not recognized.")
        raise ValueError(f"Normalization method '{method}' not recognized.")

    final_df = pd.DataFrame(scaled_data, index=omics_df.index, columns=omics_df.columns)
    logger.info("Normalization complete.")
    return final_df

def clean_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a numeric DataFrame by handling infinite values, imputing NaNs, and dropping zero-variance columns.

    Infinite values are replaced with NaN, all NaNs are imputed using the column median, and any features with zero variance are removed.

    Args:

        df (pd.DataFrame): Input DataFrame containing numeric columns, potentially with inf and NaN values.

    Returns:

        pd.DataFrame: Cleaned DataFrame with finite values, no NaNs, and only columns with non-zero variance.

    """

    df = df.copy()

    inf_count = df.isin([np.inf, -np.inf]).sum().sum()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    nan_before = df.isna().sum().sum()
    med = df.median(axis=0, skipna=True)
    df.fillna(med, inplace=True)

    var_before = df.shape[1]
    df = df.loc[:, df.std(axis=0, ddof=0) > 0]
    var_after = df.shape[1]

    # log
    logger.info(f"[Inf]: Replaced {inf_count} infinite values")
    logger.info(f"[NaN]: Replaced {nan_before} NaNs after median imputation")
    logger.info(f"[Zero-Var]: {var_before-var_after} columns dropped due to zero variance")

    return df

def clean_internal(df: pd.DataFrame, nan_threshold: float = 0.5) -> pd.DataFrame:
    """Clean a numeric DataFrame by dropping sparse and constant columns and imputing remaining NaNs.

    Columns with a fraction of missing values above nan_threshold are dropped, columns with zero variance are removed, and any remaining NaNs are imputed using the column median.

    Args:

        df (pd.DataFrame): Input numeric DataFrame to be cleaned.
        nan_threshold (float): Maximum allowed fraction of NaNs per column before the column is dropped (e.g., 0.5 drops columns with >50% missing).

    Returns:

        pd.DataFrame: Cleaned DataFrame with dense, non-constant columns and no remaining NaN values.

    """

    col_nan_percent = df.isna().mean()
    cols_to_drop = col_nan_percent[col_nan_percent > nan_threshold].index
    if not cols_to_drop.empty:
        logger.info(f"Dropping {len(cols_to_drop)} numeric columns due to >{nan_threshold*100}% missing values.")
        df = df.drop(columns=cols_to_drop)

    # Drop columns with zero variance (all the same value)
    cols_zero_variance = df.columns[df.std(axis=0, ddof=0) == 0]
    if not cols_zero_variance.empty:
        logger.info(f"Dropping {len(cols_zero_variance)} numeric columns with zero variance.")
        df = df.drop(columns=cols_zero_variance)

    if df.empty:
        logger.warning("No numeric features left after cleaning.")
        return df

    imputer = SimpleImputer(strategy='median')
    df_imputed = imputer.fit_transform(df)

    df_clean = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)

    return df_clean

def preprocess_clinical(X: pd.DataFrame, top_k: Optional[int] = 10, scale: bool = False, ignore_columns=None, nan_threshold: float = 0.5) -> pd.DataFrame:
    """Preprocess clinical data by cleaning, encoding, scaling, and selecting features by variance.

    Numeric columns are cleaned via clean_internal, optionally scaled with RobustScaler, and categorical columns are one-hot encoded. Zero-variance features are removed, missing values are filled, and if top_k is set the top-variance features are selected.

    Args:

        X (pd.DataFrame): Clinical feature matrix with samples as rows and features as columns.
        top_k (int | None): Number of highest-variance features to keep after preprocessing; if None, all features are retained.
        scale (bool): If True, apply RobustScaler to cleaned numeric features before combining with encoded categorical features.
        ignore_columns (list | None): Column names to drop from X before preprocessing; useful for IDs or label columns.
        nan_threshold (float): Maximum allowed fraction of NaNs per numeric column before it is dropped in clean_internal.

    Returns:

        pd.DataFrame: Fully cleaned, numeric feature matrix with optional scaling and variance-based feature selection applied.

    """

    ignore_columns = ignore_columns or []
    missing = set(ignore_columns) - set(X.columns)
    if missing:
        raise KeyError(f"Ignored columns not in X: {missing}")

    logger.info(f"Ignoring {len(ignore_columns)} columns.")
    X = X.drop(columns=ignore_columns, errors='ignore')

    df_numeric = X.select_dtypes(include="number")
    df_categorical = X.select_dtypes(include=["object", "category", "bool"])

    if not df_numeric.empty:
        df_numeric_clean = clean_internal(df_numeric, nan_threshold=nan_threshold)
        if scale:
            scaler = RobustScaler()
            scaled_array = scaler.fit_transform(df_numeric_clean)
            df_numeric_scaled = pd.DataFrame(scaled_array, columns=df_numeric_clean.columns, index=df_numeric_clean.index)
        else:
            df_numeric_scaled = df_numeric_clean.copy()
    else:
        logger.warning("No numeric data found to process.")
        df_numeric_scaled = pd.DataFrame(index=X.index)

    if not df_categorical.empty:
        df_cat_filled = df_categorical.fillna("Missing").astype(str)
        df_cat_encoded = pd.get_dummies(df_cat_filled, drop_first=True, dtype=int)
    else:
        logger.info("No categorical data found to encode.")
        df_cat_encoded = pd.DataFrame(index=X.index)


    df_combined = pd.concat([df_numeric_scaled, df_cat_encoded], axis=1, join="outer")
    df_features = df_combined.loc[:, df_combined.std(axis=0, ddof=0) > 0]
    df_features = df_features.fillna(0)

    if top_k is not None:
        if top_k > len(df_features.columns):
            logger.warning(
                f"top_k ({top_k}) is larger than available features ({len(df_features.columns)}). "
                "Using all features."
            )
            return df_features

        logger.info(f"Selecting top {top_k} features by variance.")
        variances = df_features.var(axis=0)
        selected_columns = variances.nlargest(top_k).index
        df_final = df_features[selected_columns]

    else:
        logger.info("Skipping variance-based feature selection, using all features.")
        df_final = df_features

    logger.info(f"Clinical data cleaning complete. Final shape: {df_final.shape}")
    return df_final

def select_top_k_variance(df: pd.DataFrame, k: int = 1000, ddof: int = 0) -> pd.DataFrame:
    """Select the top-k features with the highest variance after cleaning.

    The input is first cleaned with clean_inf_nan, then numeric columns are ranked by variance and the top k features are selected (or all if fewer than k are available).

    Args:

        df (pd.DataFrame): Input DataFrame from which to select high-variance numeric features.
        k (int): Maximum number of top-variance features to keep in the output.
        ddof (int): Delta degrees of freedom used in the variance computation; passed to DataFrame.var.

    Returns:

        pd.DataFrame: Numeric DataFrame containing only the top-k highest-variance features after cleaning.

    """

    df_clean = clean_inf_nan(df)
    num = df_clean.select_dtypes(include=[np.number]).copy()
    variances = num.var(axis=0, ddof=ddof)

    k = min(k, len(variances))
    top_cols = variances.nlargest(k).index.tolist()
    logger.info(f"Selected top {len(top_cols)} features by variance")

    return num[top_cols]

def select_top_k_correlation(X: pd.DataFrame, y: pd.Series | None = None, top_k: int = 1000) -> pd.DataFrame:
    """Select top-k features by correlation in supervised or unsupervised mode.

    In supervised mode (y provided), features are ranked by their absolute correlation with the target. In unsupervised mode, features are ranked by their mean absolute correlation with all other features to reduce redundancy, and selection is applied after basic cleaning.

    Args:

        X (pd.DataFrame): Numeric feature matrix with samples as rows and features as columns.
        y (pd.Series | None): Optional target vector; if provided, supervised selection is used, otherwise unsupervised redundancy-based selection.
        top_k (int): Number of features to select, capped at the number of available numeric features.

    Returns:

        pd.DataFrame: Numeric subset of X containing the selected features ordered by correlation-based ranking.

    """

    clean_df = clean_inf_nan(X)
    numbers_only = clean_df.select_dtypes(include=[np.number]).copy()

    # if y is provided then is supervised
    if y is not None:
        logger.info("Selecting features by supervised correlation with y")
        # input validation for y
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y must be a Series or single-column DataFrame")
            y = y.iloc[:, 0]
        elif not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series or DataFrame")

        correlations = {}
        for column in numbers_only.columns:
            col = numbers_only[column].corr(y)
            if pd.isna(col):
                correlations[column] = 0.0
            else:
                correlations[column] = abs(col)

        # descending correlations
        def key_fn(k: str) -> float:
            return correlations[k]

        features = list(correlations.keys())
        features.sort(key=key_fn, reverse=True)
        selected = features[:top_k]

    # unsupervised
    else:
        logger.info("Selecting features by unsupervised correlation")
        # full absolute correlationm matrix
        correlations_matrix = numbers_only.corr().abs()

        # zeroing out the diagonal
        for i in range(correlations_matrix.shape[0]):
            correlations_matrix.iat[i, i] = 0.0

        # mean correlation for each column
        correlations_avg = {}
        columns = list(correlations_matrix.columns)
        for col in columns:
            total = 0.0

            for others in columns:
                total += correlations_matrix.at[col, others]
            avg = total / (len(columns) - 1)
            correlations_avg[col] = avg

        def key_fn(k: str) -> float:
            return correlations_avg[k]

        features = list(correlations_avg.keys())
        features.sort(key=key_fn, reverse=True)
        selected = features[:top_k]

    logger.info(f"Selected {len(selected)} features by correlation")

    return numbers_only[selected]

def select_top_randomforest(X: pd.DataFrame, y: pd.Series, top_k: int = 1000, seed: int = 119) -> pd.DataFrame:
    """Select top-k features using RandomForest feature importances.

    Non-numeric columns are rejected, the remaining data are cleaned and zero-variance features removed, a RandomForest classifier or regressor is fitted depending on y, and the top_k most important features are selected based on ``feature_importances_``.

    Args:

        X (pd.DataFrame): Numeric feature matrix with samples as rows and features as columns; all columns must be numeric.
        y (pd.Series | pd.DataFrame): Target values as a Series or single-column DataFrame used to determine classification vs regression.
        top_k (int): Maximum number of most important features to keep according to the RandomForest model.
        seed (int): Random seed for initializing the RandomForest estimator.

    Returns:

        pd.DataFrame: Cleaned numeric subset of X restricted to the top-k most important features by RandomForest importance.

    """

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be a Series or a single-column DataFrame")
        y = y.iloc[:, 0]
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or DataFrame")

    non_numeric = []

    for col, dt in X.dtypes.items():
        if not pd.api.types.is_numeric_dtype(dt):
            non_numeric.append(col)

    if non_numeric:
        raise ValueError(f"Non-numeric columns detected: {non_numeric}")

    df_num = clean_inf_nan(X)
    df_clean = df_num.loc[:, df_num.std(axis=0, ddof=0) > 0]
    is_classif = (y.nunique() <= 10)

    if is_classif:
        Model = RandomForestClassifier
    else:
        Model = RandomForestRegressor

    model = Model(n_estimators=100, random_state=seed)

    model.fit(df_clean, y)
    importances = pd.Series(model.feature_importances_, index=df_clean.columns)
    top_feats = importances.nlargest(min(top_k, len(importances))).index

    return df_clean[top_feats]

def top_anova_f_features(X: pd.DataFrame, y: pd.Series, max_features: int, alpha: float = 0.05, task: str = "classification") -> pd.DataFrame:
    """Select top features using ANOVA F-test with FDR correction.

    Numeric features are cleaned, ANOVA F-statistics and p-values are computed against y using f_classif or f_regression, p-values are adjusted with Benjamini-Hochberg FDR, and up to max_features indices are chosen by prioritizing significant features and padding with the strongest remaining ones if needed.

    Args:

        X (pd.DataFrame): Numeric feature matrix with samples as rows and features as columns.
        y (pd.Series): Target vector; categorical for classification or continuous for regression, aligned to the rows of X.
        max_features (int): Maximum number of features to return after significance-based ranking and padding.
        alpha (float): Significance threshold applied to FDR-adjusted p-values to define significant features.
        task (str): Task type, either "classification" (uses f_classif) or "regression" (uses f_regression).

    Returns:

        pd.DataFrame: Numeric subset of X with up to max_features columns ordered by F-statistic with significant features first and padded by the strongest remaining features if necessary.

    """

    X = X.copy()
    y = y.copy()
    df_clean = clean_inf_nan(X)
    num = df_clean.select_dtypes(include=[np.number]).copy()

    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or a single-column DataFrame")

    y_aligned = y.loc[num.index]
    if task == "classification":
        F_vals, p_vals = f_classif(num, y_aligned.values)
    elif task == "regression":
        F_vals, p_vals = f_regression(num, y_aligned.values)
    else:
        raise ValueError("task must be classification or regression")

    _, p_adj, _, _ = multipletests(p_vals, alpha=alpha, method="fdr_bh")
    significant = p_adj < alpha

    order_all = np.argsort(-F_vals)
    sig_idx = []
    non_sig = []
    for i in order_all:
        if significant[i]:
            sig_idx.append(i)
        else:
            non_sig.append(i)

    n_sig = len(sig_idx)
    if n_sig >= max_features:
        final_idx = sig_idx[:max_features]
        n_pad = 0
    else:
        n_pad = max_features - n_sig
        final_idx = sig_idx + non_sig[:n_pad]

    logger.info(f"Selected {len(final_idx)} features by ANOVA (task={task}), {n_sig} significant, {n_pad} padded")

    return num.iloc[:, final_idx]

def prune_network(adjacency_matrix, weight_threshold=0.0):
    """Prune a weighted network by thresholding edge weights and removing isolated nodes.

    Edges with weights below weight_threshold are removed from the input adjacency matrix, then all nodes with no remaining connections are dropped, and basic before/after graph statistics are logged.

    Args:

        adjacency_matrix (pd.DataFrame): Weighted adjacency matrix with nodes as both rows and columns.
        weight_threshold (float): Minimum edge weight to retain; edges with smaller weights are pruned.

    Returns:

        pd.DataFrame: Pruned adjacency matrix containing only edges above the threshold and nodes with at least one connection.

    """

    logger.info(f"Pruning network with weight threshold: {weight_threshold}")
    full_G = nx.from_pandas_adjacency(adjacency_matrix)
    total_nodes = full_G.number_of_nodes()
    total_edges = full_G.number_of_edges()

    G = full_G.copy()

    if weight_threshold > 0:
        edges_to_remove = []

        for u, v, d in G.edges(data=True):
            weight = d.get('weight', 0)
            if weight < weight_threshold:
                edges_to_remove.append((u, v))

        G.remove_edges_from(edges_to_remove)

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    network_after_prunning =  nx.to_pandas_adjacency(G, dtype=float)
    current_nodes = G.number_of_nodes()
    current_edges = G.number_of_edges()

    logger.info(f"Pruning network with weight threshold: {weight_threshold}")
    logger.info(f"Number of nodes in full network: {total_nodes}")
    logger.info(f"Number of edges in full network: {total_edges}")
    logger.info(f"Number of nodes after pruning: {current_nodes}")
    logger.info(f"Number of edges after pruning: {current_edges}")

    return network_after_prunning

def prune_network_by_quantile(adjacency_matrix, quantile=0.5):
    """Prune a weighted network using a quantile-based edge-weight threshold.

    A global weight threshold is computed as the given quantile of all edge weights, edges below this threshold are removed, and isolated nodes are dropped from the resulting adjacency matrix.

    Args:

        adjacency_matrix (pd.DataFrame): Weighted adjacency matrix with nodes as both rows and columns.
        quantile (float): Quantile in [0, 1] used to determine the global weight cutoff for pruning.

    Returns:

        pd.DataFrame: Adjacency matrix with low-weight edges and isolated nodes removed based on the quantile threshold.

    """

    logger.info(f"Pruning network using quantile: {quantile}")
    G = nx.from_pandas_adjacency(adjacency_matrix)

    weights = []

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0)
        weights.append(weight)

    if len(weights) == 0:
         logger.warning("Network contains no edges")
         return nx.to_pandas_adjacency(G, dtype=float)

    weight_threshold = np.quantile(weights, quantile)
    logger.info(f"Computed weight threshold: {weight_threshold} for quantile: {quantile}")

    edges_to_remove = []

    for u, v, data in G.edges(data=True):
        if data.get('weight', 0) < weight_threshold:
            edges_to_remove.append((u, v))

    G.remove_edges_from(edges_to_remove)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    pruned_adjacency = nx.to_pandas_adjacency(G, dtype=float)
    logger.info(f"Number of nodes after pruning: {G.number_of_nodes()}")
    logger.info(f"Number of edges after pruning: {G.number_of_edges()}")

    return pruned_adjacency

def network_remove_low_variance(network: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
    """Remove nodes from an adjacency matrix whose connectivity pattern has very low variance.

    Column-wise variances are computed across the adjacency matrix, and any row/column pair whose variance is at or below the given threshold is removed, preserving a square node-by-node structure.

    Args:

        network (pd.DataFrame): Square adjacency matrix with identical row and column labels.
        threshold (float): Minimum allowed variance for a node's connectivity profile; nodes below this are dropped.

    Returns:

        pd.DataFrame: Filtered adjacency matrix restricted to nodes with variance greater than the specified threshold.

    """

    logger.info(f"Removing low-variance rows/columns with threshold {threshold}.")
    variances = network.var(axis=0)
    valid_indices = variances[variances > threshold].index
    filtered_network = network.loc[valid_indices, valid_indices]
    logger.info(f"Original network shape: {network.shape}, Filtered shape: {filtered_network.shape}")
    return filtered_network

def network_remove_high_zero_fraction(network: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove nodes from an adjacency matrix with a high fraction of zero entries.

    For each node, the fraction of zero entries in its corresponding column is computed, nodes whose zero fraction is greater than or equal to the threshold are removed, and the matrix is reduced to the remaining indices.

    Args:

        network (pd.DataFrame): Square adjacency matrix with identical row and column labels.
        threshold (float): Maximum allowed fraction of zeros per node; nodes with higher zero fraction are dropped.

    Returns:

        pd.DataFrame: Filtered adjacency matrix restricted to nodes with sufficiently non-zero connectivity.

    """

    logger.info(f"Removing high zero fraction features with threshold: {threshold}.")

    zero_fraction = (network == 0).sum(axis=0) / network.shape[0]
    valid_indices = zero_fraction[zero_fraction < threshold].index
    filtered_network = network.loc[valid_indices, valid_indices]
    logger.info(f"Original network shape: {network.shape}, Filtered shape: {filtered_network.shape}")

    return filtered_network
