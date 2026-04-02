from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer

from .logger import get_logger
logger = get_logger(__name__)

def m_transform(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Converts methylation Beta-values to M-values using log2 transformation.

    M-values follow a normal distribution, improving statistical analysis by transforming the constrained [0, 1] Beta scale to an unbounded log-transformed scale.

    Args:

        df (pd.DataFrame): The input DataFrame containing Beta-values (0 to 1).
        eps (float): A small epsilon value used to clip Beta-values away from 0 and 1, preventing logarithm errors.

    Returns:

        pd.DataFrame: A new DataFrame containing the log2-transformed M-values.

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

def impute_simple(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """Imputes missing values (NaNs) in the DataFrame using a specified strategy.

    Args:

        df (pd.DataFrame): The input DataFrame containing missing values.
        method (str): The imputation strategy to use. Must be 'mean', 'median', or 'zero'.

    Returns:

        pd.DataFrame: The DataFrame with missing values filled.

    Raises:

        ValueError: If the specified imputation method is not recognized.

    """
    if method == "mean":
        return df.fillna(df.mean())
    elif method == "median":
        return df.fillna(df.median())
    elif method == "zero":
        return df.fillna(0)
    else:
        raise ValueError(f"Imputation method '{method}' not recognized.")

def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Imputes missing values (NaNs) using the K-Nearest Neighbors (KNN) approach.

    KNN imputation replaces missing values with the average value from the
    'n_neighbors' most similar samples.
    NOTE: Input data should be scaled/normalized prior to imputation.

    Args:
        df (pd.DataFrame): The input DataFrame containing missing values.
        n_neighbors (int): The number of nearest neighbors to consider.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled using KNN.
    """

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if not non_numeric_cols.empty:
        err_msg = f"KNNImputer requires numeric data. Non-numeric columns found: {list(non_numeric_cols)}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    n_missing_before = df.isna().sum().sum()

    if n_missing_before == 0:
        logger.info(f"No NaNs found in DataFrame of shape {df.shape}. Skipping imputation.")
        return df

    logger.info(
        f"Starting KNN imputation (k={n_neighbors}) on DataFrame "
        f"with shape {df.shape} and {n_missing_before} NaNs."
    )

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(df.values)

    imputed_df = pd.DataFrame(imputed_data, index=df.index, columns=df.columns)

    n_missing_after = imputed_df.isna().sum().sum()

    logger.info(f"New shape after imputation: {imputed_df.shape}")
    logger.info(
        f"KNN imputation complete. Imputed {n_missing_before} values; "
        f"remaining NaNs: {n_missing_after}."
    )

    return imputed_df

def normalize(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """Scales or transforms feature data using common normalization techniques.

    Args:

        df (pd.DataFrame): The input DataFrame.
        method (str): The scaling strategy. Must be 'standard' (Z-score), 'minmax', or 'log2'.

    Returns:

        pd.DataFrame: The DataFrame with features normalized according to the specified method.

    Raises:

        ValueError: If the specified normalization method is not recognized.

    """
    logger.info(f"Starting normalization on DataFrame (shape: {df.shape}) using method: '{method}'.")
    data = df.values

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

    final_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
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

    logger.info(f"[Inf]: Replaced {inf_count} infinite values")
    logger.info(f"[NaN]: Replaced {nan_before} NaNs after median imputation")
    logger.info(f"[Zero-Var]: {var_before-var_after} columns dropped due to zero variance")

    return df

def clean_internal(df: pd.DataFrame, nan_threshold: float = 0.5) -> pd.DataFrame:
    """Clean a numeric DataFrame by dropping sparse and constant columns and imputing remaining NaNs.

    Columns with a fraction of missing values above nan_threshold are dropped, columns with zero variance are removed, and any remaining NaNs are imputed using the column median.

    Args:

        df (pd.DataFrame): Input numeric DataFrame to be cleaned.
        nan_threshold (float): Maximum allowed fraction of NaNs per column before the column is dropped.

    Returns:

        pd.DataFrame: Cleaned DataFrame with dense, non-constant columns and no remaining NaN values.

    """
    col_nan_percent = df.isna().mean()
    cols_to_drop = col_nan_percent[col_nan_percent > nan_threshold].index
    if not cols_to_drop.empty:
        logger.info(f"Dropping {len(cols_to_drop)} numeric columns due to >{nan_threshold*100}% missing values.")
        df = df.drop(columns=cols_to_drop)

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

def preprocess_clinical(X: pd.DataFrame, scale: bool = False, drop_columns: Optional[list] = None, ordinal_mappings: Optional[dict] = None, continuous_columns: Optional[list] = None, impute:float = False) -> pd.DataFrame:
    """Preprocess clinical data by cleaning, mapping ordinals, encoding nominals, and scaling.

    This function provides a generalized pipeline for standardizing clinical datasets. It removes specified non-informative columns, maps ordinal variables to numeric ranks, safely coerces continuous variables, and applies one-hot encoding to nominal categories while tracking missing records. Optionally, it handles median imputation and scaling.

    Args:

    X (pd.DataFrame): The raw clinical feature matrix with patients as rows and variables as columns.
    scale (bool): If True, applies RobustScaler to the numeric columns.
    drop_columns (list | None): List of column names to drop prior to processing.
    ordinal_mappings (dict | None): Nested dictionary mapping string categories to numeric ranks.
    continuous_columns (list | None): List of strictly continuous column names to coerce to numeric.
    impute (bool): If True, applies median imputation to missing numeric and ordinal values.

    Returns:

    pd.DataFrame: Processed clinical feature matrix containing valid numeric types with zero-variance columns removed.

    """
    drop_columns = drop_columns or []

    logger.info(f"Ignoring {len(drop_columns)} columns.")
    X = X.copy()
    X = X.drop(columns=drop_columns, errors='ignore')

    if ordinal_mappings:
        for col, mapping in ordinal_mappings.items():
            if col in X.columns:
                X[col] = X[col].astype(str).str.lower().str.strip()
                X[col] = X[col].map(mapping)
                if impute:
                    X[col] = X[col].fillna(X[col].median())

    if continuous_columns:
        for col in continuous_columns:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')

    df_numeric = X.select_dtypes(include="number").copy()
    df_categorical = X.select_dtypes(exclude="number").copy()

    if not df_numeric.empty:
        if impute:
            for col in df_numeric.columns:
                df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

        if scale:
            scaler = RobustScaler()
            scaled_array = scaler.fit_transform(df_numeric)
            df_numeric_scaled = pd.DataFrame(scaled_array, columns=df_numeric.columns, index=df_numeric.index)
        else:
            df_numeric_scaled = df_numeric.copy()
    else:
        logger.warning("No numeric data found to process.")
        df_numeric_scaled = pd.DataFrame(index=X.index)

    if not df_categorical.empty:
        for col in df_categorical.columns:
            df_categorical[col] = df_categorical[col].astype(str).str.lower().str.strip()
            df_categorical[col] = df_categorical[col].replace('nan', np.nan)

        df_cat_encoded = pd.get_dummies(df_categorical, dummy_na=True, drop_first=True, dtype=int)
    else:
        logger.info("No categorical data found to encode.")
        df_cat_encoded = pd.DataFrame(index=X.index)

    df_combined = pd.concat([df_numeric_scaled, df_cat_encoded], axis=1, join="outer")
    df_final = df_combined.loc[:, df_combined.std(axis=0, ddof=0) > 0]

    df_final = df_final.astype(np.float32)

    logger.info(f"Clinical data cleaning complete. Final shape: {df_final.shape}")
    return df_final

def prune_network(adjacency_matrix: pd.DataFrame, weight_threshold: float = 0.0) -> pd.DataFrame:
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

    pruned_adjacency = nx.to_pandas_adjacency(G, dtype=float)
    current_nodes = G.number_of_nodes()
    current_edges = G.number_of_edges()

    logger.info(f"Pruning network with weight threshold: {weight_threshold}")
    logger.info(f"Number of nodes in full network: {total_nodes}")
    logger.info(f"Number of edges in full network: {total_edges}")
    logger.info(f"Number of nodes after pruning: {current_nodes}")
    logger.info(f"Number of edges after pruning: {current_edges}")

    return pruned_adjacency

def prune_network_by_quantile(adjacency_matrix: pd.DataFrame, quantile: float = 0.5) -> pd.DataFrame:
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

    weight_threshold = float(np.quantile(weights, quantile))
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
