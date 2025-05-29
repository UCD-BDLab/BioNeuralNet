import pandas as pd
import numpy as np
import networkx as nx

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression
from statsmodels.stats.multitest import multipletests

from .logger import get_logger
logger = get_logger(__name__)

def preprocess_clinical(X: pd.DataFrame, y: pd.Series, top_k: int = 10, scale: bool = False, ignore_columns=None) -> pd.DataFrame:
    """
    Preprocess clinical data, handling numeric and categorical features, cleaning, optional scaling, and selecting top features by RandomForest importance.

    Args:

        - X (pd.DataFrame): Clinical feature matrix (samples x features) including numeric and categorical columns.
        - y (pd.Series or pd.DataFrame): Target values; single-column DataFrame or Series of length n_samples.
        - top_k (int): Number of features to select based on importance.
        - scale (bool): If True, scale numeric features using RobustScaler; default is False.
        - ignore_columns (list): List of columns to ignore during preprocessing; default is None.

    Returns:

        - pd.DataFrame: Subset of the original features with the selected top_k features plus ignored columns.
    """
    # Align and validate y
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be a Series or single-column DataFrame")
        y_series = y.iloc[:, 0]
    elif isinstance(y, pd.Series):
        y_series = y.copy()
    else:
        raise ValueError("y must be a pandas Series or single-column DataFrame")

    ignore_columns = ignore_columns or []
    missing = set(ignore_columns) - set(X.columns)
    if missing:
        raise KeyError(f"Ignored columns not in X: {missing}")
    df_ignore = X[ignore_columns].copy()
    X = X.drop(columns=ignore_columns)

    df_numeric = X.select_dtypes(include="number")
    df_categorical = X.select_dtypes(include=["object", "category", "bool"])
    df_numeric_clean = clean_inf_nan(df_numeric)

    if scale:
        scaler = RobustScaler()
        scaled_array = scaler.fit_transform(df_numeric_clean)
        df_numeric_scaled = pd.DataFrame(scaled_array,columns=df_numeric_clean.columns,index=df_numeric_clean.index)
    else:
        df_numeric_scaled = df_numeric_clean.copy()

    if not df_categorical.empty:
        df_cat_filled = df_categorical.fillna("Missing").astype(str)
        df_cat_encoded = pd.get_dummies(df_cat_filled, drop_first=True)
    else:
        df_cat_encoded = pd.DataFrame(index=df_numeric_scaled.index)

    df_combined = pd.concat([df_numeric_scaled, df_cat_encoded],axis=1,join="inner")
    df_features = df_combined.loc[:, df_combined.std(axis=0) > 0]

    if y_series.nunique() <= 10:
        model = RandomForestClassifier(n_estimators=150,random_state=119,class_weight="balanced")
    else:
        model = RandomForestRegressor(n_estimators=150,random_state=119)

    model.fit(df_features, y_series)
    importances = model.feature_importances_
    feature_names = df_features.columns.tolist()

    order = list(np.argsort(importances))
    descending = []

    for i in range(len(order) - 1, -1, -1):
        descending.append(order[i])

    if top_k < len(descending):
        count = top_k
        logger.info(f"Selected top {count} features by RandomForest importance")
    else:
        count = len(descending)
        logger.info(f"Selected all {count} features by RandomForest importance")

    selected_idx = []
    for i in range(count):
        selected_idx.append(descending[i])

    selected_columns = []
    for idx in selected_idx:
        selected_columns.append(feature_names[idx])

    df_selected = df_features[selected_columns].copy()

    for col in df_ignore.columns:
        if col in selected_columns:
            df_selected[col] = df_ignore[col]

    return df_features[selected_columns]

def clean_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinite values with NaN, impute NaNs with the column median, and drop zero-variance columns.

    Args:

        - df (pd.DataFrame): Input DataFrame containing numeric columns.

    Returns:

        - pd.DataFrame: Cleaned DataFrame with no infinite or NaN values and no zero-variance columns.
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

def select_top_k_variance(df: pd.DataFrame, k: int = 1000, ddof: int = 0) -> pd.DataFrame:
    """
    Select the top k features with the highest variance.,

    Args:

        - df (pd.DataFrame): Input DataFrame; non-numeric columns will be ignored.
        - k (int): Number of top-variance features to select.
        - ddof (int): Delta degrees of freedom for varianceg calculation; default is 0.

    Returns:

        - pd.DataFrame: DataFrame containing only the top k features by variance.
    """
    df_clean = clean_inf_nan(df)
    num = df_clean.select_dtypes(include=[np.number]).copy()
    variances = num.var(axis=0, ddof=ddof)

    k = min(k, len(variances))
    top_cols = variances.nlargest(k).index.tolist()
    logger.info(f"Selected top {len(top_cols)} features by variance")

    return num[top_cols]

def select_top_k_correlation(X: pd.DataFrame, y: pd.Series = None, top_k: int = 1000) -> pd.DataFrame:
    """
    Select the top k features by correlation, either supervised (with respect to y) or unsupervised (redundancy minimization).

    Args:

        - X (pd.DataFrame): Numeric feature matrix (samples x features).
        - y (pd.Series, optional): Target values for supervised selection; if None, performs unsupervised selection.
        - top_k (int): Number of features to select.

    Returns:

        - pd.DataFrame: Subset of X containing the selected features.

    Note:

        - Correlation computation can be expensive for large datasets.
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
    """
    Select the top k features using RandomForest feature importances.

    Args:

        - X (pd.DataFrame): Numeric feature matrix (samples x features); must contain only numeric columns.
        - y (pd.Series or pd.DataFrame): Target values; single-column DataFrame or Series.
        - top_k (int): Number of features to select.
        - seed (int): Random seed for the RandomForest model; default is 119.

    Returns:

        - pd.DataFrame: Subset of X containing the selected top_k features by importance.
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

def top_anova_f_features(X: pd.DataFrame, y: pd.Series,max_features: int, alpha: float = 0.05, task: str = "classification") -> pd.DataFrame:
    """
    Select top features based on ANOVA F-test (with false recovery rate correction).
    This function is suitable for both classification and regression tasks.

    Args:

        - X (pd.DataFrame): Numeric feature matrix (samples x features).
        - y (pd.Series): Target vector; categorical for classification or continuous for regression.
        - max_features (int): Maximum number of features to return.
        - alpha (float): Significance threshold for false recovery rate correction; default is 0.05.
        - task (str): 'classification' to use f_classif or 'regression' to use f_regression.

    Returns:

        - pd.DataFrame: Subset of X with the selected features, padded if necessary.
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
    """
    Prune a network based on a weight threshold, removing nodes with weak connections.

    Parameters:

        - adjacency_matrix (pd.DataFrame): The adjacency matrix of the network.
        - weight_threshold (float): Minimum weight to keep an edge (default: 0.0).

    Returns:

        - pd.DataFrame:
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
    """
    Prune a network by removing edges below a quantile-based weight threshold and dropping isolated nodes.

    Args:

        - adjacency_matrix (pd.DataFrame): Weighted adjacency matrix (nodes x nodes).
        - quantile (float): Quantile in [0,1] to compute weight threshold; default is 0.5.

    Returns:

        - pd.DataFrame: Pruned adjacency matrix with edges below the quantile threshold removed.
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
    """
    Remove rows and columns from adjacency matrix where the variance is below a threshold.

    Parameters:

        network (pd.DataFrame): Adjacency matrix.
        threshold (float): Variance threshold.

    Returns:

        pd.DataFrame: Filtered adjacency matrix.
    """
    logger.info(f"Removing low-variance rows/columns with threshold {threshold}.")
    variances = network.var(axis=0)
    valid_indices = variances[variances > threshold].index
    filtered_network = network.loc[valid_indices, valid_indices]
    logger.info(f"Original network shape: {network.shape}, Filtered shape: {filtered_network.shape}")
    return filtered_network

def network_remove_high_zero_fraction(network: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove rows and columns from adjacency matrix where the fraction of zero entries is higher than the threshold.

    Parameters:

        network (pd.DataFrame): Adjacency matrix.
        threshold (float): Zero-fraction threshold.

    Returns:

        pd.DataFrame: Filtered adjacency matrix.
    """
    logger.info(f"Removing high zero fraction features with threshold: {threshold}.")

    zero_fraction = (network == 0).sum(axis=0) / network.shape[0]
    valid_indices = zero_fraction[zero_fraction < threshold].index
    filtered_network = network.loc[valid_indices, valid_indices]
    logger.info(f"Original network shape: {network.shape}, Filtered shape: {filtered_network.shape}")

    return filtered_network
