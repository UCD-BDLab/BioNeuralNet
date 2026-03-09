import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression
from statsmodels.stats.multitest import multipletests

from .preprocess import clean_inf_nan
from .logger import get_logger

logger = get_logger(__name__)

def variance_threshold(df: pd.DataFrame, k: int = 1000, ddof: int = 0) -> pd.DataFrame:
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

def mad_filter(df: pd.DataFrame, n_keep: int) -> pd.DataFrame:
    """Select the top features by Median Absolute Deviation (MAD).

    The Median Absolute Deviation is calculated across samples for each feature, and the features with the highest MAD scores are retained.

    Args:

        df (pd.DataFrame): Input DataFrame from which to select high-MAD numeric features.
        n_keep (int): Maximum number of top features to keep in the output.

    Returns:

        pd.DataFrame: DataFrame containing only the top n_keep features ranked by MAD.

    """
    if n_keep >= df.shape[1]:
        return df
    
    mad = (df - df.median(axis=0)).abs().median(axis=0)
    top_features = mad.nlargest(n_keep).index
    
    return df[top_features]

def pca_loadings(df: pd.DataFrame, n_keep: int, n_components: int = 50, seed: int = 1883) -> pd.DataFrame:
    """Select features with the highest absolute loadings across the top principal components.

    The input data is scaled and PCA is applied. Feature importance is determined by weighting each principal component's loadings by its explained variance ratio and taking the maximum across all selected components.

    Args:

        df (pd.DataFrame): Input DataFrame from which to select features.
        n_keep (int): Maximum number of top features to keep in the output.
        n_components (int): Number of principal components to evaluate.

    Returns:

        pd.DataFrame: DataFrame containing only the top n_keep features with the highest PCA loadings.

    """
    if n_keep >= df.shape[1]:
        return df

    n_components = min(n_components, df.shape[0], df.shape[1])

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df.values)

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(scaled_values) 

    weighted_loadings = np.abs(pca.components_).T * pca.explained_variance_ratio_
    feature_importance = weighted_loadings.max(axis=1)
    
    importance_series = pd.Series(feature_importance, index=df.columns)
    top_features = importance_series.nlargest(n_keep).index
    
    return df[top_features]

def laplacian_score(df: pd.DataFrame, n_keep: int, k_neighbors: int = 5) -> pd.DataFrame:
    """Unsupervised feature selection via the Laplacian Score to address dimensionality.

    Evaluates a feature's ability to preserve the local manifold structure of the data. 
    The score is computed as the sum of squared differences between connected samples 
    weighted by the global network (W_ij), divided by the feature's variance. 
    Lower scores indicate higher importance (smoothness on the graph).

    Args:

        df (pd.DataFrame): Input DataFrame from which to select features
        n_keep (int): Maximum number of top features to retain.
        k_neighbors (int): Number of neighbors to use when building the k-NN graph.

    Returns:
    
        pd.DataFrame: DataFrame containing only the top n_keep features.
    """
    if n_keep >= df.shape[1]:
        return df

    X = StandardScaler().fit_transform(df.values)
    A = kneighbors_graph(X, n_neighbors=k_neighbors, mode='connectivity', metric='cosine', include_self=False)
    W = A.maximum(A.T)
    
    D = np.array(W.sum(axis=1)).flatten()
    D_sum = D.sum()
    
    mean_f = np.dot(X.T, D) / D_sum
    X_centered = X - mean_f[np.newaxis, :]
    
    den = np.dot(D, X_centered**2)
    W_X = W.dot(X_centered)
    f_W_f = np.sum(X_centered * W_X, axis=0)
    num = den - f_W_f
    
    laplacian_scores = num / (den + 1e-8)
    
    importance_series = pd.Series(laplacian_scores, index=df.columns)
    top_features = importance_series.nsmallest(n_keep).index
    
    return df[top_features]

def correlation_filter(X: pd.DataFrame, y: pd.Series | None = None, top_k: int = 1000) -> pd.DataFrame:
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

    if y is not None:
        logger.info("Selecting features by supervised correlation with y")
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

        def key_fn(k: str) -> float:
            return correlations[k]

        features = list(correlations.keys())
        features.sort(key=key_fn, reverse=True)
        selected = features[:top_k]

    else:
        logger.info("Selecting features by unsupervised correlation")
        correlations_matrix = numbers_only.corr().abs()

        for i in range(correlations_matrix.shape[0]):
            correlations_matrix.iat[i, i] = 0.0

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

def importance_rf(X: pd.DataFrame, y: pd.Series, top_k: int = 1000, seed: int = 119) -> pd.DataFrame:
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