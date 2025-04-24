import pandas as pd
import numpy as np
import networkx as nx

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif
from statsmodels.stats.multitest import multipletests

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

from .logger import get_logger
logger = get_logger(__name__)

def preprocess_clinical(X: pd.DataFrame, y: pd.Series, top_k: int = 10) -> pd.DataFrame:
    """Steps:
        Split out numerics and cats
        Clean numeric with clean_inf_nan
        Fill and encode categoricals
        Recombine, drop any constant columns
        Select top_k features with RandomForest importances
    """
    # y is a Series
    if isinstance(y, pd.Series):
        y = y.copy()
    elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    else:
        raise ValueError("y must be a Series or a DataFrame with one column")
    
    df_num = X.select_dtypes(include="number")
    df_cat = X.select_dtypes(include=["object", "category", "bool"])

    df_num = clean_inf_nan(df_num)
    
    if not df_cat.empty:
        df_cat = df_cat.fillna("Missing").astype(str)
        df_cat = pd.get_dummies(df_cat, drop_first=True)
    
    df_clean = pd.concat([df_num, df_cat], axis=1)
    df_clean = df_clean.loc[:, df_clean.std(axis=0) > 0]

    is_classif = (y.nunique() <= 10)
    model = RandomForestClassifier(n_estimators=100, random_state=119) if is_classif else RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(df_clean, y)
    importances = pd.Series(model.feature_importances_, index=df_clean.columns)
    top_features = importances.nlargest(min(top_k, len(importances))).index

    return df_clean[top_features]

def clean_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace Inf with NaN, fill NaNs with median, drop zero-variance columns.
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

def robust_scaler(X: pd.DataFrame) -> pd.DataFrame:
    """
    Simple wrapper to normalize the features using RobustScaler.
    
    Parameters:
        X (pd.DataFrame): Input DataFrame to be normalized.
        
    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

def select_top_k_variance(df: pd.DataFrame, k: int = 1000, ddof: int = 0) -> pd.DataFrame:
    """
    Select the top k features with highest variance from a DataFrame
    """
    df_clean = clean_inf_nan(df)
    num = df_clean.select_dtypes(include=[np.number]).copy()
    variances = num.var(axis=0, ddof=ddof)

    k = min(k, len(variances))
    top_cols = variances.nlargest(k).index.tolist()
    logger.info(f"Selected top {len(top_cols)} features by variance")

    return num[top_cols]

def top_anova_f_features(X: pd.DataFrame, y: pd.Series,max_features: int, alpha: float = 0.05) -> pd.DataFrame:
    """
    Select top features based on ANOVA F-test

    """
    df_clean = clean_inf_nan(X)
    num = df_clean.select_dtypes(include=[np.number]).copy()

    y_aligned = y.loc[num.index]
    F_vals, p_vals = f_classif(num, y_aligned.values)

    _, p_adj, _, _ = multipletests(p_vals, alpha=alpha, method="fdr_bh")
    significant = p_adj < alpha
    order_all = np.argsort(-F_vals)

    sig_idx = []
    non_sig_idx = []

    for i in order_all:
        if significant[i]:
            sig_idx.append(i)
        else:
            non_sig_idx.append(i)

    n_sig = significant.sum()
    chosen_sig = sig_idx[:max_features]

    if n_sig >= max_features:
        final_idx = chosen_sig
        n_pad = 0
    else:
        n_pad = max_features - n_sig
        final_idx = chosen_sig + non_sig_idx[:n_pad]

    logger.info(f"Selected {len(final_idx)} features by ANOVA: {n_sig} significant, {n_pad} padded")

    selected_cols = num.columns[final_idx]
    return num[selected_cols]

def top_features_autoencoder(
    X: pd.DataFrame,
    top_k: int = 1000,
    hidden_dim: int = 128,
    epochs: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
    log_interval: int = 100
) -> pd.DataFrame:
    """
    Select the top_k features using autoencoder.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    df_clean = clean_inf_nan(X)
    df_num = df_clean.select_dtypes(include=[np.number]).copy()

    Xt = torch.tensor(df_num.values, dtype=torch.float32).to(device)
    dataset = TensorDataset(Xt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # we could reuse autencoder from somef the other classes
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, hdim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hdim),
                nn.ReLU()
            )
            self.decoder = nn.Linear(hdim, input_dim)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder(df_num.shape[1], hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(Xt)
        if epoch % log_interval == 0 or epoch == 1:
            logger.info(f"AE epoch: {epoch}/{epochs} | with loss: {avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        weights = model.encoder[0].weight.abs().sum(dim=0).cpu().numpy()
    idx = np.argsort(-weights)[: min(top_k, len(weights))]
    selected_cols = df_num.columns[idx]

    return df_num[selected_cols]


def top_k_cov_tgt(df: pd.DataFrame, target: pd.Series, k: int = 100) -> pd.DataFrame:
    """
    Keep top k features by covariance with target.
    """
    df = df.copy()
    df = clean_inf_nan(df)

    tgt = target.loc[df.index].astype(float)
    num = df.select_dtypes(include="number")

    def compute_cov(col: pd.Series) -> float:
        return col.cov(tgt)

    cov = num.apply(compute_cov).abs()
    k = min(k, len(cov))
    top = cov.nlargest(k).index.tolist()
    logger.info(f"Top {k} features selected by covariance with target")

    return df.loc[:, top]

def top_k_avg_cov(df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """
    Keep top-k features by average covariance with all others.
    """
    df = df.copy()
    df = clean_inf_nan(df)

    num = df.select_dtypes(include="number")

    covmat = num.cov().abs()
    avg_cov = (covmat.sum(axis=1)- covmat.values.diagonal()) / (covmat.shape[0] - 1)
    k = min(k, len(avg_cov))
    top = avg_cov.nlargest(k).index.tolist()
    logger.info(f"Top {k} features selected by average covariance")

    return df.loc[:, top]

def prune_network(adjacency_matrix, weight_threshold=0.0):
    """
    Prune a network based on a weight threshold, removing nodes with weak connections.
    Parameters:

        adjacency_matrix (pd.DataFrame): The adjacency matrix of the network.
        weight_threshold (float): Minimum weight to keep an edge (default: 0.0).

    Returns:
    
        pd.DataFrame:
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
    Prune a network based on a quantile threshold for edge weights
    Expects a adjacency matrix 
    returns pd.DataFrame: The pruned network adjacency matrix
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
