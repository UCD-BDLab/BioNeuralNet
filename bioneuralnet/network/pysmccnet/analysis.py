import os
import torch
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from collections import defaultdict
from typing import Union, List, Optional

from .math_helpers import r_vec_mult_sum

def data_preprocess(X: Union[pd.DataFrame, np.ndarray], covariates: Optional[Union[pd.DataFrame, np.ndarray]] = None, is_cv: bool = False, cv_quantile: float = 0.0, center: bool = True, scale: bool = True, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64) -> pd.DataFrame:
    """PyTorch version of data_preprocess for omics dataset preparation.

    Args:

        X (pd.DataFrame | np.ndarray): Input omics data matrix.
        covariates (pd.DataFrame | np.ndarray | None): Optional covariates to regress out.
        is_cv (bool): If True, filter features based on coefficient of variation.
        cv_quantile (float): Quantile threshold for CV filtering; required if is_cv is True.
        center (bool): If True, center columns to mean zero.
        scale (bool): If True, scale columns to unit variance.
        device (torch.device | None): Calculation device; defaults to GPU if available.
        dtype (torch.dtype): Data type for tensor computations.

    Returns:

        pd.DataFrame: Preprocessed data frame.

    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    data = X.copy()

    # cv filtering
    if is_cv:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_tensor = torch.tensor(data.values, dtype=dtype, device=device)
        N = data_tensor.shape[0]
        col_mean = data_tensor.mean(dim=0)
        col_std = torch.sqrt(torch.sum((data_tensor - col_mean.unsqueeze(0)) ** 2, dim=0) / (N - 1))

        safe_mean = torch.where(col_mean == 0, torch.ones_like(col_mean), col_mean)
        cv_values = torch.abs(col_std / safe_mean).cpu().numpy()

        if cv_quantile is None:
            raise ValueError("cv filtering quantile must be provided!")

        thresh = np.quantile(cv_values, cv_quantile)
        keep_mask = cv_values > thresh
        data = data.loc[:, keep_mask]

    # center and scale
    if center:
        data = data - data.mean(axis=0)

    if scale:
        std_devs = data.std(axis=0, ddof=1)
        std_devs[std_devs == 0] = 1
        data = data / std_devs

    # covariate adjustment
    if covariates is not None:
        if not isinstance(covariates, pd.DataFrame):
            covariates = pd.DataFrame(covariates)

        if covariates.shape[1] == 0:
            raise ValueError('Covariate dataframe must have at least 1 column!')

        if not data.index.equals(covariates.index):
            common_idx = data.index.intersection(covariates.index)
            data = data.loc[common_idx]
            covariates = covariates.loc[common_idx]

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_tensor = torch.tensor(data.values, dtype=dtype, device=device)
        cov_tensor = torch.tensor(covariates.values, dtype=dtype, device=device)

        ones_col = torch.ones((cov_tensor.shape[0], 1), device=device, dtype=dtype)
        X_cov = torch.cat([ones_col, cov_tensor], dim=1)

        result = torch.linalg.lstsq(X_cov, data_tensor)
        beta = result.solution

        residuals = data_tensor - torch.matmul(X_cov, beta)

        data = pd.DataFrame(residuals.cpu().numpy(), index=data.index, columns=data.columns)

    return data

def get_can_cor_multi(X: List[torch.Tensor], cc_coef: Union[np.ndarray, torch.Tensor, List[float]], cc_weight: List[Union[torch.Tensor, np.ndarray]], Y: Union[torch.Tensor, np.ndarray]) -> float:
    """PyTorch version of get_can_cor_multi calculating canonical correlation value on GPU.

    Args:

        X (List[torch.Tensor]): List of data matrices.
        cc_coef (np.ndarray | torch.Tensor | List[float]): Correlation coefficients / weights.
        cc_weight (List[torch.Tensor | np.ndarray]): List of weight vectors for projection.
        Y (torch.Tensor | np.ndarray): Phenotype data vector.

    Returns:

        float: Total canonical correlation (between-omics + omics-phenotype).

    """
    device = X[0].device
    dtype = X[0].dtype

    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(np.array(Y), device=device, dtype=dtype)

    if not isinstance(cc_coef, torch.Tensor):
        cc_coef = torch.tensor(np.array(cc_coef), device=device, dtype=dtype)

    num_datasets = len(X)

    # projections
    projections = []
    for i in range(num_datasets):
        w = cc_weight[i] if isinstance(cc_weight[i], torch.Tensor) else torch.tensor(np.array(cc_weight[i]), device=device, dtype=dtype)

        if w.ndim == 1:
            w = w.view(-1, 1)

        proj = torch.matmul(X[i], w)
        projections.append(proj.reshape(-1, 1))

    omics_projection = torch.cat(projections, dim=1)

    # correlation matrix
    k = omics_projection.shape[1]
    centered = omics_projection - omics_projection.mean(dim=0, keepdim=True)

    N = omics_projection.shape[0]
    cov_mat = torch.matmul(centered.t(), centered) / (N - 1)
    std_vec = torch.sqrt(torch.diag(cov_mat))

    std_outer = std_vec.unsqueeze(1) * std_vec.unsqueeze(0)
    std_outer = torch.where(std_outer == 0, torch.ones_like(std_outer), std_outer)

    omics_cor_mat = cov_mat / std_outer

    # between-omics correlations
    if k > 1:
        row_idx, col_idx = torch.triu_indices(k, k, offset=1)
        omics_cor_vec = omics_cor_mat[row_idx, col_idx]
    else:
        omics_cor_vec = torch.tensor([], device=device, dtype=dtype)

    cc_coef_between = cc_coef[:num_datasets]
    cc_between = r_vec_mult_sum(cc_coef_between, omics_cor_vec)

    # omics-phenotype correlations
    y_flat = Y.flatten()
    y_centered = y_flat - torch.mean(y_flat)
    y_norm = torch.sqrt(torch.sum(y_centered ** 2))

    pheno_cor_list = []
    for i in range(k):
        col = omics_projection[:, i]
        col_centered = col - torch.mean(col)
        col_norm = torch.sqrt(torch.sum(col_centered ** 2))

        denom = col_norm * y_norm
        if denom == 0:
            corr = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            corr = torch.sum(col_centered * y_centered) / denom

        if torch.isnan(corr):
            corr = torch.tensor(0.0, device=device, dtype=dtype)

        pheno_cor_list.append(corr)

    pheno_cor_vec = torch.stack(pheno_cor_list)
    cc_coef_pheno = cc_coef[num_datasets:]
    pheno_cor_val = r_vec_mult_sum(pheno_cor_vec, cc_coef_pheno)

    return cc_between + pheno_cor_val


def get_abar(ws: Union[pd.DataFrame, np.ndarray, torch.Tensor, List[float]], feature_label: Optional[List[str]] = None) -> pd.DataFrame:
    """PyTorch equivalent of get_abar performing matrix multiplication on GPU.

    Args:

        ws (pd.DataFrame | np.ndarray | torch.Tensor | List[float]): Weight matrix or vector.
        feature_label (List[str] | None): List of feature names for the output DataFrame.

    Returns:

        pd.DataFrame: Adjacency matrix (A-bar) representing feature similarity.

    """
    # convert to tensor
    if isinstance(ws, pd.DataFrame):
        ws = torch.tensor(ws.values, dtype=torch.float32)
    elif not isinstance(ws, torch.Tensor):
        ws = torch.tensor(ws, dtype=torch.float32)

    w_abs = torch.abs(ws)

    # compute similarity matrix
    if w_abs.ndim == 1:
        abar = torch.outer(w_abs, w_abs)
    else:
        abar = torch.matmul(w_abs, w_abs.T)

    # zero diagonal
    abar.fill_diagonal_(0)

    # normalize
    max_val = torch.max(abar)
    if max_val > 0:
        abar = abar / max_val

    # format output
    if feature_label is None:
        raise ValueError("Need to provide FeatureLabel.")

    if len(feature_label) != abar.shape[0]:
        raise ValueError(f"FeatureLabel length ({len(feature_label)}) does not match matrix rows ({abar.shape[0]}).")

    abar_cpu = abar.detach().cpu().numpy()

    return pd.DataFrame(abar_cpu, index=feature_label, columns=feature_label)

def get_omics_modules(Abar: pd.DataFrame, cut_height: float = 1 - 0.1**10) -> List[List[int]]:
    """Extract omics modules via hierarchical clustering on the similarity matrix.

    Args:

        Abar (pd.DataFrame): Similarity/adjacency matrix for all features.
        cut_height (float): Height threshold for hierarchical tree cutting.

    Returns:

        List[List[int]]: Each inner list contains 0-based feature indices belonging to a module.

    """
    abar_mat = Abar.to_numpy().copy()
    np.fill_diagonal(abar_mat, 0)

    # distance = 1 - similarity
    dist_mat = 1.0 - abar_mat
    dist_mat = (dist_mat + dist_mat.T) / 2.0
    np.fill_diagonal(dist_mat, 0)
    dist_mat = np.clip(dist_mat, 0, None)

    dist_condensed = squareform(dist_mat, checks=False)

    # complete linkage
    Z = linkage(dist_condensed, method='complete')

    labels = fcluster(Z, t=cut_height, criterion='distance')

    # Logic here only keep leaf nodes from merges below cut_height
    merged_below = Z[:, 2] < cut_height

    n = abar_mat.shape[0]
    lower_leaves = set()
    for i, below in enumerate(merged_below):
        if below:
            left, right = int(Z[i, 0]), int(Z[i, 1])
            if left < n:
                lower_leaves.add(left)
            if right < n:
                lower_leaves.add(right)

    lower_leaves = sorted(lower_leaves)

    if len(lower_leaves) == 0:
        return []

    # group by cluster assignment
    leaf_labels = {leaf: labels[leaf] for leaf in lower_leaves}
    groups = defaultdict(list)
    for leaf in lower_leaves:
        groups[leaf_labels[leaf]].append(leaf)

    return [sorted(indices) for indices in groups.values() if len(indices) > 0]


def summarize_netshy(X: Union[pd.DataFrame, np.ndarray], A: Union[pd.DataFrame, np.ndarray], npc: int = 1) -> dict:
    """NetSHy network summarization via hybrid approach leveraging topological properties.

    Summarizes a subnetwork by projecting omics data through the graph Laplacian, then extracting principal components from the projected space.

    Source: summarizeNetSHy (Vu et al., Bioinformatics 2023)

    Args:

        X (pd.DataFrame | np.ndarray): Data matrix of shape (n_samples, n_features).
        A (pd.DataFrame | np.ndarray): Adjacency matrix of shape (n_features, n_features).
        npc (int): Number of principal components for summarization.

    Returns:

        dict: Keys are 'scores' (n, npc), 'importance' (sdev, variance_pct, cumulative_pct), and 'loadings' (n_features, npc).

    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()

    X = X.astype(float)
    A = A.astype(float)

    # unnormalized Laplacian: L = D - A
    D = np.diag(np.sum(A, axis=1))
    L = D - A

    X_scaled = StandardScaler().fit_transform(X)

    # Laplacian-weighted projection (temp = X %*% L2)
    X_L = X_scaled @ L

    # PCA on projected data
    n_components = min(npc, X_L.shape[1], X_L.shape[0])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_L)

    importance = {
        'sdev': pca.singular_values_ / np.sqrt(X_L.shape[0] - 1),
        'variance_pct': pca.explained_variance_ratio_,
        'cumulative_pct': np.cumsum(pca.explained_variance_ratio_)
    }

    # loadings are (features, components)
    loadings = pca.components_.T

    return {
        'scores': scores[:, :npc],
        'importance': importance,
        'loadings': loadings[:, :npc]
    }


def prune_modules(Abar: pd.DataFrame, X_combined: np.ndarray, Y: np.ndarray, modules: List[List[int]], feature_labels: List[str], min_size: int = 10, max_size: int = 100, summarization: str = 'NetSHy', saving_dir: str = '.') -> List[dict]:
    """Prune network modules to target size range and compute summarization scores.

    For each module from hierarchical clustering, iteratively removes the lowest-degree node until the module fits within [min_size, max_size]. Then computes NetSHy summarization scores and per-feature phenotype correlations.

    Args:

        Abar (pd.DataFrame): Global adjacency matrix.
        X_combined (np.ndarray): Column-bound omics data of shape (n_samples, n_total_features).
        Y (np.ndarray): Phenotype vector of shape (n_samples,).
        modules (List[List[int]]): Feature index groups from get_omics_modules.
        feature_labels (List[str]): Feature names matching columns of X_combined.
        min_size (int): Minimum module size to retain.
        max_size (int): Maximum module size; larger modules are pruned down.
        summarization (str): Summarization method. Currently only 'NetSHy' is supported.
        saving_dir (str): Directory to save per-module pickle files.

    Returns:

        List[dict]: One dict per valid module with keys: module_id, nodes, node_indices, adjacency, correlation, pc_correlations, netshy, omics_correlation.

    """
    import pickle

    if summarization != 'NetSHy':
        raise ValueError(f"Unsupported summarization method '{summarization}'. Only 'NetSHy' is supported.")

    abar_mat = Abar.to_numpy().copy()
    Y = np.array(Y).flatten()

    print(f"\nThere are {len(modules)} network modules before pruning")

    results = []

    for mod_idx, module_indices in enumerate(modules):
        if len(module_indices) < min_size:
            continue

        print(f" Now extracting subnetwork for network module {mod_idx + 1}")

        current_indices = list(module_indices)

        # iteratively remove lowest-degree node until within max_size
        while len(current_indices) > max_size:
            sub_abar = abar_mat[np.ix_(current_indices, current_indices)]
            degrees = np.sum(sub_abar != 0, axis=1)

            min_degree = np.min(degrees)
            min_candidates = np.where(degrees == min_degree)[0]

            # break ties by lowest total edge weight
            if len(min_candidates) > 1:
                weight_sums = np.array([np.sum(sub_abar[c, :]) for c in min_candidates])
                remove_local = min_candidates[np.argmin(weight_sums)]
            else:
                remove_local = min_candidates[0]

            current_indices.pop(remove_local)

        if len(current_indices) < min_size:
            continue

        # extract subnetwork
        node_names = [feature_labels[i] for i in current_indices]
        sub_abar = abar_mat[np.ix_(current_indices, current_indices)]
        sub_abar_df = pd.DataFrame(sub_abar, index=node_names, columns=node_names)

        X_sub = X_combined[:, current_indices]

        # NetSHy summarization with 3 PCs
        netshy_result = summarize_netshy(X_sub, sub_abar, npc=3)

        # correlation of each PC with phenotype
        pc_correlations = []
        for pc_idx in range(netshy_result['scores'].shape[1]):
            pc_score = netshy_result['scores'][:, pc_idx]
            try:
                corr, _ = pearsonr(pc_score, Y)
                pc_correlations.append(abs(corr))
            except Exception:
                pc_correlations.append(0.0)

        max_corr = max(pc_correlations) if pc_correlations else 0.0

        # per-feature correlation with phenotype
        omics_corr = {}
        for j, feat_idx in enumerate(current_indices):
            try:
                c, _ = pearsonr(X_combined[:, feat_idx], Y)
                omics_corr[feature_labels[feat_idx]] = c
            except Exception:
                omics_corr[feature_labels[feat_idx]] = 0.0

        print(f"Network module {mod_idx + 1} Result:"
              f"The final network size is: {len(current_indices)} "
              f"with maximum PC correlation w.r.t. phenotype to be: {max_corr:.3f}")

        module_result = {
            'module_id': mod_idx + 1,
            'nodes': node_names,
            'node_indices': current_indices,
            'adjacency': sub_abar_df,
            'correlation': max_corr,
            'pc_correlations': pc_correlations,
            'netshy': netshy_result,
            'omics_correlation': pd.Series(omics_corr)
        }
        results.append(module_result)

        # save subnetwork
        save_name = f"size_{len(current_indices)}_net_{mod_idx + 1}.pkl"
        with open(os.path.join(saving_dir, save_name), 'wb') as f:
            pickle.dump(module_result, f)

    return results
