import torch
import pandas as pd
import numpy as np
from typing import Optional
import torch.nn.functional as F
from sklearn.covariance import GraphicalLasso
from .logger import get_logger

logger = get_logger(__name__)

def gen_similarity_graph(X:pd.DataFrame, k:int = 15, metric:str = "cosine", mutual:bool = False, per_node:bool = True, self_loops:bool = False) -> pd.DataFrame:
    """
    Build a normalized k-nearest neighbors (kNN) similarity graph from feature vectors.

    The function computes pairwise `cosine` or `Euclidean` distances, sparsifies the matrix by keeping `top-k` neighbours per node (or by applying a global threshold), optionally prunes edges to mutual neighbours, and can add self-loops.

    Args:

        - X: Dataframe of shape (N, D) where, N(`rows`) is the number of subjects/samples and D(`columns`) represents the multi-omics features.
        - k: Number of neighbors to keep per node.
        - metric: `cosine` or `Euclidean` (uses gaussian kernel on distances).
        - mutual: If `True`, retain only mutual edges (i->j and j->i).
        - per_node: If `True`, use per-node `k`, else apply a global cutoff.
        - self_loops: If `True`, add a self-loop weight of 1.

    Returns:

        - DataFrame of shape (D, D) the normalized feature-feature (multi-omics) adjacency matrix (network).

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(X, pd.DataFrame):
        X = X.T
        nodes = X.index
        number_of_omics = len(nodes)
        x_torch = torch.tensor(X.values, dtype=torch.float32, device=device)
    else:
        raise TypeError("X must be a pandas.DataFrame")

    N = x_torch.size(0)
    k = min(k, N-1)

    # full similarity matrix
    if metric == "cosine":
        X_normalized = F.normalize(x_torch, p=2, dim=1)
        S = torch.mm(X_normalized, X_normalized.t())
    else:
        D2 = torch.cdist(x_torch, x_torch).pow(2)
        median_d2 = D2.median()
        S = torch.exp(-D2 / (median_d2 + 1e-8))

    # building the knn graph or global threshold mask
    if per_node:
        _, index = torch.topk(S, k=k+1, dim=1)
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        for i in range(N):
            for j in index[i, 1:k+1]:
                mask[i, j] = True
    else:
        flat = S.reshape(-1)
        threshold = torch.kthvalue(flat, k * N).values
        mask = S >= threshold
        mask.fill_diagonal_(False)

    # mutual pruning option
    if mutual:
        mask = torch.logical_and(mask, mask.t())

    # mask and add self loops if set to true
    A = S * mask.float()
    if self_loops:
        A = A + torch.eye(N, device=device, dtype=x_torch.dtype)

    # row normalize
    A = F.normalize(A, p=1, dim=1)

    A_numpy = A.cpu().numpy()
    final_graph =pd.DataFrame(A_numpy, index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(f"Please make sure your input X follows the description: A DataFrame (N, D) where, N(rows) is the number of subjects/samples and D(columns) represents the multi-omics features.")
        raise ValueError(f"Generated graph shape {final_graph.shape} does not match expected shape ({number_of_omics}, {number_of_omics}).")

    return final_graph


def gen_correlation_graph(X: pd.DataFrame, k: int = 15,method: str = 'pearson', mutual: bool = False, per_node: bool = True,threshold: Optional[float] = None, self_loops:bool = False) -> pd.DataFrame:
    """
    Build a normalized k-nearest neighbors (kNN) correlation graph from feature vectors.

    The function computes pairwise `pearson` or `spearman` correlations, sparsifies the matrix by keeping `top-k` neighbours per node (or by applying a global threshold), optionally prunes edges to mutual neighbours, and can add self-loops.

    Args:

        - X: Dataframe of shape (N, D) where, N(`rows`) is the number of subjects/samples and D(`columns`) represents the multi-omics features.
        - k: Number of neighbors to keep per node.
        - method: `pearson` or `spearman`.
        - mutual: If `True`, retain only mutual edges (i->j and j->i).
        - per_node: If `True`, use per-node `k`, else apply a global cutoff.
        - threshold: Correlation cutoff when `per_node` is `False`.
        - self_loops: If `True`, add a self-loop weight of 1.

    Returns:

        - DataFrame of shape (D, D) the normalized feature-feature (multi-omics) adjacency matrix (network).

    Note:

        - Correlation can be expensive to compute; this function is not recommended for very large datasets.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(X, pd.DataFrame):
        X = X.T
        nodes = X.index
        number_of_omics = len(nodes)
        x_torch = torch.tensor(X.values, dtype=torch.float32, device=device)
    else:
        raise TypeError("X must be a pandas.DataFrame")

    N = x_torch.size(0)

    # rank transform for Spearman
    if method == 'spearman':
        x_ranked = x_torch.argsort(dim=1).argsort(dim=1).float()
        x_correlation = x_ranked - x_ranked.mean(dim=1, keepdim=True)
    else:
        x_correlation = x_torch - x_torch.mean(dim=1, keepdim=True)

    # correlation with clamping and normalization
    num = torch.mm(x_correlation, x_correlation.t())
    sum_sq = (x_correlation**2).sum(dim=1, keepdim=True)
    denom = torch.sqrt(torch.mm(sum_sq, sum_sq.t())).clamp(min=1e-8)
    C = num / denom
    S = C.abs()

    # build mask
    if per_node:
        _, index = torch.topk(S, k=k+1, dim=1)
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        for i in range(N):
            for j in index[i, 1:k+1]:
                mask[i, j] = True
    else:
        if threshold is not None:
            mask = S >= threshold
            mask.fill_diagonal_(False)
        else:
            flat = S.reshape(-1)
            thresh = torch.kthvalue(flat, k * N).values
            mask = S >= thresh
            mask.fill_diagonal_(False)

    if mutual:
        mask = torch.logical_and(mask, mask.t())

    W = S * mask.float()
    if self_loops:
        W.fill_diagonal_(1.0)

    W = F.normalize(W, p=1, dim=1)
    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(f"Please make sure your input X follows the description: A DataFrame (N, D) where, N(rows) is the number of subjects/samples and D(columns) represents the multi-omics features.")
        raise ValueError(f"Generated graph shape {final_graph.shape} does not match expected shape ({number_of_omics}, {number_of_omics}).")

    return final_graph


def gen_threshold_graph(X:pd.DataFrame, b: float = 6.0,k: int = 15, mutual: bool = False, self_loops: bool = False) -> pd.DataFrame:
    """
    Build a normalized k-nearest neighbors (kNN) soft-threshold co-expression graph, similar to the network-construction step in WGCNA.

    The function computes absolute pair-wise Pearson correlations, applies apower-law soft threshold with exponent `b`, sparsifies the matrix bykeeping `top-k` neighbours per node, optionally prunes edges to mutualneighbours, and can add self-loops.

    Args:

        - X: Dataframe of shape (N, D) where, N(`rows`) is the number of subjects/samples and D(`columns`) represents the multi-omics features.
        - b: Soft-threshold exponent applied to absolute correlations.
        - k: Number of neighbors to keep per node.
        - mutual: If `True`, retain only mutual edges (i->j and j->i).
        - self_loops: If `True`, add a self-loop weight of 1.

    Returns:

        - DataFrame of shape (D, D) the normalized feature-feature (multi-omics) adjacency matrix (network).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(X, pd.DataFrame):
        X = X.T
        nodes = X.index
        number_of_omics = len(nodes)
        x_torch = torch.tensor(X.values, dtype=torch.float32, device=device)
    else:
        raise TypeError("X must be a pandas.DataFrame")

    N = x_torch.size(0)

    # pearson correlation matrix

    Xc = x_torch - x_torch.mean(dim=1, keepdim=True)
    num = torch.mm(Xc, Xc.t())
    sum_sq = (Xc**2).sum(dim=1, keepdim=True)
    denom = torch.sqrt(torch.mm(sum_sq, sum_sq.t())).clamp(min=1e-8)
    C = num / denom

    # threshold
    S = C.abs().pow(b)

    # mask building
    _, index = torch.topk(S, k=k+1, dim=1)
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)

    for i in range(N):
        for j in index[i, 1:k+1]:
            mask[i, j] = True
    if mutual:
        mask = torch.logical_and(mask, mask.t())

    # apply mask (and self-loops if set)
    W = S * mask.float()

    if self_loops:
        W.fill_diagonal_(1.0)

    W = F.normalize(W, p=1, dim=1)
    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(f"Please make sure your input X follows the description: A DataFrame (N, D) where, N(rows) is the number of subjects/samples and D(columns) represents the multi-omics features.")
        raise ValueError(f"Generated graph shape {final_graph.shape} does not match expected shape ({number_of_omics}, {number_of_omics}).")

    return final_graph

def gen_gaussian_knn_graph(X: pd.DataFrame,k: int = 15,sigma: Optional[float] = None ,mutual: bool = False, self_loops: bool = True) -> pd.DataFrame:
    """
    Build a normalized k-nearest neighbors (kNN) similarity graph using a Gaussian(RBF) kernel.

    The function computes pairwise Euclidean distances, converts them to similarities with a Gaussian kernel (bandwidth `sigma`; if `None`, the median-distance heuristic is used), sparsifies the matrix by keeping `top-k` neighbours per node, optionally prunes edges to mutual neighbours, and can add self-loops.

    Args:

        - X: Dataframe of shape (N, D) where, N(`rows`) is the number of subjects/samples and D(`columns`) represents the multi-omics features.
        - k: Number of neighbors to keep per node.
        - sigma: Bandwidth of the Gaussian kernel; if `None`, uses the median squared distance.
        - mutual: If `True`, retain only mutual edges (i->j and j->i).
        - self_loops: If `True`, add a self-loop weight of 1.

    Returns:

        - DataFrame of shape (D, D) the normalized feature-feature (multi-omics) adjacency matrix (network).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(X, pd.DataFrame):
        X = X.T
        nodes = X.index
        number_of_omics = len(nodes)
        X_torch = torch.tensor(X.values, dtype=torch.float32, device=device)
    else:
        raise TypeError("X must be a pandas.DataFrame")

    N = X_torch.size(0)

    D2 = torch.cdist(X_torch, X_torch).pow(2)
    if sigma is None:
        sigma = D2.median().item()

    S = torch.exp(-D2 / (2 * sigma))

    _, index = torch.topk(S, k=k+1, dim=1)
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)

    for i in range(N):
        for j in index[i, 1:k+1]:
            mask[i, j] = True

    if mutual:
        mask = torch.logical_and(mask, mask.t())

    W = S * mask.float()

    if self_loops:
        W.fill_diagonal_(1.0)

    W = F.normalize(W, p=1, dim=1)
    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(f"Please make sure your input X follows the description: A DataFrame (N, D) where, N(rows) is the number of subjects/samples and D(columns) represents the multi-omics features.")
        raise ValueError(f"Generated graph shape {final_graph.shape} does not match expected shape ({number_of_omics}, {number_of_omics}).")

    return final_graph


def gen_lasso_graph(X: pd.DataFrame, alpha: float = 0.01, self_loops: bool = False, max_iter:int = 500) -> pd.DataFrame:
    """
    Build a sparse network using Graphical Lasso (inverse-covariance estimation).

    The function fits a precision matrix with L1 regularization (`alpha`),converts the non-zero entries to edge weights, can add self-loops, and row-normalizes the result.

    Args:

        - X: Dataframe of shape (N, D) where, N(`rows`) is the number of subjects/samples and D(`columns`) represents the multi-omics features.
        - alpha: Regularization strength for Graphical Lasso; larger values yield sparser graphs.
        - self_loops: If `True`, add a self-loop weight of 1.

    Returns:

        - DataFrame of shape (D, D) the normalized feature-feature (multi-omics) adjacency matrix (network).
    """

    if isinstance(X, pd.DataFrame):
        nodes = X.columns
        number_of_omics = len(nodes)
        x_numpy = X.values
    else:
        raise TypeError("X must be a pandas.DataFrame")

    try:
        model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
        model.fit(x_numpy)

        P = torch.from_numpy(model.precision_).abs()
        W = (P + P.t()) / 2

    except FloatingPointError:
        logger.warning("Graphical Lasso failed to converge, using identity matrix instead.")
        N = len(nodes)
        W = torch.eye(N, dtype=torch.float32)

    if self_loops:
        diag_indices = torch.arange(W.size(0), device=W.device)
        W[diag_indices, diag_indices] += 1.0

    W = F.normalize(W, p=1, dim=1)
    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(f"Please make sure your input X follows the description: A DataFrame (N, D) where, N(rows) is the number of subjects/samples and D(columns) represents the multi-omics features.")
        raise ValueError(f"Generated graph shape {final_graph.shape} does not match expected shape ({number_of_omics}, {number_of_omics}).")

    return final_graph

def gen_mst_graph(X: pd.DataFrame, self_loops: bool = False) -> pd.DataFrame:
    """
    Build a minimum-spanning-tree (MST) graph from feature vectors.

    The function computes pairwise Euclidean distances, extracts the MST, can add self-loops, and row-normalizes the result.

    Args:

        - X: Dataframe of shape (N, D) where, N(`rows`) is the number of subjects/samples and D(`columns`) represents the multi-omics features.
        - self_loops: If `True`, add a self-loop weight of 1.

    Returns:

        - DataFrame of shape (D, D) the normalized feature-feature (multi-omics) adjacency matrix (network).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(X, pd.DataFrame):
        X = X.T
        nodes = X.index
        number_of_omics = len(nodes)
        X_torch = torch.tensor(X.values, dtype=torch.float32, device=device)
    else:
        raise TypeError("X must be a pandas.DataFrame")

    N = X_torch.size(0)
    D = torch.cdist(X_torch, X_torch)

    visited = torch.zeros(N, dtype=torch.bool, device=device)
    visited[0] = True

    min_cost = D[0].clone()
    closest = torch.zeros(N, dtype=torch.long, device=device)
    mst = torch.zeros((N, N), dtype=X_torch.dtype, device=device)

    for tmp in range(N - 1):
        cost_masked = min_cost.clone()
        cost_masked[visited] = float('inf')

        j = torch.argmin(cost_masked)
        i = closest[j].item()
        w = D[i, j]

        mst[i, j] = w
        mst[j, i] = w

        visited[j] = True
        distance_j = D[j]

        update = torch.logical_not(visited) & (distance_j < min_cost)
        min_cost = torch.where(update, distance_j, min_cost)
        closest = torch.where(update, j, closest)

    W = mst
    if self_loops:
        W.fill_diagonal_(1.0)

    W = F.normalize(W, p=1, dim=1)
    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(f"Please make sure your input X follows the description: A DataFrame (N, D) where, N(rows) is the number of subjects/samples and D(columns) represents the multi-omics features.")
        raise ValueError(f"Generated graph shape {final_graph.shape} does not match expected shape ({number_of_omics}, {number_of_omics}).")

    return final_graph

def gen_snn_graph(X: pd.DataFrame,k: int = 15,mutual: bool = False, self_loops: bool = False) -> pd.DataFrame:
    """
    Build a shared-nearest-neighbor (SNN) graph from feature vectors.

    The function first finds the `top-k` nearest neighbours for each node,counts how many neighbours two nodes share, converts that count to anSNN similarity score, optionally prunes edges to mutual neighbours, and can add self-loops.

    Args:

        - X: Dataframe of shape (N, D) where, N(`rows`) is the number of subjects/samples and D(`columns`) represents the multi-omics features.
        - k: Number of neighbors to keep per node.
        - mutual: If `True`, retain only mutual edges (i->j and j->i).
        - self_loops: If `True`, add a self-loop weight of 1.

    Returns:

        - DataFrame of shape (D, D) the normalized feature-feature (multi-omics) adjacency matrix (network).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(X, pd.DataFrame):
        X = X.T
        nodes = X.index
        number_of_omics = len(nodes)
        X_torch = torch.tensor(X.values, dtype=torch.float32, device=device)
    else:
        raise TypeError("X must be a pandas.DataFrame")

    N = X_torch.size(0)
    S = torch.mm(X_torch, X_torch.t())
    _, index = torch.topk(S, k=k+1, dim=1)

    neighbors = []
    for i in range(N):
        neighbors.append(set(index[i, 1:k+1].tolist()))

    W = torch.zeros((N, N), dtype=X_torch.dtype, device=device)

    for i in range(N):
        for j in index[i, 1:k+1]:
            j = j.item()
            shared = len(neighbors
            [i].intersection(neighbors[j]))
            W[i, j] = shared

    if mutual:
        W = W.clone()
        W = torch.min(W, W.t())

    if self_loops:
        W.fill_diagonal_(1.0)

    W = F.normalize(W, p=1, dim=1)
    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(f"Please make sure your input X follows the description: A DataFrame (N, D) where, N(rows) is the number of subjects/samples and D(columns) represents the multi-omics features.")
        raise ValueError(f"Generated graph shape {final_graph.shape} does not match expected shape ({number_of_omics}, {number_of_omics}).")

    return final_graph
