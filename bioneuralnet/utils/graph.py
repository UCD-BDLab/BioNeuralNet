import torch
import numpy as np
import pandas as pd
from typing import Optional
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLasso

from .logger import get_logger
logger = get_logger(__name__)

def gen_similarity_graph(X: pd.DataFrame, k: int = 15, metric: str = "cosine", mutual: bool = False, per_node: bool = True, self_loops: bool = False, normalize: bool = True) -> pd.DataFrame:
    """Build a k-nearest neighbors similarity graph from feature vectors.

    Pairwise similarities are computed using either cosine similarity or a Gaussian kernel on Euclidean distances. The similarity matrix is sparsified by keeping top-k neighbors per node (or via a global cutoff), optionally restricted to mutual neighbors, with optional self-loops and row-normalization.

    Args:

        X (pd.DataFrame): Input data of shape (N, D) where N is the number of samples and D is the number of features.
        k (int): Number of neighbors to keep per node, or approximate neighbors per node when using a global cutoff.
        metric (str): Similarity metric; either "cosine" or "euclidean" (case-insensitive) where the latter uses a Gaussian kernel on squared distances.
        mutual (bool): If True, retain only edges where i is in the kNN of j and j is in the kNN of i.
        per_node (bool): If True, apply kNN per node; if False, apply a global threshold to keep approximately k edges per node.
        self_loops (bool): If True, add self-loop weights of 1 on the diagonal of the adjacency matrix.
        normalize (bool): If True, row-normalize the adjacency so each row sums to 1.

    Returns:

        pd.DataFrame: Adjacency matrix of shape (D, D) representing the feature-feature similarity graph.

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
    k = min(k, N - 1)

    # Full similarity matrix
    if metric == "cosine":
        X_normalized = F.normalize(x_torch, p=2, dim=1)
        S = torch.mm(X_normalized, X_normalized.t())
    else:
        D2 = torch.cdist(x_torch, x_torch).pow(2)
        median_d2 = D2.median()
        S = torch.exp(-D2 / (median_d2 + 1e-8))

    # kNN mask or global threshold
    if per_node:
        _, index = torch.topk(S, k=k + 1, dim=1)
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        for i in range(N):
            for j in index[i, 1:k + 1]:
                mask[i, j] = True
    else:
        flat = S.reshape(-1)
        threshold = torch.kthvalue(flat, k * N).values
        mask = S >= threshold
        mask.fill_diagonal_(False)

    #  pruning option
    if mutual:
        mask = torch.logical_and(mask, mask.t())

    # mask and add self-loops
    A = S * mask.float()
    if self_loops:
        A = A + torch.eye(N, device=device, dtype=x_torch.dtype)

    # row normalization
    if normalize:
        A = F.normalize(A, p=1, dim=1)

    A_numpy = A.cpu().numpy()
    final_graph = pd.DataFrame(A_numpy, index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(
            "Please make sure your input X follows the description: "
            "A DataFrame (N, D) where N (rows) is the number of subjects/samples "
            "and D (columns) represents the multi-omics features."
        )
        raise ValueError(
            f"Generated graph shape {final_graph.shape} does not match expected "
            f"shape ({number_of_omics}, {number_of_omics})."
        )

    return final_graph

def gen_correlation_graph(X: pd.DataFrame, k: Optional[int] = 15, method: str = "pearson", signed: bool = True, normalize: bool = True, mutual: bool = False, per_node: bool = True, threshold: Optional[float] = None, self_loops: bool = False) -> pd.DataFrame:
    """Build a correlation-based graph from feature vectors with optional kNN sparsification.

    Pairwise correlations (Pearson or Spearman) are computed between features, mapped to similarity scores in [0, 1], and then optionally sparsified using per-node kNN or a global cutoff. Mutual pruning, self-loops, and row-normalization can be applied to obtain a final adjacency matrix.

    Args:

        X (pd.DataFrame): Input data of shape (N, D) where N is the number of samples and D is the number of features.
        k (int | None): Number of neighbors for sparsification; when per_node is True this is per node, otherwise used to approximate k*N edges globally, and if None with threshold=None a fully connected graph is returned subject to self_loops.
        method (str): Correlation method; "pearson" for standard correlation or "spearman" for rank-based correlation.
        signed (bool): If True, use signed correlations mapped to [0, 1] via (C + 1)/2; if False, use absolute correlations in [0, 1].
        normalize (bool): If True, row-normalize the adjacency; if False, keep raw similarity weights.
        mutual (bool): If True, retain only edges that are present in both directions (i->j and j->i).
        per_node (bool): If True, apply kNN per node; if False, use a global cutoff determined by k or threshold.
        threshold (float | None): Similarity cutoff; when provided and per_node is False, overrides the k-based global cutoff.
        self_loops (bool): If True, add self-loop weights of 1 on the diagonal of the adjacency matrix.

    Returns:

        pd.DataFrame: Adjacency matrix of shape (D, D) representing the feature-feature correlation graph.

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

    # Rank transform for Spearman, otherwise mean-center for Pearson
    if method == "spearman":
        x_ranked = x_torch.argsort(dim=1).argsort(dim=1).float()
        x_correlation = x_ranked - x_ranked.mean(dim=1, keepdim=True)
    else:
        x_correlation = x_torch - x_torch.mean(dim=1, keepdim=True)

    num = torch.mm(x_correlation, x_correlation.t())
    sum_sq = (x_correlation ** 2).sum(dim=1, keepdim=True)
    denom = torch.sqrt(torch.mm(sum_sq, sum_sq.t())).clamp(min=1e-8)
    C = num / denom

    # mapping to similarity matrix S in [0, 1]
    if signed:
        S = (C + 1) / 2
    else:
        S = C.abs()

    # fully connected, per-node kNN, or global threshold
    if k is None and threshold is None:
        # Fully connected mode
        mask = torch.ones(N, N, dtype=torch.bool, device=device)
        if not self_loops:
            mask.fill_diagonal_(False)

    elif per_node:
        if k is None:
            raise ValueError("k must be an integer when per_node is True.")
        k_to_use = min(k + 1, N)
        _, index = torch.topk(S, k=k_to_use, dim=1)
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        mask.scatter_(1, index[:, 1:k_to_use], True)

    else:
        # cutoff mode
        if threshold is not None:
            mask = S >= threshold
            mask.fill_diagonal_(False)
        else:
            k_global = min(k * N, N * N - N)
            flat_off_diag = S[~torch.eye(N, dtype=torch.bool, device=device)].reshape(-1)
            thresh_val = torch.kthvalue(
                flat_off_diag, len(flat_off_diag) - k_global
            ).values
            mask = S >= thresh_val
            mask.fill_diagonal_(False)

    if mutual:
        mask = torch.logical_and(mask, mask.t())

    W = S * mask.float()
    if self_loops:
        W.fill_diagonal_(1.0)

    if normalize:
        W = F.normalize(W, p=1, dim=1)

    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(
            "Please make sure your input X follows the description: "
            "A DataFrame (N, D) where N (rows) is the number of subjects/samples "
            "and D (columns) represents the multi-omics features."
        )
        raise ValueError(
            f"Generated graph shape {final_graph.shape} does not match expected "
            f"shape ({number_of_omics}, {number_of_omics})."
        )

    return final_graph

def gen_threshold_graph(X: pd.DataFrame, b: float = 6.0, k: int = 15, mutual: bool = False, self_loops: bool = False, normalize: bool = True) -> pd.DataFrame:
    """Build a soft-thresholded kNN co-expression graph, similar to WGCNA-style networks.

    Absolute Pearson correlations between features are raised to a power b to obtain soft-thresholded similarities. A kNN mask keeps the top-k neighbors per node, optionally restricted to mutual neighbors, with optional self-loops and row-normalization.

    Args:

        X (pd.DataFrame): Input data of shape (N, D) where N is the number of samples and D is the number of features.
        b (float): Soft-threshold exponent applied to absolute correlations to control network sparsity and hub emphasis.
        k (int): Number of neighbors to keep per node in the kNN graph.
        mutual (bool): If True, retain only edges where i and j are mutual kNN neighbors.
        self_loops (bool): If True, add self-loop weights of 1 on the diagonal of the adjacency matrix.
        normalize (bool): If True, row-normalize the adjacency so each row sums to 1.

    Returns:

        pd.DataFrame: Adjacency matrix of shape (D, D) representing the soft-thresholded co-expression graph.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(X, pd.DataFrame):
        X = X.T  # features as nodes
        nodes = X.index
        number_of_omics = len(nodes)
        x_torch = torch.tensor(X.values, dtype=torch.float32, device=device)
    else:
        raise TypeError("X must be a pandas.DataFrame")

    N = x_torch.size(0)

    # Pearson correlation matrix
    Xc = x_torch - x_torch.mean(dim=1, keepdim=True)
    num = torch.mm(Xc, Xc.t())
    sum_sq = (Xc ** 2).sum(dim=1, keepdim=True)
    denom = torch.sqrt(torch.mm(sum_sq, sum_sq.t())).clamp(min=1e-8)

    C = num / denom
    S = C.abs().pow(b)

    _, index = torch.topk(S, k=k + 1, dim=1)
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(N):
        for j in index[i, 1:k + 1]:
            mask[i, j] = True

    if mutual:
        mask = torch.logical_and(mask, mask.t())

    W = S * mask.float()
    if self_loops:
        W.fill_diagonal_(1.0)

    if normalize:
        W = F.normalize(W, p=1, dim=1)

    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(
            "Please make sure your input X follows the description: "
            "A DataFrame (N, D) where N (rows) is the number of subjects/samples "
            "and D (columns) represents the multi-omics features."
        )
        raise ValueError(
            f"Generated graph shape {final_graph.shape} does not match expected "
            f"shape ({number_of_omics}, {number_of_omics})."
        )

    return final_graph

def gen_gaussian_knn_graph(X: pd.DataFrame, k: int = 15, sigma: Optional[float] = None, mutual: bool = False, self_loops: bool = True, normalize: bool = True) -> pd.DataFrame:
    """Build a Gaussian (RBF) kNN similarity graph from feature vectors.

    Pairwise Euclidean distances between features are converted to similarities using a Gaussian kernel with bandwidth sigma (or a median-distance heuristic). The graph is sparsified by keeping top-k neighbors per node, optionally restricted to mutual neighbors, with optional self-loops and row-normalization.

    Args:

        X (pd.DataFrame): Input data of shape (N, D) where N is the number of samples and D is the number of features.
        k (int): Number of neighbors to keep per node in the kNN graph.
        sigma (float | None): Bandwidth parameter for the Gaussian kernel; if None, a median squared distance heuristic is used.
        mutual (bool): If True, retain only edges where i and j are mutual kNN neighbors.
        self_loops (bool): If True, add self-loop weights of 1 on the diagonal of the adjacency matrix.
        normalize (bool): If True, row-normalize the adjacency so each row sums to 1.

    Returns:

        pd.DataFrame: Adjacency matrix of shape (D, D) representing the Gaussian-kernel feature similarity graph.

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

    # Pairwise squared distances
    D2 = torch.cdist(x_torch, x_torch).pow(2)

    if sigma is None:
        sigma = D2.median().item()

    # Gaussian kernel
    S = torch.exp(-D2 / (2 * sigma))

    # kNN mask
    _, index = torch.topk(S, k=k + 1, dim=1)
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(N):
        for j in index[i, 1:k + 1]:
            mask[i, j] = True

    if mutual:
        mask = torch.logical_and(mask, mask.t())

    # mask and self-loops
    W = S * mask.float()
    if self_loops:
        W.fill_diagonal_(1.0)

    if normalize:
        W = F.normalize(W, p=1, dim=1)

    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(
            "Please make sure your input X follows the description: "
            "A DataFrame (N, D) where N (rows) is the number of subjects/samples "
            "and D (columns) represents the multi-omics features."
        )
        raise ValueError(
            f"Generated graph shape {final_graph.shape} does not match expected "
            f"shape ({number_of_omics}, {number_of_omics})."
        )

    return final_graph

def gen_lasso_graph(X: pd.DataFrame, alpha: float = 0.01, tolerance: float = 0.004, self_loops: bool = False, max_iter: int = 500) -> pd.DataFrame:
    """
    Build a sparse network using Graphical Lasso (inverse-covariance estimation).

    - X: DataFrame (N samples, D features). Nodes = features.
    - If D is too large, falls back to correlation-based graph.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas.DataFrame")

    nodes = X.columns
    number_of_omics = len(nodes)
    x_numpy = X.values

    MAX_FEATURES_GLASSO = 1500
    if number_of_omics > MAX_FEATURES_GLASSO:
        logger.warning(
            f"gen_lasso_graph: n_features={number_of_omics} > {MAX_FEATURES_GLASSO}. "
            "Skipping Graphical Lasso and using correlation-based graph instead."
        )
        corr = torch.from_numpy(np.corrcoef(x_numpy.T)).float().abs()
        W = corr
        W.fill_diagonal_(0.0)
    else:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_numpy)

        try:
            model = GraphicalLasso(alpha=alpha, tol=tolerance, max_iter=max_iter)
            model.fit(x_scaled)

            P = torch.from_numpy(model.precision_).float()

            P_abs = P.abs()
            W = (P_abs + P_abs.t()) / 2
            W.fill_diagonal_(0.0)

            logger.info(
                f"Lasso: Non-zero off-diagonal elements: {(W > 0).sum().item()}"
            )
            if (W > 0).any():
                logger.info(
                    f"Lasso: Edge weight range: "
                    f"[{W[W > 0].min().item():.6f}, {W.max().item():.6f}]"
                )

        except Exception as e:
            logger.warning(f"Graphical Lasso failed: {e}. Using correlation-based graph.")
            corr = torch.from_numpy(np.corrcoef(x_numpy.T)).float().abs()
            W = corr
            W.fill_diagonal_(0.0)

    if self_loops:
        W.fill_diagonal_(1.0)

    row_sums = W.sum(dim=1, keepdim=True)
    W = W / torch.clamp(row_sums, min=1e-10)

    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(
            "Please make sure your input X follows the description: "
            "A DataFrame (N, D) where N is #samples and D is #multi-omics features."
        )
        raise ValueError(
            f"Generated graph shape {final_graph.shape} does not match expected "
            f"shape ({number_of_omics}, {number_of_omics})."
        )

    return final_graph

def gen_mst_graph(X: pd.DataFrame, self_loops: bool = False) -> pd.DataFrame:
    """
    Build a minimum-spanning-tree (MST) graph from feature vectors.

    - X: DataFrame (N samples, D features). Nodes = features.
    - Uses a vectorized Prim-style algorithm (O(N^2)) over pairwise distances.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas.DataFrame")

    X_feat = X.T
    nodes = X_feat.index
    number_of_omics = len(nodes)
    X_torch = torch.tensor(X_feat.values, dtype=torch.float32, device=device)

    N = X_torch.size(0)
    if N <= 1:
        raise ValueError("Need at least 2 features to build an MST.")

    D = torch.cdist(X_torch, X_torch)

    visited = torch.zeros(N, dtype=torch.bool, device=device)
    parent = torch.full((N,), -1, dtype=torch.long, device=device)

    visited[0] = True
    best_dist = D[0].clone()
    best_dist[0] = float("inf")
    parent[:] = 0

    for _ in range(N - 1):
        masked = best_dist.clone()
        masked[visited] = float("inf")
        min_val, j = torch.min(masked, dim=0)
        j = int(j.item())

        if not torch.isfinite(min_val) or visited[j]:
            break

        visited[j] = True

        new_dists = D[j]
        improved = (new_dists < best_dist) & (~visited)
        best_dist[improved] = new_dists[improved]
        parent[improved] = j

    mst = torch.zeros_like(D)
    for v in range(1, N):
        p = int(parent[v].item())
        if p >= 0 and p != v:
            w = D[v, p]
            mst[v, p] = w
            mst[p, v] = w

    W = torch.where(mst > 0, 1.0 / (mst + 1e-8), torch.zeros_like(mst))

    logger.info(f"MST: Number of edges: {(W > 0).sum().item() // 2}")
    if (W > 0).any():
        logger.info(
            f"MST: Edge weight range: "
            f"[{W[W > 0].min().item():.6f}, {W[W > 0].max().item():.6f}]"
        )

    if self_loops:
        W.fill_diagonal_(1.0)
    else:
        W.fill_diagonal_(0.0)

    row_sums = W.sum(dim=1, keepdim=True)
    W = W / torch.clamp(row_sums, min=1e-10)

    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(
            "Please make sure your input X follows the description: "
            "A DataFrame (N, D) where N is #samples and D is #multi-omics features."
        )
        raise ValueError(
            f"Generated graph shape {final_graph.shape} does not match expected "
            f"shape ({number_of_omics}, {number_of_omics})."
        )

    return final_graph

def gen_snn_graph(X: pd.DataFrame, k: int = 15, mutual: bool = False,self_loops: bool = False) -> pd.DataFrame:
    """
    Build a shared-nearest-neighbor (SNN) graph from feature vectors.

    - X: DataFrame (N samples, D features). Nodes = features.
    - k: number of neighbors per node.
    - mutual: keep only mutual SNN edges if True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas.DataFrame")

    X_feat = X.T
    nodes = X_feat.index
    number_of_omics = len(nodes)
    X_torch = torch.tensor(X_feat.values, dtype=torch.float32, device=device)

    N = X_torch.size(0)
    if N <= 1:
        raise ValueError("Need at least 2 features to build an SNN graph.")

    k = max(1, min(k, N - 1))

    S = torch.mm(X_torch, X_torch.t())

    _, index = torch.topk(S, k=k + 1, dim=1)
    index_np = index[:, 1 : k + 1].cpu().numpy()

    neighbors = [set(row) for row in index_np]

    W = torch.zeros((N, N), dtype=X_torch.dtype, device=device)

    for i in range(N):
        n_i = neighbors[i]
        for j in index_np[i]:
            shared = len(n_i.intersection(neighbors[j]))
            if shared > 0:
                W[i, j] = float(shared)

    if mutual:
        W = torch.min(W, W.t())

    if self_loops:
        W.fill_diagonal_(1.0)

    W = F.normalize(W, p=1, dim=1)

    final_graph = pd.DataFrame(W.cpu().numpy(), index=nodes, columns=nodes)

    if final_graph.shape != (number_of_omics, number_of_omics):
        logger.info(
            "Please make sure your input X follows the description: "
            "A DataFrame (N, D) where N is #samples and D is #multi-omics features."
        )
        raise ValueError(
            f"Generated graph shape {final_graph.shape} does not match expected "
            f"shape ({number_of_omics}, {number_of_omics})."
        )

    return final_graph
