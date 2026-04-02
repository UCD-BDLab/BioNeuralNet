import numpy as np
import pandas as pd
import networkx as nx
import warnings
from typing import Optional, Dict, List, Union
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, ParameterGrid
from .generate import similarity_network,correlation_network,threshold_network,gaussian_knn_network
from ..utils.logger import get_logger

import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np

logger = get_logger(__name__)

# while computing eigenvector centrality, ignore warnings about k >= N - 1. This does not break the functionality.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*k >= N - 1.*"
)

class NetworkAnalyzer:
    """
    Performs GPU-accelerated network analysis.

    This class leverages PyTorch tensors to speed up graph statistics, clustering computations, and edge analysis for large-scale omics networks.

    Args:

        adjacency_matrix (pd.DataFrame): The input weighted adjacency matrix representing network connections.
        source_omics (list): Optional list of original DataFrames used to build the network to dynamically assign omics types.
        device (str): The target computing device, defaulting to 'cuda' if available.
    """
    def __init__(self, adjacency_matrix: pd.DataFrame, source_omics: Optional[list] = None, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.feature_names = adjacency_matrix.index.tolist()
        self.n_nodes = len(self.feature_names)

        self.omics_types: Dict[str, List[str]] = {}
        self.feature_to_omic: Dict[str, str] = {}

        if source_omics is not None:
            for i, df in enumerate(source_omics):
                omic_name = f"omic_{i+1}"
                self.omics_types[omic_name] = []
                for col in df.columns:
                    if col in self.feature_names:
                        self.omics_types[omic_name].append(col)
                        self.feature_to_omic[col] = omic_name
        else:
            for feat in self.feature_names:
                omics = feat.split('_')[0]
                if omics not in self.omics_types:
                    self.omics_types[omics] = []
                self.omics_types[omics].append(feat)
                self.feature_to_omic[feat] = omics

        self.A = torch.tensor(
            adjacency_matrix.values,
            dtype=torch.float32,
            device=self.device
        )

        logger.info(f"Initialized on {self.device.upper()}")
        logger.info(f"Nodes: {self.n_nodes:,}")
        logger.info(f"Omics types: {list(self.omics_types.keys())}")

    def threshold_network(self, threshold: float) -> torch.Tensor:
        """
        Generates a binary adjacency matrix by applying a hard threshold to the connection weights.

        This converts continuous edge weights into a binary structure suitable for standard graph topology metrics.

        Args:

            threshold (float): The cutoff value above which an edge is considered to exist.

        Returns:

            torch.Tensor: A binary tensor where 1 indicates an edge and 0 indicates no edge.
        """
        return (self.A > threshold).float()

    def basic_statistics(self, threshold: float = 0.5) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Computes fundamental graph metrics including density, degree statistics, and node isolation counts.

        This provides a high-level overview of the network topology and connectivity at a specific threshold.

        Args:

            threshold (float): The threshold used to binarize the network before analysis.

        Returns:

            dict: A dictionary containing node count, edge count, density, average/max/min degree, and isolated node count.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"BASIC NETWORK STATISTICS (threshold > {threshold})")
        logger.info(f"{'='*60}")

        A_bin = self.threshold_network(threshold)

        num_nodes = self.n_nodes
        num_edges = A_bin.sum().item() / 2
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_possible_edges

        degrees = A_bin.sum(dim=1)
        avg_degree = degrees.mean().item()
        max_degree = degrees.max().item()
        min_degree = degrees.min().item()

        isolated = (degrees == 0).sum().item()

        logger.info(f"Nodes: {num_nodes:,}")
        logger.info(f"Edges: {int(num_edges):,}")
        logger.info(f"Density: {density:.6f}")
        logger.info(f"Avg Degree: {avg_degree:.2f}")
        logger.info(f"Max Degree: {int(max_degree)}")
        logger.info(f"Min Degree: {int(min_degree)}")
        logger.info(f"Isolated Nodes: {isolated:,} ({100*isolated/num_nodes:.1f}%)")

        return {
            'nodes': num_nodes,
            'edges': int(num_edges),
            'density': density,
            'avg_degree': avg_degree,
            'max_degree': int(max_degree),
            'isolated': isolated,
            'degrees': degrees.cpu().numpy()
        }

    def degree_distribution(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Calculates the frequency distribution of node degrees across the entire network.

        This helps identify if the network follows a scale-free power law or a random graph distribution.

        Args:

            threshold (float): The threshold used to binarize the network.

        Returns:

            pd.DataFrame: A DataFrame with columns for degree, count, and percentage of total nodes.
        """
        A_bin = self.threshold_network(threshold)
        degrees = A_bin.sum(dim=1).cpu().numpy().astype(int)

        unique, counts = np.unique(degrees, return_counts=True)

        return pd.DataFrame({
            'degree': unique,
            'count': counts,
            'percentage': 100 * counts / len(degrees)
        })

    def hub_analysis(self, threshold: float = 0.5, top_n: int = 10) -> pd.DataFrame:
        """
        Identifies and ranks the most highly connected 'hub' nodes in the network.

        This is critical for finding central regulatory features or bottlenecks in the omics network.

        Args:

            threshold (float): The threshold used to define network edges.
            top_n (int): The number of top degree nodes to retrieve.

        Returns:

            pd.DataFrame: A table of the top N nodes including their rank, feature name, omics type, and degree.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"TOP {top_n} HUB NODES (threshold > {threshold})")
        logger.info(f"{'='*60}")

        A_bin = self.threshold_network(threshold)
        degrees = A_bin.sum(dim=1)

        top_values, top_indices = torch.topk(degrees, top_n)

        results = []
        for i, (idx, deg) in enumerate(zip(top_indices.cpu().numpy(), top_values.cpu().numpy())):
            feat_name = self.feature_names[idx]
            omics_type = self.feature_to_omic.get(feat_name, 'unknown')
            actual_name = feat_name

            results.append({
                'rank': i + 1,
                'feature': feat_name,
                'gene/probe': actual_name,
                'omics': omics_type,
                'degree': int(deg)
            })
            logger.info(f"{i+1:2d}. {feat_name:<40s} | {omics_type:<6s} | degree: {int(deg)}")

        return pd.DataFrame(results)

    def clustering_coefficient_gpu(self, threshold: float = 0.5, sample_size: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Computes the local clustering coefficient for nodes using GPU-optimized matrix operations.

        This measures the degree to which nodes tend to cluster together, using random sampling for efficiency on large graphs.

        Args:

            threshold (float): The threshold used to define valid edges.
            sample_size (Optional[int]): The number of nodes to sample for calculation to save memory on massive graphs.

        Returns:

            dict: Statistics including average, max, and min clustering coefficients, plus raw values and sample indices.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CLUSTERING COEFFICIENT ANALYSIS (threshold > {threshold})")
        logger.info(f"{'='*60}")

        A_bin = self.threshold_network(threshold)
        degrees = A_bin.sum(dim=1)

        if sample_size is None:
            valid_mask = degrees >= 2
            n_valid = valid_mask.sum().item()

            if n_valid > 5000:
                logger.info(f"Large network ({n_valid} valid nodes). Sampling 5000 nodes...")
                sample_size = 5000

        if sample_size:
            valid_indices = torch.where(degrees >= 2)[0]
            if len(valid_indices) > sample_size:
                perm = torch.randperm(len(valid_indices), device=self.device)[:sample_size]
                sample_indices = valid_indices[perm]
            else:
                sample_indices = valid_indices
        else:
            sample_indices = torch.where(degrees >= 2)[0]

        logger.info(f"Computing clustering for {len(sample_indices)} nodes...")

        clustering_coeffs = torch.zeros(self.n_nodes, device=self.device)

        batch_size = 500
        n_batches = (len(sample_indices) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(sample_indices))
            batch_nodes = sample_indices[start:end]

            for node_idx in batch_nodes:
                node = node_idx.item()
                neighbors = torch.where(A_bin[node] > 0)[0]
                k = len(neighbors)

                if k >= 2:
                    neighbor_subgraph = A_bin[neighbors][:, neighbors]
                    triangles = neighbor_subgraph.sum().item() / 2
                    max_triangles = k * (k - 1) / 2
                    clustering_coeffs[node] = triangles / max_triangles

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed batch {batch_idx + 1}/{n_batches}")

        valid_cc = clustering_coeffs[sample_indices]
        avg_cc = valid_cc.mean().item()
        max_cc = valid_cc.max().item()
        min_cc = valid_cc[valid_cc > 0].min().item() if (valid_cc > 0).any() else 0

        logger.info(f"\nClustering Coefficient Statistics:")
        logger.info(f" Average: {avg_cc:.4f}")
        logger.info(f" Maximum: {max_cc:.4f}")
        logger.info(f" Minimum (non-zero): {min_cc:.4f}")
        logger.info(f" Nodes with CC > 0: {(valid_cc > 0).sum().item()}")

        return {
            'average': avg_cc,
            'max': max_cc,
            'coefficients': clustering_coeffs.cpu().numpy(),
            'sample_indices': sample_indices.cpu().numpy()
        }

    def cross_omics_analysis(self, threshold: float = 0.5) -> Dict[tuple, Dict]:
        """
        Quantifies the connectivity density between different omics layers (e.g., RNA vs Protein).

        This reveals whether the network structure is driven by within-omics correlations or cross-omics interactions.

        Args:

            threshold (float): The threshold used to count valid edges between features.

        Returns:

            dict: A nested dictionary mapping omics pairs to their edge counts and density statistics.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CROSS-OMICS CONNECTIVITY (threshold > {threshold})")
        logger.info(f"{'='*60}")

        A_bin = self.threshold_network(threshold)

        omics_indices = {}
        for omics, features in self.omics_types.items():
            omics_indices[omics] = [self.feature_names.index(f) for f in features]

        results = {}
        omics_list = list(self.omics_types.keys())

        logger.info(f"\n{'Omics Pair':<20s} | {'Edges':>10s} | {'Max Possible':>12s} | {'Density':>10s}")
        logger.info("-" * 60)

        for i, om1 in enumerate(omics_list):
            for j, om2 in enumerate(omics_list):
                if i <= j:
                    idx1 = torch.tensor(omics_indices[om1], device=self.device, dtype=torch.long)
                    idx2 = torch.tensor(omics_indices[om2], device=self.device, dtype=torch.long)

                    submatrix = A_bin[idx1][:, idx2]
                    n_edges = submatrix.sum().item()

                    if i == j:
                        n_edges = n_edges / 2
                        max_edges = len(idx1) * (len(idx1) - 1) / 2
                    else:
                        max_edges = len(idx1) * len(idx2)

                    density = n_edges / max_edges if max_edges > 0 else 0

                    pair_name = f"{om1}-{om2}" if i != j else f"{om1} (within)"
                    results[(om1, om2)] = {
                        'edges': int(n_edges),
                        'max_edges': int(max_edges),
                        'density': density
                    }

                    logger.info(f"{pair_name:<20s} | {int(n_edges):>10,} | {int(max_edges):>12,} | {density:>10.6f}")

        return results

    def edge_weight_analysis(self) -> Optional[np.ndarray]:
        """
        Analyzes the statistical distribution of edge weights across the entire network.

        This is useful for determining appropriate threshold values and understanding signal strength distribution.

        Args:

            None.

        Returns:

            Optional[np.ndarray]: An array of all non-zero edge weights, or None if no edges exist.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"EDGE WEIGHT DISTRIBUTION")
        logger.info(f"{'='*60}")

        upper_tri = torch.triu(self.A, diagonal=1)
        weights = upper_tri[upper_tri > 0]

        if len(weights) == 0:
            logger.info("No edges found!")
            return None

        weights_cpu = weights.cpu().numpy()

        logger.info(f"Total edges (weight > 0): {len(weights_cpu):,}")
        logger.info(f"Weight statistics:")
        logger.info(f" Mean: {weights_cpu.mean():.6f}")
        logger.info(f" Std: {weights_cpu.std():.6f}")
        logger.info(f" Median: {np.median(weights_cpu):.6f}")
        logger.info(f" Min: {weights_cpu.min():.6f}")
        logger.info(f" Max: {weights_cpu.max():.6f}")

        logger.info(f"\nPercentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(weights_cpu, p)
            logger.info(f"  {p}th: {val:.6f}")

        logger.info(f"\nEdges at different biological thresholds:")
        for thresh in [0.001, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
            n_edges = (weights_cpu > thresh).sum()
            logger.info(f"  > {thresh}: {n_edges:,} edges")

        return weights_cpu

    def find_strongest_edges(self, top_n: int = 50) -> pd.DataFrame:
        """
        Retrieves the strongest edges in the network sorted by weight magnitude.

        This isolates the most significant pairwise interactions between features.

        Args:

            top_n (int): The number of top weighted edges to return.

        Returns:

            pd.DataFrame: A DataFrame detailing the top interactions, including feature names and weights.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"TOP {top_n} STRONGEST EDGES")
        logger.info(f"{'='*60}")

        upper_tri = torch.triu(self.A, diagonal=1)

        flat = upper_tri.flatten()
        top_values, top_flat_indices = torch.topk(flat, top_n)

        n = self.n_nodes
        row_indices = top_flat_indices // n
        col_indices = top_flat_indices % n

        results = []
        logger.info(f"{'Rank':<5s} | {'Feature 1':<35s} | {'Feature 2':<35s} | {'Weight':>10s}")
        logger.info("-" * 95)

        for i in range(top_n):
            row = row_indices[i].item()
            col = col_indices[i].item()
            weight = top_values[i].item()

            feat1 = self.feature_names[row]
            feat2 = self.feature_names[col]

            results.append({
                'rank': i + 1,
                'feature1': feat1,
                'feature2': feat2,
                'omics1': self.feature_to_omic.get(feat1, 'unknown'),
                'omics2': self.feature_to_omic.get(feat2, 'unknown'),
                'weight': weight
            })

            logger.info(f"{i+1:<5d} | {feat1:<35s} | {feat2:<35s} | {weight:>10.6f}")

        return pd.DataFrame(results)

    def connected_components(self, threshold: float = 0.5) -> Dict[str, Union[int, np.ndarray, List[int]]]:
        """
        Identifies isolated subgraphs within the network using Breadth-First Search logic.

        This computation is performed on the CPU using scipy due to the sequential nature of traversal algorithms.

        Args:

            threshold (float): The threshold used to define connectivity.

        Returns:

            dict: Contains the count of components, label assignments for each node, and a size distribution list.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CONNECTED COMPONENTS (threshold > {threshold})")
        logger.info(f"{'='*60}")

        A_bin = self.threshold_network(threshold)

        A_cpu = A_bin.cpu().numpy()

        A_sparse = csr_matrix(A_cpu)
        n_components, labels = connected_components(A_sparse, directed=False)

        unique, counts = np.unique(labels, return_counts=True)
        component_sizes = sorted(counts, reverse=True)

        logger.info(f"Number of components: {n_components}")
        logger.info(f"Largest component: {component_sizes[0]} nodes ({100*component_sizes[0]/self.n_nodes:.1f}%)")

        if n_components > 1:
            logger.info(f"Second largest: {component_sizes[1]} nodes")
            logger.info(f"\nTop 10 component sizes: {component_sizes[:10]}")

        isolated = (counts == 1).sum()
        logger.info(f"Isolated nodes: {isolated}")

        return {
            'n_components': n_components,
            'labels': labels,
            'sizes': component_sizes
        }

def network_search(
    omics_data: pd.DataFrame,
    y_labels,
    methods: list = ["correlation", "threshold", "similarity", "gaussian"],
    seed: int = 1883,
    verbose: bool = True,
    trials: Optional[int] = None,
    centrality_mode: str = "eigenvector",
    topology_weight: float = 0.15,
    scoring: str = "f1_macro"
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Search over graph-construction hyperparameters using a structural proxy.

    Each candidate configuration builds a graph, scores it with a fast centrality-weighted Ridge classifier proxy, and blends that score with a topological quality term (average clustering coefficient) to favour well-connected, informative graphs.

    Args:

        omics_data: Feature matrix of shape (n_samples, n_features).
        y_labels: Target labels for stratified CV evaluation.
        methods: Graph-construction methods to search over.
        seed: Random seed for reproducibility.
        verbose: Log per-configuration progress.
        trials: Optional cap on evaluated configurations (random subset).
        centrality_mode: Centrality used for feature weighting in the proxy; one of ``"eigenvector"`` or ``"degree"``.
        topology_weight: Blending factor in [0, 1] that controls how much the topological quality term contributes to the final score. ``0`` ignores topology; ``1`` ignores the proxy F1.

    Returns:

        A 3-tuple of (best_graph, best_params, results_df).

    Raises:

        RuntimeError: If every configuration fails.
    """
    y_vec = _to_array(y_labels)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(omics_data.values),
        index=omics_data.index,
        columns=omics_data.columns,
    )

    all_configs = _build_config_grid(methods, seed, trials)

    logger.info(f"Total configurations to evaluate: {len(all_configs)}")
    if verbose:
        logger.info(f"Starting Network Search (n_configs={len(all_configs)})")

    best_score = -np.inf
    best_config = None
    results: list[dict] = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    dispatch = {
        "threshold": threshold_network,
        "gaussian": gaussian_knn_network,
        "correlation": correlation_network,
        "similarity": similarity_network,
    }

    for idx, (method_name, gen_params) in enumerate(all_configs, start=1):
        builder = dispatch.get(method_name)
        if builder is None or not callable(builder):
            continue
        try:
            G = builder(omics_data, **{**gen_params, "self_loops": False})

            f1_score = _feature_proxy(G, X_scaled, y_vec, cv, mode=centrality_mode, scoring=scoring)
            topo_score = _topology_quality(G)
            topo_norm = np.clip(topo_score, 0, 1)
            combined = ((1.0 - topology_weight) * f1_score+ topology_weight * topo_norm)

        except Exception:
            continue

        if verbose:
            logger.info(
                f"[{idx}/{len(all_configs)}] {method_name[:4].upper()} "
                f"| F1={f1_score:.3f} "
                f"| Topology={topo_score:.3f} "
                f"| Score={combined:.3f}"
            )

        if combined > best_score:
            best_score = combined
            best_config = {
                "method": method_name,
                "graph": G,
                "params": gen_params,
                "stats": f"F1: {f1_score:.3f}",
                "proxy_score": f1_score,
                "topology_score": topo_score,
                "combined_score": combined,
            }

        results.append({
            "method": method_name,
            "params": gen_params,
            "score": combined,
            "f1": f1_score,
            "topology": topo_score,
        })

    results_df = pd.DataFrame(results)

    if best_config is None:
        raise RuntimeError(
            "network_search: every configuration failed. "
            "Check that omics_data has sufficient samples and features."
        )

    logger.info(f"Best topology: {best_config['method'].upper()}")
    logger.info(f"Performance: {best_config['stats']}")
    logger.info(f"Topology: {best_config['topology_score']:.3f}")

    return best_config["graph"], best_config["params"], results_df


_PARAM_GRIDS = {
    "gaussian": {
        "k": list(range(5, 30)),
        "sigma": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, None],
        "mutual": [True, False],
    },
    "similarity": {
        "k": list(range(5, 30)),
        "metric": ["cosine", "euclidean"],
        "mutual": [True, False],
    },
    "correlation": {
        "k": list(range(5, 30)),
        "method": ["pearson", "spearman"],
        "signed": [False],
        "threshold": [None],
    },
    "threshold": {
        "b": [5.0,5.5,6.0,6.25,6.5,6.75,7.0,7.25,7.5,7.75,8.0,8.5,9.0],
        "k": list(range(5, 30)),
        "mutual": [True, False],
    },
}


def _to_array(y_labels) -> np.ndarray:
    """Coerce labels to a flat numpy array."""
    if isinstance(y_labels, pd.Series):
        return y_labels.values
    return np.asarray(y_labels).ravel()


def _build_config_grid(methods: list[str],seed: int,trials: Optional[int],) -> list[tuple[str, dict]]:
    """Expand parameter grids, shuffle, and optionally subsample."""
    configs = []
    for method in methods:
        grid = _PARAM_GRIDS.get(method)
        if grid is None:
            continue
        for params in ParameterGrid(grid):
            configs.append((method, params))

    rng = np.random.RandomState(seed)
    rng.shuffle(configs)

    if trials and trials < len(configs):
        configs = configs[:trials]

    return configs

def _topology_quality(adj_df: pd.DataFrame) -> float:
    A = adj_df.fillna(0.0).values.copy()
    np.fill_diagonal(A, 0.0)
    n_nodes = max(A.shape[0], 1)

    degrees = np.count_nonzero(A, axis=1)
    connectivity = np.sum(degrees > 0) / n_nodes

    G = nx.from_numpy_array(A)

    if n_nodes > 0:
        largest_cc_size = len(max(nx.connected_components(G), key=len))
        lcc_ratio = largest_cc_size / n_nodes
    else:
        lcc_ratio = 0.0

    return 0.5 * connectivity + 0.5 * lcc_ratio

def _feature_proxy(adj_df, X_df, y, cv, mode="laplacian", scoring="f1_macro"):
    A = adj_df.fillna(0.0).values.copy()
    np.fill_diagonal(A, 0.0)

    if mode == "laplacian":
        deg = A.sum(axis=1)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        A_norm = deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]
        X_smooth = X_df.values + A_norm @ X_df.values
    else:
        G_nx = nx.from_pandas_adjacency(adj_df)
        try:
            weights = nx.eigenvector_centrality_numpy(G_nx, weight="weight")
        except Exception:
            weights = dict(G_nx.degree(weight=None))
        w_vec = np.log1p(np.maximum(0.0,
                    np.array([weights.get(c, 0.0) for c in X_df.columns])))
        X_smooth = X_df.values * w_vec[None, :]

    scores = cross_val_score(
        RidgeClassifier(class_weight="balanced"),
        X_smooth, y, cv=cv, scoring=scoring
    )
    return float(scores.mean())
