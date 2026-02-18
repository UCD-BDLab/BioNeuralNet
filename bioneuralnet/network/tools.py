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

logger = get_logger(__name__)

# while computing eigenvector centrality, ignore warnings about k >= N - 1. This does not break the functionality.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*k >= N - 1.*"
)

class GPUNetworkAnalyzer:
    """
    Performs GPU-accelerated network analysis operations on SmCCNet adjacency matrices.

    This class leverages PyTorch tensors to speed up graph statistics, clustering computations, and edge analysis for large-scale omics networks.

    Args:

        adjacency_matrix (pd.DataFrame): The input weighted adjacency matrix representing network connections.
        device (str): The target computing device, defaulting to 'cuda' if available.
    """

    def __init__(self, adjacency_matrix: pd.DataFrame, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.feature_names = adjacency_matrix.index.tolist()
        self.n_nodes = len(self.feature_names)
        
        self.omics_types = {}
        for feat in self.feature_names:
            omics = feat.split('_')[0]
            if omics not in self.omics_types:
                self.omics_types[omics] = []
            self.omics_types[omics].append(feat)
        
        self.A = torch.tensor(
            adjacency_matrix.values, 
            dtype=torch.float32, 
            device=self.device
        )
        
        print(f"Initialized on {self.device.upper()}")
        print(f"Nodes: {self.n_nodes:,}")
        print(f"Omics types: {list(self.omics_types.keys())}")

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

    def basic_statistics(self, threshold: float = 0.001) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Computes fundamental graph metrics including density, degree statistics, and node isolation counts.

        This provides a high-level overview of the network topology and connectivity at a specific threshold.

        Args:

            threshold (float): The threshold used to binarize the network before analysis.

        Returns:

            dict: A dictionary containing node count, edge count, density, average/max/min degree, and isolated node count.
        """
        print(f"\n{'='*60}")
        print(f"BASIC NETWORK STATISTICS (threshold > {threshold})")
        print(f"{'='*60}")
        
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
        
        print(f"Nodes: {num_nodes:,}")
        print(f"Edges: {int(num_edges):,}")
        print(f"Density: {density:.6f}")
        print(f"Avg Degree: {avg_degree:.2f}")
        print(f"Max Degree: {int(max_degree)}")
        print(f"Min Degree: {int(min_degree)}")
        print(f"Isolated Nodes: {isolated:,} ({100*isolated/num_nodes:.1f}%)")
        
        return {
            'nodes': num_nodes,
            'edges': int(num_edges),
            'density': density,
            'avg_degree': avg_degree,
            'max_degree': int(max_degree),
            'isolated': isolated,
            'degrees': degrees.cpu().numpy()
        }

    def degree_distribution(self, threshold: float = 0.01) -> pd.DataFrame:
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

    def hub_analysis(self, threshold: float = 0.01, top_n: int = 20) -> pd.DataFrame:
        """
        Identifies and ranks the most highly connected 'hub' nodes in the network.

        This is critical for finding central regulatory features or bottlenecks in the omics network.

        Args:

            threshold (float): The threshold used to define network edges.
            top_n (int): The number of top degree nodes to retrieve.

        Returns:

            pd.DataFrame: A table of the top N nodes including their rank, feature name, omics type, and degree.
        """
        print(f"\n{'='*60}")
        print(f"TOP {top_n} HUB NODES (threshold > {threshold})")
        print(f"{'='*60}")
        
        A_bin = self.threshold_network(threshold)
        degrees = A_bin.sum(dim=1)
        
        top_values, top_indices = torch.topk(degrees, top_n)
        
        results = []
        for i, (idx, deg) in enumerate(zip(top_indices.cpu().numpy(), top_values.cpu().numpy())):
            feat_name = self.feature_names[idx]
            omics_type = feat_name.split('_')[0]
            actual_name = '_'.join(feat_name.split('_')[1:])
            
            results.append({
                'rank': i + 1,
                'feature': feat_name,
                'gene/probe': actual_name,
                'omics': omics_type,
                'degree': int(deg)
            })
            print(f"{i+1:2d}. {feat_name:<40s} | {omics_type:<6s} | degree: {int(deg)}")
        
        return pd.DataFrame(results)

    def clustering_coefficient_gpu(self, threshold: float = 0.01, sample_size: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Computes the local clustering coefficient for nodes using GPU-optimized matrix operations.

        This measures the degree to which nodes tend to cluster together, using random sampling for efficiency on large graphs.

        Args:

            threshold (float): The threshold used to define valid edges.
            sample_size (Optional[int]): The number of nodes to sample for calculation to save memory on massive graphs.

        Returns:

            dict: Statistics including average, max, and min clustering coefficients, plus raw values and sample indices.
        """
        print(f"\n{'='*60}")
        print(f"CLUSTERING COEFFICIENT ANALYSIS (threshold > {threshold})")
        print(f"{'='*60}")
        
        A_bin = self.threshold_network(threshold)
        degrees = A_bin.sum(dim=1)
        
        if sample_size is None:
            valid_mask = degrees >= 2
            n_valid = valid_mask.sum().item()
            
            if n_valid > 5000:
                print(f"Large network ({n_valid} valid nodes). Sampling 5000 nodes...")
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
        
        print(f"Computing clustering for {len(sample_indices)} nodes...")
        
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
                print(f"  Processed batch {batch_idx + 1}/{n_batches}")
        
        valid_cc = clustering_coeffs[sample_indices]
        avg_cc = valid_cc.mean().item()
        max_cc = valid_cc.max().item()
        min_cc = valid_cc[valid_cc > 0].min().item() if (valid_cc > 0).any() else 0
        
        print(f"\nClustering Coefficient Statistics:")
        print(f" Average:  {avg_cc:.4f}")
        print(f" Maximum:  {max_cc:.4f}")
        print(f" Minimum (non-zero): {min_cc:.4f}")
        print(f" Nodes with CC > 0: {(valid_cc > 0).sum().item()}")
        
        return {
            'average': avg_cc,
            'max': max_cc,
            'coefficients': clustering_coeffs.cpu().numpy(),
            'sample_indices': sample_indices.cpu().numpy()
        }

    def cross_omics_analysis(self, threshold: float = 0.01) -> Dict[tuple, Dict]:
        """
        Quantifies the connectivity density between different omics layers (e.g., RNA vs Protein).

        This reveals whether the network structure is driven by within-omics correlations or cross-omics interactions.

        Args:

            threshold (float): The threshold used to count valid edges between features.

        Returns:

            dict: A nested dictionary mapping omics pairs to their edge counts and density statistics.
        """
        print(f"\n{'='*60}")
        print(f"CROSS-OMICS CONNECTIVITY (threshold > {threshold})")
        print(f"{'='*60}")
        
        A_bin = self.threshold_network(threshold)
        
        omics_indices = {}
        for omics, features in self.omics_types.items():
            omics_indices[omics] = [self.feature_names.index(f) for f in features]
        
        results = {}
        omics_list = list(self.omics_types.keys())
        
        print(f"\n{'Omics Pair':<20s} | {'Edges':>10s} | {'Max Possible':>12s} | {'Density':>10s}")
        print("-" * 60)
        
        for i, om1 in enumerate(omics_list):
            for j, om2 in enumerate(omics_list):
                if i <= j:
                    idx1 = torch.tensor(omics_indices[om1], device=self.device)
                    idx2 = torch.tensor(omics_indices[om2], device=self.device)
                    
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
                    
                    print(f"{pair_name:<20s} | {int(n_edges):>10,} | {int(max_edges):>12,} | {density:>10.6f}")
        
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
        print(f"\n{'='*60}")
        print(f"EDGE WEIGHT DISTRIBUTION")
        print(f"{'='*60}")
        
        upper_tri = torch.triu(self.A, diagonal=1)
        weights = upper_tri[upper_tri > 0]
        
        if len(weights) == 0:
            print("No edges found!")
            return None
        
        weights_cpu = weights.cpu().numpy()
        
        print(f"Total edges (weight > 0): {len(weights_cpu):,}")
        print(f"Weight statistics:")
        print(f"  Mean:   {weights_cpu.mean():.6f}")
        print(f"  Std:    {weights_cpu.std():.6f}")
        print(f"  Median: {np.median(weights_cpu):.6f}")
        print(f"  Min:    {weights_cpu.min():.6f}")
        print(f"  Max:    {weights_cpu.max():.6f}")
        
        print(f"\nPercentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(weights_cpu, p)
            print(f"  {p}th: {val:.6f}")
        
        print(f"\nEdges at different thresholds:")
        for thresh in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            n_edges = (weights_cpu > thresh).sum()
            print(f"  > {thresh}: {n_edges:,} edges")
        
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
        print(f"\n{'='*60}")
        print(f"TOP {top_n} STRONGEST EDGES")
        print(f"{'='*60}")
        
        upper_tri = torch.triu(self.A, diagonal=1)
        
        flat = upper_tri.flatten()
        top_values, top_flat_indices = torch.topk(flat, top_n)
        
        n = self.n_nodes
        row_indices = top_flat_indices // n
        col_indices = top_flat_indices % n
        
        results = []
        print(f"{'Rank':<5s} | {'Feature 1':<35s} | {'Feature 2':<35s} | {'Weight':>10s}")
        print("-" * 95)
        
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
                'omics1': feat1.split('_')[0],
                'omics2': feat2.split('_')[0],
                'weight': weight
            })
            
            print(f"{i+1:<5d} | {feat1:<35s} | {feat2:<35s} | {weight:>10.6f}")
        
        return pd.DataFrame(results)

    def connected_components(self, threshold: float = 0.01) -> Dict[str, Union[int, np.ndarray, List[int]]]:
        """
        Identifies isolated subgraphs within the network using Breadth-First Search logic.

        This computation is performed on the CPU using scipy due to the sequential nature of traversal algorithms.

        Args:

            threshold (float): The threshold used to define connectivity.

        Returns:

            dict: Contains the count of components, label assignments for each node, and a size distribution list.
        """
        print(f"\n{'='*60}")
        print(f"CONNECTED COMPONENTS (threshold > {threshold})")
        print(f"{'='*60}")
        
        A_bin = self.threshold_network(threshold)
        
        A_cpu = A_bin.cpu().numpy()
            
        A_sparse = csr_matrix(A_cpu)
        n_components, labels = connected_components(A_sparse, directed=False)
        
        unique, counts = np.unique(labels, return_counts=True)
        component_sizes = sorted(counts, reverse=True)
        
        print(f"Number of components: {n_components}")
        print(f"Largest component: {component_sizes[0]} nodes ({100*component_sizes[0]/self.n_nodes:.1f}%)")
        
        if n_components > 1:
            print(f"Second largest: {component_sizes[1]} nodes")
            print(f"\nTop 10 component sizes: {component_sizes[:10]}")
        
        isolated = (counts == 1).sum()
        print(f"Isolated nodes: {isolated}")
        
        return {
            'n_components': n_components,
            'labels': labels,
            'sizes': component_sizes
        }

def describe_network(
        network: pd.DataFrame, 
        graph_name: str = "network", 
        omics_list: Optional[List[pd.DataFrame]] = None
) -> None:
    """Analyze and log basic topology and small components of a network.

    The adjacency matrix is converted to a NetworkX graph, summary metrics such as node and edge counts, largest component size, average degree, and clustering coefficient are logged, and small components (≤10 nodes) are reported, optionally with modality-aware counts derived from omics_list.

    Args:

        network (pd.DataFrame): Square adjacency matrix with nodes as both rows and columns.
        netowrk_name (str): Descriptive name used for log messages identifying the network.
        omics_list (list[pd.DataFrame] | None): Optional list of omics DataFrames used to map nodes to omic blocks for modality-aware summaries of small components.

    Returns:

        None: Metrics are logged via the configured logger and no value is returned.

    """
    G = nx.from_pandas_adjacency(network)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_components = nx.number_connected_components(G)
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    largest_cc_size = len(largest_cc)
    if num_nodes > 0:
        avg_degree = 2 * num_edges / num_nodes
    else:
        avg_degree = 0

    print(f"\n{'='*60}")
    print(f"GRAPH ANALYSIS: {graph_name}")
    print(f"{'='*60}")

    print(f"Nodes: {num_nodes:,} | Edges: {num_edges:,}")
    print(f"Avg degree: {avg_degree:.2f}")
    print(f"Connected components: {num_components}")
    print(
        f"Largest component: {largest_cc_size} nodes "
        f"({100 * largest_cc_size / num_nodes:.1f}%)"
    )

    component_sizes = []
    for comp in components:
        component_sizes.append(len(comp))
    component_sizes.sort(reverse=True)
    if len(component_sizes) > 1:
        if len(component_sizes) > 5:
            suffix = "..."
        else:
            suffix = ""
        print(
            f"Component sizes (5): {component_sizes[:5]}{suffix}"
        )

    use_omics_mapping = False
    node_to_omic_idx = {}

    if omics_list is not None:
        graph_nodes = set(network.index)
        all_omics_cols = set()
        for i, omic_df in enumerate(omics_list):
            for col in omic_df.columns:
                all_omics_cols.add(col)
                node_to_omic_idx[col] = i

        if len(graph_nodes) == len(all_omics_cols) and graph_nodes == all_omics_cols:
            use_omics_mapping = True
            logger.info(
                "graph_analysis: omics_list matches graph nodes exactly; "
                "using omics-based modality breakdown."
            )
        else:
            logger.warning(
                "graph_analysis: omics_list columns do not exactly match graph nodes "
                "(different sets or lengths). Skipping modality breakdown and "
                "reporting only generic graph metrics."
            )

    small_components = []
    for comp in components:
        if len(comp) <= 10:
            small_components.append(comp)
    small_comps = len(small_components)

    if small_comps > 0:
        logger.info(f"Small components (<=10 nodes): {small_comps}")
        logger.warning(f"Found {small_comps} components with < 10 nodes")

        max_components_to_log = 10
        for comp_id, comp in enumerate(small_components[:max_components_to_log], start=1):
            labels = list(comp)
            labels.sort()

            if use_omics_mapping:
                counts: dict[str, int] = {}
                for name in labels:
                    omic_idx = node_to_omic_idx.get(name, None)
                    if omic_idx is not None:
                        key = f"omic_{omic_idx}"
                    else:
                        key = "other"
                    if key in counts:
                        counts[key] += 1
                    else:
                        counts[key] = 1

                items_sorted = sorted(counts.items())
                parts = []
                for k, v in items_sorted:
                    parts.append(f"{k}={v}")
                counts_str = ", ".join(parts)

                print(
                    f"[Island #{comp_id}] size={len(labels)} | {counts_str}"
                )
            else:

                counts: dict[str, int] = {}
                for name in labels:
                    if isinstance(name, str) and "_" in name:
                        key = name.split("_")[0]
                    else:
                        key = "other"
                    
                    if key in counts:
                        counts[key] += 1
                    else:
                        counts[key] = 1

                items_sorted = sorted(counts.items())
                parts = []
                for k, v in items_sorted:
                    parts.append(f"{k}={v}")
                counts_str = ", ".join(parts)

                print(
                    f"[Island #{comp_id}] size={len(labels)} | {counts_str}"
                )

            print(
                f"[Island #{comp_id}] nodes (up to 10): {labels[:10]}"
            )

    try:
        avg_clustering = nx.average_clustering(G, weight="weight")
        print(f"Avg clustering coefficient: {avg_clustering:.3f}")
    except Exception:
        logger.warning("Could not compute clustering coefficient")


def network_search(
    omics_data: pd.DataFrame,
    y_labels,
    methods: list = ["correlation", "threshold", "similarity", "gaussian"],
    seed: int = 1883,
    verbose: bool = True,
    trials: Optional[int] = None,
    centrality_mode: str = "eigenvector",
    topology_weight: float = 0.15,
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
        if builder is None:
            continue
        try:
            G = builder(omics_data, **{**gen_params, "self_loops": False})

            mean_f1, std_f1 = _feature_proxy(
                G, X_scaled, y_vec, cv, mode=centrality_mode,
            )
            topo_score = _topology_quality(G)

            proxy_score = mean_f1 - 2.0 * std_f1
            combined = (
                (1.0 - topology_weight) * proxy_score
                + topology_weight * (topo_score * 100.0)
            )

        except Exception:
            continue

        if verbose:
            logger.info(
                f"[{idx}/{len(all_configs)}] {method_name[:4].upper()} "
                f"| F1={mean_f1:.1f}% ±{std_f1:.1f} "
                f"| Topo={topo_score:.3f} "
                f"| Score={combined:.1f}"
            )

        if combined > best_score:
            best_score = combined
            best_config = {
                "method": method_name,
                "graph": G,
                "params": gen_params,
                "stats": f"{mean_f1:.1f}% ±{std_f1:.1f}%",
                "proxy_score": proxy_score,
                "topology_score": topo_score,
                "combined_score": combined,
            }

        results.append({
            "method": method_name,
            "params": gen_params,
            "score": combined,
            "proxy_score": proxy_score,
            "f1": mean_f1,
            "std": std_f1,
            "topology": topo_score,
        })

    results_df = pd.DataFrame(results)

    if best_config is None:
        raise RuntimeError(
            "network_search: every configuration failed. "
            "Check that omics_data has sufficient samples and features."
        )

    logger.info(f"Best topology: {best_config['method'].upper()}")
    logger.info(f"Performance:   {best_config['stats']}")
    logger.info(f"Topology:      {best_config['topology_score']:.3f}")

    return best_config["graph"], best_config["params"], results_df


_PARAM_GRIDS = {
    "gaussian": {
        "k": list(range(5, 21)),
        "sigma": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        "mutual": [True, False],
    },
    "similarity": {
        "k": list(range(5, 21)),
        "metric": ["cosine"],
        "mutual": [True, False],
    },
    "correlation": {
        "k": list(range(5, 21)),
        "method": ["pearson", "spearman"],
        "signed": [False],
        "threshold": [None],
    },
    "threshold": {
        "b": [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0],
        "k": list(range(5, 21)),
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
    """Score graph topology via average clustering coefficient.

    Returns a value in [0, 1] where higher means richer local connectivity.
    Falls back to 0.0 on failure (e.g. empty graph).
    """
    try:
        G = nx.from_pandas_adjacency(adj_df)
        return nx.average_clustering(G, weight="weight")
    except Exception:
        return 0.0


def _feature_proxy(adj_df: pd.DataFrame,X_df: pd.DataFrame,y: np.ndarray,cv,mode: str = "eigenvector",) -> tuple[float, float]:
    """Fast proxy: centrality-weighted features → Ridge CV.

    Computes a centrality score per node, applies log(1 + ReLU) scaling
    as feature weights, and evaluates with cross-validated weighted F1.

    Args:
        adj_df: Adjacency matrix aligned to columns of X_df.
        X_df: Scaled feature matrix (n_samples, n_features).
        y: Target labels.
        cv: Stratified CV splitter.
        mode: ``"eigenvector"`` or ``"degree"``.

    Returns:
        (mean_f1_pct, std_f1_pct) across CV folds.
    """
    G_nx = nx.from_pandas_adjacency(adj_df)

    try:
        if mode == "eigenvector":
            weights = nx.eigenvector_centrality_numpy(G_nx, weight="weight")
        elif mode == "degree":
            weights = dict(G_nx.degree(weight="weight"))
        else:
            weights = dict(G_nx.degree(weight=None))
    except Exception:
        weights = dict(G_nx.degree(weight=None))

    w_vec = np.array([weights.get(col, 0.0) for col in X_df.columns])
    w_vec = np.log1p(np.maximum(0.0, w_vec))

    X_weighted = X_df.values * w_vec[None, :]

    scores = cross_val_score(
        RidgeClassifier(),
        X_weighted,
        y,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    return float(scores.mean() * 100.0), float(scores.std() * 100.0)