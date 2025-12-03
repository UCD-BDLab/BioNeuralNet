import numpy as np
import pandas as pd
import networkx as nx
import warnings
from typing import Optional, List
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, ParameterGrid
from .graph import gen_similarity_graph, gen_correlation_graph, gen_threshold_graph,gen_gaussian_knn_graph
from .logger import get_logger
logger = get_logger(__name__)

# while computing eigenvector centrality, ignore warnings about k >= N - 1. This does not break the functionality.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*k >= N - 1.*"
)

def graph_analysis(network: pd.DataFrame, graph_name: str, omics_list: Optional[List[pd.DataFrame]] = None) -> None:
    """Analyze and log basic topology and small components of a graph.

    The adjacency matrix is converted to a NetworkX graph, summary metrics such as node and edge counts, largest component size, average degree, and clustering coefficient are logged, and small components (≤10 nodes) are reported, optionally with modality-aware counts derived from omics_list.

    Args:

        network (pd.DataFrame): Square adjacency matrix with nodes as both rows and columns.
        graph_name (str): Descriptive name used for log messages identifying the graph.
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

    logger.info(f"GRAPH ANALYSIS: {graph_name}\n")
    logger.info(f"Nodes: {num_nodes:,} | Edges: {num_edges:,}")
    logger.info(f"Avg degree: {avg_degree:.2f}")
    logger.info(f"Connected components: {num_components}")
    logger.info(
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
        logger.info(
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

                logger.info(
                    f"[Island #{comp_id}] size={len(labels)} | {counts_str}"
                )
            else:
                counts = {"cnv": 0, "rna": 0, "meth": 0, "other": 0}
                for name in labels:
                    if isinstance(name, str) and name.startswith("cnv_"):
                        counts["cnv"] += 1
                    elif isinstance(name, str) and name.startswith("rna_"):
                        counts["rna"] += 1
                    elif isinstance(name, str) and name.startswith("meth_"):
                        counts["meth"] += 1
                    else:
                        counts["other"] += 1

                logger.info(
                    f"[Island #{comp_id}] size={len(labels)} | "
                    f"cnv={counts['cnv']}, rna={counts['rna']}, "
                    f"meth={counts['meth']}, other={counts['other']}"
                )

            logger.info(
                f"[Island #{comp_id}] nodes (up to 10): {labels[:10]}"
            )

    try:
        avg_clustering = nx.average_clustering(G, weight="weight")
        logger.info(f"Avg clustering coefficient: {avg_clustering:.3f}")
    except Exception:
        logger.warning("Could not compute clustering coefficient")


def repair_graph_connectivity(adj_df: pd.DataFrame, epsilon: float = 1e-6, selection_mode: str = "eigen", self_loops: bool = False, omics_list: Optional[List[pd.DataFrame]] = None, verbose: bool = False) -> pd.DataFrame:
    """Augment an adjacency matrix to connect all components via hub-based bridging edges.

    The adjacency matrix is decomposed into connected components, reference hubs are identified in the largest component using eigenvector centrality or degree, and each smaller component is connected back by adding symmetric edges of weight epsilon to suitable hubs, optionally guided by omics-based correlations.

    Args:

        adj_df (pd.DataFrame): Square adjacency matrix with nodes as both rows and columns.
        epsilon (float): Weight assigned to each added bridging edge between components.
        selection_mode (str): Strategy for choosing target nodes; one of "eigen", "eigen_corr", or "modality_corr".
        omics_list (list[pd.DataFrame] | None): Optional list of omics DataFrames used for correlation-based selection when selection_mode is "eigen_corr" or "modality_corr".
        verbose (bool): If True, log detailed information about each added bridging edge.

    Returns:

        pd.DataFrame: Augmented adjacency matrix in which all components are connected through newly added edges.

    """
    adj = adj_df.values.copy()
    np.fill_diagonal(adj, 0)

    G_nx = nx.from_numpy_array(adj)
    idx_to_node = list(adj_df.index)
    node_to_idx = {}
    for i, node in enumerate(idx_to_node):
        node_to_idx[node] = i

    components = list(nx.connected_components(G_nx))
    components.sort(key=len, reverse=True)

    if len(components) <= 1:
        return adj_df

    lcc_indices = list(components[0])
    G_lcc = G_nx.subgraph(lcc_indices)

    try:
        eig_lcc = nx.eigenvector_centrality_numpy(G_lcc, weight="weight")
    except Exception:
        eig_lcc = nx.degree_centrality(G_lcc)

    global_refs_sorted = sorted(eig_lcc.keys(), key=lambda n: eig_lcc[n], reverse=True)

    if not global_refs_sorted:
        return adj_df

    omics_aligned = None
    node_to_omic_idx = {}

    if selection_mode in ("eigen_corr", "modality_corr"):
        if omics_list is None:
            raise ValueError("omics_list required for correlation-based linkage.")

        omics_combined = pd.concat(omics_list, axis=1)
        for i, omic_df in enumerate(omics_list):
            for col in omic_df.columns:
                node_to_omic_idx[col] = i
        omics_aligned = omics_combined.loc[:, adj_df.index]

    minor_components = components[1:]
    minor_components_sorted = list(minor_components)
    minor_components_sorted.sort(key=len, reverse=True)

    local_centroids = []
    for comp in minor_components_sorted:
        comp_indices = list(comp)
        G_comp = G_nx.subgraph(comp_indices)

        if len(comp_indices) == 1:
            centroid_idx = comp_indices[0]
        else:
            try:
                eig_comp = nx.eigenvector_centrality_numpy(G_comp, weight="weight")
                centroid_idx = None
                best_val = None
                for node_idx, val in eig_comp.items():
                    if best_val is None or val > best_val:
                        best_val = val
                        centroid_idx = node_idx
            except Exception:
                deg_comp = dict(G_comp.degree(weight="weight"))
                centroid_idx = None
                best_deg = None
                for node_idx, deg_val in deg_comp.items():
                    if best_deg is None or deg_val > best_deg:
                        best_deg = deg_val
                        centroid_idx = node_idx

        local_centroids.append((comp_indices, centroid_idx))

    num_refs = len(global_refs_sorted)
    edges_added = 0

    for k, (comp_indices, local_centroid_idx) in enumerate(local_centroids):
        local_label = idx_to_node[local_centroid_idx]

        if selection_mode == "eigen":
            target_idx = global_refs_sorted[k % num_refs]
            target_label = idx_to_node[target_idx]
        else:
            assert omics_aligned is not None, "omics_aligned should not be None for correlation modes"
            local_vec = omics_aligned[local_label]
            candidate_labels = []
            for idx_val in global_refs_sorted:
                candidate_labels.append(idx_to_node[idx_val])

            if selection_mode == "modality_corr":
                if "_" in local_label:
                    prefix = local_label.split("_", 1)[0]
                else:
                    prefix = None

                same_prefix_labels = []
                if prefix is not None:
                    for lab in candidate_labels:
                        if lab.startswith(prefix + "_"):
                            same_prefix_labels.append(lab)

                if len(same_prefix_labels) > 0:
                    candidate_labels = same_prefix_labels
                elif omics_list is not None:
                    local_omic_id = node_to_omic_idx.get(local_label)
                    if local_omic_id is not None:
                        same_omic_labels = []
                        for lab in candidate_labels:
                            if node_to_omic_idx.get(lab) == local_omic_id:
                                same_omic_labels.append(lab)
                        if len(same_omic_labels) > 0:
                            candidate_labels = same_omic_labels

            candidate_matrix = omics_aligned[candidate_labels]
            corr_series = candidate_matrix.corrwith(local_vec)

            best_abs = None
            best_label = None
            for label_val, corr_val in corr_series.items():
                if best_abs is None or abs(corr_val) > best_abs:
                    best_abs = abs(corr_val)
                    best_label = label_val

            target_label = best_label
            target_idx = node_to_idx[best_label]

        adj[local_centroid_idx, target_idx] = epsilon
        adj[target_idx, local_centroid_idx] = epsilon
        edges_added += 1

        if verbose:
            logger.info(
                f"[Connectivity] Component_Size={len(comp_indices)} | "
                f"Local_Centroid='{local_label}' -> Global_Ref='{target_label}' (eps={epsilon:.1e})"
            )

    connected_graph = pd.DataFrame(adj, index=adj_df.index, columns=adj_df.columns)
    if self_loops:
        np.fill_diagonal(connected_graph.values, 1.0)

    return connected_graph


def find_optimal_graph(omics_data: pd.DataFrame, y_labels, methods: list = ['correlation', 'threshold', 'similarity', 'gaussian'], seed: int = 1883, verbose: bool = True, trials: Optional[int] = None, omics_list: Optional[List[pd.DataFrame]] = None, centrality_mode="eigenvector") -> tuple[pd.DataFrame | None, dict | None, pd.DataFrame]:
    """Search over graph-construction hyperparameters using a structural proxy task.

    For each requested method and parameter configuration, a graph is generated, connectivity is repaired, node weights are approximated by a structural proxy, and a Ridge classifier with stratified cross-validation is used to score performance. The best-performing topology and its parameters are returned along with a table of all evaluated configurations.

    Args:

        omics_data (pd.DataFrame): Input feature matrix with samples as rows and multi-omics features as columns.
        y_labels (pd.Series | np.ndarray | list): Target labels used for stratified cross-validation and proxy evaluation.
        methods (list[str]): Graph-construction methods to consider; valid options include "correlation", "threshold", "similarity", and "gaussian".
        seed (int): Random seed used for configuration shuffling and cross-validation splitting.
        verbose (bool): If True, print a brief progress message for each configuration's best epsilon.
        trials (int | None): Optional limit on the number of configurations taken from the full parameter grid.
        omics_list (list[pd.DataFrame] | None): Optional list of omics blocks aligned to columns of omics_data and forwarded to repair_graph_connectivity.
        centrality_mode (str): Centrality measure used for proxy feature weighting; one of "eigenvector" or "degree".

    Returns:

        tuple[pd.DataFrame | None, dict | None, pd.DataFrame]: Best repaired graph (or None if all runs fail), parameter dictionary for the best configuration (or None), and a DataFrame summarizing scores and settings for all evaluated graphs.

    """
    if isinstance(y_labels, pd.Series):
        y_vec = y_labels.values
    else:
        y_vec = np.asarray(y_labels).ravel()

    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(omics_data.values)
    X_scaled = pd.DataFrame(X_scaled_np, index=omics_data.index, columns=omics_data.columns)

    if omics_list is None:
        bridge_omics_list = [omics_data]
    else:
        bridge_omics_list = omics_list

    base_grids = {
        'gaussian': {'k': [10, 15, 20], 'sigma': [0.5, 1.0, 2.0], 'mutual': [True, False]},
        'similarity': {'k': [10, 15, 20], 'metric': ['cosine'], 'mutual': [True, False]},
        'correlation': {'k': [10, 12, 14, 16, 18, 20], 'method': ['pearson', 'spearman'], 'signed': [False], 'threshold': [None]},
        'threshold': {'b': [5.5, 6.0, 6.5, 7.0, 8.0], 'k': [5, 10, 15], 'mutual': [True, False]},
    }

    linkage_modes = ['modality_corr', 'eigen_corr']

    all_configs = []
    for method_name in methods:
        if method_name not in base_grids:
            continue
        grid = ParameterGrid(base_grids[method_name])
        for gen_params in grid:
            for mode in linkage_modes:
                all_configs.append((method_name, gen_params, mode))

    rng = np.random.RandomState(seed)
    rng.shuffle(all_configs)
    if trials:
        if trials < len(all_configs):
            all_configs = all_configs[:trials]

    logger.info(f"Total configurations to evaluate: {len(all_configs)}")
    if verbose:
        logger.info(f"Starting Topology Optimization (n_trials={len(all_configs)})")

    best_score = -np.inf
    best_config = None
    results = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for idx, (method_name, gen_params, linkage_mode) in enumerate(all_configs, start=1):
        try:
            params_clean = gen_params.copy()
            params_clean['self_loops'] = False

            if method_name == 'threshold':
                G = gen_threshold_graph(omics_data, **params_clean)
            elif method_name == 'gaussian':
                G = gen_gaussian_knn_graph(omics_data, **params_clean)
            elif method_name == 'correlation':
                G = gen_correlation_graph(omics_data, **params_clean)
            elif method_name == 'similarity':
                G = gen_similarity_graph(omics_data, **params_clean)
            else:
                continue

            eps_schedule = _find_optimal_epsilon(G, n_eps=10)

            trial_best_score = -np.inf
            trial_best_msg = None

            for epsilon in eps_schedule:
                try:
                    G_repaired = repair_graph_connectivity(G, epsilon=epsilon, selection_mode=linkage_mode, omics_list=bridge_omics_list, verbose=False)
                    mean_f1, std_f1 = _feature_proxy(G_repaired, X_scaled, y_vec, cv, mode=centrality_mode)
                    stability_score = mean_f1 - (2.0 * std_f1)

                except Exception:
                    mean_f1, std_f1, stability_score = 0.0, 0.0, -np.inf

                if stability_score > trial_best_score:
                    trial_best_score = stability_score
                    trial_best_msg = (
                        f"[{idx}/{len(all_configs)}] {method_name[:4].upper()} "
                        f"| Linkage={linkage_mode} | Eps={epsilon:.1e} "
                        f"| F1: {mean_f1:.1f}% +/- {std_f1:.1f} | Stab: {stability_score:.1f}"
                    )

                if stability_score > best_score:
                    best_score = stability_score
                    best_config = {
                        'method': method_name,
                        'graph': G_repaired,
                        'params': {**gen_params, 'linkage_mode': linkage_mode, 'epsilon': epsilon},
                        'stats': f"{mean_f1:.1f}% +/- {std_f1:.1f}%",
                        'stability_score': stability_score,
                    }

                results.append({
                    'method': method_name,
                    'params': {**gen_params, 'linkage_mode': linkage_mode, 'epsilon': epsilon},
                    'score': stability_score,
                    'f1': mean_f1,
                    'std': std_f1,
                })

            if verbose and trial_best_msg is not None:
                logger.info(trial_best_msg)

        except Exception:
            continue

    if best_config is not None:
        logger.info(f"Optimal Topology: {best_config['method'].upper()}")
        logger.info(f"Performance: {best_config['stats']}")
        return best_config['graph'], best_config['params'], pd.DataFrame(results)

    return None, None, pd.DataFrame(results)


def _find_optimal_epsilon(adj_df: pd.DataFrame, n_eps: int = 10) -> List[float]:
    """Build a dynamic epsilon schedule around the weakest positive edge in the largest component.

    The smallest positive edge weight in the largest connected component is used to define a base scale one order of magnitude smaller, and a geometric grid of ``n_eps`` epsilon values is generated around this base and clipped to ``[1e-10, 1e-1]``. If this procedure fails, a fixed fallback schedule is returned.

    Args:

        adj_df (pd.DataFrame): Adjacency matrix with nodes as both index and columns.
        n_eps (int): Number of epsilon candidates to generate in the dynamic schedule.

    Returns:

        list[float]: Sorted list of unique epsilon values to try when repairing connectivity.

    """
    try:
        A = adj_df.values.copy().astype(float)
        np.fill_diagonal(A, 0.0)

        G_nx = nx.from_pandas_adjacency(adj_df)
        components = list(nx.connected_components(G_nx))
        if not components:
            raise RuntimeError("No connected components found.")

        largest_comp = max(components, key=len)

        idx_pos = []
        for lbl in largest_comp:
            idx = adj_df.index.get_loc(lbl)
            idx_pos.append(idx)

        A_largest = A[np.ix_(idx_pos, idx_pos)]

        positive = A_largest[A_largest > 0]
        if positive.size == 0:
            raise RuntimeError("No positive edges in largest component.")

        min_w = float(positive.min())
        exp = int(np.floor(np.log10(min_w)))
        eps_base = 10.0 ** (exp - 1)

        factors = np.geomspace(0.1, 10.0, num=n_eps)

        eps_list = []
        for f in factors:
            eps = eps_base * f
            eps = float(np.clip(eps, 1e-10, 1e-1))
            eps_list.append(eps)

        unique_eps = list(set(eps_list))
        unique_eps.sort()

        return unique_eps

    except Exception as e:
        logger.warning(f"Fallback epsilon schedule: {e}")
        return [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3]


def _feature_proxy(adj_df: pd.DataFrame, X_df: pd.DataFrame, y, cv, mode: str = "eigenvector") -> tuple[float, float]:
    """Approximate a GAT-style feature weighting pipeline for fast graph topology evaluation.

    A centrality score is computed for each node in the graph, mapped to feature weights, transformed with a log(1 + ReLU)-style scaling, applied element-wise to the feature matrix, and evaluated with a Ridge classifier using cross-validated weighted F1.

    Args:

        adj_df (pd.DataFrame): Adjacency matrix with nodes matching columns in ``X_df``.
        X_df (pd.DataFrame): Feature matrix of shape (n_samples, n_features), aligned to the graph nodes.
        y: Target labels for classification; array-like compatible with scikit-learn.
        cv: Cross-validation splitter (e.g. ``StratifiedKFold``) compatible with scikit-learn.
        mode (str): Centrality mode; one of ``"weighted_pagerank"``, ``"eigenvector"``, ``"degree"``, or any other string for unweighted degree.

    Returns:

        tuple[float, float]: Mean and standard deviation of the weighted F1 score (in percent) across CV folds.

    """
    G_nx = nx.from_pandas_adjacency(adj_df)

    try:
        if mode == "weighted_pagerank":
            weights = nx.pagerank(G_nx, weight="weight", alpha=0.85, tol=1e-4)

        elif mode == "eigenvector":
            weights = nx.eigenvector_centrality_numpy(G_nx, weight="weight")

        elif mode == "degree":
            weights = dict(G_nx.degree(weight="weight"))

        else:
            weights = dict(G_nx.degree(weight=None))

    except Exception:
        weights = dict(G_nx.degree(weight=None))

    w_list = []
    for col in X_df.columns:
        value = weights.get(col, 0.0)
        w_list.append(value)

    w_vec = np.array(w_list)
    w_vec = np.log1p(np.maximum(0.0, w_vec))

    X_weighted = X_df.values * w_vec[None, :]

    clf = RidgeClassifier()
    scores = cross_val_score(clf, X_weighted, y, cv=cv, scoring="f1_weighted", n_jobs=-1)

    mean_score = float(np.mean(scores) * 100.0)
    std_score = float(np.std(scores) * 100.0)

    return mean_score, std_score
