"""
Graph construction utilities for Parkinson's disease gene-gene correlation networks.

This module builds gene-gene correlation graphs from expression data and converts
them to PyTorch Geometric format for GNN training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import networkx as nx

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError(
        "PyTorch Geometric is required for graph construction. "
        "Please install it: pip install torch-geometric"
    )

# Import BioNeuralNet utilities
import sys
from pathlib import Path as PathLib

project_root = PathLib(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.utils.graph import gen_correlation_graph
from bioneuralnet.utils import get_logger
from bioneuralnet.metrics import plot_network

logger = get_logger(__name__)


@dataclass
class GraphData:
    """
    Container for graph data compatible with PyTorch Geometric and BioNeuralNet.

    Attributes
    ----------
    data : torch_geometric.data.Data
        PyTorch Geometric Data object containing:
        - x: Node features (num_nodes × num_features)
        - edge_index: Edge connectivity (2 × num_edges)
        - edge_weight: Optional edge weights (num_edges,)
    adjacency_matrix : pd.DataFrame
        Gene-gene adjacency matrix (genes × genes).
    node_names : list
        List of gene IDs matching the node indices.
    """

    data: Data
    adjacency_matrix: pd.DataFrame
    node_names: list


def build_correlation_graph(
    expression_df: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.7,
    use_abs: bool = True,
    mutual: bool = True,
    self_loops: bool = False,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a gene-gene correlation graph using threshold-based edge selection.

    This is a simpler alternative to BioNeuralNet's kNN-based correlation graph,
    using a fixed correlation threshold instead. For large datasets, uses
    chunked computation to avoid memory issues.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Gene expression matrix (genes × samples).
    method : str, default="pearson"
        Correlation method: "pearson" or "spearman".
    threshold : float, default=0.7
        Minimum absolute correlation to create an edge.
    use_abs : bool, default=True
        Whether to use absolute correlation values.
    mutual : bool, default=True
        Whether to keep only mutual edges (symmetric graph).
    self_loops : bool, default=False
        Whether to include self-loops.
    chunk_size : int, optional
        If provided, compute correlations in chunks to save memory.
        If None, uses full matrix computation (faster but memory-intensive).

    Returns
    -------
    pd.DataFrame
        Adjacency matrix (genes × genes) with correlation values as edge weights.
    """
    n_genes = expression_df.shape[0]
    logger.info(
        f"Building correlation graph from expression matrix of shape {expression_df.shape} "
        f"using method='{method}', threshold={threshold}."
    )

    # Warn if too many genes
    if n_genes > 10000:
        logger.warning(
            f"Large number of genes ({n_genes}). Consider using chunked computation "
            f"or reducing the number of genes via HVG selection."
        )
        if chunk_size is None:
            # Auto-set chunk size for large datasets
            chunk_size = min(5000, n_genes // 4)
            logger.info(f"Auto-setting chunk_size={chunk_size} for memory efficiency.")

    # Memory-efficient chunked computation for large datasets
    if chunk_size is not None and n_genes > chunk_size:
        logger.info(f"Using chunked correlation computation (chunk_size={chunk_size}).")
        adjacency = _build_correlation_graph_chunked(
            expression_df, method, threshold, use_abs, chunk_size
        )
    else:
        # Standard full matrix computation
        logger.info("Using full matrix correlation computation.")
        corr_matrix = expression_df.T.corr(method=method)  # genes × genes

        if use_abs:
            corr_matrix = corr_matrix.abs()

        # Apply threshold
        adjacency = corr_matrix.copy()
        adjacency[adjacency < threshold] = 0.0

    # Remove self-loops if requested
    if not self_loops:
        np.fill_diagonal(adjacency.values, 0.0)

    # Make mutual (symmetric) if requested
    if mutual:
        adjacency = (adjacency + adjacency.T) / 2

    logger.info(
        f"Graph constructed: {adjacency.shape[0]} nodes, "
        f"{(adjacency > 0).sum().sum() / 2:.0f} edges (threshold={threshold})."
    )

    return adjacency


def _build_correlation_graph_chunked(
    expression_df: pd.DataFrame,
    method: str,
    threshold: float,
    use_abs: bool,
    chunk_size: int,
) -> pd.DataFrame:
    """
    Build correlation graph using chunked computation to save memory.

    Computes correlations in chunks and only stores values above threshold.
    """
    n_genes = expression_df.shape[0]
    gene_names = expression_df.index.tolist()

    # Initialize sparse adjacency matrix
    adjacency = pd.DataFrame(
        0.0, index=gene_names, columns=gene_names, dtype=np.float32
    )

    # Compute correlations in chunks
    n_chunks = (n_genes + chunk_size - 1) // chunk_size
    logger.info(f"Computing correlations in {n_chunks} chunks...")

    for i in range(0, n_genes, chunk_size):
        chunk_i = expression_df.iloc[i : i + chunk_size]
        logger.info(f"Processing chunk {i // chunk_size + 1}/{n_chunks}...")

        for j in range(i, n_genes, chunk_size):
            chunk_j = expression_df.iloc[j : j + chunk_size]

            # Compute correlation between chunks
            corr_chunk = chunk_i.T.corr(chunk_j.T, method=method)

            if use_abs:
                corr_chunk = corr_chunk.abs()

            # Apply threshold and store
            mask = corr_chunk >= threshold
            for idx_i in corr_chunk.index:
                for idx_j in corr_chunk.columns:
                    if mask.loc[idx_i, idx_j]:
                        adjacency.loc[idx_i, idx_j] = corr_chunk.loc[idx_i, idx_j]

    return adjacency


def adjacency_to_pyg(
    adjacency_matrix: pd.DataFrame,
    node_features: pd.DataFrame,
    node_names: Optional[list] = None,
) -> tuple[Data, list]:
    """
    Convert adjacency matrix to PyTorch Geometric Data object.

    Parameters
    ----------
    adjacency_matrix : pd.DataFrame
        Gene-gene adjacency matrix (genes × genes).
    node_features : pd.DataFrame
        Node feature matrix (genes × features).
    node_names : list, optional
        List of gene IDs. If None, uses adjacency_matrix.index.

    Returns
    -------
    tuple[Data, list]
        (PyG Data object, list of node names)
    """
    logger.info(
        f"Converting adjacency matrix ({adjacency_matrix.shape}) and "
        f"node features ({node_features.shape}) to PyTorch Geometric format."
    )

    # Ensure node names match
    if node_names is None:
        node_names = adjacency_matrix.index.tolist()

    # Align node features with adjacency matrix
    common_nodes = set(adjacency_matrix.index) & set(node_features.index)
    if len(common_nodes) != len(adjacency_matrix.index):
        logger.warning(
            f"Node mismatch: {len(adjacency_matrix.index)} nodes in adjacency, "
            f"{len(node_features.index)} in features, {len(common_nodes)} common."
        )
        # Filter to common nodes
        adjacency_matrix = adjacency_matrix.loc[common_nodes, common_nodes]
        node_features = node_features.loc[common_nodes]
        node_names = [n for n in node_names if n in common_nodes]

    # Convert to NetworkX graph
    G = nx.from_pandas_adjacency(adjacency_matrix)

    # Create node mapping (gene ID → integer index)
    node_mapping = {node_name: idx for idx, node_name in enumerate(node_names)}
    G = nx.relabel_nodes(G, node_mapping)

    # Build edge_index (2 × num_edges)
    edge_list = list(G.edges())
    if len(edge_list) == 0:
        logger.warning("Graph has no edges! Creating empty edge_index.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # Make undirected (add reverse edges)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Extract edge weights
        edge_weight = torch.tensor(
            [G[u][v].get("weight", 1.0) for u, v in edge_list],
            dtype=torch.float,
        )
        # Duplicate for reverse edges
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    # Node features
    # Ensure features are aligned with node order
    feature_matrix = node_features.loc[node_names].values
    x = torch.tensor(feature_matrix, dtype=torch.float)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    logger.info(
        f"PyG Data object created: {data.num_nodes} nodes, {data.num_edges} edges, "
        f"{data.num_node_features} features per node."
    )

    return data, node_names


def build_pd_graph(
    expression_df: pd.DataFrame,
    node_features: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.7,
    use_abs: bool = True,
    mutual: bool = True,
    self_loops: bool = False,
    use_bioneuralnet: bool = False,
    k_neighbors: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> GraphData:
    """
    Complete pipeline: build gene-gene correlation graph and convert to PyG format.

    This is the main function to use for building graphs from PD expression data.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Preprocessed gene expression matrix (genes × samples).
    node_features : pd.DataFrame
        Node feature matrix (genes × features) for GNN input.
    method : str, default="pearson"
        Correlation method: "pearson" or "spearman".
    threshold : float, default=0.7
        Minimum absolute correlation to create an edge (if use_bioneuralnet=False).
    use_abs : bool, default=True
        Whether to use absolute correlation values.
    mutual : bool, default=True
        Whether to keep only mutual edges.
    self_loops : bool, default=False
        Whether to include self-loops.
    use_bioneuralnet : bool, default=False
        If True, uses BioNeuralNet's gen_correlation_graph (kNN-based).
        If False, uses threshold-based correlation graph.
    k_neighbors : int, optional
        Number of neighbors per node (only used if use_bioneuralnet=True).
        If None, uses threshold-based selection.
    chunk_size : int, optional
        Chunk size for memory-efficient correlation computation.
        If None, auto-detects based on dataset size.

    Returns
    -------
    GraphData
        Container with PyG Data object, adjacency matrix, and node names.
    """
    logger.info("=" * 60)
    logger.info("Building PD gene-gene correlation graph.")
    logger.info("=" * 60)

    # Check if expression has too many genes (memory warning)
    n_genes = expression_df.shape[0]
    if n_genes > 10000:
        logger.warning(
            f"WARNING: Expression matrix has {n_genes} genes. "
            f"This may cause memory issues. Consider using preprocessed expression "
            f"with fewer genes (e.g., top 5000 HVGs)."
        )
        if chunk_size is None:
            chunk_size = min(5000, n_genes // 4)
            logger.info(f"Auto-enabling chunked computation (chunk_size={chunk_size}).")

    # Build adjacency matrix
    if use_bioneuralnet:
        logger.info("Using BioNeuralNet's gen_correlation_graph (kNN-based).")
        # BioNeuralNet expects samples × features, so we transpose
        # It also expects features to be nodes, so we transpose expression_df
        adjacency = gen_correlation_graph(
            expression_df.T,  # samples × genes
            k=k_neighbors if k_neighbors else 15,
            method=method,
            mutual=mutual,
            per_node=(k_neighbors is not None),
            threshold=threshold if k_neighbors is None else None,
            self_loops=self_loops,
        )
        # Result is genes × genes (as expected)
    else:
        logger.info("Using threshold-based correlation graph.")
        adjacency = build_correlation_graph(
            expression_df,
            method=method,
            threshold=threshold,
            use_abs=use_abs,
            mutual=mutual,
            self_loops=self_loops,
            chunk_size=chunk_size,
        )

    # Convert to PyG format
    data, node_names = adjacency_to_pyg(adjacency, node_features)

    logger.info("=" * 60)
    logger.info("Graph construction complete!")
    logger.info(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    logger.info("=" * 60)

    return GraphData(
        data=data,
        adjacency_matrix=adjacency,
        node_names=node_names,
    )


def save_graph(graph_data: GraphData, filepath: str | Path) -> None:
    """
    Save graph data to disk.

    Parameters
    ----------
    graph_data : GraphData
        Graph data to save.
    filepath : str or Path
        Path to save the graph (will save as .pt file for PyG Data,
        and .csv for adjacency matrix).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save PyG Data
    torch.save(graph_data.data, filepath.with_suffix(".pt"))
    logger.info(f"Saved PyG Data to {filepath.with_suffix('.pt')}")

    # Save adjacency matrix
    graph_data.adjacency_matrix.to_csv(filepath.with_suffix(".csv"))
    logger.info(f"Saved adjacency matrix to {filepath.with_suffix('.csv')}")


def visualize_graph(
    graph_data: GraphData,
    weight_threshold: float = 0.0,
    show_labels: bool = False,
    show_edge_weights: bool = False,
    layout: str = "kamada",
    max_nodes: Optional[int] = 1000,
    figsize: tuple[int, int] = (14, 8),
):
    """
    Visualize the gene-gene correlation graph.

    Parameters
    ----------
    graph_data : GraphData
        Graph data to visualize.
    weight_threshold : float, default=0.0
        Minimum edge weight to display.
    show_labels : bool, default=False
        Whether to show node labels (gene IDs).
    show_edge_weights : bool, default=False
        Whether to show edge weights on the graph.
    layout : str, default="kamada"
        Layout algorithm: "kamada", "spring", or "spectral".
    max_nodes : int, optional
        Maximum number of nodes to visualize. If graph is larger,
        samples a subgraph. If None, visualizes all nodes.
    figsize : tuple, default=(14, 8)
        Figure size (width, height).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    logger.info(f"Visualizing graph with {graph_data.data.num_nodes} nodes...")

    adjacency = graph_data.adjacency_matrix.copy()

    # Sample subgraph if too large
    if max_nodes is not None and adjacency.shape[0] > max_nodes:
        logger.info(
            f"Graph has {adjacency.shape[0]} nodes. Sampling {max_nodes} nodes for visualization."
        )
        # Sample nodes with highest degree
        degrees = (adjacency > 0).sum(axis=1).sort_values(ascending=False)
        top_nodes = degrees.head(max_nodes).index.tolist()
        adjacency = adjacency.loc[top_nodes, top_nodes]
        logger.info(f"Visualizing subgraph with {len(top_nodes)} nodes.")

    # Use BioNeuralNet's plot_network function
    node_mapping = plot_network(
        adjacency,
        weight_threshold=weight_threshold,
        show_labels=show_labels,
        show_edge_weights=show_edge_weights,
        layout=layout,
    )

    logger.info("Graph visualization complete.")
    return node_mapping


def load_graph(
    filepath: str | Path, node_names_file: Optional[str | Path] = None
) -> GraphData:
    """
    Load graph data from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the graph file (.pt for PyG Data, .csv for adjacency).
    node_names_file : str or Path, optional
        Path to file containing node names (if saved separately).

    Returns
    -------
    GraphData
        Loaded graph data.
    """
    filepath = Path(filepath)

    # Load PyG Data
    data = torch.load(filepath.with_suffix(".pt"))
    logger.info(f"Loaded PyG Data from {filepath.with_suffix('.pt')}")

    # Load adjacency matrix
    adjacency = pd.read_csv(
        filepath.with_suffix(".csv"), index_col=0
    )
    logger.info(f"Loaded adjacency matrix from {filepath.with_suffix('.csv')}")

    # Load node names
    if node_names_file and Path(node_names_file).exists():
        with open(node_names_file, "r") as f:
            node_names = [line.strip() for line in f]
    else:
        node_names = adjacency.index.tolist()

    return GraphData(data=data, adjacency_matrix=adjacency, node_names=node_names)
