# bioneuralnet/clustering/spectral.py

from __future__ import annotations
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from bioneuralnet.utils.logger import get_logger
import pandas as pd
from typing import Sequence, Tuple


def spectral_cluster_graph(
    G: nx.Graph,
    n_clusters: int,
    use_edge_weights: bool = True,
    random_state: int | None = 0,
) -> Tuple[np.ndarray, Sequence]:
    """
    Spectral clustering on a NetworkX graph using its adjacency
    as a precomputed affinity matrix.

    Parameters
    G : networkx.Graph Input graph.
    n_clusters : int Number of clusters to find.
    use_edge_weights : bool, default=True
    random_state : int or None, default=0
    Returns
    labels : np.ndarray, shape (n_nodes,) Cluster label for each node, in the order of `nodes_order`.
    nodes_order : list List of nodes corresponding to `labels`.
    """
    logger = get_logger(__name__)

    # fixed node order
    nodes_order = list(G.nodes())
    n = len(nodes_order)
    node_index = {node: i for i, node in enumerate(nodes_order)}

    # build dense affinity matrix
    A = np.zeros((n, n), dtype=float)

    if use_edge_weights:
        for u, v, data in G.edges(data=True):
            i = node_index[u]
            j = node_index[v]
            w = float(data.get("weight", 1.0))
            A[i, j] = w
            A[j, i] = w  # assume undirected
    else:
        for u, v in G.edges():
            i = node_index[u]
            j = node_index[v]
            A[i, j] = 1.0
            A[j, i] = 1.0
    if not np.any(A):
        raise ValueError("Graph has no edges; spectral clustering is undefined.")

    logger.info(
        f"Running SpectralClustering on graph with {n} nodes, "
        f"{G.number_of_edges()} edges, n_clusters={n_clusters}, "
        f"use_edge_weights={use_edge_weights}."
    )

    spec = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=10,
    )
    labels = spec.fit_predict(A)

    return labels, nodes_order
