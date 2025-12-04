from __future__ import annotations
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from bioneuralnet.utils.logger import get_logger
import pandas as pd
from typing import Sequence, Tuple


class Spectral_Clustering:
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
    def __init__(
        self,
        G: nx.Graph,
        n_clusters: int,
        use_edge_weights: bool = True,
        random_state: int | None = 0,
    ):
        self.logger = get_logger(__name__)
        self.G = G
        self.n_clusters = n_clusters
        self.random_state = random_state

        # fixed node order
        self.nodes_order = list(G.nodes())
        n = len(self.nodes_order)
        node_index = {node: i for i, node in enumerate(self.nodes_order)}

        # build dense affinity matrix
        self.A = np.zeros((n, n), dtype=float)

        if use_edge_weights:
            for u, v, data in G.edges(data=True):
                i = node_index[u]
                j = node_index[v]
                w = float(data.get("weight", 1.0))
                self.A[i, j] = w
                self.A[j, i] = w  # assume undirected
        else:
            for u, v in G.edges():
                i = node_index[u]
                j = node_index[v]
                self.A[i, j] = 1.0
                self.A[j, i] = 1.0
        if not np.any(self.A):
            raise ValueError("Graph has no edges; spectral clustering is undefined.")

        self.logger.info(
            f"Running SpectralClustering on graph with {n} nodes, "
            f"{G.number_of_edges()} edges, n_clusters={n_clusters}, "
            f"use_edge_weights={use_edge_weights}."
        )

    def run(self) -> Tuple[np.ndarray, Sequence]:
        """
        Execute spectral clustering and return labels and node order.

        Returns
        labels : np.ndarray, shape (n_nodes,)
            Cluster label for each node, in the order of `nodes_order`.
        nodes_order : list
            List of nodes corresponding to `labels`.
        """
        n_clusters = self.n_clusters
        random_state = self.random_state
        A = self.A
        nodes_order = self.nodes_order
        spec = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
            n_init=10,
        )
        labels = spec.fit_predict(A)

        return labels, nodes_order
