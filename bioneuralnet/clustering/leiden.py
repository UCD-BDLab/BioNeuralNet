import numpy as np
import networkx as nx
import pandas as pd
from typing import Optional, Union, Any
import torch

try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Leiden:
    """
    Leiden algorithm for community detection in graphs.

    The Leiden algorithm is an improvement over the Louvain algorithm,
    guaranteeing that communities are well-connected.

    This implementation uses the Constant Potts Model (CPM) via
    RBConfigurationVertexPartition, which supports resolution tuning.

    Attributes
    ----------
    G : nx.Graph
        NetworkX graph object.
    resolution_parameter : float
        Resolution parameter for CPM optimization (default: 1.0).
        Higher values lead to more communities.
    n_iterations : int
        Number of iterations to run the algorithm (default: -1 for auto).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        G: nx.Graph,
        resolution_parameter: float = 1.0,
        n_iterations: int = -1,
        seed: Optional[int] = None,
    ):
        if not LEIDEN_AVAILABLE:
            raise ImportError(
                "Leiden algorithm requires 'leidenalg' and 'igraph' packages. "
                "Install with: pip install leidenalg igraph"
            )

        self.logger = get_logger(__name__)
        self.G = G.copy()
        self.resolution_parameter = resolution_parameter
        self.n_iterations = n_iterations
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.logger.info(
            f"Initialized Leiden with resolution_parameter={resolution_parameter}, "
            f"n_iterations={n_iterations}, seed={seed}"
        )
        self.logger.info(
            f"Graph has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges."
        )

        self.partition: Optional[dict[Any, int]] = None

    def _nx_to_igraph(self) -> ig.Graph:
        """
        Convert NetworkX graph to igraph Graph object.

        Returns
        -------
        ig.Graph
            igraph representation of the graph.
        """
        # Get edge list with weights if available
        edges = []
        weights = []

        for u, v, data in self.G.edges(data=True):
            edges.append((u, v))
            weight = data.get("weight", 1.0)
            weights.append(weight)

        # Create igraph graph
        # Note: igraph uses integer node IDs, so we need to map node names to integers
        node_list = list(self.G.nodes())
        node_to_id = {node: idx for idx, node in enumerate(node_list)}

        # Convert edges to integer IDs
        int_edges = [(node_to_id[u], node_to_id[v]) for u, v in edges]

        # Create graph
        g = ig.Graph(int_edges, directed=False)

        # Add weights if available (check if any edge has a weight attribute)
        has_weight_attr = any("weight" in data for _, _, data in self.G.edges(data=True))
        if has_weight_attr and weights:
            g.es["weight"] = weights

        # Store node mapping for later
        g["node_names"] = node_list

        return g

    def run(self) -> dict:
        """
        Run Leiden community detection algorithm.

        Returns
        -------
        dict
            Partition dictionary mapping node names to community IDs.
        """
        self.logger.info("Running Leiden community detection...")

        # Convert NetworkX graph to igraph
        g = self._nx_to_igraph()
        node_list = g["node_names"]

        # Check if graph has weights
        has_weights = "weight" in g.es.attributes() and g.es["weight"] is not None

        # Run Leiden algorithm
        # Use RBConfigurationVertexPartition which supports resolution_parameter
        # This uses the Constant Potts Model (CPM) which allows tuning community resolution
        if has_weights:
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.resolution_parameter,
                n_iterations=self.n_iterations,
                seed=self.seed,
                weights="weight",
            )
        else:
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self.resolution_parameter,
                n_iterations=self.n_iterations,
                seed=self.seed,
            )

        # Convert partition back to node names
        membership = partition.membership
        self.partition = {node_list[i]: int(membership[i]) for i in range(len(node_list))}

        n_clusters = len(set(self.partition.values()))
        self.logger.info(f"Leiden clustering complete: {n_clusters} communities detected.")

        return self.partition

    def get_modularity(self) -> float:
        """
        Compute modularity of the current partition.

        Returns
        -------
        float
            Modularity score.
        """
        if self.partition is None:
            raise ValueError("No partition computed. Call run() first.")

        # Convert to igraph for modularity computation
        g = self._nx_to_igraph()
        node_list = g["node_names"]

        # Create membership list
        membership = [self.partition[node] for node in node_list]

        # Check if graph has weights
        has_weights = "weight" in g.es.attributes() and g.es["weight"] is not None

        if has_weights:
            modularity = g.modularity(membership, weights="weight")
        else:
            modularity = g.modularity(membership)

        return float(modularity)

    def get_cluster_sizes(self) -> dict:
        """
        Get sizes of each cluster.

        Returns
        -------
        dict
            Dictionary mapping cluster ID to number of nodes.
        """
        if self.partition is None:
            raise ValueError("No partition computed. Call run() first.")

        cluster_sizes = {}
        for node, cluster_id in self.partition.items():
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        return cluster_sizes
