import numpy as np
import networkx as nx
import pandas as pd
import torch
from typing import Optional, Union, Any

try:
    import igraph as ig
    import leidenalg as la
    _LEIDEN_AVAILABLE = True
except ImportError:
    _LEIDEN_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Leiden:
    """
    Leiden Class for Community Detection using the Leiden Algorithm.
    
    Attributes:
        G (nx.Graph): NetworkX graph object.
        resolution_parameter (float): Resolution parameter for modularity optimization.
        n_iterations (int): Number of iterations for the algorithm.
        seed (Optional[int]): Random seed for reproducibility.
        partition_type: Type of partition to optimize (default: ModularityVertexPartition).
    """
    
    def __init__(
        self,
        G: nx.Graph,
        resolution_parameter: float = 1.0,
        n_iterations: int = 2,
        partition_type: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Leiden algorithm.
        
        Args:
            G (nx.Graph): NetworkX graph object.
            resolution_parameter (float): Resolution parameter for modularity optimization.
                Higher values lead to more communities. Default is 1.0.
            n_iterations (int): Number of iterations for the algorithm. Default is 2.
            partition_type (Optional[str]): Type of partition to optimize.
                Options: 'ModularityVertexPartition', 'RBERVertexPartition', 'CPMVertexPartition'.
                Default is 'ModularityVertexPartition'.
            seed (Optional[int]): Random seed for reproducibility.
        """
        if not _LEIDEN_AVAILABLE:
            raise ImportError(
                "leidenalg and python-igraph are required for Leiden algorithm. "
                "Install them with: pip install leidenalg python-igraph"
            )
        
        self.logger = get_logger(__name__)
        self.G = G.copy()
        self.resolution_parameter = resolution_parameter
        self.n_iterations = n_iterations
        self.seed = seed
        
        # Set partition type
        if partition_type is None:
            self.partition_type = la.ModularityVertexPartition
        elif partition_type == 'ModularityVertexPartition':
            self.partition_type = la.ModularityVertexPartition
        elif partition_type == 'RBERVertexPartition':
            self.partition_type = la.RBERVertexPartition
        elif partition_type == 'CPMVertexPartition':
            self.partition_type = la.CPMVertexPartition
        else:
            self.logger.warning(
                f"Unknown partition type '{partition_type}'. Using ModularityVertexPartition."
            )
            self.partition_type = la.ModularityVertexPartition
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        self.partition = None
        self.quality = None
        
        self.logger.info(
            f"Initialized Leiden algorithm with resolution_parameter={resolution_parameter}, "
            f"n_iterations={n_iterations}, partition_type={self.partition_type.__name__}"
        )
        self.logger.info(
            f"Graph has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges."
        )
    
    def _nx_to_igraph(self, nx_graph: nx.Graph) -> ig.Graph:
        """
        Convert a NetworkX graph to an igraph Graph.
        
        Args:
            nx_graph (nx.Graph): NetworkX graph.
            
        Returns:
            ig.Graph: igraph graph object.
        """
        # Get node list to preserve order
        node_list = list(nx_graph.nodes())
        
        # Create mapping from node names to indices
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Get edge list with indices and weights
        edges = []
        weights = []
        
        for u, v, data in nx_graph.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            weight = data.get('weight', 1.0)
            weights.append(float(weight))
        
        # Create igraph graph
        ig_graph = ig.Graph(edges=edges, directed=nx_graph.is_directed(), n=len(node_list))
        
        # Add node names as attribute
        ig_graph.vs['name'] = node_list
        
        # Add edge weights if present
        if weights:
            ig_graph.es['weight'] = weights
        
        # Add node attributes if present
        for node in nx_graph.nodes():
            node_idx = node_to_idx[node]
            for attr, value in nx_graph.nodes[node].items():
                if attr != 'name':  # Avoid overwriting name attribute
                    if attr not in ig_graph.vs.attributes():
                        ig_graph.vs[attr] = [None] * len(ig_graph.vs)
                    ig_graph.vs[node_idx][attr] = value
        
        return ig_graph
    
    def _partition_to_dict(self, partition, node_names) -> dict:
        """
        Convert leidenalg partition to a dictionary mapping node names to community IDs.
        
        Args:
            partition: leidenalg partition object.
            node_names: List of node names corresponding to igraph vertex indices.
            
        Returns:
            dict: Dictionary mapping node names to community IDs.
        """
        partition_dict = {}
        for node_idx, community_id in enumerate(partition.membership):
            node_name = node_names[node_idx]
            partition_dict[node_name] = community_id
        return partition_dict
    
    def run(self, as_dfs: bool = False, B: Optional[pd.DataFrame] = None) -> Union[dict, list]:
        """
        Run the Leiden algorithm to detect communities.
        
        Args:
            as_dfs (bool): If True, returns a list of adjacency matrices (DataFrames)
                representing clusters with more than 2 nodes.
                If False, returns the partition dictionary. Default is False.
            B (Optional[pd.DataFrame]): Omics data DataFrame. Required if as_dfs=True.
                Used to create adjacency matrices for each cluster.
        
        Returns:
            Union[dict, list]: 
                - If as_dfs=False: Dictionary mapping node names to community IDs.
                - If as_dfs=True: List of adjacency matrices (DataFrames) for clusters with >2 nodes.
        """
        self.logger.info("Running Leiden algorithm...")
        
        # Convert NetworkX graph to igraph
        ig_graph = self._nx_to_igraph(self.G)
        
        # Get node names in order (should match igraph vertex order)
        node_names = [ig_graph.vs[idx]['name'] for idx in range(len(ig_graph.vs))]
        
        # Run Leiden algorithm
        try:
            partition = la.find_partition(
                ig_graph,
                self.partition_type,
                resolution_parameter=self.resolution_parameter,
                n_iterations=self.n_iterations,
                seed=self.seed,
            )
            
            # Convert partition to dictionary
            self.partition = self._partition_to_dict(partition, node_names)
            
            # Calculate quality (modularity)
            self.quality = partition.modularity
            
            self.logger.info(
                f"Leiden algorithm completed. Found {len(set(self.partition.values()))} communities. "
                f"Modularity: {self.quality:.4f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error running Leiden algorithm: {e}")
            raise
        
        if as_dfs:
            if B is None:
                raise ValueError("B (omics data) is required when as_dfs=True")
            return self.partition_to_adjacency(self.partition, B)
        else:
            return self.partition
    
    def partition_to_adjacency(self, partition: dict, B: pd.DataFrame) -> list:
        """
        Convert the partition dictionary into a list of adjacency matrices (DataFrames),
        where each adjacency matrix represents a cluster with more than 2 nodes.
        
        Args:
            partition (dict): Dictionary mapping node names to community IDs.
            B (pd.DataFrame): Omics data DataFrame.
        
        Returns:
            list: List of adjacency matrices (DataFrames) for clusters with more than 2 nodes.
        """
        # Group nodes by community
        clusters = {}
        for node, community_id in partition.items():
            clusters.setdefault(community_id, []).append(node)
        
        self.logger.info(f"Total detected clusters: {len(clusters)}")
        
        adjacency_matrices = []
        for community_id, nodes in clusters.items():
            if len(nodes) > 2:
                # Get valid nodes that exist in B columns
                valid_nodes = [node for node in nodes if str(node) in B.columns]
                if valid_nodes:
                    adjacency_matrix = B.loc[:, valid_nodes].fillna(0)
                    adjacency_matrices.append(adjacency_matrix)
                    self.logger.debug(
                        f"Cluster {community_id} size: {len(nodes)} "
                        f"(valid nodes in B: {len(valid_nodes)})"
                    )
        
        self.logger.info(f"Clusters with >2 nodes: {len(adjacency_matrices)}")
        return adjacency_matrices
    
    def get_quality(self) -> float:
        """
        Get the modularity quality score of the partition.
        
        Returns:
            float: Modularity score.
        """
        if self.quality is None:
            raise ValueError("No partition computed. Call run() first.")
        return self.quality
    
    def get_communities(self) -> dict:
        """
        Get communities as a dictionary mapping community IDs to lists of nodes.
        
        Returns:
            dict: Dictionary mapping community IDs to lists of node names.
        """
        if self.partition is None:
            raise ValueError("No partition computed. Call run() first.")
        
        communities = {}
        for node, community_id in self.partition.items():
            communities.setdefault(community_id, []).append(node)
        
        return communities

