import networkx as nx
import pandas as pd
import numpy as np
from typing import Union, Optional
import torch
import igraph as ig
import leidenalg
from sklearn.cluster import KMeans

from bioneuralnet.clustering.correlated_pagerank import CorrelatedPageRank
from bioneuralnet.clustering.correlated_louvain import CorrelatedLouvain

from ..utils.logger import get_logger

logger = get_logger(__name__)

class Leiden:
    """
    Leiden Class that clustering community detection.

    Attributes:

        G (nx.Graph): NetworkX graph object.
        embeddings (np.ndarray): Graph embeddings
        
    """
    def __init__(
        self,
        G: nx.Graph,
        embeddings: np.ndarray
        ):

        self.logger = get_logger(__name__)
        self.G = G
        self.embeddings = embeddings
        # Convert networkx to igraph
        ig_graph = ig.Graph.Adjacency((nx.to_numpy_array(G) > 0).tolist())
        # Leiden partition
        partition = leidenalg.find_partition(ig_graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1.0)
        self.labels_leiden = np.array(partition.membership)

        self.logger.info(
            f"Initialized Leiden with {len(self.G)} graph nodes."
        )

    def run(self) -> np.ndarray:
        # Hybrid approach: apply Leiden to get communities, then run KMeans on embeddings inside each community
        labels_hybrid = np.full_like(self.labels_leiden, fill_value=-1)
        next_label = 0
        for com in set(self.labels_leiden):
            idx = np.where(self.labels_leiden == com)[0]
            if len(idx) <= 2:
                labels_hybrid[idx] = next_label
                next_label += 1
                continue
            # choose local k (e.g., min(3, size//10) or 1 cluster)
            k_local = min(3, max(1, len(idx)//10))
            sub_emb = self.embeddings[idx]
            if k_local == 1:
                labels_hybrid[idx] = next_label
                next_label += 1
                continue
            km_local = KMeans(n_clusters=k_local, random_state=0).fit(sub_emb)
            for j in range(k_local):
                labels_hybrid[idx[km_local.labels_ == j]] = next_label
                next_label += 1
        
        return labels_hybrid
