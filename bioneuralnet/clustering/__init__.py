r"""Network Clustering and Subgraph Detection.

This module implements hybrid algorithms for identifying phenotype-associated 
subgraphs in multi-omics networks. It combines global modularity optimization 
with local random-walk refinement, weighted by phenotypic correlation.

Classes:
    HybridLouvain: The primary pipeline. Iteratively alternates between global partitioning 
        (Louvain) and local refinement (PageRank) to find the most significant 
        subgraph associated with a phenotype.
    CorrelatedLouvain: Extends standard Louvain by optimizing a hybrid objective:
        Q_hybrid = k_L * Modularity + (1 - k_L) * Correlation.
    CorrelatedPageRank: Performs a biased random walk (PageRank) followed by a sweep cut to 
        minimize a hybrid conductance objective:
        Phi_hybrid = k_P * Conductance + (1 - k_P) * Correlation.
    Louvain: Standard Louvain community detection (based on modularity maximization).
        Serves as the base class and baseline method.
"""

from .correlated_pagerank import CorrelatedPageRank
from .correlated_louvain import CorrelatedLouvain
from .hybrid_louvain import HybridLouvain
from .louvain import Louvain

__all__ = [
    "CorrelatedPageRank",
    "CorrelatedLouvain",
    "HybridLouvain",
    "Louvain"
]