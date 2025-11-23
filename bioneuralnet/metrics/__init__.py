"""Metrics and visualization tools for BioNeuralNet.

This module provides functions for calculating correlations between omics data and phenotypes, as well as plotting utilities for network analysis and performance visualization.
"""

from .correlation import (
    omics_correlation,
    cluster_correlation,
    louvain_to_adjacency,
)

from .plot import (
    plot_variance_distribution,
    plot_variance_by_feature,
    plot_performance_three,
    plot_performance,
    plot_multiple_metrics,
    plot_embeddings,
    plot_network,
    compare_clusters,
)

__all__ = [
    "omics_correlation",
    "cluster_correlation",
    "louvain_to_adjacency",
    "plot_variance_distribution",
    "plot_variance_by_feature",
    "plot_performance_three",
    "plot_performance",
    "plot_multiple_metrics",
    "plot_embeddings",
    "plot_network",
    "compare_clusters",
]
