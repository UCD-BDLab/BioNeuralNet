"""
Graph mining and embedding analysis for PD gene-gene networks.

This module provides:
- Embedding reduction (UMAP/t-SNE)
- Clustering (KMeans, Leiden)
- Cluster visualization
- Cluster statistics
"""

from .embedding_analysis import (
    reduce_embeddings,
    cluster_embeddings,
    visualize_clusters,
    analyze_clusters,
    embedding_analysis_pipeline,
)

__all__ = [
    "reduce_embeddings",
    "cluster_embeddings",
    "visualize_clusters",
    "analyze_clusters",
    "embedding_analysis_pipeline",
]
