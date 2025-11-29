"""
Embedding analysis and graph mining for PD gene-gene networks.

This module provides functions for:
- Dimensionality reduction (UMAP, t-SNE)
- Clustering (KMeans, Leiden)
- Cluster visualization and statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Note: Leiden clustering is not directly available in BioNeuralNet
# We can use KMeans for embedding-based clustering, or use BioNeuralNet's
# graph-based clustering methods (CorrelatedLouvain, HybridLouvain) separately
LEIDEN_AVAILABLE = False
from bioneuralnet.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterResults:
    """
    Container for clustering results.

    Attributes
    ----------
    labels : np.ndarray
        Cluster labels for each node.
    n_clusters : int
        Number of clusters found.
    silhouette_score : float
        Silhouette score (if applicable).
    cluster_sizes : dict
        Dictionary mapping cluster ID to number of nodes.
    """

    labels: np.ndarray
    n_clusters: int
    silhouette_score: Optional[float]
    cluster_sizes: dict


def reduce_embeddings(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2D for visualization.

    Parameters
    ----------
    embeddings : np.ndarray
        High-dimensional embeddings (num_nodes × embedding_dim).
    method : str, default="umap"
        Reduction method: "umap" or "tsne".
    n_components : int, default=2
        Number of dimensions for reduction (typically 2 for visualization).
    n_neighbors : int, default=15
        Number of neighbors for UMAP (ignored for t-SNE).
    min_dist : float, default=0.1
        Minimum distance for UMAP (ignored for t-SNE).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Reduced embeddings (num_nodes × n_components).
    """
    logger.info(
        f"Reducing embeddings from {embeddings.shape[1]}D to {n_components}D "
        f"using method='{method}'."
    )

    if method.lower() == "umap":
        if not UMAP_AVAILABLE:
            logger.warning(
                "UMAP not available. Install with: pip install umap-learn. "
                "Falling back to t-SNE."
            )
            method = "tsne"

        if UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            reduced = reducer.fit_transform(embeddings)
            logger.info("UMAP reduction complete.")
            return reduced

    # t-SNE fallback or explicit choice
    if method.lower() == "tsne":
        perplexity = min(30, embeddings.shape[0] - 1)
        if perplexity < 1:
            raise ValueError(
                f"Not enough samples ({embeddings.shape[0]}) for t-SNE. "
                "Need at least 2 samples."
            )

        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
        )
        reduced = reducer.fit_transform(embeddings)
        logger.info("t-SNE reduction complete.")
        return reduced

    else:
        raise ValueError(f"Unknown reduction method: {method}. Use 'umap' or 'tsne'.")


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    adjacency_matrix: Optional[pd.DataFrame] = None,
    random_state: int = 42,
) -> ClusterResults:
    """
    Cluster nodes based on their embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings (num_nodes × embedding_dim).
    method : str, default="kmeans"
        Clustering method: "kmeans" or "leiden".
    n_clusters : int, optional
        Number of clusters (for KMeans). If None, uses elbow method.
        Ignored for Leiden.
    adjacency_matrix : pd.DataFrame, optional
        Adjacency matrix (required for Leiden clustering).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    ClusterResults
        Clustering results with labels and statistics.
    """
    logger.info(f"Clustering {embeddings.shape[0]} nodes using method='{method}'.")

    if method.lower() == "kmeans":
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            # Use elbow method (simple heuristic)
            max_k = min(10, embeddings.shape[0] // 10)
            if max_k < 2:
                n_clusters = 2
            else:
                inertias = []
                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                    kmeans.fit(embeddings)
                    inertias.append(kmeans.inertia_)

                # Simple elbow detection: find point with largest decrease
                if len(inertias) > 1:
                    decreases = [
                        inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)
                    ]
                    n_clusters = decreases.index(max(decreases)) + 2
                else:
                    n_clusters = 2

            logger.info(f"Auto-selected n_clusters={n_clusters} using elbow method.")

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Compute silhouette score
        if embeddings.shape[0] > n_clusters:
            silhouette = silhouette_score(embeddings, labels)
        else:
            silhouette = None

        logger.info(f"KMeans clustering complete: {n_clusters} clusters.")

    elif method.lower() == "leiden":
        raise NotImplementedError(
            "Leiden clustering not directly available. "
            "Use 'kmeans' for embedding-based clustering, or use BioNeuralNet's "
            "graph-based clustering methods (CorrelatedLouvain, HybridLouvain) "
            "on the adjacency matrix separately."
        )

    else:
        raise ValueError(f"Unknown clustering method: {method}. Use 'kmeans' or 'leiden'.")

    # Compute cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    results = ClusterResults(
        labels=labels,
        n_clusters=n_clusters,
        silhouette_score=silhouette,
        cluster_sizes=cluster_sizes,
    )

    logger.info(f"Cluster sizes: {cluster_sizes}")

    return results


def visualize_clusters(
    reduced_embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    node_names: Optional[list] = None,
    title: str = "Gene Clusters in Embedding Space",
    figsize: Tuple[int, int] = (12, 8),
    show_legend: bool = True,
) -> plt.Figure:
    """
    Visualize clusters in 2D embedding space.

    Parameters
    ----------
    reduced_embeddings : np.ndarray
        2D reduced embeddings (num_nodes × 2).
    cluster_labels : np.ndarray
        Cluster labels for each node.
    node_names : list, optional
        Gene names for labeling (if provided, labels top nodes by degree).
    title : str, default="Gene Clusters in Embedding Space"
        Plot title.
    figsize : tuple, default=(12, 8)
        Figure size.
    show_legend : bool, default=True
        Whether to show cluster legend.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique clusters and assign colors
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            c=[colors[i]],
            label=f"Cluster {int(cluster_id)}",
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    logger.info("Cluster visualization generated.")
    return fig


def analyze_clusters(
    cluster_results: ClusterResults,
    node_names: list,
    gene_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Analyze cluster composition and generate statistics.

    Parameters
    ----------
    cluster_results : ClusterResults
        Clustering results.
    node_names : list
        List of gene IDs/names.
    gene_metadata : pd.DataFrame, optional
        Gene metadata (e.g., gene symbols, descriptions).

    Returns
    -------
    pd.DataFrame
        Cluster analysis summary with gene counts and metadata.
    """
    logger.info("Analyzing cluster composition...")

    # Create cluster assignment DataFrame
    cluster_df = pd.DataFrame(
        {"gene_id": node_names, "cluster": cluster_results.labels}
    )

    # Add gene metadata if available
    if gene_metadata is not None:
        # Align with node names
        common_genes = set(node_names) & set(gene_metadata.index)
        if len(common_genes) > 0:
            metadata_subset = gene_metadata.loc[list(common_genes)]
            cluster_df = cluster_df.merge(
                metadata_subset.reset_index(),
                left_on="gene_id",
                right_on=gene_metadata.index.name or "index",
                how="left",
            )

    # Cluster statistics
    cluster_stats = []
    for cluster_id, size in cluster_results.cluster_sizes.items():
        cluster_genes = cluster_df[cluster_df["cluster"] == cluster_id]["gene_id"].tolist()
        stats = {
            "cluster_id": cluster_id,
            "size": size,
            "genes": cluster_genes[:10],  # First 10 genes as example
        }
        cluster_stats.append(stats)

    summary_df = pd.DataFrame(cluster_stats)
    logger.info(f"Cluster analysis complete. Summary:\n{summary_df}")

    return cluster_df, summary_df


def embedding_analysis_pipeline(
    embeddings: np.ndarray,
    node_names: list,
    adjacency_matrix: Optional[pd.DataFrame] = None,
    gene_metadata: Optional[pd.DataFrame] = None,
    reduction_method: str = "umap",
    clustering_method: str = "kmeans",
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, ClusterResults, pd.DataFrame, pd.DataFrame]:
    """
    Complete embedding analysis pipeline: reduction → clustering → visualization → statistics.

    Parameters
    ----------
    embeddings : np.ndarray
        High-dimensional node embeddings.
    node_names : list
        List of gene IDs/names.
    adjacency_matrix : pd.DataFrame, optional
        Graph adjacency matrix (required for Leiden clustering).
    gene_metadata : pd.DataFrame, optional
        Gene metadata for cluster analysis.
    reduction_method : str, default="umap"
        Embedding reduction method: "umap" or "tsne".
    clustering_method : str, default="kmeans"
        Clustering method: "kmeans" or "leiden".
    n_clusters : int, optional
        Number of clusters (for KMeans).
    random_state : int, default=42
        Random seed.

    Returns
    -------
    Tuple
        (reduced_embeddings, cluster_results, cluster_df, summary_df)
    """
    logger.info("=" * 60)
    logger.info("Starting embedding analysis pipeline.")
    logger.info("=" * 60)

    # Step 1: Reduce embeddings
    reduced_emb = reduce_embeddings(
        embeddings, method=reduction_method, random_state=random_state
    )

    # Step 2: Cluster
    cluster_results = cluster_embeddings(
        embeddings,
        method=clustering_method,
        n_clusters=n_clusters,
        adjacency_matrix=adjacency_matrix,
        random_state=random_state,
    )

    # Step 3: Analyze
    cluster_df, summary_df = analyze_clusters(
        cluster_results, node_names, gene_metadata
    )

    # Step 4: Visualize
    visualize_clusters(reduced_emb, cluster_results.labels, node_names)

    logger.info("=" * 60)
    logger.info("Embedding analysis pipeline complete!")
    logger.info("=" * 60)

    return reduced_emb, cluster_results, cluster_df, summary_df
