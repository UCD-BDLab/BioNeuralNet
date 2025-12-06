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

# Import Leiden clustering from BioNeuralNet
try:
    from bioneuralnet.clustering import Leiden
    import networkx as nx
    LEIDEN_AVAILABLE = True
    LeidenClass = Leiden
except ImportError:
    LEIDEN_AVAILABLE = False
    LeidenClass = None  # type: ignore

# Import CorrelatedLouvain and HybridLouvain from BioNeuralNet
try:
    from bioneuralnet.clustering import CorrelatedLouvain, HybridLouvain
    CORRELATED_LOUVAIN_AVAILABLE = True
    CorrelatedLouvainClass = CorrelatedLouvain
    HybridLouvainClass = HybridLouvain
except ImportError:
    CORRELATED_LOUVAIN_AVAILABLE = False
    CorrelatedLouvainClass = None  # type: ignore
    HybridLouvainClass = None  # type: ignore

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
    if method.lower() == "umap":
        if not UMAP_AVAILABLE:
            method = "tsne"

        if UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            return reducer.fit_transform(embeddings)

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
        return reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Unknown reduction method: {method}. Use 'umap' or 'tsne'.")


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    adjacency_matrix: Optional[pd.DataFrame] = None,
    resolution_parameter: float = 1.0,
    random_state: int = 42,
    omics_data: Optional[pd.DataFrame] = None,
    phenotype_data: Optional[pd.Series] = None,
    k3: float = 0.2,
    k4: float = 0.8,
    max_iter: int = 3,
    gpu: bool = False,
) -> ClusterResults:
    """
    Cluster nodes based on their embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings (num_nodes × embedding_dim).
    method : str, default="kmeans"
        Clustering method: "kmeans", "leiden", "correlated_louvain", or "hybrid_louvain".
    n_clusters : int, optional
        Number of clusters (for KMeans). If None, uses elbow method.
        Ignored for graph-based methods.
    adjacency_matrix : pd.DataFrame, optional
        Adjacency matrix (required for Leiden, CorrelatedLouvain, HybridLouvain).
    resolution_parameter : float, default=1.0
        Resolution parameter for Leiden algorithm. Higher values lead to more communities.
        Ignored for other methods.
    random_state : int, default=42
        Random seed for reproducibility.
    omics_data : pd.DataFrame, optional
        Omics data (samples × genes). Required for CorrelatedLouvain and HybridLouvain.
        Columns should be gene IDs matching adjacency matrix nodes.
    phenotype_data : pd.Series, optional
        Phenotype data (condition labels). Required for CorrelatedLouvain and HybridLouvain.
        Index should match omics_data index (sample IDs).
    k3 : float, default=0.2
        Weight for modularity in CorrelatedLouvain/HybridLouvain.
    k4 : float, default=0.8
        Weight for correlation in CorrelatedLouvain/HybridLouvain.
    max_iter : int, default=3
        Maximum iterations for HybridLouvain.
    gpu : bool, default=False
        Use GPU for CorrelatedLouvain/HybridLouvain.

    Returns
    -------
    ClusterResults
        Clustering results with labels and statistics.
    """

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

    elif method.lower() == "leiden":
        if not LEIDEN_AVAILABLE:
            raise ImportError(
                "Leiden clustering requires 'leidenalg' and 'igraph' packages. "
                "Install with: pip install leidenalg igraph"
            )

        if adjacency_matrix is None:
            raise ValueError(
                "Leiden clustering requires an adjacency matrix. "
                "Please provide 'adjacency_matrix' parameter."
            )

        # Convert adjacency matrix to NetworkX graph
        G = nx.from_pandas_adjacency(adjacency_matrix)

        # Run Leiden algorithm
        if LeidenClass is None:
            raise ImportError("Leiden clustering is not available. Install leidenalg and igraph.")
        leiden = LeidenClass(
            G=G,
            resolution_parameter=resolution_parameter,
            n_iterations=-1,
            seed=random_state,
        )
        partition = leiden.run()

        # Convert partition dictionary to labels array
        # Ensure labels are in the same order as node_names/embeddings
        node_names = list(adjacency_matrix.index)
        labels = np.array([partition.get(node, -1) for node in node_names])

        # Ensure labels are non-negative and consecutive
        unique_labels = np.unique(labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = np.array([label_mapping[label] for label in labels])

        n_clusters = len(unique_labels)

        # Compute silhouette score on embeddings (if possible)
        if embeddings.shape[0] > n_clusters and n_clusters > 1:
            try:
                silhouette = silhouette_score(embeddings, labels)
            except Exception:
                silhouette = None
        else:
            silhouette = None

        logger.info(f"Leiden clustering complete: {n_clusters} communities detected.")

    elif method.lower() in ["correlated_louvain", "correlatedlouvain"]:
        if not CORRELATED_LOUVAIN_AVAILABLE:
            raise ImportError(
                "CorrelatedLouvain clustering requires BioNeuralNet clustering module. "
                "Please ensure bioneuralnet.clustering is available."
            )

        if adjacency_matrix is None:
            raise ValueError(
                "CorrelatedLouvain clustering requires an adjacency matrix. "
                "Please provide 'adjacency_matrix' parameter."
            )

        if omics_data is None:
            raise ValueError(
                "CorrelatedLouvain clustering requires omics data. "
                "Please provide 'omics_data' parameter (samples × genes)."
            )

        if phenotype_data is None:
            raise ValueError(
                "CorrelatedLouvain clustering requires phenotype data. "
                "Please provide 'phenotype_data' parameter."
            )

        # Convert adjacency matrix to NetworkX graph
        G = nx.from_pandas_adjacency(adjacency_matrix)

        # Ensure omics_data columns match graph nodes
        graph_nodes = set(G.nodes())
        omics_cols = set(omics_data.columns.astype(str))
        common_genes = list(graph_nodes & omics_cols)

        if len(common_genes) == 0:
            raise ValueError(
                "No common genes between adjacency matrix and omics data. "
                "Check that gene identifiers match."
            )

        # Filter omics data to common genes
        B_filtered = omics_data[common_genes].copy()

        # Align phenotype data with omics data
        common_samples = list(set(B_filtered.index) & set(phenotype_data.index))
        if len(common_samples) == 0:
            raise ValueError(
                "No common samples between omics data and phenotype data. "
                "Check that sample identifiers match."
            )

        B_filtered = B_filtered.loc[common_samples]
        Y_filtered = phenotype_data.loc[common_samples]

        # Filter graph to common genes
        G_filtered = G.subgraph(common_genes).copy()

        logger.info(
            f"Running CorrelatedLouvain: {G_filtered.number_of_nodes()} nodes, "
            f"{B_filtered.shape[0]} samples, k3={k3}, k4={k4}"
        )

        # Run CorrelatedLouvain
        if CorrelatedLouvainClass is None:
            raise ImportError("CorrelatedLouvain clustering is not available.")
        correlated_louvain = CorrelatedLouvainClass(
            G=G_filtered,
            B=B_filtered,
            Y=Y_filtered,
            k3=k3,
            k4=k4,
            weight="weight",
            tune=False,
            gpu=gpu,
            seed=random_state,
        )
        partition_result = correlated_louvain.run()
        # Handle union type: run() returns Union[dict, list], but we expect dict
        if isinstance(partition_result, dict):
            partition = partition_result
        else:
            # If it's a list, convert to dict (assuming list of cluster assignments)
            raise ValueError("CorrelatedLouvain.run() returned a list, expected dict")

        # Convert partition dictionary to labels array
        node_names = list(adjacency_matrix.index)
        labels = np.array([partition.get(node, -1) for node in node_names])

        # Ensure labels are non-negative and consecutive
        unique_labels = np.unique(labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = np.array([label_mapping[label] for label in labels])

        n_clusters = len(unique_labels)

        # Compute silhouette score on embeddings (if possible)
        if embeddings.shape[0] > n_clusters and n_clusters > 1:
            try:
                silhouette = silhouette_score(embeddings, labels)
            except Exception:
                silhouette = None
        else:
            silhouette = None

        logger.info(f"CorrelatedLouvain clustering complete: {n_clusters} communities detected.")

    elif method.lower() in ["hybrid_louvain", "hybridlouvain"]:
        if not CORRELATED_LOUVAIN_AVAILABLE:
            raise ImportError(
                "HybridLouvain clustering requires BioNeuralNet clustering module. "
                "Please ensure bioneuralnet.clustering is available."
            )

        if adjacency_matrix is None:
            raise ValueError(
                "HybridLouvain clustering requires an adjacency matrix. "
                "Please provide 'adjacency_matrix' parameter."
            )

        if omics_data is None:
            raise ValueError(
                "HybridLouvain clustering requires omics data. "
                "Please provide 'omics_data' parameter (samples × genes)."
            )

        if phenotype_data is None:
            raise ValueError(
                "HybridLouvain clustering requires phenotype data. "
                "Please provide 'phenotype_data' parameter."
            )

        # Convert adjacency matrix to NetworkX graph
        G = nx.from_pandas_adjacency(adjacency_matrix)

        # Ensure omics_data columns match graph nodes
        graph_nodes = set(G.nodes())
        omics_cols = set(omics_data.columns.astype(str))
        common_genes = list(graph_nodes & omics_cols)

        if len(common_genes) == 0:
            raise ValueError(
                "No common genes between adjacency matrix and omics data. "
                "Check that gene identifiers match."
            )

        # Filter omics data to common genes
        B_filtered = omics_data[common_genes].copy()

        # Align phenotype data with omics data
        common_samples = list(set(B_filtered.index) & set(phenotype_data.index))
        if len(common_samples) == 0:
            raise ValueError(
                "No common samples between omics data and phenotype data. "
                "Check that sample identifiers match."
            )

        B_filtered = B_filtered.loc[common_samples]
        Y_filtered = phenotype_data.loc[common_samples]

        # Filter graph to common genes
        G_filtered = G.subgraph(common_genes).copy()

        logger.info(
            f"Running HybridLouvain: {G_filtered.number_of_nodes()} nodes, "
            f"{B_filtered.shape[0]} samples, k3={k3}, k4={k4}, max_iter={max_iter}"
        )

        # Run HybridLouvain
        if HybridLouvainClass is None:
            raise ImportError("HybridLouvain clustering is not available.")
        hybrid_louvain = HybridLouvainClass(
            G=G_filtered,
            B=B_filtered,
            Y=Y_filtered,
            k3=k3,
            k4=k4,
            max_iter=max_iter,
            weight="weight",
            gpu=gpu,
            seed=random_state,
            tune=False,
        )
        result_raw = hybrid_louvain.run()

        # HybridLouvain returns a dict with 'curr' (current partition) and 'clus' (all clusters)
        # Handle union type: run() returns Union[dict, list]
        if isinstance(result_raw, dict):
            result = result_raw
            partition = result.get("curr", {})
        else:
            raise ValueError("HybridLouvain.run() returned a list, expected dict")

        # Convert partition dictionary to labels array
        node_names = list(adjacency_matrix.index)
        labels = np.array([partition.get(node, -1) for node in node_names])

        # Ensure labels are non-negative and consecutive
        unique_labels = np.unique(labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = np.array([label_mapping[label] for label in labels])

        n_clusters = len(unique_labels)

        # Compute silhouette score on embeddings (if possible)
        if embeddings.shape[0] > n_clusters and n_clusters > 1:
            try:
                silhouette = silhouette_score(embeddings, labels)
            except Exception:
                silhouette = None
        else:
            silhouette = None

        logger.info(f"HybridLouvain clustering complete: {n_clusters} communities detected.")

    else:
        raise ValueError(
            f"Unknown clustering method: {method}. "
            "Use 'kmeans', 'leiden', 'correlated_louvain', or 'hybrid_louvain'."
        )

    # Compute cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    results = ClusterResults(
        labels=labels,
        n_clusters=n_clusters,
        silhouette_score=silhouette,
        cluster_sizes=cluster_sizes,
    )

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
    # Use tab20 colormap if available, otherwise use default
    try:
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_clusters))
    except (AttributeError, ValueError):
        # Fallback to default colormap
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, n_clusters))

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

    return cluster_df, summary_df


def embedding_analysis_pipeline(
    embeddings: np.ndarray,
    node_names: list,
    adjacency_matrix: Optional[pd.DataFrame] = None,
    gene_metadata: Optional[pd.DataFrame] = None,
    reduction_method: str = "umap",
    clustering_method: str = "kmeans",
    n_clusters: Optional[int] = None,
    resolution_parameter: float = 1.0,
    random_state: int = 42,
    omics_data: Optional[pd.DataFrame] = None,
    phenotype_data: Optional[pd.Series] = None,
    k3: float = 0.2,
    k4: float = 0.8,
    max_iter: int = 3,
    gpu: bool = False,
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
        Graph adjacency matrix (required for Leiden, CorrelatedLouvain, HybridLouvain).
    gene_metadata : pd.DataFrame, optional
        Gene metadata for cluster analysis.
    reduction_method : str, default="umap"
        Embedding reduction method: "umap" or "tsne".
    clustering_method : str, default="kmeans"
        Clustering method: "kmeans", "leiden", "correlated_louvain", or "hybrid_louvain".
    n_clusters : int, optional
        Number of clusters (for KMeans).
    resolution_parameter : float, default=1.0
        Resolution parameter for Leiden algorithm. Higher values lead to more communities.
        Ignored for other methods.
    random_state : int, default=42
        Random seed.
    omics_data : pd.DataFrame, optional
        Omics data (samples × genes). Required for CorrelatedLouvain and HybridLouvain.
    phenotype_data : pd.Series, optional
        Phenotype data (condition labels). Required for CorrelatedLouvain and HybridLouvain.
    k3 : float, default=0.2
        Weight for modularity in CorrelatedLouvain/HybridLouvain.
    k4 : float, default=0.8
        Weight for correlation in CorrelatedLouvain/HybridLouvain.
    max_iter : int, default=3
        Maximum iterations for HybridLouvain.
    gpu : bool, default=False
        Use GPU for CorrelatedLouvain/HybridLouvain.

    Returns
    -------
    Tuple
        (reduced_embeddings, cluster_results, cluster_df, summary_df)
    """

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
        resolution_parameter=resolution_parameter,
        random_state=random_state,
        omics_data=omics_data,
        phenotype_data=phenotype_data,
        k3=k3,
        k4=k4,
        max_iter=max_iter,
        gpu=gpu,
    )

    # Step 3: Analyze
    cluster_df, summary_df = analyze_clusters(
        cluster_results, node_names, gene_metadata
    )

    # Step 4: Visualize
    visualize_clusters(reduced_emb, cluster_results.labels, node_names)


    return reduced_emb, cluster_results, cluster_df, summary_df
