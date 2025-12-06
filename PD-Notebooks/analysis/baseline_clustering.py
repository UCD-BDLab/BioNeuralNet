"""
Baseline clustering methods for gene clustering using raw features.

This module provides baseline clustering approaches that use raw expression
data (not GNN embeddings) for comparison with graph-based clustering methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.utils import get_logger

logger = get_logger(__name__)

# Import ClusterResults from embedding_analysis to avoid circular import
# We'll import it inside functions that need it
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from analysis.embedding_analysis import ClusterResults


def baseline_kmeans_clustering(
    expression_df: pd.DataFrame,
    node_names: Optional[list] = None,
    n_clusters: Optional[int] = None,
    use_pca: bool = True,
    n_components: Optional[int] = None,
    random_state: int = 42,
):
    """
    Perform K-means clustering on raw expression features (baseline method).

    This is a baseline that clusters genes based on their raw expression patterns
    across samples, without using graph structure or GNN embeddings.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Gene expression matrix (genes × samples).
        Rows are genes, columns are samples.
    node_names : list, optional
        List of gene IDs/names matching expression_df index.
        If None, uses expression_df.index.
    n_clusters : int, optional
        Number of clusters. If None, uses elbow method to auto-detect.
    use_pca : bool, default=True
        Whether to apply PCA dimensionality reduction before clustering.
        Helps with high-dimensional data and reduces noise.
    n_components : int, optional
        Number of PCA components to use. If None, uses enough components
        to explain 95% of variance (or max 50 components).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    ClusterResults
        Clustering results with labels and statistics.
    """
    from analysis.embedding_analysis import ClusterResults

    if node_names is None:
        node_names = list(expression_df.index)

    # Prepare features: transpose to samples × genes, then transpose back to genes × features
    # Each gene is represented by its expression vector across samples
    X = expression_df.values  # genes × samples

    # Standardize features (across samples for each gene)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA if requested
    if use_pca:
        if n_components is None:
            # Use enough components to explain 95% variance, but cap at 50
            pca = PCA(n_components=min(50, X_scaled.shape[1] - 1))
            X_pca = pca.fit_transform(X_scaled)
            # Find number of components explaining 95% variance
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
            if n_components_95 < X_pca.shape[1]:
                pca = PCA(n_components=n_components_95)
                X_pca = pca.fit_transform(X_scaled)

        else:
            n_components = min(n_components, X_scaled.shape[1] - 1)
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

        features = X_pca
    else:
        features = X_scaled

    if n_clusters is None:
        max_k = min(20, len(node_names) // 10)
        if max_k < 2:
            n_clusters = 2
        else:
            inertias = []
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)

            # Simple elbow detection: find point with largest decrease
            if len(inertias) > 1:
                decreases = [
                    inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)
                ]
                n_clusters = decreases.index(max(decreases)) + 2
            else:
                n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)

    if len(node_names) > n_clusters and n_clusters > 1:
        try:
            silhouette = silhouette_score(features, labels)
        except Exception:
            silhouette = None
    else:
        silhouette = None

    # Compute cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    return ClusterResults(
        labels=labels,
        n_clusters=n_clusters,
        silhouette_score=silhouette,
        cluster_sizes=cluster_sizes,
    )


def baseline_clustering_pipeline(
    expression_df: pd.DataFrame,
    node_names: Optional[list] = None,
    gene_metadata: Optional[pd.DataFrame] = None,
    n_clusters: Optional[int] = None,
    use_pca: bool = True,
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[ClusterResults, pd.DataFrame, pd.DataFrame]:
    """
    Complete baseline clustering pipeline: K-means on raw features.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Gene expression matrix (genes × samples).
    node_names : list, optional
        List of gene IDs/names matching expression_df index.
    gene_metadata : pd.DataFrame, optional
        Gene metadata for cluster analysis.
    n_clusters : int, optional
        Number of clusters. If None, uses elbow method.
    use_pca : bool, default=True
        Whether to apply PCA before clustering.
    n_components : int, optional
        Number of PCA components (if use_pca=True).
    random_state : int, default=42
        Random seed.

    Returns
    -------
    Tuple
        (cluster_results, cluster_df, summary_df)
    """
    from analysis.embedding_analysis import analyze_clusters

    cluster_results = baseline_kmeans_clustering(
        expression_df=expression_df,
        node_names=node_names,
        n_clusters=n_clusters,
        use_pca=use_pca,
        n_components=n_components,
        random_state=random_state,
    )

    if node_names is None:
        node_names = list(expression_df.index)

    cluster_df, summary_df = analyze_clusters(
        cluster_results, node_names, gene_metadata
    )

    return cluster_results, cluster_df, summary_df
