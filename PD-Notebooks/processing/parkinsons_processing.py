"""
Preprocessing utilities for Parkinson's disease transcriptomics data.

This module provides PD-specific preprocessing functions that work with
the BioNeuralNet framework, including transformations, feature selection,
and node feature construction for GNN models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import BioNeuralNet utilities
import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.utils.data import normalize_omics
from bioneuralnet.utils import get_logger, select_top_k_variance

logger = get_logger(__name__)


def log_transform_counts(
    expression_df: pd.DataFrame,
    method: str = "log2",
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """
    Apply log transformation to count data.

    Common for RNA-seq count data to reduce the impact of highly expressed genes
    and make the data more suitable for downstream analysis.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Gene expression matrix (genes × samples) with raw counts.
    method : str, default="log2"
        Transformation method. Options: "log2", "log10", "ln" (natural log).
    pseudocount : float, default=1.0
        Value to add before log transformation to avoid log(0).

    Returns
    -------
    pd.DataFrame
        Log-transformed expression matrix with same index/columns as input.
    """
    logger.info(
        f"Applying {method} transformation with pseudocount={pseudocount} "
        f"to expression matrix of shape {expression_df.shape}."
    )

    if method == "log2":
        transformed = np.log2(expression_df + pseudocount)
    elif method == "log10":
        transformed = np.log10(expression_df + pseudocount)
    elif method == "ln":
        transformed = np.log(expression_df + pseudocount)
    else:
        raise ValueError(f"Unknown log method: {method}. Use 'log2', 'log10', or 'ln'.")

    result = pd.DataFrame(
        transformed, index=expression_df.index, columns=expression_df.columns
    )
    logger.info(f"Log transformation complete. Shape: {result.shape}")
    return result


def select_hvg(
    expression_df: pd.DataFrame,
    n_top: int = 5000,
    method: str = "variance",
    min_mean: float = 0.0,
    max_mean: float = np.inf,
) -> pd.DataFrame:
    """
    Select highly variable genes (HVGs) from expression data.

    HVGs are genes that show high variability across samples, which are often
    more informative for downstream analysis and reduce computational burden.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Gene expression matrix (genes × samples).
    n_top : int, default=5000
        Number of top highly variable genes to select.
    method : str, default="variance"
        Selection method. Options:
        - "variance": Select by variance (uses BioNeuralNet's select_top_k_variance)
        - "cv": Select by coefficient of variation (std/mean)
    min_mean : float, default=0.0
        Minimum mean expression threshold (filters lowly expressed genes).
    max_mean : float, default=np.inf
        Maximum mean expression threshold (filters highly expressed outliers).

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix containing only the selected HVGs.
    """
    logger.info(
        f"Selecting {n_top} highly variable genes from {expression_df.shape[0]} genes "
        f"using method='{method}'."
    )

    # Filter by mean expression thresholds
    mean_expr = expression_df.mean(axis=1)
    valid_genes = (mean_expr >= min_mean) & (mean_expr <= max_mean)
    filtered_df = expression_df.loc[valid_genes]

    logger.info(
        f"After mean expression filtering ({min_mean} <= mean <= {max_mean}): "
        f"{filtered_df.shape[0]} genes remaining."
    )

    if method == "variance":
        # Use BioNeuralNet utility
        # Note: select_top_k_variance expects samples × features, so we transpose
        # (genes × samples) -> (samples × genes)
        filtered_df_T = filtered_df.T  # samples × genes
        hvg_df_T = select_top_k_variance(filtered_df_T, k=n_top)
        hvg_df = hvg_df_T.T  # genes × samples
    elif method == "cv":
        # Coefficient of variation: std / mean
        std_expr = filtered_df.std(axis=1)
        cv = std_expr / (mean_expr.loc[filtered_df.index] + 1e-8)  # avoid div by zero
        top_genes = cv.nlargest(min(n_top, len(cv))).index
        hvg_df = filtered_df.loc[top_genes]
        logger.info(f"Selected {len(top_genes)} genes by coefficient of variation.")
    else:
        raise ValueError(f"Unknown HVG selection method: {method}")

    logger.info(f"Final HVG selection: {hvg_df.shape[0]} genes selected.")
    return hvg_df


def preprocess_for_gnn(
    expression_df: pd.DataFrame,
    log_transform: bool = True,
    log_method: str = "log2",
    select_hvgs: bool = True,
    n_hvgs: int = 5000,
    normalize: bool = True,
    normalize_method: str = "standard",
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for GNN input.

    Applies log transformation, HVG selection, and normalization in sequence.
    This prepares the expression data for graph construction and GNN training.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Raw gene expression matrix (genes × samples).
    log_transform : bool, default=True
        Whether to apply log transformation.
    log_method : str, default="log2"
        Log transformation method (see log_transform_counts).
    select_hvgs : bool, default=True
        Whether to select highly variable genes.
    n_hvgs : int, default=5000
        Number of HVGs to select (if select_hvgs=True).
    normalize : bool, default=True
        Whether to normalize the data.
    normalize_method : str, default="standard"
        Normalization method. Options: "standard" (Z-score), "minmax", "log2".
        Uses BioNeuralNet's normalize_omics function.

    Returns
    -------
    pd.DataFrame
        Preprocessed expression matrix ready for graph construction.
    """
    logger.info("Starting GNN preprocessing pipeline.")
    processed = expression_df.copy()

    # Step 1: Log transformation
    if log_transform:
        processed = log_transform_counts(processed, method=log_method)
        logger.info("Step 1/3: Log transformation complete.")

    # Step 2: HVG selection
    if select_hvgs:
        processed = select_hvg(processed, n_top=n_hvgs, method="variance")
        logger.info("Step 2/3: HVG selection complete.")

    # Step 3: Normalization
    if normalize:
        # Note: normalize_omics expects samples × features, so we transpose
        # then transpose back
        processed_T = processed.T  # samples × genes
        processed_T_norm = normalize_omics(processed_T, method=normalize_method)
        processed = processed_T_norm.T  # genes × samples
        logger.info("Step 3/3: Normalization complete.")

    logger.info(
        f"Preprocessing complete. Final shape: {processed.shape} "
        f"(genes × samples)."
    )
    return processed


def build_node_features(
    expression_df: pd.DataFrame,
    feature_type: str = "mean_variance",
    n_pca_components: Optional[int] = None,
    standardize_features: bool = True,
) -> pd.DataFrame:
    """
    Build node features for GNN models from gene expression data.

    Each gene (node) gets a feature vector that can be used as input to a GNN.
    Options include mean/variance statistics or PCA-reduced expression vectors.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Preprocessed expression matrix (genes × samples).
    feature_type : str, default="mean_variance"
        Type of node features to construct. Options:
        - "mean_variance": Mean and variance of expression across samples
        - "pca": PCA-reduced expression vectors (requires n_pca_components)
        - "full": Use full expression vector (one feature per sample)
        - "combined": Mean, variance, and PCA components
    n_pca_components : int, optional
        Number of PCA components to use (if feature_type includes "pca").
        If None and PCA is requested, uses min(n_samples, 50).
    standardize_features : bool, default=True
        Whether to standardize the final feature matrix (Z-score per feature).

    Returns
    -------
    pd.DataFrame
        Node feature matrix (genes × features) ready for GNN input.
        Index matches expression_df.index (gene IDs).
    """
    logger.info(
        f"Building node features from expression matrix of shape {expression_df.shape} "
        f"using feature_type='{feature_type}'."
    )

    features_list = []

    # Mean and variance features
    if feature_type in ["mean_variance", "combined"]:
        mean_expr = expression_df.mean(axis=1)
        var_expr = expression_df.var(axis=1)
        features_list.append(pd.DataFrame({"mean": mean_expr, "variance": var_expr}))
        logger.info("Added mean and variance features.")

    # PCA features
    if feature_type in ["pca", "combined"]:
        if n_pca_components is None:
            n_pca_components = min(expression_df.shape[1], 50)
        n_components: int = min(n_pca_components, expression_df.shape[1])

        # PCA: reduce dimensionality of each gene's expression profile
        # Input: genes × samples
        # We want to reduce the sample dimension, so we do PCA on samples × genes
        # This finds principal components in the sample space
        # Then we project each gene's expression vector onto these components
        X_T = expression_df.T.values  # samples × genes
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_T)  # Fit on samples × genes

        # Transform each gene's expression profile (across samples) to PCA space
        # For each gene (column in X_T), get its projection
        X_pca = pca.transform(X_T)  # samples × n_components
        # Now transpose to get genes × n_components
        # Each row is a gene, each column is a PC
        X_pca_genes = X_pca.T  # n_components × samples
        # Actually, we need to think about this differently
        # X_T is samples × genes, so after transform we get samples × n_components
        # We want genes × n_components, so we need to transpose the result
        # But the transform gives us the projection of samples onto PCs
        # We want the projection of genes onto PCs
        # So we need to use the components differently
        # Let's use the inverse: project genes onto the PC space
        # Actually simpler: use the components to transform the gene vectors
        components = pca.components_  # n_components × genes
        X_genes = expression_df.values  # genes × samples
        # Project: genes × samples @ (samples × n_components via components.T)
        # components is n_components × genes, so components.T is genes × n_components
        X_pca_genes = X_genes @ pca.components_.T  # genes × n_components

        pca_df = pd.DataFrame(
            X_pca_genes,
            index=expression_df.index,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )
        features_list.append(pca_df)
        logger.info(
            f"Added {n_components} PCA components "
            f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})."
        )


    # Full expression vector
    if feature_type == "full":
        features_list.append(expression_df.copy())
        logger.info("Using full expression vector as features.")

    # Combine all features
    if len(features_list) == 0:
        raise ValueError(f"Invalid feature_type: {feature_type}")

    node_features = pd.concat(features_list, axis=1)

    # Standardize if requested
    if standardize_features:
        scaler = StandardScaler()
        node_features_scaled = scaler.fit_transform(node_features.values)
        node_features = pd.DataFrame(
            node_features_scaled,
            index=node_features.index,
            columns=node_features.columns,
        )
        logger.info("Standardized node features (Z-score per feature).")

    logger.info(
        f"Node features constructed. Shape: {node_features.shape} "
        f"(genes × features)."
    )
    return node_features


def preprocess_pipeline(
    expression_df: pd.DataFrame,
    sample_metadata: Optional[pd.DataFrame] = None,
    log_transform: bool = True,
    log_method: str = "log2",
    select_hvgs: bool = True,
    n_hvgs: int = 5000,
    normalize: bool = True,
    normalize_method: str = "standard",
    build_features: bool = True,
    feature_type: str = "mean_variance",
    n_pca_components: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline: expression preprocessing + node features.

    This is a convenience function that combines preprocess_for_gnn and
    build_node_features into a single pipeline.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Raw gene expression matrix (genes × samples).
    sample_metadata : pd.DataFrame, optional
        Sample metadata (not used in processing, but kept for reference).
    log_transform : bool, default=True
        Whether to apply log transformation.
    log_method : str, default="log2"
        Log transformation method.
    select_hvgs : bool, default=True
        Whether to select highly variable genes.
    n_hvgs : int, default=5000
        Number of HVGs to select.
    normalize : bool, default=True
        Whether to normalize the data.
    normalize_method : str, default="standard"
        Normalization method.
    build_features : bool, default=True
        Whether to build node features.
    feature_type : str, default="mean_variance"
        Type of node features (see build_node_features).
    n_pca_components : int, optional
        Number of PCA components (if using PCA features).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (processed_expression, node_features)
        - processed_expression: Preprocessed expression matrix
        - node_features: Node feature matrix for GNN input
    """
    logger.info("=" * 60)
    logger.info("Starting complete preprocessing pipeline for PD GNN analysis.")
    logger.info("=" * 60)

    # Step 1: Preprocess expression
    processed_expr = preprocess_for_gnn(
        expression_df,
        log_transform=log_transform,
        log_method=log_method,
        select_hvgs=select_hvgs,
        n_hvgs=n_hvgs,
        normalize=normalize,
        normalize_method=normalize_method,
    )

    # Step 2: Build node features
    if build_features:
        node_features = build_node_features(
            processed_expr,
            feature_type=feature_type,
            n_pca_components=n_pca_components,
        )
    else:
        # If not building features, return processed expression as features
        node_features = processed_expr.copy()

    logger.info("=" * 60)
    logger.info("Preprocessing pipeline complete!")
    logger.info(f"Processed expression shape: {processed_expr.shape}")
    logger.info(f"Node features shape: {node_features.shape}")
    logger.info("=" * 60)

    return processed_expr, node_features
