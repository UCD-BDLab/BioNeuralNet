"""
Multi-omics preprocessing for Parkinson's disease data.

Extends the single-omic preprocessing to handle RNA + proteomics integration
at the gene level for building multi-omics graphs.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from processing.parkinsons_processing import (
    log_transform_counts,
    select_hvg,
    preprocess_for_gnn,
    build_node_features,
)
from processing.multiomics_loader import MultiOmicsData, OmicsData
from bioneuralnet.utils.data import normalize_omics
from bioneuralnet.utils import get_logger

logger = get_logger(__name__)


def preprocess_omic(
    omic_data: OmicsData,
    log_transform: bool = True,
    select_hvgs: bool = True,
    n_hvgs: int = 5000,
    normalize: bool = True,
    normalize_method: str = "standard",
) -> pd.DataFrame:
    """
    Preprocess a single omic dataset.

    Parameters
    ----------
    omic_data : OmicsData
        Single omic data container
    log_transform : bool, default=True
        Whether to apply log transformation
    select_hvgs : bool, default=True
        Whether to select highly variable features
    n_hvgs : int, default=5000
        Number of HVGs to select
    normalize : bool, default=True
        Whether to normalize
    normalize_method : str, default="standard"
        Normalization method

    Returns
    -------
    pd.DataFrame
        Preprocessed expression matrix (features × samples)
    """
    logger.info(f"Preprocessing {omic_data.omic_type} data...")

    processed = omic_data.expression.copy()

    # Log transform
    if log_transform:
        processed = log_transform_counts(processed, method="log2")

    # HVG selection (for RNA, skip for proteomics if too few features)
    if select_hvgs and processed.shape[0] > n_hvgs:
        processed = select_hvg(processed, n_top=n_hvgs, method="variance")

    # Normalize
    if normalize:
        processed_T = processed.T  # samples × features
        processed_T_norm = normalize_omics(processed_T, method=normalize_method)
        processed = processed_T_norm.T  # features × samples

    logger.info(f"Preprocessed {omic_data.omic_type}: {processed.shape}")
    return processed


def build_multiomic_node_features(
    multiomics: MultiOmicsData,
    common_genes: Optional[list] = None,
    feature_type: str = "mean_variance",
    n_pca_components: Optional[int] = None,
    standardize_features: bool = True,
) -> pd.DataFrame:
    """
    Build node features for multi-omics graph.

    For each gene, creates features from:
    - RNA expression (mean, variance, PCA components)
    - Proteomics abundance (mean, variance, PCA components)
    - Combined statistics

    Parameters
    ----------
    multiomics : MultiOmicsData
        Multi-omics data container
    common_genes : list, optional
        List of genes to include. If None, uses multiomics.common_genes
    feature_type : str, default="mean_variance"
        Type of features: "mean_variance", "pca", "combined"
    n_pca_components : int, optional
        Number of PCA components (if using PCA)
    standardize_features : bool, default=True
        Whether to standardize final features

    Returns
    -------
    pd.DataFrame
        Node feature matrix (genes × features)
        Index: gene symbols
    """
    logger.info("Building multi-omic node features...")

    if common_genes is None:
        common_genes = multiomics.common_genes

    if not common_genes:
        logger.warning(
            "No common genes found. Building node features separately for each omic. "
            "Graph will be built separately for each omic and combined."
        )
        # Use all genes from all omics
        all_genes = set()
        if multiomics.rna is not None:
            if 'gene_symbol' in multiomics.rna.feature_metadata.columns:
                all_genes.update(multiomics.rna.feature_metadata['gene_symbol'].dropna())
            else:
                all_genes.update(multiomics.rna.expression.index)
        if multiomics.proteomics is not None:
            all_genes.update(multiomics.proteomics.expression.index)
        common_genes = list(all_genes)
        logger.info(f"Using all genes from all omics: {len(common_genes)} total genes")

    features_list = []

    # RNA features
    if multiomics.rna is not None:
        # Map gene symbols to expression features
        rna_expr = multiomics.rna.expression.copy()

        # If RNA has probe IDs, need to map to gene symbols
        if 'gene_symbol' in multiomics.rna.feature_metadata.columns:
            # Aggregate probes per gene (take mean)
            rna_expr = rna_expr.groupby(
                multiomics.rna.feature_metadata['gene_symbol']
            ).mean()

        # Filter to common genes
        rna_common = set(rna_expr.index) & set(common_genes)
        rna_expr = rna_expr.loc[list(rna_common)]

        # Build features
        rna_features = build_node_features(
            rna_expr,
            feature_type=feature_type,
            n_pca_components=n_pca_components,
            standardize_features=False,  # Standardize later
        )

        # Rename columns to indicate omic
        rna_features.columns = [f"rna_{col}" for col in rna_features.columns]
        features_list.append(("rna", rna_features))
        logger.info(f"RNA features: {rna_features.shape}")

    # Proteomics features
    if multiomics.proteomics is not None:
        prot_expr = multiomics.proteomics.expression.copy()

        # Filter to common genes
        prot_common = set(prot_expr.index) & set(common_genes)
        prot_expr = prot_expr.loc[list(prot_common)]

        # Build features
        prot_features = build_node_features(
            prot_expr,
            feature_type=feature_type,
            n_pca_components=n_pca_components,
            standardize_features=False,
        )

        # Rename columns
        prot_features.columns = [f"prot_{col}" for col in prot_features.columns]
        features_list.append(("proteomics", prot_features))
        logger.info(f"Proteomics features: {prot_features.shape}")

    # Combine features
    if not features_list:
        raise ValueError("No omics data available for feature building.")

    # Merge on gene symbols
    node_features: Optional[pd.DataFrame] = None
    for omic_name, features in features_list:
        if node_features is None:
            node_features = features
        else:
            # Outer join to include all genes
            node_features = node_features.join(
                features, how='outer', rsuffix=f'_{omic_name}'
            )

    # Fill missing values with 0 (genes not present in an omic)
    if node_features is None:
        raise ValueError("No node features were built from omics data.")
    node_features = node_features.fillna(0)

    # Standardize if requested
    if standardize_features:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        node_features_scaled = scaler.fit_transform(node_features.values)
        node_features = pd.DataFrame(
            node_features_scaled,
            index=node_features.index,
            columns=node_features.columns,
        )
        logger.info("Standardized node features (Z-score per feature).")

    logger.info(f"Multi-omic node features: {node_features.shape}")
    return node_features


def build_multiomic_expression_matrix(
    multiomics: MultiOmicsData,
    common_genes: Optional[list] = None,
    combine_method: str = "concatenate",
) -> pd.DataFrame:
    """
    Build a combined expression matrix for graph construction.

    For each gene, combines expression from multiple omics.
    Options:
    - "concatenate": Stack features from each omic
    - "mean": Average across omics (requires same samples)
    - "weighted_mean": Weighted average

    Parameters
    ----------
    multiomics : MultiOmicsData
        Multi-omics data container
    common_genes : list, optional
        List of genes to include
    combine_method : str, default="concatenate"
        Method to combine omics: "concatenate", "mean", "weighted_mean"

    Returns
    -------
    pd.DataFrame
        Combined expression matrix (genes × features)
        For concatenate: features are [rna_sample1, rna_sample2, ..., prot_sample1, ...]
    """
    logger.info(f"Building multi-omic expression matrix (method={combine_method})...")

    if common_genes is None:
        common_genes = multiomics.common_genes

    if not common_genes:
        raise ValueError("No common genes found.")

    if combine_method == "concatenate":
        # Stack expression vectors from each omic
        combined_expr = None

        if multiomics.rna is not None:
            rna_expr = multiomics.rna.expression.copy()

            # Map to gene symbols if needed
            if 'gene_symbol' in multiomics.rna.feature_metadata.columns:
                rna_expr = rna_expr.groupby(
                    multiomics.rna.feature_metadata['gene_symbol']
                ).mean()

            rna_common = set(rna_expr.index) & set(common_genes)
            rna_expr = rna_expr.loc[list(rna_common)]
            rna_expr.columns = [f"rna_{col}" for col in rna_expr.columns]

            if combined_expr is None:
                combined_expr = rna_expr
            else:
                combined_expr = combined_expr.join(rna_expr, how='outer')

        if multiomics.proteomics is not None:
            prot_expr = multiomics.proteomics.expression.copy()
            prot_common = set(prot_expr.index) & set(common_genes)
            prot_expr = prot_expr.loc[list(prot_common)]
            prot_expr.columns = [f"prot_{col}" for col in prot_expr.columns]

            if combined_expr is None:
                combined_expr = prot_expr
            else:
                combined_expr = combined_expr.join(prot_expr, how='outer')

        # Fill missing with 0
        if combined_expr is None:
            raise ValueError("No omics data available for expression matrix construction.")
        combined_expr = combined_expr.fillna(0)

        logger.info(f"Combined expression matrix: {combined_expr.shape}")
        return combined_expr

    else:
        raise NotImplementedError(
            f"Combine method '{combine_method}' not yet implemented. "
            "Use 'concatenate' for now."
        )


def preprocess_multiomics_pipeline(
    multiomics: MultiOmicsData,
    log_transform: bool = True,
    select_hvgs: bool = True,
    n_hvgs: int = 5000,
    normalize: bool = True,
    normalize_method: str = "standard",
    build_features: bool = True,
    feature_type: str = "mean_variance",
    n_pca_components: Optional[int] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Complete preprocessing pipeline for multi-omics data.

    Parameters
    ----------
    multiomics : MultiOmicsData
        Multi-omics data container
    log_transform : bool, default=True
        Whether to log transform
    select_hvgs : bool, default=True
        Whether to select HVGs
    n_hvgs : int, default=5000
        Number of HVGs
    normalize : bool, default=True
        Whether to normalize
    normalize_method : str, default="standard"
        Normalization method
    build_features : bool, default=True
        Whether to build node features
    feature_type : str, default="mean_variance"
        Feature type
    n_pca_components : int, optional
        Number of PCA components

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], pd.DataFrame]
        (processed_omics, node_features)
        - processed_omics: dict mapping omic_type -> processed expression
        - node_features: combined node features for graph
    """
    logger.info("=" * 60)
    logger.info("Starting multi-omics preprocessing pipeline")
    logger.info("=" * 60)

    processed_omics = {}

    # Preprocess each omic
    if multiomics.rna is not None:
        processed_rna = preprocess_omic(
            multiomics.rna,
            log_transform=log_transform,
            select_hvgs=select_hvgs,
            n_hvgs=n_hvgs,
            normalize=normalize,
            normalize_method=normalize_method,
        )
        processed_omics['rna'] = processed_rna

    if multiomics.proteomics is not None:
        processed_prot = preprocess_omic(
            multiomics.proteomics,
            log_transform=log_transform,
            select_hvgs=select_hvgs and multiomics.proteomics.expression.shape[0] > n_hvgs,
            n_hvgs=n_hvgs,
            normalize=normalize,
            normalize_method=normalize_method,
        )
        processed_omics['proteomics'] = processed_prot

    # Build node features
    if build_features:
        node_features = build_multiomic_node_features(
            multiomics,
            feature_type=feature_type,
            n_pca_components=n_pca_components,
        )
    else:
        # Use combined expression as features
        node_features = build_multiomic_expression_matrix(multiomics)

    logger.info("=" * 60)
    logger.info("Multi-omics preprocessing complete!")
    logger.info(f"Processed omics: {list(processed_omics.keys())}")
    logger.info(f"Node features shape: {node_features.shape}")
    logger.info("=" * 60)

    return processed_omics, node_features
