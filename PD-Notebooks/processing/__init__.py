"""
PD-specific preprocessing utilities for Parkinson's disease transcriptomics and multi-omics data.

This module provides functions for:
- Log-transformation and variance-stabilizing transforms
- Highly variable gene (HVG) selection
- Normalization and standardization for GNN input
- Node feature construction (mean, variance, PCA embeddings)
- Multi-omics data loading and integration
- Multi-omics preprocessing and feature building
"""

from .parkinsons_processing import (
    log_transform_counts,
    select_hvg,
    select_hvg_with_pd_genes,
    preprocess_for_gnn,
    build_node_features,
    preprocess_pipeline,
)

from .multiomics_loader import (
    OmicsData,
    MultiOmicsData,
    parse_geo_series_matrix,
    load_proteomics_csv,
    load_rna_brain_data,
    load_proteomics_brain_data,
    integrate_multiomics,
    load_multiomics_brain_data,
)

from .multiomics_processing import (
    preprocess_omic,
    build_multiomic_node_features,
    build_multiomic_expression_matrix,
    preprocess_multiomics_pipeline,
)

__all__ = [
    # Single-omic processing
    "log_transform_counts",
    "select_hvg",
    "select_hvg_with_pd_genes",
    "preprocess_for_gnn",
    "build_node_features",
    "preprocess_pipeline",
    # Multi-omics data structures
    "OmicsData",
    "MultiOmicsData",
    # Multi-omics loaders
    "parse_geo_series_matrix",
    "load_proteomics_csv",
    "load_rna_brain_data",
    "load_proteomics_brain_data",
    "integrate_multiomics",
    "load_multiomics_brain_data",
    # Multi-omics processing
    "preprocess_omic",
    "build_multiomic_node_features",
    "build_multiomic_expression_matrix",
    "preprocess_multiomics_pipeline",
]
