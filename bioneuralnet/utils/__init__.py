"""Utility functions for BioNeuralNet.

This module provides a collection of helper functions for data preprocessing, statistical analysis, graph generation, and reproducibility.
"""

from .logger import get_logger
from .rdata_convert import rdata_to_df
from .reproducibility import set_seed

from .graph_tools import (
    graph_analysis,
    repair_graph_connectivity,
    find_optimal_graph,
)

from .data import (
    variance_summary,
    zero_fraction_summary,
    expression_summary,
    correlation_summary,
    explore_data_stats,
)

from .graph import (
    gen_similarity_graph,
    gen_correlation_graph,
    gen_threshold_graph,
    gen_gaussian_knn_graph,
    gen_lasso_graph,
    gen_mst_graph,
    gen_snn_graph,
)

from .preprocess import (
    preprocess_clinical,
    clean_inf_nan,
    select_top_k_variance,
    select_top_k_correlation,
    select_top_randomforest,
    top_anova_f_features,
    prune_network,
    prune_network_by_quantile,
    network_remove_low_variance,
    network_remove_high_zero_fraction,
    impute_omics,
    impute_omics_knn,
    normalize_omics,
    beta_to_m,
)

__all__ = [
    "get_logger",
    "rdata_to_df",
    "set_seed",
    "graph_analysis",
    "repair_graph_connectivity",
    "find_optimal_graph",
    "variance_summary",
    "zero_fraction_summary",
    "expression_summary",
    "correlation_summary",
    "explore_data_stats",
    "gen_similarity_graph",
    "gen_correlation_graph",
    "gen_threshold_graph",
    "gen_gaussian_knn_graph",
    "gen_lasso_graph",
    "gen_mst_graph",
    "gen_snn_graph",
    "preprocess_clinical",
    "clean_inf_nan",
    "select_top_k_variance",
    "select_top_k_correlation",
    "select_top_randomforest",
    "top_anova_f_features",
    "prune_network",
    "prune_network_by_quantile",
    "network_remove_low_variance",
    "network_remove_high_zero_fraction",
    "impute_omics",
    "impute_omics_knn",
    "normalize_omics",
    "beta_to_m",
]
