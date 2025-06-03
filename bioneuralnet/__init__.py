"""
BioNeuralNet: A Python Package for Multi-Omics Integration and Neural Network Embeddings.
BioNeuralNet offers a comprehensive suite of tools designed to transform complex biological data into meaningful low-dimensional representations. The framework facilitates the integration of omics data with advanced neural network embedding methods, enabling downstream applications such as clustering, subject representation, and disease prediction.

Key Features:

    - **Network Embedding**: Generate lower-dimensional representations using Graph Neural Networks (GNNs).
    - **Subject Representation**: Combine network-derived embeddings with raw omics data to produce enriched subject-level profiles.
    - **Correlated Clustering**: BioNeuralNet includes internal modules for performing correlated clustering on complex networks to identify functional modules and informative biomarkers.
    - **Downstream Prediction**: Execute end-to-end pipelines (DPMON) for disease phenotype prediction using network information.
    - **External Integration**: Easily interface with external tools (WGCNA, SmCCNet, Node2Vec, among others.) for network construction, visualization, and advanced analysis.
    - **Evaluation Metrics**: Evaluate the quality of embeddings, clustering results, and network performance using a variety of metrics and visualizations.
    - **Data Handling**: Streamline data preprocessing, filtering, and conversion tasks to ensure seamless integration with the BioNeuralNet framework.
    - **Example Datasets**: Access synthetic datasets for testing and demonstration purposes, enabling users to explore the package's capabilities.
    - **Logging and Configuration**: Utilize built-in logging and configuration utilities to manage experiments, track progress, and optimize workflows.
    - **Comprehensive Documentation**: Detailed API documentation and usage examples to guide users through the package's functionalities.
    - **Open-Source and Extensible**: BioNeuralNet is open-source and designed to be easily extensible, allowing users to customize and enhance its capabilities.
    - **Community Support**: Engage with the BioNeuralNet community for assistance, feedback, and collaboration on biological data analysis projects.

Modules:

    - `network_embedding`: Generates network embeddings via GNNs and Node2Vec.
    - `subject_representation`: Integrates network embeddings into omics data.
    - `downstream_task`: Contains pipelines for disease prediction (e.g., DPMON).
    - `metrics`: Provides tools for evaluating embeddings, cluster comparison, plotting functions, and network performance.
    - `clustering`: Implements clustering algorithms for network analysis.
    - `external_tools`: Wraps external packages (e.g.WGCNA and SmCCNet) for quick integration.
    - `utils`: Utilities for configuration, logging, file handling, converting .Rdata files to dataframes, and variance filtering.
    - `datasets`: Contains example (synthetic) datasets for testing and demonstration purposes.
"""

__version__ = "1.0.9"

from .network_embedding import GNNEmbedding
from .downstream_task import SubjectRepresentation
from .downstream_task import DPMON
from .clustering import CorrelatedPageRank
from .clustering import CorrelatedLouvain
from .clustering import HybridLouvain
from .datasets import DatasetLoader
from .external_tools import SmCCNet

from .metrics import omics_correlation
from .metrics import cluster_correlation
from .metrics import louvain_to_adjacency
from .metrics import evaluate_rf
from .metrics import evaluate_model
from .metrics import evaluate_f1m
from .metrics import evaluate_f1w
from .metrics import plot_performance_three
from .metrics import plot_variance_distribution
from .metrics import plot_variance_by_feature
from .metrics import plot_performance
from .metrics import plot_embeddings
from .metrics import plot_network
from .metrics import plot_multiple_metrics
from .metrics import compare_clusters


from .utils import get_logger
from .utils import rdata_to_df
from .utils import variance_summary
from .utils import zero_fraction_summary
from .utils import expression_summary
from .utils import correlation_summary
from .utils import explore_data_stats
from .utils import preprocess_clinical
from .utils import clean_inf_nan
from .utils import select_top_k_variance
from .utils import select_top_k_correlation
from .utils import select_top_randomforest
from .utils import top_anova_f_features
from .utils import prune_network
from .utils import prune_network_by_quantile
from .utils import network_remove_low_variance
from .utils import network_remove_high_zero_fraction
from .utils import gen_similarity_graph
from .utils import gen_correlation_graph
from .utils import gen_threshold_graph
from .utils import gen_gaussian_knn_graph
from .utils import gen_lasso_graph
from .utils import gen_mst_graph
from .utils import gen_snn_graph


__all__: list = [
    "__version__",
    "GNNEmbedding",
    "SubjectRepresentation",
    "DPMON",
    "CorrelatedPageRank",
    "CorrelatedLouvain",
    "HybridLouvain",
    "omics_correlation",
    "cluster_correlation",
    "louvain_to_adjacency",
    "evaluate_rf",
    "plot_performance_three",
    "plot_variance_distribution",
    "plot_variance_by_feature",
    "plot_performance",
    "plot_embeddings",
    "plot_network",
    "compare_clusters",
    "plot_multiple_metrics",
    "evaluate_model",
    "evaluate_f1m",
    "evaluate_f1w",
    "get_logger",
    "rdata_to_df",
    "variance_summary",
    "zero_fraction_summary",
    "expression_summary",
    "correlation_summary",
    "explore_data_stats",
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
    "gen_similarity_graph",
    "gen_correlation_graph",
    "gen_threshold_graph",
    "gen_gaussian_knn_graph",
    "gen_lasso_graph",
    "gen_mst_graph",
    "gen_snn_graph",
    "DatasetLoader",
    "SmCCNet",
    "evaluate_model",
]
