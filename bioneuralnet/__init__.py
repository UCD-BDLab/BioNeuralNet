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

__version__ = "1.0.1"

from .network_embedding import GNNEmbedding
from .subject_representation import GraphEmbedding
from .downstream_task import DPMON
from .clustering import CorrelatedPageRank
from .clustering import CorrelatedLouvain
from .clustering import HybridLouvain

from .metrics import omics_correlation
from .metrics import cluster_correlation
from .metrics import louvain_to_adjacency
from .metrics import evaluate_rf
from .metrics import plot_performance_three
from .metrics import plot_variance_distribution
from .metrics import plot_variance_by_feature
from .metrics import plot_performance
from .metrics import plot_embeddings
from .metrics import plot_network
from .metrics import compare_clusters

from .utils import clean_inf_nan
from .utils import preprocess_clinical
from .utils import prune_network
from .utils import prune_network_by_quantile
from .utils import select_top_k_variance
from .utils import top_anova_f_features
from .utils import top_features_autoencoder
from .utils import zero_fraction_summary
from .utils import correlation_summary
from .utils import network_remove_low_variance
from .utils import network_filter
from .utils import variance_summary
from .utils import explore_data_stats
from .utils import expression_summary
from .utils import rdata_to_df
from .utils import get_logger

from .datasets import DatasetLoader
from .external_tools import SmCCNet
from .external_tools import WGCNA
from .external_tools import Node2Vec

__all__: list = [
    "__version__",
    "GNNEmbedding",
    "GraphEmbedding",
    "DPMON",
    "CorrelatedPageRank",
    "CorrelatedLouvain",
    "HybridLouvain",
    "omics_correlation",
    "cluster_correlation",
    "louvain_to_adjacency",
    "evaluate_rf",
    "network_filter",
    "rdata_to_df",
    "variance_summary",
    "explore_data_stats",
    "network_remove_low_variance",
    "zero_fraction_summary",
    "expression_summary",
    "correlation_summary",
    "clean_inf_nan",
    "preprocess_clinical",
    "prune_network",
    "prune_network_by_quantile",
    "select_top_k_variance",
    "top_anova_f_features",
    "top_features_autoencoder",
    "get_logger",
    "plot_performance",
    "plot_performance_three",
    "plot_variance_distribution",
    "plot_variance_by_feature",
    "plot_embeddings",
    "plot_network",
    "compare_clusters",
    "DatasetLoader",
    "SmCCNet",
    "WGCNA",
    "Node2Vec",
]
