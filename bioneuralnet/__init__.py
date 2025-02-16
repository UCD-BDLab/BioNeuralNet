"""
BioNeuralNet: A Python Package for Multi-Omics Integration and Neural Network Embeddings.

BioNeuralNet offers a comprehensive suite of tools designed to transform complex biological data into meaningful low-dimensional representations. The framework facilitates the integration of omics data with advanced neural network embedding methods, enabling downstream applications such as clustering, subject representation, and disease prediction.

Key Features:
    - **Network Embedding**: Generate lower-dimensional representations using Graph Neural Networks (GNNs).
    - **Subject Representation**: Combine network-derived embeddings with raw omics data to produce enriched subject-level profiles.
    - **Correlated Clustering**: BioNeuralNet includes internal modules for performing correlated clustering on complex networks to identify functional modules and informative biomarkers.
    - **Downstream Prediction**: Execute end-to-end pipelines (DPMON) for disease phenotype prediction using network information.
    - **External Integration**: Easily interface with external tools (WGCNA, SmCCNet, Node2Vec, among others.) for network construction, visualization, and advanced analysis.

Modules:
    - `network_embedding`: Generates network embeddings via GNNs and Node2Vec.
    - `subject_representation`: Integrates network embeddings into omics data.
    - `downstream_task`: Contains pipelines for disease prediction (e.g., DPMON).
    - `metrics`: Provides tools for checking the variance of a dataset and computing correlation.
    - `external_tools`: Wraps external packages (e.g.WGCNA and SmCCNet) for quick integration.
    - `utils`: Utilities for configuration, logging, file handling, converting .Rdata files to csv and more.
    - `datasets`: Contains example (synthetic) datasets for testing and demonstration purposes.
"""

__version__ = "0.2.0b2"

from .network_embedding import GNNEmbedding
from .subject_representation import GraphEmbedding
from .downstream_task import DPMON
from .clustering import CorrelatedPageRank
from .clustering import CorrelatedLouvain
from .clustering import HybridLouvain
from .metrics import compute_cluster_correlation_from_df, compute_correlation, convert_louvain_to_adjacency
from .metrics import CheckVariance, network_remove_low_variance, network_remove_high_zero_fraction

from .datasets import DatasetLoader
from .utils import get_logger
from .utils import rdata_to_csv_file
from .utils import evaluate_rf_regressor,evaluate_rf_classifier,plot_embeddings, plot_network, plot_performance,compare_clusters

from .external_tools import SmCCNet
from .external_tools import WGCNA
from .external_tools import Node2Vec

__all__: list = [
    "__version__",
    "GNNEmbedding",
    "GraphEmbedding",
    "DPMON",
    "PageRank",
    "CorrelatedPageRank",
    "CorrelatedLouvain",
    "correlation",
    "logger",
    "Correlation",
    "HybridLouvain",
    "CheckVariance",
    "utils",
    "Node2Vec",
    "get_logger",
    "rdata_to_csv_file",
    "evaluate_rf",
    "plot_embeddings",
    "plot_network",
    "plot_performance",
    "compute_cluster_correlation_from_df",
    "compute_correlation",
    "network_remove_low_variance",
    "network_remove_high_zero_fraction",
    "SmCCNet",
    "WGCNA",
    "DatasetLoader",
    "compare_clusters",
    "evaluate_rf_regressor",
    "evaluate_rf_classifier",
    "convert_louvain_to_adjacency"
]
