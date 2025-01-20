"""
BioNeuralNet: A Python Package for Multi-Omics Integration and Neural Network Embeddings.

BioNeuralNet offers a comprehensive suite of tools designed to transform complex biological data into meaningful low-dimensional representations. The framework facilitates the integration of omics data with advanced neural network embedding methods, enabling downstream applications such as clustering, subject representation, and disease prediction.

Key Features:
    - **Network Embedding**: Generate lower-dimensional representations using Graph Neural Networks (GNNs).
    - **Subject Representation**: Combine network-derived embeddings with raw omics data to produce enriched subject-level profiles.
    - **Clustering & Analysis**: Leverage clustering algorithms and feature selection methods to identify functional modules and informative biomarkers.
    - **Downstream Prediction**: Execute end-to-end pipelines (DPMON) for disease phenotype prediction using network information.
    - **External Integration**: Easily interface with external tools (WGCNA, SmCCNet, Node2Vec, among others.) for network
      construction, visualization, and advanced analysis.

Modules:
    - `network_embedding`: Generates network embeddings via GNNs and Node2Vec.
    - `subject_representation`: Integrates network embeddings into omics data.
    - `downstream_task`: Contains pipelines for disease prediction (e.g., DPMON).
    - `analysis`: Provides tools for feature selection and clustering.
    - `external_tools`: Wraps external packages (e.g., WGCNA, SmCCNet, Node2Vec, FeatureSelector, StaticVisualizer, DynamicVisualizer, HierarchicalClustering) for quick integration.
    - `utils`: Utilities for configuration, logging, file handling, and more.

"""

__version__: str = "0.1.0"

from .network_embedding import GNNEmbedding
from .subject_representation import GraphEmbedding
from .downstream_task import DPMON

from .external_tools import DynamicVisualizer
from .external_tools import FeatureSelector
from .external_tools import HierarchicalClustering
from .external_tools import StaticVisualizer
from .external_tools import SmCCNet
from .external_tools import WGCNA
from .external_tools import Node2Vec

__all__: list = [
    "network_embedding",
    "subject_representation",
    "utils",
    "__version__",
    "GNNEmbedding",
    "Node2Vec",
    "GraphEmbedding",
    "DPMON",
    "FeatureSelector",
    "StaticVisualizer",
    "DynamicVisualizer",
    "HierarchicalClustering",
    "SmCCNet",
    "WGCNA",
]
