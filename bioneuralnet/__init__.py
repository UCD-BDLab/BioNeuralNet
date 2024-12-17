"""
BioNeuralNet: A Python Package for Integrating Omics Data with Neural Network Embeddings.

BioNeuralNet provides a suite of tools and components designed to facilitate the integration of omics
data with advanced neural network embeddings. The package supports various embedding techniques,
data preprocessing, and subject representation methods tailored for bioinformatics and computational
biology applications.

Key Features:
    - **Network Embedding**: Generate embeddings using Graph Neural Networks (GNNs) and Node2Vec.
    - **Subject Representation**: Integrate network embeddings into omics data for enhanced subject representations.
    - **Data Utilities**: Tools for loading, validating, and combining omics datasets.
    - **Configuration Management**: Simplified configuration loading from dictionaries or YAML files.
    - **Logging and Path Utilities**: Robust logging setup and path validation utilities for streamlined workflows.

Modules:
    - `network_embedding`: Tools for generating network embeddings using GNNs and Node2Vec.
    - `subject_representation`: Methods for integrating embeddings into omics data.
    - `analysis`: Feature selection and visualization tools.
    - `utils`: Utility functions for configuration, logging, file handling, and more.
"""

__version__: str = '0.1.0'

# Import key classes and functions for easy access
from .network_embedding import GnnEmbedding
from .network_embedding import Node2VecEmbedding
from .subject_representation import GraphEmbedding
from .integrated_tasks import DPMON
from .analysis import FeatureSelector
from .analysis import StaticVisualizer
from .analysis import DynamicVisualizer
from .utils.data_utils import combine_omics_data

# Define the public API of the package
__all__: list = [
    'network_embedding',
    'subject_representation',
    'utils',
    '__version__',
    'GnnEmbedding',
    'Node2VecEmbedding',
    'GraphEmbedding',
    'DPMON',
    'combine_omics_data',
    'FeatureSelector',
    'StaticVisualizer',
    'DynamicVisualizer'
]
