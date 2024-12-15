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

Example Usage:
    ```python
    import pandas as pd
    from bioneuralnet import SubjectRepresentationEmbedding

    # Load data
    adjacency_matrix = pd.read_csv('input/adjacency_matrix.csv', index_col=0)
    omics_data = pd.read_csv('input/omics_data.csv', index_col=0)
    phenotype_data = pd.read_csv('input/phenotype_data.csv', index_col=0).squeeze()
    clinical_data = pd.read_csv('input/clinical_data.csv', index_col=0)
    
    # Initialize and run SubjectRepresentationEmbedding with GNNs
    subject_rep = SubjectRepresentationEmbedding(
        adjacency_matrix=adjacency_matrix,
        omics_data=omics_data,
        phenotype_data=phenotype_data,
        clinical_data=clinical_data,
        embedding_method='GNNs'  # Change to 'Node2Vec' to use Node2Vec embeddings
    )
    enhanced_omics_data = subject_rep.run()
    
    print(enhanced_omics_data.head())
    ```
"""

__version__: str = '0.1.0'

# Import key classes and functions for easy access
from .network_embedding.gnns import GNNEmbedding
from .network_embedding.node2vec import Node2VecEmbedding
from .subject_representation.subject_representation import SubjectRepresentationEmbedding
from .analysis.feature_selector import FeatureSelector
from .analysis.static_visualization import StaticVisualizer
from .analysis.dynamic_visualization import DynamicVisualizer
from .utils.data_utils import combine_omics_data

# Define the public API of the package
__all__: list = [
    'network_embedding',
    'subject_representation',
    'utils',
    '__version__',
    'GNNEmbedding',
    'Node2VecEmbedding',
    'SubjectRepresentationEmbedding',
    'combine_omics_data',
    'FeatureSelector',
    'StaticVisualizer',
    'DynamicVisualizer'
]
