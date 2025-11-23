"""Network embedding modules for BioNeuralNet.

This module contains classes for generating embeddings using various GNN architectures including GCN, GAT, SAGE, and GIN.
"""

from .gnn_embedding import GNNEmbedding
from .gnn_models import (
    GCN,
    GAT,
    SAGE,
    GIN,
)

__all__ = [
    "GNNEmbedding",
    "GCN",
    "GAT",
    "SAGE",
    "GIN",
]
