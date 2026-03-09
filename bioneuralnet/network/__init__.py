"""Network Construction and Analysis.

This module provides tools for generating, searching, and analyzing multi-omics 
networks. It includes methods for building networks from raw tabular data using 
similarity, correlation, thresholding, and Gaussian KNN, as well as 
phenotype-driven strategies like PySmCCNet.
"""

from .tools import (
    NetworkAnalyzer,
    network_search
)

from .generate import (
    similarity_network,
    correlation_network,
    threshold_network,
    gaussian_knn_network,
)
from .pysmccnet import (
    auto_pysmccnet
)

__all__ = [
    "NetworkAnalyzer",
    "network_search",
    "similarity_network",
    "correlation_network",
    "threshold_network",
    "gaussian_knn_network",
    "auto_pysmccnet",
]