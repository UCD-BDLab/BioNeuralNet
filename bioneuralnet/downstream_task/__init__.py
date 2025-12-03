"""Downstream task pipelines for BioNeuralNet.

This module implements high-level workflows for analyzing patient data using network-derived insights. It includes **DPMON** (Disease Prediction using Multi-Omics Networks), an end-to-end pipeline that leverages GNNs (GCN, GAT, SAGE, GIN) to learn feature importance weights for supervised phenotype prediction. Additionally, it provides **SubjectRepresentation**, a class for fusing learned network embeddings with raw omics data via dimensionality reduction (AutoEncoder or PCA) to generate enriched patient profiles.
"""

from .dpmon import DPMON
from .subject_representation import SubjectRepresentation

__all__ = ["DPMON", "SubjectRepresentation"]
