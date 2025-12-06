"""
GNN models for PD gene-gene graph analysis.

This module provides GNN training and embedding generation for PD transcriptomics.
Uses BioNeuralNet's GNNEmbedding for supervised learning.
"""

from .gnn_trainer import train_gnn_pd, GNNResults

__all__ = ["train_gnn_pd", "GNNResults"]
