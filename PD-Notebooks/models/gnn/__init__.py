"""
GNN models for PD gene-gene graph analysis.

This module provides GNN training and embedding generation for PD transcriptomics.
"""

from .gnn_trainer import GNNTrainer, train_gnn_pd

__all__ = ["GNNTrainer", "train_gnn_pd"]
