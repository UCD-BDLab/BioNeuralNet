"""
GNN trainer for PD gene-gene correlation graphs.

This module provides training and embedding generation using BioNeuralNet's GNN models.
Now uses GNNEmbedding from bioneuralnet.network_embedding for supervised learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.network_embedding.gnn_embedding import GNNEmbedding
from bioneuralnet.utils import get_logger

logger = get_logger(__name__)


@dataclass
class GNNResults:
    """
    Container for GNN training results and embeddings.

    Attributes
    ----------
    embeddings : np.ndarray
        Final node embeddings (num_nodes × hidden_dim).
    training_losses : list
        Training loss per epoch (may be empty if not available).
    model : nn.Module
        Trained GNN model.
    """

    embeddings: np.ndarray
    training_losses: List[float]
    model: nn.Module


def train_gnn_pd(
    graph_data: Data,
    adjacency_matrix: Optional[pd.DataFrame] = None,
    processed_omics: Optional[Dict[str, pd.DataFrame]] = None,
    sample_metadata: Optional[pd.DataFrame] = None,
    model_type: str = "GCN",
    hidden_dim: int = 64,
    layer_num: int = 2,
    num_epochs: int = 100,
    device: str = "cuda" or "cpu",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    activation: str = "elu",
    seed: int = 42,
    **trainer_kwargs,
) -> Tuple[GNNEmbedding, GNNResults]:
    """
    Convenience function to train GNN using BioNeuralNet's GNNEmbedding.

    This function uses supervised learning with phenotype labels (PD vs Control).
    The GNNEmbedding class handles node feature preparation, training, and embedding generation.

    Parameters
    ----------
    graph_data : torch_geometric.data.Data
        PyG Data object with graph structure and node features (for compatibility).
    adjacency_matrix : pd.DataFrame, optional
        Adjacency matrix (genes × genes). Required for GNNEmbedding.
    processed_omics : Dict[str, pd.DataFrame], optional
        Dictionary of processed omics data (e.g., {'rna': df, 'proteomics': df}).
        Each DataFrame should be (genes × samples). Required for GNNEmbedding.
    sample_metadata : pd.DataFrame, optional
        Sample metadata with 'condition' column (PD vs CONTROL).
        Index should be sample IDs matching omics data columns. Required for GNNEmbedding.
    model_type : str, default="GCN"
        GNN model type: "GCN", "GAT", "SAGE", or "GIN".
    hidden_dim : int, default=64
        Hidden dimension of GNN layers.
    layer_num : int, default=2
        Number of GNN layers.
    num_epochs : int, default=100
        Number of training epochs.
    device : str, default="cuda"
        Device to use ("cuda" or "cpu").
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-4
        Weight decay (L2 regularization).
    dropout : float, default=0.5
        Dropout rate.
    activation : str, default="elu"
        Activation function: "relu", "elu", or "leaky_relu".
    seed : int, default=42
        Random seed for reproducibility.
    **trainer_kwargs
        Additional keyword arguments for GNNEmbedding.

    Returns
    -------
    Tuple[GNNEmbedding, GNNResults]
        (trained GNNEmbedding instance, results with embeddings)
    """
    # Validate inputs
    if adjacency_matrix is None:
        raise ValueError("adjacency_matrix is required. Please provide the gene-gene adjacency matrix.")

    if processed_omics is None or len(processed_omics) == 0:
        raise ValueError("processed_omics is required. Please provide processed omics data.")

    if sample_metadata is None:
        raise ValueError("sample_metadata is required. Please provide sample metadata with 'condition' column.")

    # Combine omics data (prioritize proteomics if available, else use RNA)
    if 'proteomics' in processed_omics:
        omics_data = processed_omics['proteomics'].copy()
    elif 'rna' in processed_omics:
        omics_data = processed_omics['rna'].copy()
    else:
        # Use first available omics
        omics_data = list(processed_omics.values())[0].copy()

    # Ensure omics_data is samples × genes (transpose if needed)
    # GNNEmbedding expects samples × genes
    if omics_data.shape[0] > omics_data.shape[1]:
        # Likely genes × samples, need to transpose
        omics_data = omics_data.T

    # Create phenotype data from sample metadata
    if 'condition' not in sample_metadata.columns:
        raise ValueError("sample_metadata must have 'condition' column with PD/CONTROL labels.")

    # Align sample metadata with omics data
    common_samples = list(set(omics_data.index) & set(sample_metadata.index))
    if len(common_samples) == 0:
        raise ValueError("No common samples between omics_data and sample_metadata.")

    omics_data = omics_data.loc[common_samples]
    phenotype_data = sample_metadata.loc[common_samples, 'condition']

    # Convert condition to numeric (PD=1, CONTROL=0) for regression
    # GNNEmbedding uses correlation with phenotype, so we can use binary encoding
    phenotype_numeric = (phenotype_data == 'PD').astype(float)

    logger.info(f"Using GNNEmbedding with {len(common_samples)} samples, {omics_data.shape[1]} genes")
    logger.info(f"Phenotype distribution: {phenotype_data.value_counts().to_dict()}")

    # Initialize GNNEmbedding
    gnn_embedding = GNNEmbedding(
        adjacency_matrix=adjacency_matrix,
        omics_data=omics_data,
        phenotype_data=phenotype_numeric,  # Use numeric phenotype
        model_type=model_type,
        hidden_dim=hidden_dim,
        layer_num=layer_num,
        dropout=dropout,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        activation=activation,
        gpu=(device == "cuda"),
        seed=seed,
        tune=False,
        **trainer_kwargs,
    )

    # Train the model
    logger.info("Training GNN model...")
    gnn_embedding.fit()

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings_tensor = gnn_embedding.embed(as_df=False)

    # Convert to numpy if tensor
    if isinstance(embeddings_tensor, torch.Tensor):
        embeddings = embeddings_tensor.cpu().numpy()
    else:
        embeddings = embeddings_tensor.values if isinstance(embeddings_tensor, pd.DataFrame) else embeddings_tensor

    # Extract training losses from the model (if available)
    # GNNEmbedding doesn't expose losses directly, so we'll create a placeholder
    training_losses = []
    if hasattr(gnn_embedding, 'training_losses'):
        training_losses = gnn_embedding.training_losses
    else:
        # Try to get from model if available
        training_losses = []

    results = GNNResults(
        embeddings=embeddings,
        training_losses=training_losses,
        model=gnn_embedding.model,
    )

    return gnn_embedding, results
