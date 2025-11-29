"""
GNN trainer for PD gene-gene correlation graphs.

This module provides training and embedding generation using BioNeuralNet's GNN models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.network_embedding.gnn_models import GCN, GAT, SAGE, GIN
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
        Training loss per epoch.
    model : nn.Module
        Trained GNN model.
    """

    embeddings: np.ndarray
    training_losses: List[float]
    model: nn.Module


class GNNTrainer:
    """
    GNN trainer for PD gene-gene correlation graphs.

    Trains GNN models (GCN, GAT, SAGE, GIN) on gene-gene correlation graphs
    and generates node embeddings for downstream analysis.
    """

    def __init__(
        self,
        model_type: str = "GCN",
        hidden_dim: int = 64,
        layer_num: int = 2,
        dropout: float = 0.5,
        activation: str = "elu",
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        num_epochs: int = 100,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize GNN trainer.

        Parameters
        ----------
        model_type : str, default="GCN"
            GNN model type: "GCN", "GAT", "SAGE", or "GIN".
        hidden_dim : int, default=64
            Hidden dimension of GNN layers.
        layer_num : int, default=2
            Number of GNN layers.
        dropout : float, default=0.5
            Dropout rate.
        activation : str, default="elu"
            Activation function: "relu", "elu", or "leaky_relu".
        lr : float, default=0.01
            Learning rate.
        weight_decay : float, default=5e-4
            Weight decay (L2 regularization).
        num_epochs : int, default=100
            Number of training epochs.
        device : str, optional
            Device to use ("cuda" or "cpu"). If None, auto-detects.
        seed : int, default=42
            Random seed for reproducibility.
        """
        self.model_type = model_type.upper()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.dropout = dropout
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.seed = seed

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Validate device availability
            if device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA requested but not available. "
                        "Your PyTorch installation may not have CUDA support. "
                        "Please install PyTorch with CUDA: "
                        "https://pytorch.org/get-started/locally/"
                    )
                self.device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device(device)

        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.training_losses: List[float] = []

        logger.info(
            f"Initialized GNNTrainer: model={model_type}, hidden_dim={hidden_dim}, "
            f"layers={layer_num}, device={self.device}"
        )

    def _create_model(self, input_dim: int) -> nn.Module:
        """Create GNN model based on model_type."""
        model_kwargs = {
            "input_dim": input_dim,
            "hidden_dim": self.hidden_dim,
            "layer_num": self.layer_num,
            "dropout": self.dropout,
            "final_layer": "regression",  # Will use get_embeddings() instead
            "activation": self.activation,
            "seed": self.seed,
        }

        if self.model_type == "GCN":
            return GCN(**model_kwargs)
        elif self.model_type == "GAT":
            model_kwargs["heads"] = 1  # Single attention head for simplicity
            return GAT(**model_kwargs)
        elif self.model_type == "SAGE":
            return SAGE(**model_kwargs)
        elif self.model_type == "GIN":
            return GIN(**model_kwargs)
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                "Use 'GCN', 'GAT', 'SAGE', or 'GIN'."
            )

    def train(
        self,
        graph_data: Data,
        supervision_targets: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
    ) -> GNNTrainer:
        """
        Train GNN model on graph data.

        Parameters
        ----------
        graph_data : torch_geometric.data.Data
            PyG Data object with x, edge_index, edge_attr.
        supervision_targets : torch.Tensor, optional
            Node-level targets for supervised learning.
            If None, uses unsupervised embedding learning.
        train_mask : torch.Tensor, optional
            Boolean mask for training nodes.
        val_mask : torch.Tensor, optional
            Boolean mask for validation nodes.

        Returns
        -------
        GNNTrainer
            Self (for method chaining).
        """
        logger.info("=" * 60)
        logger.info("Training GNN model.")
        logger.info("=" * 60)

        # Move data to device
        graph_data = graph_data.to(self.device)
        input_dim = graph_data.x.shape[1]

        # Create model
        self.model = self._create_model(input_dim).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if supervision_targets is not None:
            supervision_targets = supervision_targets.to(self.device)
            criterion = nn.MSELoss() if supervision_targets.dtype == torch.float else nn.CrossEntropyLoss()
            logger.info("Using supervised learning with targets.")
        else:
            # Unsupervised: use reconstruction loss or contrastive loss
            # For simplicity, we'll use a simple embedding regularization
            criterion = None
            logger.info("Using unsupervised embedding learning.")

        # Training loop
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model and optimizer must be initialized before training.")

        self.model.train()
        self.training_losses = []

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()

            # Forward pass
            if supervision_targets is not None:
                # Supervised: predict targets
                out = self.model(graph_data)
                if train_mask is not None:
                    loss = criterion(out[train_mask], supervision_targets[train_mask])
                else:
                    loss = criterion(out, supervision_targets)
            else:
                # Unsupervised: use embedding regularization
                embeddings = self.model.get_embeddings(graph_data)
                # Simple regularization: encourage smooth embeddings
                # (neighbors should have similar embeddings)
                edge_index = graph_data.edge_index
                if edge_index.shape[1] > 0:
                    src, dst = edge_index[0], edge_index[1]
                    emb_src = embeddings[src]
                    emb_dst = embeddings[dst]
                    # L2 distance between connected nodes
                    loss = torch.mean((emb_src - emb_dst) ** 2)
                else:
                    # No edges: use L2 regularization
                    loss = torch.mean(embeddings ** 2) * 0.01

            # Backward pass
            loss.backward()
            self.optimizer.step()

            self.training_losses.append(loss.item())

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.4f}")

        logger.info("=" * 60)
        logger.info("GNN training complete!")
        logger.info("=" * 60)

        return self

    def get_embeddings(self, graph_data: Data) -> np.ndarray:
        """
        Generate node embeddings from trained model.

        Parameters
        ----------
        graph_data : torch_geometric.data.Data
            PyG Data object.

        Returns
        -------
        np.ndarray
            Node embeddings (num_nodes × hidden_dim).
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating embeddings.")

        self.model.eval()
        graph_data = graph_data.to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_embeddings(graph_data)

        return embeddings.cpu().numpy()

    def evaluate(
        self,
        graph_data: Data,
        supervision_targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Evaluate model on supervision targets.

        Parameters
        ----------
        graph_data : torch_geometric.data.Data
            PyG Data object.
        supervision_targets : torch.Tensor
            Ground truth targets.
        mask : torch.Tensor, optional
            Boolean mask for evaluation nodes.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        self.model.eval()
        graph_data = graph_data.to(self.device)
        supervision_targets = supervision_targets.to(self.device)

        with torch.no_grad():
            predictions = self.model(graph_data)

        if mask is not None:
            predictions = predictions[mask]
            targets = supervision_targets[mask]
        else:
            targets = supervision_targets

        # Compute metrics based on task type
        if targets.dtype == torch.long:
            # Classification
            from sklearn.metrics import accuracy_score, f1_score
            pred_labels = predictions.argmax(dim=1).cpu().numpy()
            true_labels = targets.cpu().numpy()
            return {
                "accuracy": accuracy_score(true_labels, pred_labels),
                "f1_score": f1_score(true_labels, pred_labels, average="macro"),
            }
        else:
            # Regression
            from sklearn.metrics import mean_squared_error, r2_score
            pred_vals = predictions.cpu().numpy()
            true_vals = targets.cpu().numpy()
            return {
                "mse": mean_squared_error(true_vals, pred_vals),
                "r2": r2_score(true_vals, pred_vals),
            }


def train_gnn_pd(
    graph_data: Data,
    model_type: str = "GCN",
    hidden_dim: int = 64,
    layer_num: int = 2,
    num_epochs: int = 100,
    supervision_targets: Optional[torch.Tensor] = None,
    device: str = "cuda" or "cpu",
    **trainer_kwargs,
) -> Tuple[GNNTrainer, GNNResults]:
    """
    Convenience function to train GNN and generate embeddings.

    Parameters
    ----------
    graph_data : torch_geometric.data.Data
        PyG Data object with graph structure and node features.
    model_type : str, default="GCN"
        GNN model type: "GCN", "GAT", "SAGE", or "GIN".
    hidden_dim : int, default=64
        Hidden dimension of GNN layers.
    layer_num : int, default=2
        Number of GNN layers.
    num_epochs : int, default=100
        Number of training epochs.
    supervision_targets : torch.Tensor, optional
        Node-level targets for supervised learning.
    **trainer_kwargs
        Additional keyword arguments for GNNTrainer.

    Returns
    -------
    Tuple[GNNTrainer, GNNResults]
        (trained trainer, results with embeddings)
    """
    trainer = GNNTrainer(
        model_type=model_type,
        hidden_dim=hidden_dim,
        layer_num=layer_num,
        num_epochs=num_epochs,
        device=device,
        **trainer_kwargs,
    )

    trainer.train(graph_data, supervision_targets=supervision_targets)
    embeddings = trainer.get_embeddings(graph_data)

    results = GNNResults(
        embeddings=embeddings,
        training_losses=trainer.training_losses,
        model=trainer.model,
    )

    return trainer, results
