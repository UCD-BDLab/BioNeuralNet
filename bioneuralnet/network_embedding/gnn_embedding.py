from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from .gnn_models import GCN, GAT, SAGE, GIN
from ..utils.logger import get_logger


class GNNEmbedding:
    """
    GNNEmbedding Class for Generating Graph Neural Network (GNN) Based Embeddings.
    -------------------------------------------------------------------------
    1) X: For each node (omics feature), build a feature vector by correlating that feature
       with each clinical variable.
    2) Y: For each node, assign a label = correlation(node, phenotype).
    3) Pass (X, Y) + adjacency to a GNN (GCN, GAT, SAGE, GIN), each returning a single float
       per node for MSE regression.
    4) Train with MSELoss over 'num_epochs'.
    5) Return final node embeddings (penultimate layer).
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        omics_data: pd.DataFrame,
        phenotype_data: pd.DataFrame,
        clinical_data: pd.DataFrame,
        phenotype_col: str = "finalgold_visit",
        model_type: str = "GAT",
        hidden_dim: int = 64,
        layer_num: int = 2,
        dropout: bool = True,
        num_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gpu: bool = False,
    ):
        """
        Parameters
        ----------
        adjacency_matrix : pd.DataFrame
            NxN adjacency matrix for nodes (omics features).
        omics_data : pd.DataFrame
            (samples x features) table, columns must match adjacency index.
        phenotype_data : pd.DataFrame
            (samples x 1) table containing 'phenotype_col' for each sample.
        clinical_data : pd.DataFrame
            (samples x clinical_vars) table.
        phenotype_col : str
            Column in phenotype_data used for correlation-based node label.
        model_type : str
            One of {"GCN", "GAT", "SAGE", "GIN"}.
        hidden_dim : int
            Hidden dimension for the GNN layers.
        layer_num : int
            Number of GNN layers.
        dropout : bool
            Whether to apply dropout in the GNN.
        num_epochs : int
            Number of epochs for MSE training.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay (L2 regularization).
        gpu : bool
            Whether to use GPU if available.
        """
        if adjacency_matrix.empty:
            raise ValueError("Adjacency matrix cannot be empty.")
        if omics_data.empty:
            raise ValueError("Omics data cannot be empty.")
        if phenotype_data.empty or phenotype_col not in phenotype_data.columns:
            raise ValueError(f"Phenotype data must have column '{phenotype_col}'.")
        if clinical_data.empty:
            raise ValueError("Clinical data cannot be empty.")

        self.adjacency_matrix = adjacency_matrix
        self.omics_data = omics_data
        self.phenotype_data = phenotype_data
        self.clinical_data = clinical_data
        self.phenotype_col = phenotype_col

        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.logger = get_logger(__name__)
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.logger.info(
            f"Initialized GNNEmbedding for regression. device={self.device}"
        )

    def run(self) -> Dict[str, torch.Tensor]:
        """
        Generate GNN-based embeddings from the provided adjacency matrix and node features.

        **Returns**: Dict[str, torch.Tensor]
            - A dictionary where keys are graph names (e.g., 'graph') and values are PyTorch tensors of shape
              `(num_nodes, embedding_dim)` containing the node embeddings.

        **Raises**:
            - **ValueError**: If node features cannot be computed or if required nodes are missing.
            - **Exception**: For any unforeseen errors during node feature preparation, model inference, or embedding generation.

        **Notes**:
            - Ensure the adjacency matrix aligns with nodes in omics_data.
            - Node features are built by correlating each feature (column) with clinical_data.
            - Node labels are built by correlating each feature with `phenotype_col`.
        """
        self.logger.info("Running GNN Embedding process for node-level regression.")

        node_features = self._prepare_node_features()
        node_labels = self._prepare_node_labels()
        data = self._build_pyg_data(node_features, node_labels)
        model = self._initialize_gnn_model().to(self.device)
        self._train_gnn(model, data)
        embeddings = self._generate_embeddings(model, data)
        return {"graph": embeddings}

    def _prepare_node_features(self) -> pd.DataFrame:
        """
        Build node features by correlating each omics feature with each clinical variable.

        Returns
        -------
        pd.DataFrame
            Shape (num_nodes, num_clinical_vars), each entry is correlation(feature_i, clinical_var_j).
        """
        self.logger.info(
            "Preparing node features by correlating each omic with clinical vars."
        )
        common_samples = self.omics_data.index.intersection(
            self.phenotype_data.index
        ).intersection(self.clinical_data.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples among omics, phenotype, clinical.")

        omics_filtered = self.omics_data.loc[common_samples]
        clinical_filtered = self.clinical_data.loc[common_samples]
        node_names = self.adjacency_matrix.index.tolist()

        node_features_list = []
        clinical_cols = clinical_filtered.columns.tolist()
        for node in node_names:
            if node not in omics_filtered.columns:
                raise ValueError(f"Node {node} not found in omics_data.")
            corr_vector = []
            for cvar in clinical_cols:
                corr_val = omics_filtered[node].corr(clinical_filtered[cvar])
                corr_vector.append(corr_val)
            node_features_list.append(corr_vector)

        node_features_df = pd.DataFrame(
            node_features_list, index=node_names, columns=clinical_cols
        ).fillna(0.0)
        return node_features_df

    def _prepare_node_labels(self) -> pd.Series:
        """
        Build node labels by correlating each omics feature with the specified phenotype column.

        Returns
        -------
        pd.Series
            Index=node, value=correlation(feature, phenotype).
        """
        self.logger.info(
            f"Preparing node labels from correlation with phenotype_col='{self.phenotype_col}'."
        )
        common_samples = self.omics_data.index.intersection(self.phenotype_data.index)
        omics_filtered = self.omics_data.loc[common_samples]
        phen_filtered = self.phenotype_data.loc[common_samples, self.phenotype_col]

        labels_dict = {}
        node_names = self.adjacency_matrix.index.tolist()
        for node in node_names:
            if node not in omics_filtered.columns:
                raise ValueError(f"Node {node} not in omics_data columns.")
            corr_val = omics_filtered[node].corr(phen_filtered)
            labels_dict[node] = corr_val

        labels_series = pd.Series(labels_dict, index=node_names).fillna(0.0)
        return labels_series

    def _build_pyg_data(
        self, node_features: pd.DataFrame, node_labels: pd.Series
    ) -> Data:
        """
        Construct a PyTorch Geometric Data object:
         - data.x = node_features
         - data.y = node_labels
         - data.edge_index from adjacency

        Returns
        -------
        Data
            PyG Data object with x, y, edge_index.
        """
        self.logger.info("Constructing PyG Data (edge_index, data.x, data.y).")
        G = nx.from_pandas_adjacency(self.adjacency_matrix)
        node_mapping = {name: i for i, name in enumerate(node_features.index)}
        G = nx.relabel_nodes(G, node_mapping)

        data = from_networkx(G)
        node_order = list(node_features.index)
        data.x = torch.tensor(node_features.loc[node_order].values, dtype=torch.float)
        data.y = torch.tensor(node_labels.loc[node_order].values, dtype=torch.float)
        return data

    def _initialize_gnn_model(self) -> nn.Module:
        """
        Create a GNN model (GCN, GAT, SAGE, GIN) for node-level regression.

        Returns
        -------
        nn.Module
        """
        self.logger.info(
            f"Initializing GNN of type {self.model_type}, hidden_dim={self.hidden_dim}"
        )
        # node_features width = num_clinical_vars
        input_dim = self.clinical_data.shape[1]
        if self.model_type == "GCN":
            return GCN(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                layer_num=self.layer_num,
                dropout=self.dropout,
            )
        elif self.model_type == "GAT":
            return GAT(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                layer_num=self.layer_num,
                dropout=self.dropout,
            )
        elif self.model_type == "SAGE":
            return SAGE(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                layer_num=self.layer_num,
                dropout=self.dropout,
            )
        elif self.model_type == "GIN":
            return GIN(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                layer_num=self.layer_num,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _train_gnn(self, model: nn.Module, data: Data) -> None:
        """
        Train the GNN model (MSE regression). Each node's label is correlation in [-1..+1].
        """
        self.logger.info("Training GNN for regression using MSELoss.")
        data = data.to(self.device)
        model.to(self.device)
        model.train()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            # [num_nodes, 1]
            out = model(data)
            # [num_nodes]
            out = out.view(-1)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch [{epoch+1}/{self.num_epochs}], MSE Loss: {loss.item():.4f}"
                )

    def _generate_embeddings(self, model: nn.Module, data: Data) -> torch.Tensor:
        """
        Get node embeddings from the penultimate layer (hidden_dim) after training.

        Returns
        -------
        torch.Tensor
            Shape [num_nodes, hidden_dim].
        """
        self.logger.info("Generating final node embeddings (penultimate layer).")
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            embeddings = model.get_embeddings(data)
        return embeddings.cpu()
