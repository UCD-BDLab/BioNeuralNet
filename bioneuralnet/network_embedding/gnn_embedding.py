import os
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from datetime import datetime
from .gnn_models import GCN, GAT, SAGE, GIN
from ..utils.logger import get_logger


class GnnEmbedding:
    """
    GnnEmbedding Class for Generating Graph Neural Network (GNN) Based Embeddings.

    Accepts dataframes directly (omics_data, clinical_data) and does not load any files internally.
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        omics_data: pd.DataFrame,
        clinical_data: pd.DataFrame,
        model_type: str = 'GCN',
        gnn_input_dim: Optional[int] = None,
        gnn_hidden_dim: int = 64,
        gnn_layer_num: int = 2,
        dropout: bool = True,
        output_dir: Optional[str] = None,
    ):
        if adjacency_matrix.empty:
            raise ValueError("Adjacency matrix cannot be empty.")
        if omics_data.empty:
            raise ValueError("Omics data cannot be empty.")
        if clinical_data.empty:
            raise ValueError("Clinical data cannot be empty.")

        self.adjacency_matrix = adjacency_matrix
        self.omics_data = omics_data
        self.clinical_data = clinical_data
        self.model_type = model_type
        self.gnn_input_dim = gnn_input_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_layer_num = gnn_layer_num
        self.dropout = dropout
        self.output_dir = output_dir #if output_dir else self._create_output_dir()

        self.logger = get_logger(__name__)
        self.logger.info("Initialized GnnEmbedding with direct data inputs.")

    def _create_output_dir(self) -> str:
        base_dir = "gnn_embedding_output"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def run(self) -> Dict[str, torch.Tensor]:
        """
        Generate GNN-based embeddings from the provided adjacency matrix and node features.

        Steps:
        1. **Node Feature Preparation**: Computes correlations between omics nodes and clinical variables.
        2. **Building PyG Data Object**: Converts the adjacency matrix and node features into a PyTorch Geometric Data object.
        3. **Model Inference**: Runs the specified GNN model (GCN, GAT, SAGE, or GIN) to compute embeddings.
        4. **Saving Embeddings**: Stores the resulting embeddings to a file for future use.

        Returns:
            Dict[str, torch.Tensor]:
                A dictionary where keys are graph names (e.g., 'graph') and values are PyTorch tensors of shape
                (num_nodes, embedding_dim) containing the node embeddings.

        Raises:
            ValueError: If node features cannot be computed or if required nodes are missing.
            Exception: For any other unforeseen errors.

        Notes:
            - Ensure that the adjacency matrix aligns with the nodes present in the omics data.
            - Clinical variables should be properly correlated with omics features.
            - Adjust `model_type`, `gnn_hidden_dim`, or `gnn_layer_num` as needed to alter the embedding process.
        """
        self.logger.info("Running GNN Embedding process.")
        # Prepare node features
        node_features = self._prepare_node_features()

        if self.gnn_input_dim is None:
            self.gnn_input_dim = node_features.shape[1]
            self.logger.info(f"GNN input dimension set to {self.gnn_input_dim}")

        data = self._adjacency_to_data(node_features)
        model = self._initialize_gnn_model(
            model_type=self.model_type,
            input_dim=self.gnn_input_dim,
            hidden_dim=self.gnn_hidden_dim,
            layer_num=self.gnn_layer_num,
            dropout=self.dropout
        )

        embeddings = self._generate_embeddings(model, data)
        #embeddings_file = os.path.join(self.output_dir, "gnn_embeddings.pt")
        #torch.save(embeddings, embeddings_file)
        #self.logger.info(f"GNN embeddings saved to {embeddings_file}")

        return {'graph': embeddings}


    def _prepare_node_features(self) -> pd.DataFrame:
        self.logger.info("Preparing node features from omics_data and clinical_data.")

        # Check that nodes in adjacency_matrix are present in omics_data
        nodes_in_network = self.adjacency_matrix.index.tolist()
        missing_nodes = set(nodes_in_network) - set(self.omics_data.columns)
        if missing_nodes:
            raise ValueError(f"Nodes missing in omics_data: {missing_nodes}")

        # Align samples between omics_data and clinical_data
        common_samples = self.omics_data.index.intersection(self.clinical_data.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between omics_data and clinical_data.")

        node_data = self.omics_data.loc[common_samples, nodes_in_network]
        clinical_data_aligned = self.clinical_data.loc[common_samples]

        correlations = []
        for node in node_data.columns:
            corr_values = []
            for clinical_var in clinical_data_aligned.columns:
                corr = node_data[node].corr(clinical_data_aligned[clinical_var])
                corr_values.append(corr)
            correlations.append(corr_values)

        node_features = pd.DataFrame(correlations, index=node_data.columns, columns=clinical_data_aligned.columns)
        return node_features

    def _adjacency_to_data(self, node_features: pd.DataFrame) -> Data:
        self.logger.info("Converting adjacency matrix to PyTorch Geometric Data object.")
        G = nx.from_pandas_adjacency(self.adjacency_matrix)
        node_mapping = {node_name: idx for idx, node_name in enumerate(self.adjacency_matrix.index)}
        G = nx.relabel_nodes(G, node_mapping)

        data = from_networkx(G)
        node_order = [self.adjacency_matrix.index[idx] for idx in range(len(self.adjacency_matrix))]
        data.x = torch.tensor(node_features.loc[node_order].values, dtype=torch.float)
        return data

    def _initialize_gnn_model(self, model_type: str, input_dim: int, hidden_dim: int, layer_num: int, dropout: bool) -> nn.Module:
        if model_type == 'GCN':
            model = GCN(input_dim, hidden_dim, layer_num, dropout)
        elif model_type == 'GAT':
            model = GAT(input_dim, hidden_dim, layer_num, dropout)
        elif model_type == 'SAGE':
            model = SAGE(input_dim, hidden_dim, layer_num, dropout)
        elif model_type == 'GIN':
            model = GIN(input_dim, hidden_dim, layer_num, dropout)
        else:
            raise ValueError(f"Unsupported GNN model type: {model_type}")
        return model

    def _generate_embeddings(self, model: nn.Module, data: Data) -> torch.Tensor:
        self.logger.info("Generating embeddings using the GNN model.")
        model.eval()
        with torch.no_grad():
            embeddings = model(data)
        return embeddings
