import os
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.data_utils import combine_omics_data


class GNNEmbedding:
    """
    GNNEmbedding Class for Generating Graph Neural Network (GNN) Based Embeddings.

    This class handles the loading and processing of omics data, initialization of GNN models,
    execution of the embedding process, and saving of the resulting embeddings.
    """

    def __init__(
        self,
        omics_list: List[str],
        phenotype_file: str,
        clinical_data_file: str,
        adjacency_matrix: pd.DataFrame,
        model_type: str = 'GCN',
        gnn_input_dim: Optional[int] = None,
        gnn_hidden_dim: int = 64,
        gnn_layer_num: int = 2,
        dropout: bool = True,
        output_dir: Optional[str] = None,
    ):
        """
        Initializes the GNNEmbedding instance with direct parameters.

        Args:
            omics_list (List[str]): List of paths to omics data CSV files.
            phenotype_file (str): Path to the phenotype CSV file.
            clinical_data_file (str): Path to the clinical data CSV file.
            adjacency_matrix (pd.DataFrame): The adjacency matrix representing the network.
            model_type (str): Type of the GNN model ('GCN', 'GAT', 'SAGE', 'GIN').
            gnn_input_dim (int, optional): Dimension of input node features. If None, will be determined from data.
            gnn_hidden_dim (int): Dimension of hidden layers in the GNN.
            gnn_layer_num (int): Number of GNN layers.
            dropout (bool): Whether to apply dropout after each layer.
            output_dir (str, optional): Directory to save outputs. If None, creates a unique directory.
        """
        # Assign parameters
        self.omics_list = omics_list
        self.phenotype_file = phenotype_file
        self.clinical_data_file = clinical_data_file
        self.adjacency_matrix = adjacency_matrix
        self.model_type = model_type
        self.gnn_input_dim = gnn_input_dim 
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_layer_num = gnn_layer_num
        self.dropout = dropout
        self.output_dir = output_dir if output_dir else self._create_output_dir()

        # Initialize logger
        self.logger = get_logger(__name__)
        self.logger.info("Initialized GNNEmbedding with the following parameters:")
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"GNN Hidden Dimension: {self.gnn_hidden_dim}")
        self.logger.info(f"GNN Layer Number: {self.gnn_layer_num}")
        self.logger.info(f"Dropout: {self.dropout}")
        self.logger.info(f"Output Directory: {self.output_dir}")

    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the current GNNEmbedding run.

        Returns:
            str: Path to the created output directory.
        """
        base_dir = "gnn_embedding_output"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def run(self) -> Dict[str, torch.Tensor]:
        """
        Perform GNN-based embedding on the provided network.

        Returns:
            dict:
                A dictionary where keys are graph names and values are embeddings as tensors.
        """
        self.logger.info("Running GNN Embedding")

        try:
            # Step 1: Load and combine omics data
            omics_data = self._load_and_combine_omics_data(self.omics_list, self.phenotype_file)

            # Step 2: Prepare node features based on correlations
            node_features = self._prepare_node_features(omics_data)

            # Step 3: Update GNN input dimension if not set
            if self.gnn_input_dim is None:
                self.gnn_input_dim = node_features.shape[1]
                self.logger.info(f"GNN input dimension set to {self.gnn_input_dim}")

            # Step 4: Convert adjacency matrix to PyTorch Geometric Data object
            data = self._adjacency_to_data(self.adjacency_matrix, node_features)

            # Step 5: Initialize GNN model
            model = self._initialize_gnn_model(
                model_type=self.model_type,
                input_dim=self.gnn_input_dim,
                hidden_dim=self.gnn_hidden_dim,
                layer_num=self.gnn_layer_num,
                dropout=self.dropout
            )

            # Step 6: Generate embeddings
            embeddings = self._generate_embeddings(model, data)

            # Step 7: Save embeddings
            embeddings_file = os.path.join(self.output_dir, "gnn_embeddings.pt")
            torch.save(embeddings, embeddings_file)
            self.logger.info(f"GNN embeddings saved to {embeddings_file}")

            return {'graph': embeddings}

        except Exception as e:
            self.logger.error(f"Error in GNN Embedding: {e}")
            raise

    def _load_and_combine_omics_data(self, omics_list: List[str], phenotype_file: str) -> pd.DataFrame:
        """
        Load and combine omics data with phenotype data.

        Args:
            omics_list (List[str]): List of omics data file paths.
            phenotype_file (str): Phenotype data file path.

        Returns:
            pd.DataFrame: Combined omics and phenotype data.
        """
        self.logger.info("Loading and combining omics data.")
        omics_data = combine_omics_data(omics_list)
        phenotype_data = pd.read_csv(phenotype_file, index_col=0)
        combined_data = omics_data.merge(phenotype_data, left_index=True, right_index=True)

        # Clean column names to match network nodes
        combined_data.columns = self._clean_column_names(combined_data.columns)

        return combined_data

    def _prepare_node_features(self, omics_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare node features for the GNN by computing correlations between nodes and clinical variables.

        Args:
            omics_data (pd.DataFrame): Combined omics data.

        Returns:
            pd.DataFrame: Node features dataframe.
        """
        self.logger.info("Preparing node features based on correlations with clinical data.")

        # Ensure that nodes in adjacency matrix are present in omics data columns
        nodes_in_network = self.adjacency_matrix.index.tolist()
        missing_nodes = set(nodes_in_network) - set(omics_data.columns)
        if missing_nodes:
            self.logger.error(f"Nodes missing in omics data: {missing_nodes}")
            raise ValueError(f"Nodes missing in omics data: {missing_nodes}")

        # Extract node data
        node_data = omics_data[nodes_in_network]

        # Load clinical data
        clinical_data = pd.read_csv(self.clinical_data_file, index_col=0)

        # Ensure that samples align between omics data and clinical data
        common_samples = node_data.index.intersection(clinical_data.index)
        if len(common_samples) == 0:
            self.logger.error("No common samples between omics data and clinical data.")
            raise ValueError("No common samples between omics data and clinical data.")

        # Align data
        node_data = node_data.loc[common_samples]
        clinical_data = clinical_data.loc[common_samples]

        # Compute correlations between each node and each clinical variable
        correlations = []
        for node in node_data.columns:
            corr_values = []
            for clinical_var in clinical_data.columns:
                corr = node_data[node].corr(clinical_data[clinical_var])
                corr_values.append(corr)
            correlations.append(corr_values)

        node_features = pd.DataFrame(
            correlations,
            index=node_data.columns,
            columns=clinical_data.columns
        )

        return node_features

    def _adjacency_to_data(self, adjacency_matrix: pd.DataFrame, node_features: pd.DataFrame) -> Data:
        """
        Convert an adjacency matrix to a PyTorch Geometric Data object.

        Args:
            adjacency_matrix (pd.DataFrame): Adjacency matrix.
            node_features (pd.DataFrame): Node features.

        Returns:
            Data: PyTorch Geometric Data object.
        """
        self.logger.info("Converting adjacency matrix to PyTorch Geometric Data object.")

        # Convert adjacency matrix to NetworkX graph
        G = nx.from_pandas_adjacency(adjacency_matrix)
        self.logger.debug(f"NetworkX graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        # Map node names to indices
        node_mapping = {node_name: idx for idx, node_name in enumerate(adjacency_matrix.index)}
        G = nx.relabel_nodes(G, node_mapping)

        # Convert to PyTorch Geometric Data object
        data = from_networkx(G)

        # Assign node features
        node_order = [adjacency_matrix.index[idx] for idx in range(len(adjacency_matrix))]
        data.x = torch.tensor(node_features.loc[node_order].values, dtype=torch.float)

        return data

    def _initialize_gnn_model(
        self,
        model_type: str,
        input_dim: int,
        hidden_dim: int,
        layer_num: int,
        dropout: bool
    ) -> nn.Module:
        """
        Initialize a GNN model based on the specified type.

        Args:
            model_type (str): Type of the GNN model.
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            layer_num (int): Number of layers.
            dropout (bool): Apply dropout.

        Returns:
            nn.Module: GNN model.
        """
        self.logger.info(f"Initializing {model_type} model.")
        if model_type == 'GCN':
            model = GCN(input_dim, hidden_dim, layer_num, dropout)
        elif model_type == 'GAT':
            model = GAT(input_dim, hidden_dim, layer_num, dropout)
        elif model_type == 'SAGE':
            model = SAGE(input_dim, hidden_dim, layer_num, dropout)
        elif model_type == 'GIN':
            model = GIN(input_dim, hidden_dim, layer_num, dropout)
        else:
            self.logger.error(f"Unsupported GNN model type: {model_type}")
            raise ValueError(f"Unsupported GNN model type: {model_type}")

        return model

    def _generate_embeddings(self, model: nn.Module, data: Data) -> torch.Tensor:
        """
        Generate embeddings using the GNN model.

        Args:
            model (nn.Module): GNN model.
            data (Data): PyTorch Geometric Data object.

        Returns:
            torch.Tensor: Node embeddings.
        """
        self.logger.info("Generating embeddings using the GNN model.")

        model.eval()
        with torch.no_grad():
            embeddings = model(data)

        return embeddings

    def _clean_column_names(self, columns: pd.Index) -> pd.Index:
        """
        Clean column names to match node names in the network.

        Args:
            columns (pd.Index): Original column names.

        Returns:
            pd.Index: Cleaned column names.
        """
        import re
        clean_columns = []
        for col in columns:
            col_clean = re.sub(r'[^0-9a-zA-Z_]', '.', col)
            if not col_clean[0].isalpha():
                col_clean = 'X' + col_clean
            clean_columns.append(col_clean)
        return pd.Index(clean_columns)


# GNN Model Classes
class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) Model.
    """

    def __init__(self, input_dim: int, hidden_dim: int, layer_num: int = 2, dropout: bool = True):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(layer_num - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network (GAT) Model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_num: int = 2,
        dropout: bool = True,
        heads: int = 1
    ):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
        for _ in range(layer_num - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x


class SAGE(nn.Module):
    """
    GraphSAGE Model.
    """

    def __init__(self, input_dim: int, hidden_dim: int, layer_num: int = 2, dropout: bool = True):
        super(SAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(layer_num - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x


class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) Model.
    """

    def __init__(self, input_dim: int, hidden_dim: int, layer_num: int = 2, dropout: bool = True):
        super(GIN, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(layer_num):
            nn_module = nn.Sequential(
                nn.Linear(hidden_dim if i > 0 else input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_module))
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        return x
