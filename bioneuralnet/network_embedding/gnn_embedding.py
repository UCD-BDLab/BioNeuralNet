from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from .gnn_models import GCN, GAT, SAGE, GIN
from ..utils.logger import get_logger


class GNNEmbedding:
    """
    GNNEmbedding Class for Generating Graph Neural Network (GNN) Based Embeddings.
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        omics_data: pd.DataFrame,
        phenotype_data: pd.DataFrame,
        clinical_data: Optional[pd.DataFrame] = None,
        phenotype_col: str = "phenotype",
        model_type: str = "GAT",
        hidden_dim: int = 64,
        layer_num: int = 2,
        dropout: bool = True,
        num_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gpu: bool = False,
        random_feature_dim: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Initializes the GNNEmbedding instance.

        Parameters
        adjacency_matrix : pd.DataFrame
        omics_data : pd.DataFrame
        phenotype_data : pd.DataFrame
        clinical_data : Optional[pd.DataFrame], default=None
        phenotype_col : str, optional
        model_type : str, optional
        hidden_dim : int, optional
        layer_num : int, optional
        dropout : bool, optional
        num_epochs : int, optional
        lr : float, optional
        weight_decay : float, optional
        gpu : bool, optional
        random_feature_dim : int, optional
        seed : Optional[int], default=None
        """
        self.logger = get_logger(__name__)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # input validation
        if adjacency_matrix.empty:
            raise ValueError("Adjacency matrix cannot be empty.")
        if omics_data.empty:
            raise ValueError("Omics data cannot be empty.")
        if phenotype_data.empty or phenotype_col not in phenotype_data.columns:
            raise ValueError(f"Phenotype data must have column '{phenotype_col}'.")
        if clinical_data is not None and clinical_data.empty:
            self.logger.warning(
                "Clinical data provided is empty ... using random features."
            )
            clinical_data = None

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
        self.random_feature_dim = random_feature_dim

        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.logger.info(
            f"Initialized GNNEmbedding for regression. device={self.device}"
        )

        self.model = None
        self.data = None
        self.embeddings = None

    def fit(self) -> None:
        """
        Trains the GNN model using the provided data.
        """
        self.logger.info("Starting training process.")
        try:
            node_features = self._prepare_node_features()
            node_labels = self._prepare_node_labels()
            self.data = self._build_pyg_data(node_features, node_labels)
            self.model = self._initialize_gnn_model().to(self.device)
            self._train_gnn(self.model, self.data)
            self.logger.info("Training completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    def embed(self) -> torch.Tensor:
        """
        Generates and retrieves node embeddings from the trained GNN model.

        Returns:
            torch.Tensor

        Raises:
            ValueError
        """
        self.logger.info("Generating node embeddings.")
        if self.model is None or self.data is None:
            self.logger.error(
                "Model has not been trained. Call 'fit()' before 'embed()'."
            )
            raise ValueError(
                "Model has not been trained. Call 'fit()' before 'embed()'."
            )

        try:
            self.embeddings = self._generate_embeddings(self.model, self.data)
            self.logger.info("Node embeddings generated successfully.")
            return self.embeddings
        except Exception as e:
            self.logger.error(f"Error during embedding generation: {e}")
            raise

    def _prepare_node_features(self) -> pd.DataFrame:
        """
        Build node features by correlating each omics feature with each clinical variable
        (if clinical data is provided). Otherwise, initialize node features randomly.

        Returns:
            pd.DataFrame
        """
        self.logger.info("Preparing node features.")
        node_names = self.adjacency_matrix.index.tolist()

        if self.clinical_data is not None:
            self.logger.info(
                "Clinical data provided. Preparing node features based on correlations."
            )
            common_samples = self.omics_data.index.intersection(
                self.phenotype_data.index
            ).intersection(self.clinical_data.index)
            if len(common_samples) == 0:
                raise ValueError(
                    "No common samples among omics, phenotype, and clinical data."
                )

            omics_filtered = self.omics_data.loc[common_samples]
            clinical_filtered = self.clinical_data.loc[common_samples]
            clinical_cols = clinical_filtered.columns.tolist()

            node_features_list = []
            for node in node_names:
                if node not in omics_filtered.columns:
                    raise ValueError(f"Node '{node}' not found in omics_data.")
                corr_vector = []
                for cvar in clinical_cols:
                    corr_val = omics_filtered[node].corr(clinical_filtered[cvar])
                    corr_vector.append(corr_val if not pd.isna(corr_val) else 0.0)
                node_features_list.append(corr_vector)

            node_features_df = pd.DataFrame(
                node_features_list, index=node_names, columns=clinical_cols
            ).fillna(0.0)
            self.logger.info("Node features prepared based on clinical data.")
            return node_features_df

        else:
            self.logger.info(
                "No clinical data provided. Initializing node features randomly."
            )
            num_nodes = len(node_names)
            num_features = self.random_feature_dim
            random_features = np.random.rand(num_nodes, num_features)
            node_features_df = pd.DataFrame(
                random_features,
                index=node_names,
                columns=[f"RandomFeat_{i+1}" for i in range(num_features)],
            )
            self.logger.info(
                f"Random node features initialized with shape {node_features_df.shape}."
            )
            return node_features_df

    def _prepare_node_labels(self) -> pd.Series:
        """
        Build node labels by correlating each omics feature with the specified phenotype column.

        Returns:
            pd.Series
        """
        self.logger.info(
            f"Preparing node labels by correlating each omics feature with phenotype column '{self.phenotype_col}'."
        )
        common_samples = self.omics_data.index.intersection(self.phenotype_data.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between omics data and phenotype data.")

        omics_filtered = self.omics_data.loc[common_samples]
        phen_filtered = self.phenotype_data.loc[common_samples, self.phenotype_col]

        labels_dict = {}
        node_names = self.adjacency_matrix.index.tolist()
        for node in node_names:
            if node not in omics_filtered.columns:
                raise ValueError(f"Node '{node}' not found in omics_data columns.")
            corr_val = omics_filtered[node].corr(phen_filtered)
            labels_dict[node] = corr_val if not pd.isna(corr_val) else 0.0

        labels_series = pd.Series(labels_dict, index=node_names).fillna(0.0)
        self.logger.info("Node labels prepared successfully.")
        return labels_series

    def _build_pyg_data(
        self, node_features: pd.DataFrame, node_labels: pd.Series
    ) -> Data:
        """
        Construct a PyTorch Geometric Data object:

        - data.x = node_features
        - data.y = node_labels
        - data.edge_index from adjacency

        Returns:
            PyG Data object with x, y, edge_index.
        """
        self.logger.info("Constructing PyTorch Geometric Data object.")
        G = nx.from_pandas_adjacency(self.adjacency_matrix)
        node_mapping = {name: i for i, name in enumerate(node_features.index)}
        G = nx.relabel_nodes(G, node_mapping)

        data = from_networkx(G)
        node_order = list(node_features.index)
        data.x = torch.tensor(node_features.loc[node_order].values, dtype=torch.float)
        data.y = torch.tensor(node_labels.loc[node_order].values, dtype=torch.float)
        self.logger.info("PyTorch Geometric Data object constructed successfully.")
        return data

    def _initialize_gnn_model(self) -> nn.Module:
        """
        Initialize the GNN model based on the specified type.

        Returns:
            nn.Module

        """
        self.logger.info(
            f"Initializing GNN model of type '{self.model_type}' with hidden_dim={self.hidden_dim} and layer_num={self.layer_num}."
        )

        if self.data is None or not hasattr(self.data, "x") or self.data.x is None:
            raise ValueError("Data is not initialized or is missing the 'x' attribute.")

        input_dim = self.data.x.shape[1]

        if self.model_type.upper() == "GCN":
            return GCN(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                layer_num=self.layer_num,
                dropout=self.dropout,
            )
        elif self.model_type.upper() == "GAT":
            return GAT(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                layer_num=self.layer_num,
                dropout=self.dropout,
            )
        elif self.model_type.upper() == "SAGE":
            return SAGE(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                layer_num=self.layer_num,
                dropout=self.dropout,
            )
        elif self.model_type.upper() == "GIN":
            return GIN(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                layer_num=self.layer_num,
                dropout=self.dropout,
            )
        else:
            self.logger.error(f"Unsupported model_type: {self.model_type}")
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _train_gnn(self, model: nn.Module, data: Data) -> None:
        """
        Train the GNN model using MSE loss.
        """
        self.logger.info("Starting GNN training process.")
        data = data.to(self.device)
        model.to(self.device)
        model.train()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        for epoch in range(1, self.num_epochs + 1):
            optimizer.zero_grad()
            out = model(data)
            out = out.view(-1)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(
                    f"Epoch [{epoch}/{self.num_epochs}], MSE Loss: {loss.item():.4f}"
                )

        self.logger.info("GNN training process completed.")

    def _generate_embeddings(self, model: nn.Module, data: Data) -> torch.Tensor:
        """
        Retrieve node embeddings from the penultimate layer of the trained GNN model.

        Returns:
            torch.Tensor
        """
        self.logger.info("Generating node embeddings from the trained GNN model.")
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            embeddings = model.get_embeddings(data)
        return embeddings.cpu()
