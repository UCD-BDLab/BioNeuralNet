from typing import Optional, Union
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

from ray import tune
from ray.tune import CLIReporter
from ray.air import session
from ray.tune.schedulers import ASHAScheduler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os
import json


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
        layer_num: int = 4,
        dropout: bool = True,
        num_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gpu: bool = False,
        seed: Optional[int] = None,
        tune: Optional[bool] = False,
    ):
        """
        Initializes the GNNEmbedding instance.

        Parameters:
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
        #self.random_feature_dim = random_feature_dim

        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Initialized GNNEmbedding. device={self.device}")

        self.model = None
        self.data = None
        self.embeddings = None
        self.tune = tune

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

    def embed(self, as_df: bool = False) -> Union[torch.Tensor, pd.DataFrame]:
        self.logger.info("Generating node embeddings.")
        if self.tune:
            self.logger.info("Tuning is enabled. Running hyperparameter tuning.")
            best_config = self.run_gnn_embedding_tuning(num_samples=5)
            self.logger.info(f"Best tuning config: {best_config}")
        if self.model is None or self.data is None:
            self.logger.error("Model has not been trained. Call 'fit()' before 'embed()'.")
            raise ValueError("Model has not been trained. Call 'fit()' before 'embed()'.")
        try:
            self.embeddings = self._generate_embeddings(self.model, self.data)
            self.logger.info("Node embeddings generated successfully.")
            if as_df:
                embeddings_df = self._tensor_to_df(self.embeddings, self.adjacency_matrix)
                return embeddings_df
            else:
                return self.embeddings
        except Exception as e:
            self.logger.error(f"Error during embedding generation: {e}")
            raise


    def _tensor_to_df(self, embeddings_tensor: torch.Tensor, network: pd.DataFrame) -> pd.DataFrame:
        """
        Convert embeddings tensor to DataFrame with node (feature) names as the index,
        and embedding dimension labels as columns.
        """
        try:
            self.logger.info("Converting embeddings tensor to DataFrame.")
            if embeddings_tensor is None:
                raise ValueError("Embeddings tensor is empty (None).")
            if network is None:
                raise ValueError("Network (adjacency matrix) is empty (None).")

            if embeddings_tensor.shape[0] != len(network.index):
                raise ValueError(
                    f"Mismatch: embeddings tensor has {embeddings_tensor.shape[0]} rows, "
                    f"but network index has {len(network.index)} rows."
                )
            self.logger.debug(f"Embeddings tensor shape: {embeddings_tensor.shape}")

            embeddings_df = pd.DataFrame(
                embeddings_tensor.numpy(),
                index=network.index,
                columns=[f"Embed_{i+1}" for i in range(embeddings_tensor.shape[1])]
            )
            return embeddings_df

        except Exception as e:
            self.logger.error(f"Error during conversion: {e}")
            raise
        
    def _prepare_node_features(self) -> pd.DataFrame:
        """
        Build node feature vector by computing, for each omics feature (node) in the network,
        the Pearson correlation (across patients) between the features values (from omics_data)
        and each clinical variable (from clinical_data).

        The network (adjacency matrix) is assumed to capture relationships between omics features,
        so its nodes should match the columns of the omics_data. Patient-level data (rows) in
        omics_data, phenotype_data, and clinical_data are assumed to be aligned.

        Returns:
            pd.DataFrame: A node feature matrix where rows are features (nodes) and columns are clinical variables.
        """
        self.logger.info("Preparing node features.")

        network_features = set(self.adjacency_matrix.index)
        omics_features = set(self.omics_data.columns)
        common_features = list(network_features.intersection(omics_features))
        if len(common_features) == 0:
            raise ValueError(
                "No common features found between the network and omics data."
            )
        if len(common_features) != len(network_features):
            raise ValueError(
                "Ther are duplicate features in the network or omics data."
            )
        self.logger.info(
            f"Found {len(common_features)} common features between network and omics data."
        )

        omics_in_network = self.omics_data[self.adjacency_matrix.columns]

        self.adjacency_matrix = self.adjacency_matrix.loc[
            omics_in_network.columns, omics_in_network.columns
        ]

        clinical_cols = (
            list(self.clinical_data.columns) if self.clinical_data is not None else []
        )

        node_features_list = []
        for node in common_features:
            if node not in self.omics_data.columns:
                raise ValueError(f"Feature '{node}' not found in omics_data columns.")
            corr_vector = []
            for cvar in clinical_cols:
                corr_val = self.omics_data[node].corr(self.clinical_data[cvar])
                corr_vector.append(corr_val if not pd.isna(corr_val) else 0.0)
            node_features_list.append(corr_vector)

        node_features_df = pd.DataFrame(
            node_features_list, index=common_features, columns=clinical_cols
        ).fillna(0.0)
        self.logger.info("Node features prepared based on clinical data.")
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
        self.logger.info(common_samples)

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
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
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

    def _tune_helper(self, config):
        tuned_instance = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            phenotype_data=self.phenotype_data,
            clinical_data=self.clinical_data,
            phenotype_col=self.phenotype_col,
            model_type=config.get("model_type", self.model_type),
            hidden_dim=config["hidden_dim"],
            layer_num=config["layer_num"],
            dropout=config["dropout"],
            num_epochs=config["num_epochs"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            gpu=(self.device.type == "cuda"),
            seed=None,
            tune=False,
        )
        tuned_instance.fit()
        node_emb = tuned_instance.embed()

        node_labels = self._prepare_node_labels().values
        y = (node_labels > node_labels.mean()).astype(int)
        X = node_emb.detach().numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        session.report({"accuracy": acc})

    def run_gnn_embedding_tuning(self, num_samples=10):
        """
        Runs hyperparameter tuning for the GNNEmbedding module using Ray Tune
        with accuracy as classification metric.
        """
        config = {
            "model_type": tune.choice(["GCN", "GAT", "GAT", "SAGE", "GIN"]),
            "hidden_dim": tune.choice([8, 16, 32, 64, 128]),
            "layer_num": tune.choice([2, 3, 4, 5, 6]),
            "dropout": tune.choice([True, False]),
            "num_epochs": tune.choice([16, 64, 256, 512, 1024]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-4, 1e-1),
        }

        scheduler = ASHAScheduler(
            metric="accuracy", mode="max", grace_period=1, reduction_factor=2
        )
        reporter = CLIReporter(metric_columns=["accuracy", "training_iteration"])

        def short_dirname_creator(trial):
            return f"_{trial.trial_id}"

        result = tune.run(
            tune.with_parameters(self._tune_helper),
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            verbose=0,
            progress_reporter=reporter,
            storage_path=os.path.expanduser("~/gnn"),
            trial_dirname_creator=short_dirname_creator,
            name="e",
        )

        best_trial = result.get_best_trial("accuracy", "max", "last")
        self.logger.info(f"Best trial config: {best_trial.config}")
        self.logger.info(f"Best trial final accuracy: {best_trial.last_result['accuracy']}")
        
        best_params_file = "embeddings_best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(best_trial.config, f, indent=4)
        self.logger.info(f"Best embedding parameters saved to {best_params_file}")
        
        return best_trial.config
