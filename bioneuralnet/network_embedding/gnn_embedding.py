import os
import json
import pandas as pd
import networkx as nx
import numpy as np
from typing import Optional,Union
from datetime import datetime
from pathlib import Path
import tempfile

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx
    from torch_geometric.utils import add_self_loops
    from torch_geometric.utils import degree
except ModuleNotFoundError:
    raise ImportError(
        "GNNEmbedding requires PyTorch Geometric. "
        "Please install it by following the instructions at: "
        "https://bioneuralnet.readthedocs.io/en/latest/installation.html"
    )

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from .gnn_models import GCN, GAT, SAGE, GIN, process_dropout
from ..utils.logger import get_logger
from scipy.stats import skew

class GNNEmbedding:
    """
    GNNEmbedding Class for Generating Graph Neural Network (GNN) Based Embeddings.

    Attributes:

        adjacency_matrix : pd.DataFrame
        omics_data : pd.DataFrame
        phenotype_data : pd.DataFrame
        clinical_data : Optional[pd.DataFrame]
        phenotype_col : str
        model_type : str
        hidden_dim : int
        layer_num : int
        dropout: Union[bool, float] (if bool, True maps to 0.5, False to 0.0)
        num_epochs : int
        lr : float
        weight_decay : float
        gpu : bool
        seed : Optional[int]
        tune : Optional[bool]

    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        omics_data: pd.DataFrame,
        phenotype_data: Union[pd.Series, pd.DataFrame],
        clinical_data: Optional[pd.DataFrame] = None,
        phenotype_col: str = "phenotype",
        model_type: str = "GAT",
        hidden_dim: int = 64,
        layer_num: int = 4,
        dropout: Union[bool, float] = True,
        num_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gpu: bool = False,
        activation: str = "relu",
        seed: Optional[int] = None,
        tune: Optional[bool] = False,
        output_dir: Optional[Union[str, Path]] = None,
        ):

        """
        Initializes the GNNEmbedding instance.
        """
        self.logger = get_logger(__name__)

        # Input validation
        if adjacency_matrix.empty:
            raise ValueError("Adjacency matrix cannot be empty.")
        if omics_data.empty:
            raise ValueError("Omics data cannot be empty.")
        if clinical_data is not None and clinical_data.empty:
            self.logger.warning("Clinical data was provided but is empty, setting to None.")
            clinical_data = None

        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        if not adjacency_matrix.index.equals(adjacency_matrix.columns):
            raise ValueError("Adjacency matrix must have the same index and columns.")
        if omics_data.shape[1] != adjacency_matrix.shape[0]:
            raise ValueError("Number of omics features must match adjacency matrix size.")

        if isinstance(phenotype_data, pd.Series):
            self.phenotype_data = phenotype_data.copy(deep=True)

        elif isinstance(phenotype_data, pd.DataFrame):
            if phenotype_col and phenotype_col in phenotype_data.columns:
                self.phenotype_data = phenotype_data[phenotype_col].copy(deep=True)
            elif phenotype_data.shape[1] == 1:
                self.phenotype_data = phenotype_data.iloc[:, 0].copy(deep=True)
            else:
                raise ValueError(f"Cannot determine phenotype column. "f"Either provide a single-column DataFrame or set 'phenotype_col' to a valid column name.")
        else:
            raise ValueError("Phenotype data must be a Series or a DataFrame.")

        #lastly we check if phenotype columns is in omics data
        if phenotype_col in omics_data.columns:
            raise ValueError(f"Phenotype column '{phenotype_col}' is present in omics data. Please remove it.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.seed = seed
        self.adjacency_matrix = adjacency_matrix.copy(deep=True)
        self.omics_data = omics_data.copy(deep=True)
        self.clinical_data = clinical_data.copy(deep=True) if clinical_data is not None else None
        self.phenotype_col = phenotype_col

        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.dropout = process_dropout(dropout)
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.gpu = gpu

        self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized GNNEmbedding. device={self.device}")

        self.model = None
        self.data = None
        self.embeddings = None
        self.tune = tune

        if output_dir is None:
            self.temp_dir_obj = tempfile.TemporaryDirectory()
            self.output_dir = Path(self.temp_dir_obj.name)
            self.logger.info(f"No output_dir provided; using temporary directory: {self.output_dir}")
        else:
            self.output_dir = Path(output_dir)
            self.logger.info(f"Output directory set to: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)

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
        """
        Generates node embeddings.
        If tuning is enabled, runs hyperparameter tuning and uses the best configuration.
        """
        self.logger.info("Generating node embeddings.")
        if not self.tune and (self.model is None or self.data is None):
            raise ValueError("Model not trained. Call fit() first.")

        if self.tune:
            self.logger.info("Tuning is enabled. Running hyperparameter tuning.")
            best_config = self.run_gnn_embedding_tuning()
            self.logger.info(f"Best tuning config: {best_config}")

            self.model_type = best_config["model_type"]
            self.hidden_dim = best_config["hidden_dim"]
            self.layer_num = best_config["layer_num"]
            self.dropout = best_config["dropout"]
            self.num_epochs = best_config["num_epochs"]
            self.lr = best_config["lr"]
            self.weight_decay = best_config["weight_decay"]
            self.activation = best_config["activation"]
            self.tune = False

            self.logger.info(f"Retraining with best config: {best_config}")
            self.fit()
            self.logger.info("Model retrained with best hyperparameters.")

        try:
            self.embeddings = self._generate_embeddings(self.model, self.data)
            self.logger.info("Node embeddings generated.")
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
        Convert embeddings tensor to a DataFrame with node (feature) names as the index,
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
        Prepare node features by computing centrality measures, statistical features and/or correlation.
        1. Align network & omics nodes.
        2. Compute graph-centralities (pagerank, eigenvector, katz).
        3. If clinical_data exists:

            compute Pearson correlations vs. each clinical var.
        Else:

            compute mean, log-skew, median-abs-dev of omics per node.
        4. Standardize every feature to zero mean and unit variance.
        5. Save and return the final features DataFrame.
        """
        self.logger.info("Preparing node features.")

        network_features = self.adjacency_matrix.columns
        nodes = sorted(network_features.intersection(self.omics_data.columns))

        if len(nodes) == 0:
            raise ValueError("No common features found between the network and omics data.")

        if len(nodes) != len(network_features):
            missing = set(network_features) - set(nodes)
            self.logger.warning(f"Length of common features: {len(nodes)}")
            self.logger.warning(f"Length of network features: {len(network_features)}")
            self.logger.warning(f"Missing features: {missing}")
            raise ValueError("Mismatch between network features and omics data columns.")

        self.logger.info(f"Found {len(nodes)} common features between network and omics data.")
        omics_filtered = self.omics_data[nodes]
        network_filtered = self.adjacency_matrix.loc[nodes, nodes]

        G = nx.from_pandas_adjacency(network_filtered)

        pagerank = nx.pagerank(G, weight="weight")
        katz = nx.katz_centrality_numpy(G, weight="weight")

        eigenvector = {}
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            try:
                ec = nx.eigenvector_centrality(sub, weight="weight")
            except nx.PowerIterationFailedConvergence:
                ec = {}
                self.logger.warning(
                    f"Eigenvector centrality failed for component size {len(sub)}; defaulting to 0."
                )
                for node in sub.nodes():
                    ec[node] = 0.0

            eigenvector.update(ec)

        nodes = list(network_filtered.index)
        centralities_df = pd.DataFrame({"pagerank": pagerank,"eigenvector": eigenvector,"katz": katz}).reindex(nodes)

        #due to the different magnitudes from centralities and stats, we need to scale them
        centralities_df = (centralities_df - centralities_df.mean()) / centralities_df.std()

        if self.clinical_data is not None and not self.clinical_data.empty:
            clinical_cols = list(self.clinical_data.columns)
            common_index = self.clinical_data.index.intersection(omics_filtered.index)
            if common_index.empty:
                raise ValueError("No common indices between omics and clinical data.")

            node_features_dict = {}
            for node in nodes:
                vec = pd.to_numeric(omics_filtered[node].loc[common_index], errors="coerce")
                corr_vector = {}
                for cvar in clinical_cols:
                    clinical_series = self.clinical_data[cvar].loc[common_index]
                    corr_val = vec.corr(clinical_series)
                    if pd.isna(corr_val):
                        z = 0.0
                    else:
                        r_clip = np.clip(corr_val, -0.999999, 0.999999)
                        z = np.arctanh(r_clip)
                    corr_vector[cvar] = z

                full_feature_vec = {
                    "pagerank": centralities_df.at[node, "pagerank"],
                    "eigenvector": centralities_df.at[node, "eigenvector"],
                    "katz": centralities_df.at[node, "katz"],
                }

                full_feature_vec.update(corr_vector)
                node_features_dict[node] = full_feature_vec


            node_features_df = pd.DataFrame.from_dict(node_features_dict, orient="index")
            self.logger.info(f"Built feature matrix with clinical correlations shape: {node_features_df.shape}")

        else:
            self.logger.warning("No clinical data found. Using centrality measures and statistical features.")
            if self.phenotype_data is None or self.phenotype_data.empty:
                raise ValueError("No phenotype data available for statistical features.")
            pheno = self.phenotype_data.loc[omics_filtered.index].dropna()

            stat_features = {}
            for node in nodes:
                vec = omics_filtered[node].loc[pheno.index].dropna()
                if vec.empty:
                    mean_val = np.nan
                    skew_val = np.nan
                    mad_val = np.nan
                    log_skew_val = np.nan
                else:
                    mean_val = vec.mean()
                    skew_val = skew(vec)
                    log_skew_val = np.sign(skew_val) * np.log1p(abs(skew_val)) if not np.isnan(skew_val) else 0.0
                    mad_val = np.median(np.abs(vec - np.median(vec)))
                stat_features[node] = {"mean": mean_val, "log_skew": log_skew_val, "mad": mad_val}

            stat_df = pd.DataFrame.from_dict(stat_features, orient="index")

            node_features_df = stat_df.join(centralities_df)
            self.logger.info(f"Built statistical feature matrix shape: {node_features_df.shape}")

        # ranks = node_features_df.rank(method="average")
        # scale_den = (ranks.max() - ranks.min()).replace(0, 1)
        # scaled_ranks = 2 * (ranks - ranks.min()) / scale_den - 1
        # node_features_df = scaled_ranks
        node_features_df = (node_features_df - node_features_df.mean()) / node_features_df.std()

        timestamp = datetime.now().strftime("%m%d_%H_%M_%S")
        labels_file = self.output_dir / f"features_{network_filtered.shape[0]}_{timestamp}.txt"
        with open(labels_file, "w") as f:
            f.write(node_features_df.to_string())

        self.logger.info(f"Node features prepared successfully and saved to {labels_file}.")

        return node_features_df


    def _prepare_node_labels(self) -> pd.Series:
        """
        Build node labels using either Pearson correlation OR mutual information
        between each omics feature and the specified phenotype column.
        """
        self.logger.info("Preparing node labels.")
        nodes = sorted(self.adjacency_matrix.index.intersection(self.omics_data.columns))

        samples = self.omics_data.index.intersection(self.phenotype_data.index)
        omics_data =  self.omics_data.loc[samples, nodes]
        pheno = self.phenotype_data.loc[samples]

        if len(samples)==0:
            raise ValueError("No overlapping samples between omics and phenotype.")
        if len(nodes)==0:
            raise ValueError("No overlapping nodes between adjacency and omics.")

        labels_dict = {}

        for node in nodes:
            r = pd.to_numeric(omics_data[node], errors="coerce").corr(pheno)
            r = 0.0 if pd.isna(r) else np.clip(r, -0.999999, 0.999999)
            z = np.arctanh(r)
            labels_dict[node] = z

        #labels_series = pd.Series(labels_dict, index=nodes)
        # ranks = labels_series.rank(method="average")
        # den = (ranks.max() - ranks.min()) or 1
        # scaled = 2*(ranks - ranks.min())/den - 1
        labels = pd.Series(labels_dict, index=nodes)
        labels = (labels - labels.mean()) / labels.std()

        timestamp = datetime.now().strftime("%m%d_%H_%M_%S")
        labels_file = self.output_dir / f"labels_{self.adjacency_matrix.shape[0]}_{timestamp}.txt"

        with open(labels_file, "w") as f:
            f.write(labels.to_string())

        self.logger.info(f"Node labels prepared successfully and saved to {labels_file}.")

        return labels

    def _build_pyg_data(self, node_features: pd.DataFrame, node_labels: pd.Series) -> Data:
        self.logger.info("Constructing PyTorch Geometric Data object.")
        if not node_labels.index.equals(node_features.index):
            raise ValueError("`node_labels` must have the same index and order as `node_features`.")

        nodes = node_features.index
        adj = self.adjacency_matrix.loc[nodes, nodes]

        G = nx.from_pandas_adjacency(adj)
        node_mapping = {name: i for i, name in enumerate(nodes)}
        G = nx.relabel_nodes(G, node_mapping)

        data = from_networkx(G)
        data.num_nodes = len(nodes)

        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            data.edge_weight = edge_attr.view(-1)
            del data.edge_attr
        else:
            # no edge_attr
            data.edge_weight = torch.ones(data.edge_index.size(1))

        data.edge_index, data.edge_weight = add_self_loops(data.edge_index,data.edge_weight,fill_value=data.edge_weight.mean(),num_nodes=data.num_nodes)
        row, col = data.edge_index
        deg = degree(row, data.num_nodes, dtype=data.edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_w = deg_inv_sqrt[row] * data.edge_weight * deg_inv_sqrt[col]
        data.edge_weight = norm_w

        data.x = torch.tensor(node_features.loc[nodes].values, dtype=torch.float)
        data.y = torch.tensor(node_labels.loc[nodes].values, dtype=torch.float)

        self.logger.info("PyTorch Geometric Data object constructed successfully.")
        return data

    def _initialize_gnn_model(self) -> nn.Module:
        """
        Initialize the GNN model based on the specified type.

        Returns:

            nn.Module
        """
        self.logger.info(f"Initializing GNN model of type '{self.model_type}' with hidden_dim={self.hidden_dim} and layer_num={self.layer_num}.")
        if self.data is None or not hasattr(self.data, "x") or self.data.x is None:
            raise ValueError("Data is not initialized or is missing the 'x' attribute.")
        input_dim = self.data.x.shape[1]
        if self.model_type.upper() == "GCN":
            return GCN(input_dim=input_dim, hidden_dim=self.hidden_dim, layer_num=self.layer_num, dropout=self.dropout,activation=self.activation, seed = self.seed, self_loop_and_norm=False)
        elif self.model_type.upper() == "GAT":
            return GAT(input_dim=input_dim, hidden_dim=self.hidden_dim, layer_num=self.layer_num, dropout=self.dropout,activation=self.activation, seed = self.seed, self_loop_and_norm=False)
        elif self.model_type.upper() == "SAGE":
            return SAGE(input_dim=input_dim, hidden_dim=self.hidden_dim, layer_num=self.layer_num, dropout=self.dropout,activation=self.activation, seed = self.seed, self_loop_and_norm=False)
        elif self.model_type.upper() == "GIN":
            return GIN(input_dim=input_dim, hidden_dim=self.hidden_dim, layer_num=self.layer_num, dropout=self.dropout, activation=self.activation, seed = self.seed, self_loop_and_norm=False)
        else:
            self.logger.error(f"Unsupported model_type: {self.model_type}")
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _train_gnn(self, model: nn.Module, data: Data) -> None:
        """
        Train the GNN model using MSE loss to predict node labels with early stopping.
        """
        self.logger.info("Starting GNN training with early stopping.")

        data = data.to(self.device)
        model = model.to(self.device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        patience = 50
        counter = 0

        for epoch in range(1, self.num_epochs + 1):
            optimizer.zero_grad()

            output = model(data)
            output = output.view(-1)
            target = data.y.to(self.device)

            loss = loss_fn(output, target)

            if torch.isnan(loss):
                self.logger.error("NaN loss encountered. Stopping early.")
                break

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1

            if epoch % 100 == 0 or epoch == 1 or epoch == self.num_epochs:
                self.logger.info(f"Epoch [{epoch}/{self.num_epochs}] | Loss: {loss.item():.4f} | EarlyStop: {counter}/{patience}")

            if counter >= patience:
                self.logger.warning(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.4f}")
                break

        self.logger.info("GNN training finished.")

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
        """
        The function that each Ray Tune trial calls.
        """
        try:
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
                gpu=self.device.type,
                seed=self.seed,
                tune=False,
                activation=config["activation"],
                output_dir=self.output_dir,
            )

            tuned_instance.fit()
            node_embeddings = tuned_instance.embed()

            X = node_embeddings.detach().cpu().numpy()

            dim_stds = np.std(X, axis=0)
            keep_dims = dim_stds >= 1e-5
            num_dims_kept = np.sum(keep_dims)

            if num_dims_kept == 0:
                self.logger.warning("All embedding dimensions are nearly constant. Discarding trial.")
                tune.report({"mse": 1e8,"composite_score": 1e8,"mean_dim_std": 0.0})
                return

            X = X[:, keep_dims]
            new_dim_stds = dim_stds[keep_dims]
            mean_dim_std = np.mean(new_dim_stds)

            y = tuned_instance._prepare_node_labels().values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.seed)
            reg = RandomForestRegressor()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            composite_score = mse / (mean_dim_std + 1e-6)
            tune.report({
                "mse": mse,
                "composite_score": composite_score,
                "mean_dim_std": mean_dim_std,
                "dims_original": len(dim_stds),
                "dims_dropped": int(len(dim_stds) - num_dims_kept)
            })

        except Exception as e:
            self.logger.error(f"[Tuning Trial Error] config={config}")
            self.logger.error(f"[Tuning Trial Error] Exception: {e}")
            import traceback
            traceback.print_exc()
            tune.report({"mse": 1e8, "composite_score": 1e8, "mean_dim_std": 0.0})

    def run_gnn_embedding_tuning(self, num_samples=20):
        """
        Run hyperparameter tuning with Ray Tune.
        """
        num_nodes = self.adjacency_matrix.shape[0]
        config = {
            "model_type": tune.choice(["GAT","GCN","SAGE","GIN"]),
            "hidden_dim": tune.choice([16, 32, 64, 128, 256, 512, 1024]),
            "layer_num": tune.choice([2, 3, 4, 5, 6, 7]),
            "dropout": tune.choice([0.0, 0.2, 0.4, 0.6]),
            "num_epochs": tune.choice([256, 512, 1024, 2048, 4096]),
            "lr": tune.loguniform(1e-6, 1e-2),
            "weight_decay": tune.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
            "activation": tune.choice(["relu", "elu", "leaky_relu"]),
        }

        scheduler = ASHAScheduler(metric="mse", mode="min", grace_period=1, reduction_factor=2)
        reporter = CLIReporter(metric_columns=["mse", "composite_score", "mean_dim_std", "training_iteration"])

        def short_dirname_creator(trial):
            return f"_{trial.trial_id}"

        resources = {"cpu": 1, "gpu": 1} if self.device.type == "cuda" else {"cpu": 1, "gpu": 0}

        result = tune.run(
            tune.with_parameters(self._tune_helper),
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            verbose=1,
            progress_reporter=reporter,
            storage_path=os.path.expanduser("~/gnn"),
            trial_dirname_creator=short_dirname_creator,
            resources_per_trial=resources,
            name="e",
        )

        timestamp = datetime.now().strftime("%m%d_%H_%M_%S")
        save_dir = Path(self.output_dir)/"tuning_results"
        os.makedirs(save_dir, exist_ok=True)

        # we grab the logged results
        try:
            df = result.get_dataframe()
        except AttributeError:
            df = result.dataframe()

        # only pull the last row per trial
        df_last = (df.sort_values("training_iteration").groupby("trial_id", as_index=False).last())
        K = min(3, len(df_last))
        top_k = df_last.nsmallest(K, "mse")

        # lowest composite_score among the top K trials
        best_row = top_k.loc[top_k.composite_score.idxmin()]
        best_trial_id = best_row["trial_id"]
        best_trial = None
        for trial in result.trials:
            if trial.trial_id == best_trial_id:
                best_trial = trial
                break

        if best_trial is None:
            raise RuntimeError(f"Could not find trial with id {best_trial_id}")

        summary_file = save_dir / f"summary_{num_nodes}_{timestamp}.txt"

        with open(summary_file, "w") as f:
            f.write(f"Top {K} by MSE:\n")
            f.write(top_k.to_string(index=False))
            f.write("\n\nFinal pick:\n")
            f.write(json.dumps(best_trial.config, indent=4))

        self.logger.info(f"Full trial summary saved to {summary_file}")

        # best trial results
        self.logger.info(f"Best trial config: {best_trial.config}")
        self.logger.info(f"Best trial final MSE: {best_trial.last_result['mse']}")

        # best config as a JSON file
        timestamp = datetime.now().strftime("%m%d_%H_%M_%S")
        best_params_file = save_dir / f"emb_tuned_{num_nodes}_{timestamp}.json"
        with open(best_params_file, "w") as f:
            json.dump(best_trial.config, f, indent=4)

        self.logger.info(f"Best embedding parameters saved to {best_params_file}")

        return best_trial.config
