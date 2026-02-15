from __future__ import annotations

import os
import re
import statistics
import tempfile
import shutil
import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional, List
from pathlib import Path
from typing import Optional, List, Tuple
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.data import Data
except ModuleNotFoundError:
    raise ImportError(
        "DPMON (Disease Prediction using Multi-Omics Networks) requires PyTorch Geometric. "
        "Please install it by following the instructions at: "
        "https://bioneuralnet.readthedocs.io/en/latest/installation.html"
    )

from ray import train
from ray import tune
from ray.tune import Checkpoint
from ray.tune import CLIReporter
from ray.tune.error import TuneError
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from sklearn.model_selection import train_test_split,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.preprocessing import label_binarize
from scipy.stats import pointbiserialr
from sklearn.metrics import f1_score, roc_auc_score, recall_score,average_precision_score, matthews_corrcoef

from bioneuralnet.utils import get_logger
from bioneuralnet.utils import set_seed
from bioneuralnet.network_embedding import GCN, GAT, SAGE, GIN

logger= get_logger(__name__)

class DPMON:
    """DPMON (Disease Prediction using Multi-Omics Networks) end-to-end pipeline for multi-omics disease prediction.

    Instead of node-level MSE regression, DPMON aggregates node embeddings with patient-level omics data and feeds them to a downstream classification head (e.g., a softmax layer with CrossEntropyLoss) for sample-level disease prediction. This end-to-end setup leverages both local (node-level) and global (patient-level) network information.

    Attributes:

        adjacency_matrix (pd.DataFrame): Adjacency matrix of the feature-level network; index/columns are feature names.
        omics_list (List[pd.DataFrame] | pd.DataFrame): List of omics data matrices or a single merged omics DataFrame (samples x features).
        phenotype_data (pd.DataFrame | pd.Series): Phenotype labels used for supervision.
        clinical_data (Optional[pd.DataFrame]): Optional clinical covariates (samples x clinical features); may be None.
        phenotype_col (str): Column name in phenotype_data that stores the target labels.
        model (str): GNN backbone; one of {"GCN", "GAT", "SAGE", "GIN"}.
        gnn_hidden_dim (int): Hidden dimension size of GNN layers.
        gnn_layer_num (int): Number of stacked GNN layers.
        gnn_dropout (float): Dropout rate applied within the GNN.
        gnn_activation (str): Non-linear activation used in GNN layers (e.g., "relu").
        dim_reduction (str): Dimensionality reduction strategy for omics input (e.g., "ae" for autoencoder).
        ae_encoding_dim (int): Encoding dimension of the autoencoder bottleneck if dim_reduction="ae".
        nn_hidden_dim1 (int): Hidden dimension of the first fully connected layer in the downstream classifier.
        nn_hidden_dim2 (int): Hidden dimension of the second fully connected layer in the downstream classifier.
        num_epochs (int): Number of training epochs per run.
        repeat_num (int): Number of repeated training runs (for repeated train/test splits or repeated CV).
        n_folds (int): Number of folds to use when cv=True.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): L2 weight decay (regularization) coefficient.
        tune (bool): If True, perform hyperparameter tuning before final training.
        tune_trails (int): Number of trials to perform if tune=True.
        gpu (bool): If True, use GPU if available.
        cv (bool): If True, use K-fold cross-validation; otherwise use repeated train/test splits.
        cuda (int): CUDA device index to use when gpu=True.
        seed (int): Random seed for reproducibility.
        seed_trials (bool): If True, use a fixed seed for hyperparameter sampling to ensure reproducibility across trials.
        output_dir (Path): Directory where logs, checkpoints, and results are written.
    """
    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        omics_list: List[pd.DataFrame],
        phenotype_data: pd.DataFrame,
        clinical_data: Optional[pd.DataFrame] = None,
        model: str = "GAT",
        phenotype_col: str = "phenotype",
        gnn_hidden_dim: int = 16,
        gnn_layer_num: int = 4,
        gnn_dropout: float = 0.1,
        gnn_activation: str = "relu",
        dim_reduction: str = "ae",
        ae_encoding_dim: int = 8,
        nn_hidden_dim1: int = 16,
        nn_hidden_dim2: int = 8,
        num_epochs: int = 100,
        repeat_num: int = 1,
        n_folds: int = 5,
        lr: float = 1e-1,
        weight_decay: float = 1e-4,
        tune: bool = False,
        tune_trails: int = 10,
        gpu: bool = False,
        cv: bool = False,
        cuda: int = 0,
        seed: int = 1804,
        seed_trials: bool = False,
        output_dir: Optional[str] = None,
    ):
        if adjacency_matrix.empty:
            raise ValueError("Adjacency matrix cannot be empty.")

        if isinstance(omics_list, list):
            if not omics_list or any(df.empty for df in omics_list):
                raise ValueError("All provided omics data files must be non-empty.")
            self.combined_omics_input = pd.concat(omics_list, axis=1)
        elif isinstance(omics_list, pd.DataFrame):
            if omics_list.empty:
                raise ValueError("Provided omics DataFrame cannot be empty.")
            self.combined_omics_input = omics_list
        else:
            raise TypeError("omics_list must be a pandas DataFrame or a list of DataFrames.")

        if isinstance(phenotype_data, pd.DataFrame):
            if phenotype_data.empty or phenotype_col not in phenotype_data.columns:
                raise ValueError(f"Phenotype DataFrame must have a '{phenotype_col}' column.")
            self.phenotype_series = phenotype_data[phenotype_col]
        elif isinstance(phenotype_data, pd.Series):
            if phenotype_data.empty:
                raise ValueError("Phenotype Series cannot be empty.")
            self.phenotype_series = phenotype_data
        else:
            raise TypeError("phenotype_data must be a pandas DataFrame or Series.")

        if clinical_data is not None and clinical_data.empty:
            logger.warning(
                "Clinical data provided is empty => treating as None => random features."
            )
            clinical_data = None

        self.adjacency_matrix = adjacency_matrix
        self.omics_list = omics_list
        self.phenotype_data = phenotype_data
        self.clinical_data = clinical_data
        self.phenotype_col = phenotype_col
        self.model = model
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_layer_num = gnn_layer_num
        self.gnn_dropout = gnn_dropout
        self.gnn_activation = gnn_activation
        self.dim_reduction = dim_reduction
        self.ae_encoding_dim = ae_encoding_dim
        self.nn_hidden_dim1 = nn_hidden_dim1
        self.nn_hidden_dim2 = nn_hidden_dim2
        self.num_epochs = num_epochs
        self.repeat_num = repeat_num
        self.n_folds = n_folds
        self.lr = lr
        self.weight_decay = weight_decay
        self.tune = tune
        self.tune_trails = tune_trails
        self.gpu = gpu
        self.cuda = cuda
        self.seed = seed
        self.seed_trials = seed_trials
        self.cv = cv

        if output_dir is None:
            self.output_dir = Path(os.getcwd()) / "dpmon"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")
        logger.info(f"Initialized DPMON with model: {self.model}")

    def run(self) -> Tuple[pd.DataFrame, object, torch.Tensor | None]:
        """Execute the DPMON pipeline.

        This method aligns the graph and omics features, optionally performs hyperparameter tuning, and then trains and evaluates the chosen GNN model using either K-fold cross-validation (cv=True) or repeated train/test splits (cv=False). It returns prediction outputs, a metrics/config object, and optionally the learned embeddings.

        Returns:

            Tuple[pd.DataFrame, object, torch.Tensor | None]: A tuple (predictions_df, metrics, embeddings) where:
                predictions_df (pd.DataFrame): If cv=False, per-sample predictions with actual vs predicted labels; if cv=True, aggregated CV performance or fold-level results depending on the backend
                metrics (object): Dictionary or configuration object containing evaluation metrics and, when tuning is enabled, information about the selected hyperparameters.
                embeddings (torch.Tensor | None): Learned embedding tensor (e.g., node or sample embeddings) if produced by the training routine, otherwise None.
        """
        set_seed(self.seed)
        logger.info(f"Random seed set to: {self.seed}")

        dpmon_params = {
            "model": self.model,
            "phenotype_col": self.phenotype_col,
            "gnn_hidden_dim": self.gnn_hidden_dim,
            "gnn_layer_num": self.gnn_layer_num,
            "gnn_dropout":self.gnn_dropout,
            "gnn_activation":self.gnn_activation,
            "dim_reduction": self.dim_reduction,
            "ae_encoding_dim": self.ae_encoding_dim,
            "nn_hidden_dim1": self.nn_hidden_dim1,
            "nn_hidden_dim2": self.nn_hidden_dim2,
            "num_epochs": self.num_epochs,
            "n_folds": self.n_folds,
            "repeat_num": self.repeat_num,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "gpu": self.gpu,
            "cuda": self.cuda,
            "tune": self.tune,
            "tune_trials": self.tune_trails,
            "seed": self.seed,
            "seed_trials": self.seed_trials,
            "cv": self.cv,
        }

        graph_nodes = set(self.adjacency_matrix.index)
        omics_features = set(self.combined_omics_input.columns)
        common_features = list(graph_nodes.intersection(omics_features))

        if not common_features:
            raise ValueError("No common features found between adjacency matrix and omics data.")

        dropped_graph_nodes = len(graph_nodes) - len(common_features)
        dropped_omics_features = len(omics_features) - len(common_features)

        if dropped_graph_nodes > 0 or dropped_omics_features > 0:
            logger.info(
                f"Graph/omics mismatch. Aligning to {len(common_features)} common features. "
                f"Dropped {dropped_graph_nodes} from graph and {dropped_omics_features} from omics. "
                "To prevent this, ensure data is pre-aligned."
            )

        self.adjacency_matrix = self.adjacency_matrix.loc[common_features, common_features]
        combined_omics = self.combined_omics_input[common_features]

        phenotype_series = self.phenotype_series.rename(self.phenotype_col)

        if self.phenotype_col not in combined_omics.columns:
            combined_omics = combined_omics.merge(
                phenotype_series,
                left_index=True,
                right_index=True,
            )
        else:
            logger.warning(f"Column '{self.phenotype_col}' already exists in omics data. Using existing column.")

        predictions_df, metrics, embeddings = run_standard_training(
            dpmon_params,
            self.adjacency_matrix,
            combined_omics,
            self.clinical_data,
            seed=self.seed,
            cv=self.cv,
            output_dir=self.output_dir
        )

        logger.info("DPMON run completed.")
        return predictions_df, metrics, embeddings


def setup_device(gpu, cuda):
    if gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logger.info(f"Using GPU {cuda}")
        else:
            logger.warning(f"GPU {cuda} requested but not available, using CPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def slice_omics_datasets(omics_dataset: pd.DataFrame, adjacency_matrix: pd.DataFrame, phenotype_col: str = "phenotype") -> List[pd.DataFrame]:
    logger.debug("Slicing omics dataset based on network nodes.")
    omics_network_nodes_names = adjacency_matrix.index.tolist()

    # Clean omics dataset columns
    clean_columns = []
    for node in omics_dataset.columns:
        node_clean = re.sub(r"[^0-9a-zA-Z_]", ".", node)
        if not node_clean[0].isalpha():
            node_clean = "X" + node_clean
        clean_columns.append(node_clean)
    omics_dataset.columns = clean_columns

    missing_nodes = set(omics_network_nodes_names) - set(omics_dataset.columns)
    if missing_nodes:
        logger.error(f"Nodes missing in omics data: {missing_nodes}")
        raise ValueError("Missing nodes in omics dataset.")

    selected_columns = omics_network_nodes_names + [phenotype_col]
    return [omics_dataset[selected_columns]]

def prepare_node_features(adjacency_matrix: pd.DataFrame, omics_datasets: List[pd.DataFrame], clinical_data: Optional[pd.DataFrame], phenotype_col: str) -> List[Data]:
    """Build node-level features and return a PyTorch Geometric graph.

    Aligns network and omics features, constructs a NetworkX graph, and when clinical_data is provided computes Fisher Z-transformed correlations between each feature and each clinical variable as node features. The resulting feature matrix is standardized and wrapped into a PyTorch Geometric Data object with weighted edges.

    Args:

        adjacency_matrix (pd.DataFrame): Symmetric adjacency matrix with node names as both index and columns.
        omics_datasets (List[pd.DataFrame]): List of omics matrices (samples x features); only the first element is used here.
        clinical_data (Optional[pd.DataFrame]): Clinical covariates (samples x variables) used to build correlation-based node features; may be None.
        phenotype_col (str): Column name in the first omics DataFrame that stores the phenotype and should be dropped from features if present.

    Returns:

        List[Data]: Single-element list containing a PyTorch Geometric Data object with standardized node features and weighted edges.

    """
    logger.debug("Building PyTorch Geometric Data object from adjacency matrix.")

    network_features = adjacency_matrix.columns
    omics_data = omics_datasets[0]

    if phenotype_col in omics_data.columns:
        omics_feature_df = omics_data.drop(columns=[phenotype_col])
    else:
        omics_feature_df = omics_data

    nodes = sorted(network_features.intersection(omics_feature_df.columns))

    if len(nodes) == 0:
        raise ValueError("No common features found between the network and omics data.")

    omics_filtered = omics_feature_df[nodes]
    network_filtered = adjacency_matrix.loc[nodes, nodes]

    logger.info(f"Building graph with {len(nodes)} common features.")
    G = nx.from_pandas_adjacency(network_filtered)

    if clinical_data is not None and not clinical_data.empty:
        logger.debug("Adding clinical correlation features.")
        clinical_cols = list(clinical_data.columns)
        common_index = clinical_data.index.intersection(omics_filtered.index)

        if common_index.empty:
            raise ValueError("No common indices between omics_filtered and clinical_data in fold.")

        node_features_dict = {}
        for node in nodes:
            vec = pd.to_numeric(omics_filtered[node].loc[common_index], errors="coerce")
            vec = vec.dropna()

            corr_vector = {}
            for cvar in clinical_cols:
                clinical_series = clinical_data[cvar].loc[common_index]

                common_valid = vec.index.intersection(clinical_series.index)
                vec_aligned = vec.loc[common_valid]
                clinical_aligned = clinical_series.loc[common_valid]

                if (clinical_aligned.nunique() <= 1 or vec_aligned.nunique() <= 1 or len(vec_aligned) < 2):
                    corr_vector[cvar] = 0.0
                    continue

                vec_is_binary = vec_aligned.nunique() == 2
                clinical_is_binary = clinical_aligned.nunique() == 2

                try:
                    if vec_is_binary and clinical_is_binary:
                        corr_val = matthews_corrcoef(vec_aligned, clinical_aligned)
                    elif vec_is_binary or clinical_is_binary:
                        corr_val, _ = pointbiserialr(clinical_aligned, vec_aligned)
                        if pd.isna(corr_val):
                            corr_val = 0.0
                    else:
                        corr_val = vec_aligned.corr(clinical_aligned)
                        if pd.isna(corr_val):
                            corr_val = 0.0
                except Exception as e:
                    logger.debug(f"Correlation failed for {node}-{cvar}: {e}")
                    corr_val = 0.0

                if pd.isna(corr_val) or corr_val == 0.0:
                    z = 0.0
                else:
                    r_clip = np.clip(corr_val, -0.999999, 0.999999)
                    z = np.arctanh(r_clip)

                corr_vector[cvar] = z

            node_features_dict[node] = corr_vector

        node_features_df = pd.DataFrame.from_dict(node_features_dict, orient="index")
        logger.info(f"Built feature matrix with clinical correlations shape: {node_features_df.shape}")

    node_features_df = node_features_df.fillna(0.0)

    std_vals = node_features_df.std()
    std_vals = std_vals.replace(0, 1e-8)
    node_features_df = (node_features_df - node_features_df.mean()) / std_vals

    x = torch.tensor(node_features_df.values, dtype=torch.float)

    node_mapping = {}
    for i, name in enumerate(nodes):
        node_mapping[name] = i

    G_mapped = nx.relabel_nodes(G, node_mapping)

    edges_list = []
    for u, v in G_mapped.edges():
        edges_list.append((u, v))

    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    weights = []
    for _, _, data_attr in G_mapped.edges(data=True):
        weights.append(data_attr.get("weight", 1.0))

    edge_weight = torch.tensor(weights, dtype=torch.float)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    return [data]

def run_standard_training(dpmon_params, adjacency_matrix, combined_omics, clinical_data, seed, cv=False, output_dir=None):
    phenotype_col = dpmon_params["phenotype_col"]
    device = setup_device(dpmon_params["gpu"], dpmon_params["cuda"])
    omics_dataset = slice_omics_datasets(combined_omics, adjacency_matrix, phenotype_col)
    omics_dataset = omics_dataset[0]

    if not cv:
        logger.info(f"Running in standard mode (cv=False) with {dpmon_params['repeat_num']} repeats.")
        test_accuracies = []
        all_predictions_list = []
        best_accuracy = 0.0
        best_model_state = None
        best_predictions_df = None

        X = omics_dataset.drop([phenotype_col], axis=1)
        Y = omics_dataset[phenotype_col]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed, stratify=Y)

        if clinical_data is None:
            clinical_data_full = pd.DataFrame(index=X.index)
        else:
            clinical_data_full = clinical_data.reindex(X.index)

        clinical_train = clinical_data_full.loc[X_train.index]

        logger.info("Building 'clean' graph features for standard run using train split")
        omics_train_fold_list = [X_train.join(y_train.rename(phenotype_col))]

        omics_network_tg = prepare_node_features(
            adjacency_matrix,
            omics_train_fold_list,
            clinical_train,
            phenotype_col
        )[0].to(device)

        X_train_tensor = torch.FloatTensor(X_train.values).to(device)
        y_train_tensor = torch.LongTensor(y_train.values).to(device)
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
        y_test_tensor = torch.LongTensor(y_test.values).to(device)

        train_labels_dict = {
            "labels": y_train_tensor,
            "omics_network": omics_network_tg
        }

        for i in range(dpmon_params["repeat_num"]):
            logger.info(f"Training iteration {i+1}/{dpmon_params['repeat_num']}")

            model = NeuralNetwork(
                model_type=dpmon_params["model"],
                gnn_input_dim=omics_network_tg.x.shape[1],
                gnn_hidden_dim=dpmon_params["gnn_hidden_dim"],
                gnn_layer_num=dpmon_params["gnn_layer_num"],
                dim_reduction=dpmon_params["dim_reduction"],
                ae_encoding_dim=dpmon_params["ae_encoding_dim"],
                nn_input_dim=X_train_tensor.shape[1],
                nn_hidden_dim1=dpmon_params["nn_hidden_dim1"],
                nn_hidden_dim2=dpmon_params["nn_hidden_dim2"],
                nn_output_dim=Y.nunique(),
                gnn_dropout=dpmon_params["gnn_dropout"],
                gnn_activation=dpmon_params["gnn_activation"]
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=dpmon_params["lr"], weight_decay=dpmon_params["weight_decay"])

            model = train_model(
                model, criterion, optimizer,
                X_train_tensor, train_labels_dict, dpmon_params["num_epochs"]
            )

            model.eval()

            with torch.no_grad():
                predictions, _, _ = model(X_test_tensor, omics_network_tg)
                _, predicted = torch.max(predictions, 1)

                accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
                test_accuracies.append(accuracy)
                logger.info(f"Iteration {i+1} Test Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_state = model.state_dict()

                        best_predictions_df = pd.DataFrame({
                            "Actual": y_test_tensor.cpu().numpy(),
                            "Predicted": predicted.cpu().numpy(),
                            "Iteration": i + 1
                        })

                all_predictions_list.append(
                    pd.DataFrame({
                        "Actual": y_test_tensor.cpu().numpy(),
                        "Predicted": predicted.cpu().numpy(),
                        "Iteration": i + 1
                    })
                )

        if output_dir and best_model_state is not None:
            model_save_path = os.path.join(output_dir, "best_model_standard_run.pt")

            try:
                torch.save(best_model_state, model_save_path)
                logger.info(f"Successfully saved best model state to: {model_save_path}")
            except Exception as e:
                logger.error(f"Failed to save best model: {e}")

        return best_predictions_df, all_predictions_list, None

    else:
        n_folds = dpmon_params["n_folds"]
        logger.info(f"Running in Cross-Validation mode (cv=True) with {n_folds} folds.")

        #these are to track the best model across folds and then save it
        best_global_fold_accuracy = 0.0
        best_global_model_state = None
        best_global_embeddings = None

        fold_accuracies = []
        fold_f1_macros = []
        fold_f1_weighteds = []
        fold_auprs = []
        fold_aucs = []
        fold_recalls = []

        X = omics_dataset.drop([phenotype_col], axis=1)
        Y = omics_dataset[phenotype_col]

        if clinical_data is None:
            clinical_data_full = pd.DataFrame(index=X.index)
        else:
            clinical_data_full = clinical_data.reindex(X.index)


        repeat_num_val = dpmon_params.get("repeat_num", 1)
        total_splits = n_folds * repeat_num_val

        if repeat_num_val > 1:
            skf = RepeatedStratifiedKFold(
                n_splits=n_folds,
                n_repeats=repeat_num_val,
                random_state=seed
            )
            logger.info(f"CV Setup: Repeated K-Fold ({n_folds}x{repeat_num_val} = {total_splits} splits total).")
        else:
            # Fallback to single Stratified K-Fold
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            logger.info(f"CV Setup: Standard {n_folds}-fold split.")

        for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
            current_repeat = fold // n_folds + 1
            current_fold = fold % n_folds + 1

            if repeat_num_val > 1:
                logger.info(f"Starting Repeat {current_repeat}/{repeat_num_val}, Fold {current_fold}/{n_folds} (Total Split {fold + 1}/{total_splits})")
            else:
                logger.info(f"Starting Fold {current_fold}/{n_folds}")

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

            if dpmon_params['tune']:
                best_config = run_hyperparameter_tuning(
                    X_train, y_train,
                    adjacency_matrix,
                    clinical_data_full.iloc[train_index],
                    dpmon_params
                )
                dpmon_params.update(best_config)
                logger.info(f"Fold {fold+1} best config: {best_config}")

            clinical_train = clinical_data_full.iloc[train_index]
            logger.info(f"Building graph features for Fold {fold+1} using train split only")

            omics_train_fold_list = [X_train.join(y_train.rename(phenotype_col))]

            omics_network_tg = prepare_node_features(
                adjacency_matrix,
                omics_train_fold_list,
                clinical_train,
                phenotype_col
            )[0].to(device)

            X_train_tensor = torch.FloatTensor(X_train.values).to(device)
            y_train_tensor = torch.LongTensor(y_train.values).to(device)
            X_test_tensor = torch.FloatTensor(X_test.values).to(device)
            y_test_tensor = torch.LongTensor(y_test.values).to(device)

            train_labels_dict = {
                "labels": y_train_tensor,
                "omics_network": omics_network_tg
            }

            model = NeuralNetwork(
                model_type=dpmon_params["model"],
                gnn_input_dim=omics_network_tg.x.shape[1],
                gnn_hidden_dim=dpmon_params["gnn_hidden_dim"],
                gnn_layer_num=dpmon_params["gnn_layer_num"],
                ae_encoding_dim=dpmon_params["ae_encoding_dim"],
                nn_input_dim=X_train_tensor.shape[1],
                nn_hidden_dim1=dpmon_params["nn_hidden_dim1"],
                nn_hidden_dim2=dpmon_params["nn_hidden_dim2"],
                nn_output_dim=Y.nunique(),
                gnn_dropout=dpmon_params["gnn_dropout"],
                gnn_activation=dpmon_params["gnn_activation"],
                dim_reduction=dpmon_params["dim_reduction"],
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=dpmon_params["lr"], weight_decay=dpmon_params["weight_decay"])

            model = train_model(model, criterion, optimizer,X_train_tensor, train_labels_dict, dpmon_params["num_epochs"])
            model.eval()
            logger.info(f"Evaluating model for Fold {fold+1} on test set")
            with torch.no_grad():
                predictions, _, node_emb = model(X_test_tensor, omics_network_tg)
                _, predicted = torch.max(predictions, 1)
                probs = torch.softmax(predictions, dim=1)

                y_test_np = y_test_tensor.cpu().numpy()
                predicted_np = predicted.cpu().numpy()
                probs_np = probs.cpu().numpy()

                accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
                f1_ma = f1_score(y_test_np, predicted_np, average='macro', zero_division=0)
                f1_wt = f1_score(y_test_np, predicted_np, average='weighted', zero_division=0)
                recall = recall_score(y_test_np, predicted_np, average='macro', zero_division=0)

                try:
                    n_classes = probs_np.shape[1]

                    # binary
                    if n_classes == 2:
                        # Ususinge probability of positive
                        auc_score = roc_auc_score(y_test_np, probs_np[:, 1])
                        aupr = average_precision_score(y_test_np, probs_np[:, 1])
                        logger.debug(f"Binary | AUC: {auc_score:.4f}, AUPR: {aupr:.4f}")

                    else:
                        auc_score = roc_auc_score(y_test_np, probs_np, multi_class='ovr', average='macro')
                        y_test_bin = label_binarize(y_test_np, classes=range(n_classes))

                        aupr_scores = []
                        for i in range(n_classes):
                            # checking if class exists in test set
                            if np.sum(y_test_bin[:, i]) > 0:
                                ap = average_precision_score(y_test_bin[:, i], probs_np[:, i])
                                aupr_scores.append(ap)

                        aupr = np.mean(aupr_scores) if aupr_scores else 0.0
                        logger.debug(f"Multiclass | AUC: {auc_score:.4f}, AUPR: {aupr:.4f}")

                except Exception as e:
                    logger.error(f"Could not calculate AUC/AUPR: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    auc_score = 0.0
                    aupr = 0.0

                fold_predictions = {
                    'accuracy': accuracy,
                    'f1_ma': f1_ma,
                    'f1_wt': f1_wt,
                    'aupr': aupr,
                    'auc': auc_score,
                    'recall': recall
                }

                if accuracy > best_global_fold_accuracy:
                    best_global_fold_accuracy = accuracy
                    best_global_model_state = model.state_dict()
                    best_global_embeddings = node_emb.detach().cpu()


                if fold_predictions:
                    fold_accuracies.append(fold_predictions['accuracy'])
                    fold_f1_macros.append(fold_predictions['f1_ma'])
                    fold_f1_weighteds.append(fold_predictions['f1_wt'])
                    fold_auprs.append(fold_predictions['aupr'])
                    fold_aucs.append(fold_predictions['auc'])
                    fold_recalls.append(fold_predictions['recall'])

                logger.info(f"Fold {fold+1} results:")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"F1 Macro: {f1_ma:.4f}")
                logger.info(f"F1 Weighted: {f1_wt:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"AUC: {auc_score:.4f}")
                logger.info(f"AUPR: {aupr:.4f}\n")

                if dpmon_params['gpu']:
                    torch.cuda.empty_cache()

                    logger.debug(f"Clearing cuda cache for fold {fold+1} \n")

        avg_acc = statistics.mean(fold_accuracies) if fold_accuracies else 0.0
        std_acc = statistics.stdev(fold_accuracies) if len(fold_accuracies) > 1 else 0.0
        avg_f1_ma = statistics.mean(fold_f1_macros) if fold_f1_macros else 0.0
        std_f1_ma = statistics.stdev(fold_f1_macros) if len(fold_f1_macros) > 1 else 0.0
        avg_f1_wt = statistics.mean(fold_f1_weighteds) if fold_f1_weighteds else 0.0
        std_f1_wt = statistics.stdev(fold_f1_weighteds) if len(fold_f1_weighteds) > 1 else 0.0
        avg_aupr = statistics.mean(fold_auprs) if fold_auprs else 0.0
        std_aupr = statistics.stdev(fold_auprs) if len(fold_auprs) > 1 else 0.0
        avg_auc = statistics.mean(fold_aucs) if fold_aucs else 0.0
        std_auc = statistics.stdev(fold_aucs) if len(fold_aucs) > 1 else 0.0
        avg_recall = statistics.mean(fold_recalls) if fold_recalls else 0.0
        std_recall = statistics.stdev(fold_recalls) if len(fold_recalls) > 1 else 0.0

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Recall', 'AUC', 'AUPR'],
            'Average': [avg_acc, avg_f1_ma, avg_f1_wt, avg_recall, avg_auc, avg_aupr],
            'StdDev': [std_acc, std_f1_ma, std_f1_wt, std_recall, std_auc, std_aupr]
        })

        #final_cv_predictions_df = pd.concat(cv_predictions_list, ignore_index=True)
        if output_dir and best_global_model_state is not None:
            model_save_path = os.path.join(output_dir, "best_cv_model.pt")
            try:
                torch.save(best_global_model_state, model_save_path)
                logger.info(f"Successfully saved global best CV model state to: {model_save_path}")
            except Exception as e:
                logger.error(f"Failed to save best CV model: {e}")

        if output_dir and best_global_embeddings is not None:
            emb_save_path = os.path.join(output_dir, "best_cv_model_embds.pt")
            try:
                torch.save(best_global_embeddings, emb_save_path)
                logger.info(f"Successfully saved global best CV model state to: {emb_save_path}")
            except Exception as e:
                logger.error(f"Failed to save best CV model: {e}")


        logger.info("Cross-Validation Results:\n")
        logger.info(f"Avg Accuracy: {avg_acc:.4f} +/- {std_acc:.4f}")
        logger.info(f"Avg F1 Macro: {avg_f1_ma:.4f} +/- {std_f1_ma:.4f}")
        logger.info(f"Avg F1 Weighted: {avg_f1_wt:.4f} +/- {std_f1_wt:.4f}")
        logger.info(f"Avg Recall: {avg_recall:.4f} +/- {std_recall:.4f}")
        logger.info(f"Avg AUC: {avg_auc:.4f} +/- {std_auc:.4f}")
        logger.info(f"Avg AUPR: {avg_aupr:.4f} +/- {std_aupr:.4f}")

        return pd.DataFrame(), metrics_df, best_global_embeddings

def run_hyperparameter_tuning(X_train, y_train, adjacency_matrix, clinical_data, dpmon_params):
    device = setup_device(dpmon_params["gpu"], dpmon_params["cuda"])
    phenotype_col = dpmon_params["phenotype_col"]
    combined_omics_fold = X_train.join(y_train.rename(phenotype_col))
    omics_dataset = slice_omics_datasets(combined_omics_fold, adjacency_matrix, phenotype_col)
    omics_train_fold_list = [omics_dataset[0]]

    omics_network_tg = prepare_node_features(
        adjacency_matrix,
        omics_train_fold_list,
        clinical_data,
        phenotype_col
    )[0].to(device)

    pipeline_configs = {
        "gnn_layer_num": tune.choice([1, 2, 3, 4]),
        "gnn_hidden_dim": tune.choice([64, 128, 256, 512]),
        "lr": tune.loguniform(1e-5, 3e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "nn_hidden_dim1": tune.choice([128, 256, 512]),
        "nn_hidden_dim2": tune.choice([32, 64, 128]),
        "ae_encoding_dim": tune.choice([4, 8, 16]),
        "num_epochs": tune.choice([512, 1024, 2048]),
        "gnn_dropout": tune.choice([0.2, 0.3, 0.4, 0.5, 0.6]),
        "gnn_activation": tune.choice(["relu", "elu"]),
        "dim_reduction": tune.choice(["ae","linear", "mlp"]),
    }

    stopper = TrialPlateauStopper(
        metric="val_loss",
        mode="min",
        num_results=20,
        metric_threshold=0.002,
        grace_period=30,
    )

    reporter = CLIReporter(metric_columns=["train_loss", "val_loss", "val_accuracy", "training_iteration"])
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        grace_period=30,
        reduction_factor=2
    )

    best_configs = []

    omics_data = omics_dataset[0]
    logger.info(f"Starting hyperparameter tuning for dataset shape: {omics_data.shape}")

    X = omics_data.drop([phenotype_col], axis=1)
    Y = omics_data[phenotype_col]

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=dpmon_params["seed"], stratify=Y)
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.LongTensor(y_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)
    y_val_tensor = torch.LongTensor(y_val.values).to(device)

    omics_network_tg_dev = omics_network_tg.to(device)

    def tune_train_n(config):
        gnn_input_dim = omics_network_tg.x.shape[1]
        nn_input_dim = X.shape[1]
        nn_output_dim = Y.nunique()

        model = NeuralNetwork(
            model_type=dpmon_params["model"],
            gnn_hidden_dim=config["gnn_hidden_dim"],
            gnn_layer_num=config["gnn_layer_num"],
            gnn_activation=config["gnn_activation"],
            dim_reduction=config["dim_reduction"],
            gnn_input_dim=gnn_input_dim,
            gnn_dropout=config["gnn_dropout"],
            ae_encoding_dim=config["ae_encoding_dim"],
            nn_input_dim=nn_input_dim,
            nn_hidden_dim1=config["nn_hidden_dim1"],
            nn_hidden_dim2=config["nn_hidden_dim2"],
            nn_output_dim=nn_output_dim,

        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        for epoch in range(config["num_epochs"]):
            model.train()
            optimizer.zero_grad()
            outputs, _, _ = model(X_train_tensor, omics_network_tg_dev)
            train_loss = criterion(outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()

            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                val_outputs, _, _ = model(X_val_tensor, omics_network_tg_dev)
                val_loss_obj = criterion(val_outputs, y_val_tensor)
                val_loss = val_loss_obj.item()

                _, predicted = torch.max(val_outputs, 1)
                total = y_val_tensor.size(0)
                correct = (predicted == y_val_tensor).sum().item()
                val_accuracy = correct / total

            metrics = {
                "loss": val_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "train_loss": train_loss.item()
            }

            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict()},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(
                    metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir)
                )

    def short_dirname_creator(trial):
        return f"T{trial.trial_id}"

    cpu_per_trial = 2
    use_gpu = bool(dpmon_params.get("gpu", False)) and torch.cuda.is_available()
    if dpmon_params.get("gpu", False) and not torch.cuda.is_available():
        logger.warning("gpu=True but CUDA is not available; Ray Tune will run on CPU only (gpu_per_trial=0.0).")

    gpu_per_trial = 0.05 if use_gpu else 0.0

    num_samples = dpmon_params['tune_trials']
    max_retries = 4

    seed_trials = dpmon_params.get("seed_trials", False)

    if seed_trials:
        logger.debug(f"seed_trials=True: Using FIXED seed {dpmon_params['seed']} for hyperparameter sampling.")
    else:
        logger.debug("seed_trials=False: Using RANDOM hyperparameter sampling.")

    for attempt in range(max_retries):
        try:
            if seed_trials:
                search_alg = BasicVariantGenerator(random_state=np.random.RandomState(dpmon_params["seed"]))
            else:
                search_alg = None

            result = tune.run(
                tune_train_n,
                search_alg=search_alg,
                resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
                config=pipeline_configs,
                num_samples=num_samples,
                verbose=0,
                log_to_file=True,
                scheduler=scheduler,
                stop=stopper,
                name="tune_dp",
                progress_reporter=reporter,
                trial_dirname_creator=short_dirname_creator,
                checkpoint_score_attr="min-val_loss",
            )
            break
        except TuneError as e:
            msg = str(e)
            if "Trials did not complete" not in msg and "OutOfMemoryError" not in msg:
                raise

            new_num_samples = max(1, num_samples // 2)
            if new_num_samples == num_samples:
                raise

            logger.warning(
                f"Ray Tune failed with a likely resource / OOM error (attempt {attempt + 1}). "
                f"Reducing num_samples from {num_samples} to {new_num_samples} "
                f"(cpu_per_trial={cpu_per_trial}, gpu_per_trial={gpu_per_trial})."
            )
            num_samples = new_num_samples

    else:
        raise RuntimeError("Hyperparameter tuning failed after reducing resources several times.")

    best_trial = result.get_best_trial("val_loss", "min", "last")
    logger.debug("Best trial config: {}".format(best_trial.config))
    logger.debug("Best trial final val_loss: {}".format(best_trial.last_result["val_loss"]))
    logger.debug("Best trial final val_accuracy: {}".format(best_trial.last_result["val_accuracy"]))
    best_configs.append(best_trial.config)

    try:
        tune_dir = os.path.expanduser("~/ray_results/tune_dp")
        if os.path.exists(tune_dir):
            shutil.rmtree(tune_dir)
            logger.debug(f"Cleaned up tuning directory: {tune_dir}")
    except Exception as e:
        logger.warning(f"Could not clean up tuning directory: {e}")

    return best_trial.config

def train_model(model, criterion, optimizer, train_features, train_labels, epoch_num):
    network = train_labels["omics_network"]
    labels = train_labels["labels"]

    model.train()
    for epoch in range(epoch_num):
        optimizer.zero_grad()
        outputs, _, _ = model(train_features, network)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            logger.debug(f"Epoch [{epoch+1}/{epoch_num}], Loss: {loss.item():.4f}")

    return model

class NeuralNetwork(nn.Module):
    """Core DPMON model combining GNN feature weighting and sample-level prediction.

    A graph neural network first computes node embeddings on the feature graph. An autoencoder then compresses these embeddings, and a projection module maps them to scalar feature weights. These weights re-scale the sample-level omics matrix, and a downstream MLP takes the reweighted features to produce logits for the prediction task.

    The forward pass returns a tuple containing:

    1. Predictions (logits).
    2. Reweighted omics dataset.
    3. Original node embeddings from the GNN.

    Args:

        model_type (str): GNN backbone type ("GCN", "GAT", "SAGE", "GIN").
        gnn_input_dim (int): Input dimension for the GNN nodes.
        gnn_hidden_dim (int): Hidden dimension for GNN layers.
        gnn_layer_num (int): Number of GNN layers.
        ae_encoding_dim (int): Target dimension for the autoencoder compression.
        nn_input_dim (int): Input dimension for the downstream predictor.
        nn_hidden_dim1 (int): First hidden layer dimension of the predictor.
        nn_hidden_dim2 (int): Second hidden layer dimension of the predictor.
        nn_output_dim (int): Output dimension (e.g., number of classes).
        gnn_dropout (float): Dropout probability for the GNN.
        gnn_activation (str): Activation function for the GNN ("relu", etc.).
        dim_reduction (str): Reduction strategy ("ae", "linear", "mlp").

    """
    def __init__(
        self,
        model_type,
        gnn_input_dim,
        gnn_hidden_dim,
        gnn_layer_num,
        ae_encoding_dim,
        nn_input_dim,
        nn_hidden_dim1,
        nn_hidden_dim2,
        nn_output_dim,
        gnn_dropout: float = 0.2,
        gnn_activation: str = "relu",
        dim_reduction: str = "ae",
    ):
        super(NeuralNetwork, self).__init__()

        if model_type == "GCN":
            self.gnn = GCN(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                layer_num=gnn_layer_num,
                final_layer="none",
                dropout=gnn_dropout,
                activation=gnn_activation,
            )
        elif model_type == "GAT":
            self.gnn = GAT(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                layer_num=gnn_layer_num,
                final_layer="none",
                dropout=gnn_dropout,
                activation=gnn_activation,
            )
        elif model_type == "SAGE":
            self.gnn = SAGE(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                layer_num=gnn_layer_num,
                final_layer="none",
                dropout=gnn_dropout,
                activation=gnn_activation,
            )
        elif model_type == "GIN":
            self.gnn = GIN(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_hidden_dim,
                layer_num=gnn_layer_num,
                final_layer="none",
                dropout=gnn_dropout,
                activation=gnn_activation,
            )
        else:
            raise ValueError(f"Unsupported GNN model type: {model_type}")

        if dim_reduction == "ae":
            self.autoencoder = AutoEncoder(input_dim=gnn_hidden_dim, encoding_dim=1)
            self.projection = nn.Identity()

        elif dim_reduction == "linear":
            self.autoencoder = AutoEncoder(input_dim=gnn_hidden_dim, encoding_dim=ae_encoding_dim)
            self.projection = ScalarProjection(encoding_dim=ae_encoding_dim)

        elif dim_reduction == "mlp":
            self.autoencoder = AutoEncoder(input_dim=gnn_hidden_dim, encoding_dim=ae_encoding_dim)
            self.projection = MLPProjection(encoding_dim=ae_encoding_dim, hidden_dim=8)

        else:
            raise ValueError(f"Unsupported dim_reduction: {dim_reduction}. Use 'ae', 'linear', or 'mlp'")

        self.predictor = DownstreamTaskNN(
            nn_input_dim, nn_hidden_dim1, nn_hidden_dim2, nn_output_dim
        )

    def forward(self, omics_dataset, omics_network_tg):
        """Performs the forward pass of the DPMON network.

        Args:

            omics_dataset (torch.Tensor): The raw sample-level omics data [Samples x Features].
            omics_network_tg (Data): The PyG graph data object representing the feature network.

        Returns:

            tuple: (predictions, reweighted_omics, node_embeddings)
        """
        # we get node embeddings from the GNN
        omics_network_nodes_embedding = self.gnn.get_embeddings(omics_network_tg)

        # dim reduction of the embeddings
        omics_network_nodes_embedding_ae = self.autoencoder(omics_network_nodes_embedding)

        # then project to scallar weights
        feature_weights = self.projection(omics_network_nodes_embedding_ae)

        # reweight the original Omics Data (Element-wise multiplication)
        # transpose weights to match [Features x 1] -> broadcast to [Samples x Features]
        omics_dataset_with_embeddings = torch.mul(
            omics_dataset,
            feature_weights.expand(omics_dataset.shape[1], omics_dataset.shape[0]).t()
        )

        # Lasly we predict using the Downstream MLP
        predictions = self.predictor(omics_dataset_with_embeddings)

        return predictions, omics_dataset_with_embeddings, omics_network_nodes_embedding


class AutoEncoder(nn.Module):
    """Compresses high-dimensional node embeddings into a lower-dimensional latent space.

    Args:

        input_dim (int): Input feature dimension.
        encoding_dim (int): Output latent dimension.

    """
    def __init__(self, input_dim: int, encoding_dim: int):
        super(AutoEncoder, self).__init__()

        # the intermediate layer is roughly half input
        hidden_dim = max(input_dim // 2, encoding_dim * 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class ScalarProjection(nn.Module):
    """Linear projection mapping encoded features to scalar importance weights.

    Args:

        encoding_dim (int): Dimension of the input encoded features.

    """
    def __init__(self, encoding_dim):
        super(ScalarProjection, self).__init__()
        self.proj = nn.Linear(encoding_dim, 1)

    def forward(self, x):
        return self.proj(x)


class MLPProjection(nn.Module):
    """Non-linear projection mapping encoded features to scalar importance weights using an MLP.

    Args:

        encoding_dim (int): Dimension of the input encoded features.
        hidden_dim (int): Hidden layer size for the projection MLP.

    """
    def __init__(self, encoding_dim, hidden_dim=8):
        super(MLPProjection, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.mlp(x)


class DownstreamTaskNN(nn.Module):
    """Multilayer Perceptron for the final downstream prediction task.

    Args:

        input_size (int): Number of input features (omics features).
        hidden_dim1 (int): Size of the first hidden layer.
        hidden_dim2 (int): Size of the second hidden layer.
        output_dim (int): Size of the output layer (num_classes or 1 for regression).

    """
    def __init__(self, input_size, hidden_dim1, hidden_dim2, output_dim):
        super(DownstreamTaskNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
