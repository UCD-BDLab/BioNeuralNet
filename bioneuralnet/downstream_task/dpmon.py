r"""
DPMON: Optimized Network Embedding and Fusion for Disease Prediction.

This module implements an end-to-end Graph Neural Network (GNN) pipeline
integrating network topology with subject-level omics data.

References:
    Hussein, S. et al. (2024), "Learning from Multi-Omics Networks to
    Enhance Disease Prediction: An Optimized Network Embedding and
    Fusion Approach" IEEE BIBM.

Algorithm:
    The pipeline consists of three distinct phases:

    Phase 1: Task-Aware Embedding Generation
        1. Construct a multi-omics network.
        2. Initialize node features using clinical correlation vectors.
        3. Pass graph through a GNN (GAT/GCN/GIN).

    Phase 2: Dimensionality Reduction
        Compress embeddings into scalar weights via AutoEncoder/MLP.

    Phase 3: Fusion and Prediction
        Fuse embeddings with subject-level data via element-wise
        multiplication (Feature Reweighting).

Notes:
    The embedding space is optimized dynamically using the loss function:

    .. math::
        L_{total} = L_{classification} + \lambda L_{regularization}

    The fusion acts as a **Network-Guided Attention Mechanism**,
    amplifying features that are topologically central.
"""

from __future__ import annotations

import os
import re
import logging
import statistics
import tempfile
import shutil
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Optional, List, Tuple,Dict, Any

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

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune import Checkpoint
    from ray.tune.error import TuneError
    from ray.tune.stopper import TrialPlateauStopper
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.basic_variant import BasicVariantGenerator

    os.environ["TUNE_DISABLE_IPY_WIDGETS"] = "1"
    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"
    os.environ["RAY_DEDUP_LOGS"] = "0"

    for logger_name in ("ray", "raylet", "ray.train.session", "ray.tune", "torch_geometric"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

except ModuleNotFoundError:
    raise ImportError(
        "DPMON (Disease Prediction using Multi-Omics Networks) requires Ray Tune"
        "Please install it by following the instructions at: "
        "https://bioneuralnet.readthedocs.io/en/latest/installation.html"
    )

from sklearn.model_selection import train_test_split,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.preprocessing import label_binarize
from scipy.stats import pointbiserialr
from sklearn.metrics import f1_score, roc_auc_score, recall_score,precision_score,average_precision_score, matthews_corrcoef

from bioneuralnet.utils import set_seed
from bioneuralnet.network_embedding import GCN, GAT, SAGE, GIN
from ..utils import get_logger

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
        tune_trials (int): Number of trials to perform if tune=True.
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
        correlation_mode: str = "abs_pearson",
        model: str = "GAT",
        phenotype_col: str = "phenotype",
        gnn_hidden_dim: int = 16,
        gnn_layer_num: int = 4,
        gnn_dropout: float = 0.1,
        gnn_activation: str = "relu",
        dim_reduction: str = "ae",
        ae_architecture: str = "original",
        ae_encoding_dim: int = 8,
        nn_hidden_dim1: int = 16,
        nn_hidden_dim2: int = 8,
        num_epochs: int = 100,
        repeat_num: int = 1,
        n_folds: int = 5,
        lr: float = 1e-1,
        weight_decay: float = 1e-4,
        gat_heads: int = 1,
        tune: bool = False,
        tune_trials: int = 20,
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
        self.tune_trials = tune_trials
        self.gpu = gpu
        self.cuda = cuda
        self.seed = seed
        self.seed_trials = seed_trials
        self.cv = cv
        self.correlation_mode = correlation_mode
        self.ae_architecture= ae_architecture
        self.gat_heads =gat_heads

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
            "tune_trials": self.tune_trials,
            "seed": self.seed,
            "seed_trials": self.seed_trials,
            "correlation_mode": self.correlation_mode,
            "ae_architecture": self.ae_architecture,
            "gat_heads": self.gat_heads,
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
            logger.debug(f"Using GPU {cuda}")
        else:
            logger.warning(f"GPU {cuda} requested but not available, using CPU")
    else:
        device = torch.device("cpu")
        logger.debug("Using CPU")
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

def prepare_node_features(
    adjacency_matrix: pd.DataFrame,
    omics_datasets: List[pd.DataFrame],
    clinical_data: Optional[pd.DataFrame],
    phenotype_col: str,
    correlation_mode: str = "abs_pearson",
) -> List[Data]:
    """Build node-level features and return a PyTorch Geometric graph.

    Args:

        adjacency_matrix: Symmetric adjacency matrix (node names as index/columns).
        omics_datasets: List of omics matrices (samples x features); first element used.
        clinical_data: Clinical covariates for correlation-based node features; may be None.
        phenotype_col: Column name storing phenotype labels (dropped from features).
        correlation_mode: How to compute node features from clinical correlations.
            - "abs_pearson": Absolute Pearson correlation, no transforms = DPMON.
            - "adaptive": Mixed correlation types + Fisher-Z + standardization.

    Returns:

        List[Data]: Single-element list with a PyG Data object.
    """
    logger.debug(f"Building PyG Data object (correlation_mode={correlation_mode}).")

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

    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        G.remove_edges_from(self_loops)
        logger.debug(f"Removed {len(self_loops)} self-loop edges.")

    if clinical_data is not None and not clinical_data.empty:
        clinical_cols = list(clinical_data.columns)
        common_index = clinical_data.index.intersection(omics_filtered.index)

        if common_index.empty:
            raise ValueError("No common indices between omics and clinical data.")

        node_features_dict = {}

        for node in nodes:
            vec = pd.to_numeric(omics_filtered[node].loc[common_index], errors="coerce")
            vec = vec.dropna()

            corr_vector = {}
            for cvar in clinical_cols:
                clinical_series = clinical_data[cvar].loc[common_index]
                common_valid = vec.index.intersection(clinical_series.dropna().index)
                vec_aligned = vec.loc[common_valid]
                clinical_aligned = clinical_series.loc[common_valid].astype("float64")

                if clinical_aligned.nunique() <= 1 or vec_aligned.nunique() <= 1 or len(vec_aligned) < 2:
                    corr_vector[cvar] = 0.0
                    continue

                if correlation_mode == "abs_pearson":
                    # OG DPMON: abs(Pearson correlation)
                    try:
                        corr_val = abs(vec_aligned.corr(clinical_aligned))
                        if pd.isna(corr_val):
                            corr_val = 0.0
                    except Exception:
                        corr_val = 0.0
                    corr_vector[cvar] = corr_val

                elif correlation_mode == "adaptive":
                    # OPTION 2: mixed types + Fisher-Z
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

                    # Fisher-Z transform
                    if pd.isna(corr_val) or corr_val == 0.0:
                        z = 0.0
                    else:
                        r_clip = np.clip(corr_val, -0.999999, 0.999999)
                        z = np.arctanh(r_clip)
                    corr_vector[cvar] = z
                else:
                    raise ValueError(f"Unknown correlation_mode: {correlation_mode}")

            node_features_dict[node] = corr_vector

        node_features_df = pd.DataFrame.from_dict(node_features_dict, orient="index")
        node_features_df = node_features_df.fillna(0.0)

        if correlation_mode == "adaptive":
            # standardize only in adaptive mode DPMON uses raw.
            std_vals = node_features_df.std()
            std_vals = std_vals.replace(0, 1e-8)
            node_features_df = (node_features_df - node_features_df.mean()) / std_vals

        logger.info(f"Node feature matrix shape: {node_features_df.shape} (mode={correlation_mode})")

    else:
        # No clinical data -> generate random features as fallback
        logger.warning("No clinical data provided. Using random node features.")
        rng = np.random.default_rng(1998)
        node_features_df = pd.DataFrame(
            rng.standard_normal((len(nodes), 7)),
            index=nodes,
            columns=[f"rand_{i}" for i in range(7)],
        )

    # convert to PyG Data
    x = torch.tensor(node_features_df.values, dtype=torch.float)

    node_mapping = {name: i for i, name in enumerate(nodes)}
    G_mapped = nx.relabel_nodes(G, node_mapping)

    edges_list = list(G_mapped.edges())
    if not edges_list:
        logger.warning("Graph has no edges after self-loop removal.")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
    else:
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
        # Make bidirectional
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        weights = []
        for _, _, d in G_mapped.edges(data=True):
            weights.append(d.get("weight", 1.0))
        edge_weight = torch.tensor(weights, dtype=torch.float)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    return [data]

def run_standard_training(dpmon_params, adjacency_matrix, combined_omics, clinical_data, seed, cv=False, output_dir=None):
    phenotype_col = dpmon_params["phenotype_col"]
    correlation_mode = dpmon_params["correlation_mode"]
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

        f1_macros = []
        f1_weighteds = []
        recalls = []
        aucs = []
        auprs = []

        X = omics_dataset.drop([phenotype_col], axis=1)
        Y = omics_dataset[phenotype_col]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed, stratify=Y)

        if clinical_data is None:
            clinical_data_full = pd.DataFrame(index=X.index)
        else:
            clinical_data_full = clinical_data.reindex(X.index)

        clinical_train = clinical_data_full.loc[X_train.index]
        if dpmon_params['tune']:
            clinical_train_tune = clinical_data_full.loc[X_train.index]
            best_config = run_hyperparameter_tuning(
                X_train, y_train,
                adjacency_matrix,
                clinical_train_tune,
                dpmon_params
            )
            dpmon_params.update(best_config)
            logger.info(f"Best config: {best_config}")

        logger.info("Building 'clean' graph features for standard run using train split")
        omics_train_fold_list = [X_train.join(y_train.rename(phenotype_col))]

        omics_network_tg = prepare_node_features(
            adjacency_matrix,
            omics_train_fold_list,
            clinical_train,
            phenotype_col,
            correlation_mode
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
                ae_architecture=dpmon_params["ae_architecture"],
                nn_input_dim=X_train_tensor.shape[1],
                nn_hidden_dim1=dpmon_params["nn_hidden_dim1"],
                nn_hidden_dim2=dpmon_params["nn_hidden_dim2"],
                nn_output_dim=Y.nunique(),
                gnn_dropout=dpmon_params["gnn_dropout"],
                gnn_activation=dpmon_params["gnn_activation"],
                gat_heads=dpmon_params["gat_heads"]
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
                    if n_classes == 2:
                        auc_score = roc_auc_score(y_test_np, probs_np[:, 1])
                        aupr = average_precision_score(y_test_np, probs_np[:, 1])
                    else:
                        auc_score = roc_auc_score(y_test_np, probs_np, multi_class='ovr', average='macro')
                        aupr = 0.0
                except:
                    auc_score, aupr = 0.0, 0.0

                logger.info(f"Iteration {i+1} Results:")
                logger.info(f" Accuracy: {accuracy:.4f}")
                logger.info(f" F1 Macro: {f1_ma:.4f}")
                logger.info(f" F1 Weighted: {f1_wt:.4f}")
                logger.info(f" Recall: {recall:.4f}")
                logger.info(f" AUC: {auc_score:.4f}")
                logger.info(f" AUPR: {aupr:.4f}\n")

                test_accuracies.append(accuracy)
                f1_macros.append(f1_ma)
                f1_weighteds.append(f1_wt)
                recalls.append(recall)
                aucs.append(auc_score)
                auprs.append(aupr)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_state = model.state_dict()

        if test_accuracies:
            def get_stats(data_list):
                avg = statistics.mean(data_list) if data_list else 0.0
                std = statistics.stdev(data_list) if len(data_list) > 1 else 0.0
                return avg, std

            metrics_to_report = {
                'Accuracy': test_accuracies,
                'F1 Macro': f1_macros,
                'F1 Weighted': f1_weighteds,
                'Recall': recalls,
                'AUC': aucs,
                'AUPR': auprs
            }

            summary_rows = []
            for name, values in metrics_to_report.items():
                avg, std = get_stats(values)
                summary_rows.append({'Metric': name, 'Average': avg, 'StdDev': std})

            metrics_df = pd.DataFrame(summary_rows)

            logger.info("--- Standard Run Final Summary (avg across repeats) ---")
            for _, row in metrics_df.iterrows():
                logger.info(f"Avg {row['Metric']}: {row['Average']:.4f} +/- {row['StdDev']:.4f}")
            logger.info("------------------------------------------------------\n")

        else:
            metrics_df = pd.DataFrame()

        if output_dir and best_model_state is not None:
            model_save_path = os.path.join(output_dir, "best_model_standard_run.pt")

            try:
                torch.save(best_model_state, model_save_path)
                logger.info(f"Successfully saved best model state to: {model_save_path}")
            except Exception as e:
                logger.error(f"Failed to save best model: {e}")

        return best_predictions_df, all_predictions_list, metrics_df

    else:
        n_folds = dpmon_params["n_folds"]
        logger.info(f"Running in Cross-Validation mode (cv=True) with {n_folds} folds.")

        # these are to track the best model across folds and then save it
        best_global_fold_accuracy = 0.0
        best_global_fold_f1 = 0.0
        best_global_model_state = None
        best_global_embeddings = None

        fold_accuracies = []
        fold_f1_macros = []
        fold_f1_weighteds = []
        fold_auprs = []
        fold_aucs = []
        fold_recalls = []
        fold_precisions = []
        fold_best_configs = []
        all_fold_results = []

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
            # fallback to single Stratified kfold
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

                #save params
                fold_record = best_config.copy()
                fold_record['Fold'] = fold + 1
                fold_best_configs.append(fold_record)

            clinical_train = clinical_data_full.iloc[train_index]
            clinical_test = clinical_data_full.iloc[test_index]
            logger.info(f"Building graph features for Fold {fold+1} using train split only")

            omics_train_fold_list = [X_train.join(y_train.rename(phenotype_col))]

            omics_network_tg = prepare_node_features(
                adjacency_matrix,
                omics_train_fold_list,
                clinical_train,
                phenotype_col,
                correlation_mode
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
                ae_architecture=dpmon_params["ae_architecture"],
                nn_input_dim=X_train_tensor.shape[1],
                nn_hidden_dim1=dpmon_params["nn_hidden_dim1"],
                nn_hidden_dim2=dpmon_params["nn_hidden_dim2"],
                nn_output_dim=Y.nunique(),
                gnn_dropout=dpmon_params["gnn_dropout"],
                gnn_activation=dpmon_params["gnn_activation"],
                dim_reduction=dpmon_params["dim_reduction"],
                gat_heads=dpmon_params["gat_heads"]
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
                precision = precision_score(y_test_np, predicted_np, average='macro', zero_division=0)

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
                    'recall': recall,
                    'precision': precision
                }

                if accuracy > best_global_fold_accuracy:
                    best_global_fold_accuracy = accuracy
                    best_global_model_state = model.state_dict()
                    best_global_embeddings = node_emb.detach().cpu()

                # Should be a parameter that way we can decide which model to optemize/save
                # if f1_ma > best_global_fold_f1:
                #     best_global_fold_f1 = f1_ma
                #     best_global_model_state = model.state_dict()
                #     best_global_embeddings = node_emb.detach().cpu()

                if fold_predictions:
                    fold_accuracies.append(fold_predictions['accuracy'])
                    fold_f1_macros.append(fold_predictions['f1_ma'])
                    fold_f1_weighteds.append(fold_predictions['f1_wt'])
                    fold_auprs.append(fold_predictions['aupr'])
                    fold_aucs.append(fold_predictions['auc'])
                    fold_recalls.append(fold_predictions['recall'])
                    fold_precisions.append(fold_predictions["precision"])
                    all_fold_results.append(fold_predictions)

                logger.info(f"Fold {fold+1} results:")
                logger.info(f" Accuracy: {accuracy:.4f}")
                logger.info(f" F1 Macro: {f1_ma:.4f}")
                logger.info(f" F1 Weighted: {f1_wt:.4f}")
                logger.info(f" Recall: {recall:.4f}")
                logger.info(f" Precision: {precision:.4f}")
                logger.info(f" AUC: {auc_score:.4f}")
                logger.info(f" AUPR: {aupr:.4f}\n")

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
        avg_precision = statistics.mean(fold_precisions) if fold_precisions else 0.0
        std_precision = statistics.stdev(fold_precisions) if len(fold_precisions) > 1 else 0.0

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Recall', 'Precision', 'AUC', 'AUPR'],
            'Average': [avg_acc, avg_f1_ma, avg_f1_wt, avg_recall, avg_precision, avg_auc, avg_aupr],
            'StdDev': [std_acc, std_f1_ma, std_f1_wt, std_recall, std_precision, std_auc, std_aupr]
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

        if output_dir and fold_best_configs:
            try:
                results_df = pd.DataFrame(all_fold_results)
                config_save_path = os.path.join(output_dir, "cv_tuning_results.csv")
                config_df = pd.DataFrame(fold_best_configs)
                combined = pd.concat([results_df,config_df], axis=1)
                combined.to_csv(config_save_path)
                logger.info(f"Successfully saved CV tuning parameters to: {config_save_path}")
            except Exception as e:
                logger.warning(f"Failed to save CV tuning parameters: {e}")

        logger.info("Cross-Validation Results:\n")
        logger.info(f" Avg Accuracy: {avg_acc:.4f} +/- {std_acc:.4f}")
        logger.info(f" Avg F1 Macro: {avg_f1_ma:.4f} +/- {std_f1_ma:.4f}")
        logger.info(f" Avg F1 Weighted: {avg_f1_wt:.4f} +/- {std_f1_wt:.4f}")
        logger.info(f" Avg Recall: {avg_recall:.4f} +/- {std_recall:.4f}")
        logger.info(f" Avg Precision: {avg_precision:.4f} +/- {std_precision:.4f}")
        logger.info(f" Avg AUC: {avg_auc:.4f} +/- {std_auc:.4f}")
        logger.info(f" Avg AUPR: {avg_aupr:.4f} +/- {std_aupr:.4f}")

        return pd.DataFrame(), metrics_df, best_global_embeddings

def run_hyperparameter_tuning(X_train, y_train, adjacency_matrix, clinical_data, dpmon_params) -> Dict[str, Any]:
    """Run Ray Tune hyperparameter search with inner k-fold CV.

    Each trial trains one model per inner fold, epoch-synchronised,
    and reports the mean validation metrics. Asha early-stops on
    the averaged signal, which is far more stable than a single split.

    Args:

        X_train: Training features for this outer fold (pd.DataFrame).
        y_train: Training labels for this outer fold (pd.Series).
        adjacency_matrix: Feature-level adjacency matrix.
        clinical_data: Clinical covariates for the training fold.
        dpmon_params: Full DPMON parameter dictionary.

    Returns:

        Dict with the best hyperparameter configuration.
    """
    #os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"
    #os.environ["TUNE_DISABLE_IPY_WIDGETS"] = "1"

    device = setup_device(dpmon_params["gpu"], dpmon_params["cuda"])
    phenotype_col = dpmon_params["phenotype_col"]
    correlation_mode = dpmon_params["correlation_mode"]
    combined_omics_fold = X_train.join(y_train.rename(phenotype_col))
    omics_dataset = slice_omics_datasets(
        combined_omics_fold, adjacency_matrix, phenotype_col
    )
    omics_train_fold_list = [omics_dataset[0]]

    omics_network_tg = prepare_node_features(
        adjacency_matrix,
        omics_train_fold_list,
        clinical_data,
        phenotype_col,
        correlation_mode,
    )[0].to(device)

    pipeline_configs = {
        "gnn_layer_num": tune.choice([2, 3, 4]),
        "gnn_hidden_dim": tune.choice([32, 64]),
        "lr": tune.loguniform(1e-4, 8e-4),
        "weight_decay": tune.loguniform(1e-5, 5e-3),
        "nn_hidden_dim1": tune.choice([128, 256]),
        "nn_hidden_dim2": tune.choice([64]),
        "ae_encoding_dim": tune.choice([4, 8]),
        "ae_architecture": tune.choice(["original", "dynamic"]),
        "num_epochs": tune.choice([128, 256]),
        "gnn_dropout": tune.choice([0.4, 0.5, 0.6]),
        "gnn_activation": tune.choice(["relu", "elu"]),
        "dim_reduction": tune.choice(["ae", "linear", "mlp"]),
        "gat_heads": tune.choice([1, 2]),
    }

    # prepare inner k-fold splits and push to ray object store
    omics_data = omics_dataset[0]
    X = omics_data.drop([phenotype_col], axis=1)
    Y = omics_data[phenotype_col]

    n_inner_folds = dpmon_params.get("tune_inner_folds", 5)
    inner_cv = StratifiedKFold(
        n_splits=n_inner_folds, shuffle=True, random_state=dpmon_params["seed"]
    )

    # pre-tensor on every inner fold and store in ray object store, this is so trials fetch data zero-copy instead of re-splitting.
    fold_data_refs = []
    for tr_idx, val_idx in inner_cv.split(X, Y):
        y_tr = Y.iloc[tr_idx].values
        fold_tensors = {
            "X_train": torch.FloatTensor(X.iloc[tr_idx].values),
            "y_train": torch.LongTensor(y_tr),
            "X_val": torch.FloatTensor(X.iloc[val_idx].values),
            "y_val": torch.LongTensor(Y.iloc[val_idx].values),
        }

        fold_data_refs.append(ray.put(fold_tensors))

    omics_network_ref = ray.put(omics_network_tg.cpu())
    logger.info(f"Inner CV: {n_inner_folds} folds  |  X shape: {X.shape}  |  Graph nodes: {omics_network_tg.x.shape}")

    # pre-compute dims
    gnn_input_dim = omics_network_tg.x.shape[1]
    nn_input_dim = X.shape[1]
    nn_output_dim = Y.nunique()
    model_type = dpmon_params["model"]

    # trial function trains k models epoch-sync
    def tune_train_fn(config):
        device_inner = setup_device(dpmon_params["gpu"], dpmon_params["cuda"])
        omics_net = ray.get(omics_network_ref).to(device_inner)

        # load every inner fold onto device
        folds = []
        for ref in fold_data_refs:
            fd = ray.get(ref)
            fold_dict = {
                "X_train": fd["X_train"].to(device_inner),
                "y_train": fd["y_train"].to(device_inner),
                "X_val": fd["X_val"].to(device_inner),
                "y_val": fd["y_val"].to(device_inner),
                "criterion": nn.CrossEntropyLoss()
            }
            folds.append(fold_dict)

        # one model + optimizer per inner fold
        models, optimizers = [], []
        for _ in range(len(folds)):
            m = NeuralNetwork(
                model_type=model_type,
                gnn_input_dim=gnn_input_dim,
                gnn_hidden_dim=config["gnn_hidden_dim"],
                gnn_layer_num=config["gnn_layer_num"],
                gnn_dropout=config["gnn_dropout"],
                gnn_activation=config["gnn_activation"],
                dim_reduction=config["dim_reduction"],
                ae_encoding_dim=config["ae_encoding_dim"],
                ae_architecture=config["ae_architecture"],
                gat_heads=config["gat_heads"],
                nn_input_dim=nn_input_dim,
                nn_hidden_dim1=config["nn_hidden_dim1"],
                nn_hidden_dim2=config["nn_hidden_dim2"],
                nn_output_dim=nn_output_dim,
            ).to(device_inner)
            o = optim.Adam(
                m.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"],
            )
            models.append(m)
            optimizers.append(o)

        # epoch-sync training across all inner folds
        for epoch in range(config["num_epochs"]):
            epoch_val_losses = []
            epoch_val_accs   = []
            epoch_train_losses = []
            epoch_val_f1s = []
            epoch_val_auprs = []

            for fi, fold in enumerate(folds):
                # train step
                models[fi].train()
                optimizers[fi].zero_grad()
                out, _, _ = models[fi](fold["X_train"], omics_net)
                loss = fold["criterion"](out, fold["y_train"])
                loss.backward()
                optimizers[fi].step()
                epoch_train_losses.append(loss.item())

                # val step
                models[fi].eval()
                with torch.no_grad():
                    val_out, _, _ = models[fi](fold["X_val"], omics_net)
                    vl = fold["criterion"](val_out, fold["y_val"]).item()
                    _, preds = torch.max(val_out, 1)
                    probs = torch.softmax(val_out, dim=1)

                    y_true_np = fold["y_val"].cpu().numpy()
                    predicted_np = preds.cpu().numpy()
                    probs_np = probs.cpu().numpy()

                    va = (preds == fold["y_val"]).sum().item() / fold["y_val"].size(0)
                    f1_mac = f1_score(y_true_np, predicted_np, average='macro', zero_division=0)

                    try:
                        n_classes = probs_np.shape[1]
                        if n_classes == 2:
                            aupr = average_precision_score(y_true_np, probs_np[:, 1])
                        else:
                            y_bin = label_binarize(y_true_np, classes=range(n_classes))
                            aupr_scores = []
                            for i in range(n_classes):
                                if np.sum(y_bin[:, i]) > 0:
                                    aupr_scores.append(average_precision_score(y_bin[:, i], probs_np[:, i]))
                            aupr = np.mean(aupr_scores) if aupr_scores else 0.0
                    except:
                        aupr = 0.0

                epoch_val_losses.append(vl)
                epoch_val_accs.append(va)
                epoch_val_f1s.append(f1_mac)
                epoch_val_auprs.append(aupr)

            composite = 0.5 * float(np.mean(epoch_val_accs)) + 0.5 * float(np.mean(epoch_val_f1s))
            # report mean metrics
            metrics = {
                "val_loss": float(np.mean(epoch_val_losses)),
                "val_accuracy": float(np.mean(epoch_val_accs)),
                "val_f1_macro": float(np.mean(epoch_val_f1s)),
                "val_aupr": float(np.mean(epoch_val_auprs)),
                "train_loss": float(np.mean(epoch_train_losses)),
            }

            ckpt_dir = "trial_checkpoint"
            os.makedirs(ckpt_dir, exist_ok=True)

            torch.save(
                {"epoch": epoch, "model_state": models[0].state_dict()},
                os.path.join(ckpt_dir, "checkpoint.pt"),
            )
            tune.report(
                metrics=metrics,
                checkpoint=Checkpoint.from_directory(ckpt_dir),
            )

    # launch Ray Tune
    num_samples = dpmon_params.get("tune_trials", 20)
    seed_trials = dpmon_params.get("seed_trials", False)
    max_retries = 4

    if seed_trials:
        logger.debug(f"seed_trials=True: fixed seed {dpmon_params['seed']}")
    else:
        logger.debug("seed_trials=False: random hyperparameter sampling")

    scheduler = ASHAScheduler(grace_period=50, reduction_factor=2)
    stopper = TrialPlateauStopper(
        metric="val_f1_macro", mode="max",
        num_results=20, metric_threshold=0.01, grace_period=50,
    )

    def short_dirname_creator(trial):
        return f"T{trial.trial_id}"

    #dummy reporter that fixes Ray-rune screen-clearing for jupyter notebooks
    class SilentReporter(CLIReporter):
        def should_report(self, trials, done=False):
            return False

        def report(self, trials, done, *sys_info):
            pass

    use_gpu = bool(dpmon_params.get("gpu", False)) and torch.cuda.is_available()
    if dpmon_params.get("gpu", False) and not torch.cuda.is_available():
        logger.warning("gpu=True but CUDA not available; running on CPU.")

    cpu_per_trial = 2
    gpu_per_trial = 0.2 if use_gpu else 0.0

    for attempt in range(max_retries):
        try:
            search_alg = None
            if seed_trials:
                search_alg = BasicVariantGenerator(
                    random_state=np.random.RandomState(dpmon_params["seed"])
                )

            tuner = tune.Tuner(
                tune.with_resources(
                    tune_train_fn,
                    resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
                ),
                param_space=pipeline_configs,
                tune_config=tune.TuneConfig(
                    metric="val_f1_macro",
                    mode="max",
                    num_samples=num_samples,
                    scheduler=scheduler,
                    search_alg=search_alg,
                    trial_dirname_creator=short_dirname_creator,
                ),
                run_config=tune.RunConfig(
                    name="tune_dp",
                    verbose=0,
                    log_to_file=True,
                    stop=stopper,
                    storage_path=os.path.expanduser("~/ray_results"),
                    sync_config=tune.SyncConfig(sync_artifacts=False),
                    checkpoint_config=tune.CheckpointConfig(
                        num_to_keep=1,
                        checkpoint_score_attribute="val_f1_macro",
                        checkpoint_score_order="max",
                    ),progress_reporter=SilentReporter(),
                ),
            )

            results = tuner.fit()
            break

        except TuneError as e:
            msg = str(e)
            if "Trials did not complete" not in msg and "OutOfMemoryError" not in msg:
                raise
            new_num_samples = max(1, num_samples // 2)

            if use_gpu:
                new_gpu_per_trial = min(1.0, gpu_per_trial + 0.2)
            else:
                new_gpu_per_trial = 0.0

            if new_num_samples == num_samples and new_gpu_per_trial == gpu_per_trial:
                logger.error("Cannot reduce num_samples or increase gpu_per_trial any further. Aborting.")
                raise

            logger.warning(
                f"Ray Tune failed (attempt {attempt + 1}). "
                f"Adjusting resources -> num_samples: {num_samples} to {new_num_samples}, "
                f"gpu_per_trial: {gpu_per_trial:.2f} to {new_gpu_per_trial:.2f}."
            )
            num_samples = new_num_samples
            gpu_per_trial= new_gpu_per_trial
    else:
        raise RuntimeError("Hyperparameter tuning failed after max retries.")

    # extract best config
    best_result = results.get_best_result(metric="val_f1_macro", mode="max")
    best_config = best_result.config

    logger.info(f"Best trial config: {best_config}")
    logger.info(f"Best trial val_accuracy: {best_result.metrics.get('val_accuracy'):.4f}")
    logger.info(f"Best trial val_loss: {best_result.metrics.get('val_loss'):.4f}")
    logger.info(f"Best trial val_f1_macro: {best_result.metrics.get('val_f1_macro'):.4f}")
    logger.info(f"Best trial val_aupr: {best_result.metrics.get('val_aupr'):.4f}")

    # cleanup
    try:
        tune_dir = os.path.expanduser("~/ray_results/tune_dp")
        if os.path.exists(tune_dir):
            shutil.rmtree(tune_dir)
            logger.debug(f"Cleaned up tuning directory: {tune_dir}")
    except Exception as e:
        logger.warning(f"Could not clean up tuning directory: {e}")

    return best_config

def train_model(model, criterion, optimizer, train_features, train_labels, epoch_num):
    network = train_labels["omics_network"]
    labels = train_labels["labels"]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-6)
    model.train()
    for epoch in range(epoch_num):
        optimizer.zero_grad()
        outputs, _, _ = model(train_features, network)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            logger.debug(f"Epoch [{epoch+1}/{epoch_num}], Loss: {loss.item():.4f}")

    return model

class NeuralNetwork(nn.Module):
    """Core DPMON model combining GNN feature weighting and sample-level prediction.
        When using GAT with heads > 1, the GNN output is hidden_dim * heads.
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
        gnn_dropout: float = 0.,
        gnn_activation: str = "relu",
        dim_reduction: str = "ae",
        ae_architecture: str = "original",
        gat_heads: int = 1,
    ):
        super().__init__()
        self.model_type = model_type

        if model_type == "GCN":
            self.gnn = GCN(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                layer_num=gnn_layer_num,
                final_layer="none",
                dropout=gnn_dropout,
                activation=gnn_activation,
            )
            gnn_out_dim = gnn_hidden_dim

        elif model_type == "GAT":
            self.gnn = GAT(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                layer_num=gnn_layer_num,
                final_layer="none",
                dropout=gnn_dropout,
                activation=gnn_activation,
                heads=gat_heads,
            )
            # GAT output dim = hidden_dim * heads
            gnn_out_dim = gnn_hidden_dim * gat_heads

        elif model_type == "SAGE":
            self.gnn = SAGE(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                layer_num=gnn_layer_num,
                final_layer="none",
                dropout=gnn_dropout,
                activation=gnn_activation,
            )
            gnn_out_dim = gnn_hidden_dim

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
            gnn_out_dim = gnn_hidden_dim

        else:
            raise ValueError(f"Unsupported GNN model type: {model_type}")

        if dim_reduction == "ae":
            self.autoencoder = AutoEncoder(
                input_dim=gnn_out_dim, encoding_dim=1, architecture=ae_architecture
            )
            self.projection = nn.Identity()

        elif dim_reduction == "linear":
            self.autoencoder = AutoEncoder(
                input_dim=gnn_out_dim, encoding_dim=ae_encoding_dim, architecture=ae_architecture
            )
            self.projection = ScalarProjection(encoding_dim=ae_encoding_dim)

        elif dim_reduction == "mlp":
            self.autoencoder = AutoEncoder(
                input_dim=gnn_out_dim, encoding_dim=ae_encoding_dim, architecture=ae_architecture
            )
            self.projection = MLPProjection(encoding_dim=ae_encoding_dim, hidden_dim=8)

        else:
            raise ValueError(f"Unsupported dim_reduction: {dim_reduction}")

        self.predictor = DownstreamTaskNN(
            nn_input_dim, nn_hidden_dim1, nn_hidden_dim2, nn_output_dim
        )

    def forward(self, omics_dataset, omics_network_tg, clinical_tensor=None):
        # GNN embeddings
        omics_network_nodes_embedding = self.gnn.get_embeddings(omics_network_tg)

        # compress
        omics_network_nodes_embedding_ae = self.autoencoder(omics_network_nodes_embedding)

        # project to scalar weights
        feature_weights = self.projection(omics_network_nodes_embedding_ae)

        # reweight the original omics data (element-wise multiplication)
        omics_dataset_with_embeddings = torch.mul(
            omics_dataset,
            feature_weights.expand(omics_dataset.shape[1], omics_dataset.shape[0]).t(),
        )

        # leaving clinical_tensor parameter as pontential addition after dicussion with colleagues.
        # this way we would be making the final prediction on the scaled omics dataset + the clinical data.
        if clinical_tensor is not None and clinical_tensor.shape[1] > 0:
            predictor_input = torch.cat([omics_dataset_with_embeddings, clinical_tensor], dim=1)
        else:
            predictor_input = omics_dataset_with_embeddings

        # predict
        predictions = self.predictor(predictor_input)
        return predictions, omics_dataset_with_embeddings, omics_network_nodes_embedding

"""
DPMON AutoEncoder & NeuralNetwork:
    1. AutoEncoder: support both a hardcoded 3-layer encoder (input -> 8 -> 4 encoding_dim) and a 2-layer version (input -> input//2 -> encoding_dim).
    2. NeuralNetwork: Supports a `correlation_mode` passthrough for prepare_node_features.
"""

class AutoEncoder(nn.Module):
    """Compresses high-dimensional node embeddings into a lower-dimensional latent space.

    Args:

        input_dim: Input feature dimension (gnn_hidden_dim).
        encoding_dim: Output latent dimension.
        architecture: original or dynamic. "original" (input -> 8 -> 4 encoding_dim). "dynamic" (input -> input//2 -> encoding_dim).

    """

    def __init__(self, input_dim: int, encoding_dim: int, architecture: str = "original"):
        super().__init__()

        if architecture == "original":
            if encoding_dim == 1:
                # fixed bottleneck
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 8),
                    nn.ReLU(),
                    nn.Linear(8, 4),
                    nn.ReLU(),
                    nn.Linear(4, 1),
                )
            else:
                # EX1 if Tune picks 4: Flow is Input -> 8 -> 4. EX2: Tune picks 8: Flow is Input -> 16 -> 8
                intermediate_dim = encoding_dim * 2
                if intermediate_dim > input_dim:
                    intermediate_dim = input_dim

                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, intermediate_dim),
                    nn.ReLU(),
                    nn.Linear(intermediate_dim, encoding_dim),
                )

        elif architecture == "dynamic":
            if encoding_dim == 1:
                # 3-step funnel
                h1 = max(input_dim // 2, 8)
                h2 = max(input_dim // 4, 4)

                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, h1),
                    nn.ReLU(),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Linear(h2, 1),
                )
            else:
                # 2 step funnel for linear projections
                hidden_dim = max(input_dim // 2, encoding_dim * 2)
                if hidden_dim > input_dim:
                    hidden_dim = input_dim

                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, encoding_dim),
                )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x):
        return self.encoder(x)

class ScalarProjection(nn.Module):
    def __init__(self, encoding_dim):
        super().__init__()
        self.proj = nn.Linear(encoding_dim, 1)

    def forward(self, x):
        return self.proj(x)

class MLPProjection(nn.Module):
    def __init__(self, encoding_dim, hidden_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)

class DownstreamTaskNN(nn.Module):
    """MLP for final prediction - outputs raw logits."""

    def __init__(self, input_size, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
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
