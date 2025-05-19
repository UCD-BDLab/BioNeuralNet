import numpy as np
import networkx as nx
import pandas as pd
import torch
import os
from typing import Optional, Union, Any

from community.community_louvain import (
    modularity as original_modularity,
    best_partition,
)
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ..utils.logger import get_logger

from ray import tune
from ray.tune import CLIReporter
from ray.air import session
from ray.tune.schedulers import ASHAScheduler

logger = get_logger(__name__)


class CorrelatedLouvain:
    """
    CorrelatedLouvain Class for Community Detection with Correlated Omics Data.
    Attributes:

        G (nx.Graph): NetworkX graph object.
        B (pd.DataFrame): Omics data.
        Y (pd.DataFrame): Phenotype data.
        k3 (float): Weight for Correlated Louvain.
        k4 (float): Weight for Correlated Louvain.
        weight (str): Edge weight parameter name.
        tune (bool): Flag to enable tuning of parameters
    """
    def __init__(
        self,
        G: nx.Graph,
        B: pd.DataFrame,
        Y=None,
        k3: float = 0.2,
        k4: float = 0.8,
        weight: str = "weight",
        tune: bool = False,
        gpu: bool = False,
        seed: Optional[int] = None,
    ):
        self.logger = get_logger(__name__)
        self.G = G.copy()
        self.B = B.copy()
        self.Y = Y
        self.K3 = k3
        self.K4 = k4
        self.weight = weight
        self.tune = tune

        self.logger.info(
            f"Initialized CorrelatedLouvain with k3 = {self.K3}, k4 = {self.K4}, "
        )
        if self.B is not None:
            self.logger.info(f"Original omics data shape: {self.B.shape}")

        self.logger.info(f"Original graph has {self.G.number_of_nodes()} nodes.")

        if self.B is not None:
            self.logger.info(f"Final omics data shape: {self.B.shape}")
        self.logger.info(
            f"Graph has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges."
        )

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        self.seed = seed
        self.gpu = gpu
        self.clusters: dict[Any, Any] = {}

        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized Correlated Louvain. device={self.device}")


    def _compute_community_correlation(self, nodes) -> tuple:
        """
        Compute the Pearson correlation between the first principal component (PC1) of the omics data
        (for the given nodes) and the phenotype.
        Drops columns that are completely zero.
        """
        try:
            self.logger.info(
                f"Computing community correlation for {len(nodes)} nodes..."
            )
            node_cols = [str(n) for n in nodes if str(n) in self.B.columns]
            if not node_cols:
                self.logger.info(
                    "No valid columns found for these nodes; returning (0.0, 1.0)."
                )
                return 0.0, 1.0
            B_sub = self.B.loc[:, node_cols]
            zero_mask = (B_sub == 0).all(axis=0)
            num_zero_columns = int(zero_mask.sum())
            if num_zero_columns > 0:
                self.logger.info(
                    f"WARNING: {num_zero_columns} columns are all zeros in community subset."
                )
            B_sub = B_sub.loc[:, ~zero_mask]
            if B_sub.shape[1] == 0:
                self.logger.info("All columns dropped; returning (0.0, 1.0).")
                return 0.0, 1.0

            self.logger.info(
                f"B_sub shape: {B_sub.shape}, first few columns: {node_cols[:5]}"
            )
            scaler = StandardScaler()
            scaled = scaler.fit_transform(B_sub)
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(scaled).flatten()
            target = (
                self.Y.iloc[:, 0].values
                if isinstance(self.Y, pd.DataFrame)
                else self.Y.values
            )
            corr, pvalue = pearsonr(pc1, target)
            return corr, pvalue
        except Exception as e:
            self.logger.info(f"Error in _compute_community_correlation: {e}")
            raise

    def _quality_correlated(self, partition) -> float:
        """
        Compute the overall quality metric as:
            Q* = k3 * Q + k4 * avg_abs_corr,
        where Q is the standard modularity and avg_abs_corr is the average absolute Pearson correlation
        (computed over communities with at least 2 nodes).
        """
        Q = original_modularity(partition, self.G, self.weight)
        if self.B is None or self.Y is None:
            self.logger.info(
                "Omics/phenotype data not provided; returning standard modularity."
            )
            return Q
        community_corrs = []
        for com in set(partition.values()):
            nodes = [n for n in self.G.nodes() if partition[n] == com]
            if len(nodes) < 2:
                continue
            corr, _ = self._compute_community_correlation(nodes)
            community_corrs.append(abs(corr))
        avg_corr = np.mean(community_corrs) if community_corrs else 0.0
        quality = self.K3 * Q + self.K4 * avg_corr
        self.logger.info(
            f"Computed quality: Q = {Q:.4f}, avg_corr = {avg_corr:.4f}, combined = {quality:.4f}"
        )
        return quality

    def run(self, as_dfs: bool = False) -> Union[dict, list]:
        """
        Run correlated Louvain clustering.

        If as_dfs is True, returns a list of adjacency matrices (DataFrames),
        where each adjacency matrix represents a cluster with more than 2 nodes.
        Otherwise, returns the partition dictionary.

        If tune is True and as_dfs is False, hyperparameter tuning is performed and the best parameters are returned.
        If tune is True and as_dfs is True, tuning is performed, and then standard detection is run using the tuned parameters.
        """
        if self.tune and not as_dfs:
            self.logger.info("Tuning enabled. Running hyperparameter tuning...")
            best_config = self.run_tuning(num_samples=10)
            self.logger.info("Tuning completed successfully.")
            return {"best_config": best_config}

        elif self.tune and as_dfs:
            self.logger.info("Tuning enabled and adjacency matrices output requested.")
            best_config = self.run_tuning(num_samples=10)
            tuned_k4 = best_config.get("k4", 0.8)
            tuned_k3 = 1.0 - tuned_k4
            tuned_instance = CorrelatedLouvain(
                G=self.G,
                B=self.B,
                Y=self.Y,
                k3=tuned_k3,
                k4=tuned_k4,
                weight=self.weight,
                tune=False,
                gpu=self.gpu,
                seed=self.seed,
            )
            return tuned_instance.run(as_dfs=True)

        else:
            self.logger.info("Running standard community detection...")
            partition = best_partition(self.G, weight=self.weight)
            quality = self._quality_correlated(partition)
            self.logger.info(f"Final quality: {quality:.4f}")
            self.partition = partition

        if as_dfs:
            self.logger.info("Raw partition output:", self.partition)
            clusters_dfs = self.partition_to_adjacency(self.partition)
            print(f"Returning {len(clusters_dfs)} clusters after filtering")
            return clusters_dfs

        else:
            return partition

    def partition_to_adjacency(self, partition: dict) -> list:
        """
        Convert the partition dictionary into a list of adjacency matrices (DataFrames),
        where each adjacency matrix represents a cluster with more than 2 nodes.
        """

        for node, cl in partition.items():
            self.clusters.setdefault(cl, []).append(node)

        self.logger.debug(f"Total detected clusters: {len(self.clusters)}")

        adjacency_matrices = []
        for cl, nodes in self.clusters.items():
            self.logger.debug(f"Cluster {cl} size: {len(nodes)}")
            if len(nodes) > 2:
                valid_nodes = list(set(nodes).intersection(set(self.B.columns)))
                if valid_nodes:
                    adjacency_matrix = self.B.loc[:, valid_nodes].fillna(0)
                    adjacency_matrices.append(adjacency_matrix)

        print(f"Clusters with >2 nodes: {len(adjacency_matrices)}")

        return adjacency_matrices

    def get_quality(self) -> float:
        if not hasattr(self, "partition"):
            raise ValueError("No partition computed. Call run() first.")
        return self._quality_correlated(self.partition)

    def _tune_helper(self, config):
        k4 = config["k4"]
        k3 = 1.0 - k4
        tuned_instance = CorrelatedLouvain(
            G=self.G,
            B=self.B,
            Y=self.Y,
            k3=k3,
            k4=k4,
            weight=self.weight,
            gpu=self.gpu,
            seed=self.seed,
            tune=False,
        )
        tuned_instance.run()
        quality = tuned_instance.get_quality()
        session.report({"quality": quality})

    def run_tuning(self, num_samples=10):
        search_config = {"k4": tune.grid_search([0.5, 0.6, 0.7, 0.8, 0.9])}
        scheduler = ASHAScheduler(
            metric="quality",
            mode="max",
            grace_period=1,
            reduction_factor=2,
        )
        reporter = CLIReporter(metric_columns=["quality", "training_iteration"])

        def short_dirname_creator(trial):
            return f"_{trial.trial_id}"

        resources = {"cpu": 1, "gpu": 1} if self.device.type == "cuda" else {"cpu": 1, "gpu": 0}

        self.logger.info("Starting hyperparameter tuning...")
        analysis = tune.run(
            tune.with_parameters(self._tune_helper),
            config=search_config,
            verbose=0,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            storage_path=os.path.expanduser("~/cl"),
            trial_dirname_creator=short_dirname_creator,
            resources_per_trial=resources,
            name="l",
        )

        best_config = analysis.get_best_config(metric="quality", mode="max")
        self.logger.info(f"Best hyperparameters found: {best_config}")
        return best_config
