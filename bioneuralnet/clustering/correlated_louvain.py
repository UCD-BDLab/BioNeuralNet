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

from ray import tune
from ray.tune import CLIReporter
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from bioneuralnet.utils import set_seed, get_logger

logger = get_logger(__name__)

class CorrelatedLouvain:
    """Correlated Louvain community detection on an omics network.

    Extends standard Louvain modularity with an additional term that captures either phenotype correlation (supervised mode, when ``Y`` is provided) or omics cohesion (unsupervised mode, when only ``B`` is provided). The two terms are combined via weights ``k3`` (modularity) and ``k4`` (correlation/cohesion), and Ray Tune can optionally search over these weights.

    Args:

        G (nx.Graph): Input graph with nodes corresponding to omics features.
        B (pd.DataFrame): Omics data matrix (samples x features) whose columns correspond to graph nodes.
        Y (Optional[pd.Series or pd.DataFrame]): Phenotype or target values for supervised correlation; if None, runs in unsupervised cohesion mode.
        k3 (float): Weight on the standard modularity term in the combined quality score.
        k4 (float): Weight on the correlation/cohesion term in the combined quality score.
        weight (str): Edge attribute name in ``G`` to use as the weight (for example, "weight").
        tune (bool): If True, enable hyperparameter tuning over ``k4`` via Ray Tune.
        gpu (bool): If True and CUDA is available, use GPU for supported computations.
        seed (Optional[int]): Random seed for NumPy, PyTorch, and CUDA (if available).

    Attributes:

        G (nx.Graph): Copy of the input graph.
        B (pd.DataFrame): Copy of the omics data.
        Y: Stored phenotype or target data.
        K3 (float): Stored modularity weight.
        K4 (float): Stored correlation or cohesion weight.
        weight (str): Stored edge-weight attribute name.
        tune (bool): Whether tuning mode is enabled.
        device (torch.device): Selected compute device ("cpu" or "cuda").
        clusters (dict): Mapping from community id to list of node identifiers.
        partition (dict): Node-to-community mapping, populated after ``run()`` is called.

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
            f"CorrelatedLouvain(k3={self.K3}, k4={self.K4}, "
            f"nodes={self.G.number_of_nodes()}, edges={self.G.number_of_edges()}, "
            f"features={self.B.shape[1] if self.B is not None else 0})"
        )

        self.logger.debug(
            f"Initialized CorrelatedLouvain with k3 = {self.K3}, k4 = {self.K4}, "
        )
        if self.B is not None:
            self.logger.debug(f"Original omics data shape: {self.B.shape}")

        self.logger.debug(f"Original graph has {self.G.number_of_nodes()} nodes.")

        if self.B is not None:
            self.logger.debug(f"Final omics data shape: {self.B.shape}")

        self.logger.debug(
            f"Graph has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges."
        )

        self.seed = seed
        self.gpu = gpu

        if seed is not None:
            set_seed(seed)

        self.clusters: dict[Any, Any] = {}
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Initialized Correlated Louvain. device={self.device}")

    def _compute_community_cohesion(self, nodes) -> float:
        """Compute average absolute pairwise correlation of omics features within a community.

        Uses columns in ``B`` matching the given nodes, drops all-zero columns, and returns the mean of the upper-triangle absolute correlation matrix; returns 0.0 if fewer than two valid features remain.
        """
        if self.B is None:
            return 0.0

        node_cols = []
        for n in nodes:
            name = str(n)
            if name in self.B.columns:
                node_cols.append(name)

        if len(node_cols) < 2:
            return 0.0

        B_sub = self.B.loc[:, node_cols]
        B_sub = B_sub.loc[:, (B_sub != 0).any(axis=0)]
        if B_sub.shape[1] < 2:
            return 0.0

        corr_matrix = B_sub.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        return upper_tri.stack().mean()

    def _compute_community_correlation(self, nodes) -> tuple:
        """
        Compute the Pearson correlation between the first principal component (PC1) of the omics data (for the given nodes) and the phenotype.
        Drops columns that are completely zero.
        """
        try:
            self.logger.debug(
                f"Computing community correlation for {len(nodes)} nodes..."
            )
            node_cols = [str(n) for n in nodes if str(n) in self.B.columns]
            if not node_cols:
                self.logger.debug(
                    "No valid columns found for these nodes; returning (0.0, 1.0)."
                )
                return 0.0, 1.0
            B_sub = self.B.loc[:, node_cols]
            zero_mask = (B_sub == 0).all(axis=0)
            num_zero_columns = int(zero_mask.sum())
            if num_zero_columns > 0:
                self.logger.debug(
                    f"WARNING: {num_zero_columns} columns are all zeros in community subset."
                )
            B_sub = B_sub.loc[:, ~zero_mask]
            if B_sub.shape[1] == 0:
                self.logger.debug("All columns dropped; returning (0.0, 1.0).")
                return 0.0, 1.0

            self.logger.debug(
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
            self.logger.error(f"Error in _compute_community_correlation: {e}")
            raise

    def _quality_correlated(self, partition) -> float:
        """
        Compute the overall quality metric as:
            Q* = k3 * Q + k4 * avg_abs_corr (supervised)
            Q* = k3 * Q + k4 * avg_cohesion (unsupervised)
        """
        Q = original_modularity(partition, self.G, self.weight)

        # Unsupervised mode: Y is None
        if self.Y is None:
            self.logger.debug("Phenotype data not provided; using unsupervised cohesion.")

            if self.B is None:
                return Q

            community_cohesions = []
            for com in set(partition.values()):
                nodes = [n for n in self.G.nodes() if partition[n] == com]
                if len(nodes) < 2:
                    continue
                cohesion = self._compute_community_cohesion(nodes)
                community_cohesions.append(cohesion)

            avg_cohesion = np.mean(community_cohesions) if community_cohesions else 0.0
            quality = self.K3 * Q + self.K4 * avg_cohesion
            self.logger.debug(
                f"Computed quality (unsupervised): Q = {Q:.4f}, avg_cohesion = {avg_cohesion:.4f}, combined = {quality:.4f}"
            )
            return quality

        # Supervised mode: Y is provided
        if self.B is None:
            self.logger.debug(
                "Omics data not provided; returning standard modularity."
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
            f"Computed quality (supervised): Q = {Q:.4f}, avg_corr = {avg_corr:.4f}, combined = {quality:.4f}"
        )
        return quality

    def run(self, as_dfs: bool = False) -> Union[dict, list]:
        """Run correlated Louvain clustering and optionally return cluster tables.

        Behavior depends on the combination of ``tune`` and ``as_dfs``: if ``tune=True`` and ``as_dfs=False``, Ray Tune is used to search over ``k4`` and a dictionary with the best configuration is returned; if ``tune=True`` and ``as_dfs=True``, tuning is run first and then a standard detection using the tuned parameters returns per-cluster DataFrames; if ``tune=False``, a single correlated Louvain partition is computed and either the raw partition or per-cluster DataFrames are returned.

        Args:

            as_dfs (bool): If True, convert the final partition to a list of per-cluster DataFrames (one DataFrame per cluster with more than 2 nodes); if False, return the raw partition or tuning result.

        Returns:

            Union[dict, list]: If ``tune=True`` and ``as_dfs=False``, a dict of the form ``{"best_config": dict}`` with the best Ray Tune configuration; if ``as_dfs=True``, a list of ``pd.DataFrame`` objects, one per cluster with more than 2 nodes; otherwise, a partition dict mapping each node to a community id.

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

            n_clusters = len(set(partition.values()))
            self.logger.info(
                f"CorrelatedLouvain found {n_clusters} communities "
                f"(nodes={self.G.number_of_nodes()})"
            )

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
