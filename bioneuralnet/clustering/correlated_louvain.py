import numpy as np
import networkx as nx
import pandas as pd
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

# review
class CorrelatedLouvain:
    def __init__(
        self,
        G: nx.Graph,
        B: pd.DataFrame,
        Y=None,
        k3: float = 0.2,
        k4: float = 0.8,
        weight: str = "weight",
        tune: bool = False,
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

    def run(self) -> dict:
        if self.tune:
            self.logger.info("Tuning enabled. Running hyperparameter tuning...")
            best_config = self.run_tuning(num_samples=5)
            self.logger.info("Tuning completed successfully.")
            return {"best_config": best_config}
        else:
            self.logger.info("Running standard community detection...")
            partition = best_partition(self.G, weight=self.weight)
            quality = self._quality_correlated(partition)
            self.logger.info(f"Final quality: {quality:.4f}")
            self.partition = partition
            return partition

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

        analysis = tune.run(
            tune.with_parameters(self._tune_helper),
            config=search_config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            trial_dirname_creator=short_dirname_creator,
            name="l",
        )

        best_config = analysis.get_best_config(metric="quality", mode="max")
        self.logger.info(f"Best hyperparameters found: {best_config}")
        return best_config
