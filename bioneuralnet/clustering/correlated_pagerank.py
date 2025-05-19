from typing import List, Tuple, Dict, Any,Optional
import pandas as pd
import networkx as nx
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

from ray import tune
from ray.tune import CLIReporter
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
import torch

from ..utils.logger import get_logger

class CorrelatedPageRank:
    """
    PageRank Class for Clustering Nodes Based on Personalized PageRank.

    This class handles the execution of the Personalized PageRank algorithm
    and identification of clusters based on sweep cuts.

    Attributes:

        alpha (float): Damping factor for PageRank.
        max_iter (int): Maximum number of iterations for PageRank convergence.
        tol (float): Tolerance for convergence.
        k (float): Weighting factor for composite correlation-conductance score.
    """

    def __init__(
        self,
        graph: nx.Graph,
        omics_data: pd.DataFrame,
        phenotype_data: pd.DataFrame,
        alpha: float = 0.9,
        max_iter: int = 100,
        tol: float = 1e-6,
        k: float = 0.5,
        tune: bool = False,
        gpu: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initializes the PageRank instance with direct data structures.

        Args:

            graph (nx.Graph): NetworkX graph object representing the network.
            omics_data (pd.DataFrame): Omics data DataFrame.
            phenotype_data (pd.DataFrame): Phenotype data Series.
            alpha (float, optional): Damping factor for PageRank. Defaults to 0.9.
            max_iter (int, optional): Maximum iterations for PageRank. Defaults to 100.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
            k (float, optional): Weighting factor for composite score. Defaults to 0.9.
        """
        self.G = graph
        self.B = omics_data
        self.Y = phenotype_data.squeeze()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.k = k
        self.tune = tune

        self.logger = get_logger(__name__)
        self.logger.info("Initialized PageRank with the following parameters:")
        self.logger.info(
            f"Graph: NetworkX Graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges."
        )
        self.logger.info(f"Omics Data: DataFrame with shape {self.B.shape}.")
        self.logger.info(f"Phenotype Data: Series with {len(self.Y)} samples.")
        self.logger.info(f"Alpha: {self.alpha}")
        self.logger.info(f"Max Iterations: {self.max_iter}")
        self.logger.info(f"Tolerance: {self.tol}")
        self.logger.info(f"K (Composite Score Weight): {self.k}")
        self._validate_inputs()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.seed = seed
        self.gpu = gpu
        self.results: dict[str, float] = {}

        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized Correlated Louvain. device={self.device}")


    def _validate_inputs(self):
        """
        Validates the consistency of input data structures.
        """
        try:
            if not isinstance(self.G, nx.Graph):
                raise TypeError("graph must be a networkx.Graph instance.")

            if not isinstance(self.B, pd.DataFrame):
                raise TypeError("omics_data must be a pandas DataFrame.")

            if not isinstance(self.Y, pd.Series):
                raise TypeError("phenotype_data must be a pandas Series.")

            graph_nodes = set(self.G.nodes())
            omics_nodes = set(self.B.columns)
            phenotype_nodes = set(self.Y.index)

            if not graph_nodes.issubset(omics_nodes):
                missing = graph_nodes - omics_nodes
                raise ValueError(f"Omics data is missing nodes: {missing}")

        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            raise

    def phen_omics_corr(self, nodes: List[Any]= []) -> Tuple[float, str]:
        """
        Calculates the Pearson correlation between the PCA of omics data and phenotype.

        Args:
            nodes (List[Any]): List of node identifiers to include in the calculation.

        Returns:
            Tuple[float, str]: Correlation coefficient and formatted correlation with p-value.
        """
        try:

            B_sub = self.B[nodes]
            scaler = StandardScaler()
            scaled = scaler.fit_transform(B_sub)

            pca = PCA(n_components=1)
            g1 = pca.fit_transform(scaled).flatten()
            g2 = self.Y

            corr, pvalue = pearsonr(g1, g2)
            corr = round(corr, 2)
            p_value = format(pvalue, ".3g")
            corr_pvalue = f"{corr} ({p_value})"
            return corr, corr_pvalue

        except Exception as e:
            self.logger.error(f"Error in phen_omics_corr: {e}")
            raise

    def sweep_cut(
        self, p: Dict[Any, float] = {}) -> Tuple[List[Any], int, float, float, float, str]:
        try:
            best_cluster = set()
            min_comp_score = float("inf")
            best_len = 0
            best_cond = 1.0
            best_corr = 0.0
            best_corr_pval = ""

            degrees = dict(self.G.degree(weight="weight"))
            vec = sorted(
                [
                    (p[node] / degrees[node] if degrees[node] > 0 else 0, node)
                    for node in p.keys()
                ],
                reverse=True,
            )

            current_cluster = set()
            for i, (val, node) in enumerate(vec):
                self.logger.debug(
                    f"Iteration {i}: Adding node {node} with norm value {val:.3f}"
                )
                if val == 0:
                    continue

                current_cluster.add(node)

                if len(current_cluster) < len(self.G.nodes()):
                    vol_S = sum(
                        d for n, d in self.G.degree(current_cluster, weight="weight")
                    )
                    vol_T = sum(
                        d
                        for n, d in self.G.degree(
                            set(self.G.nodes()) - current_cluster, weight="weight"
                        )
                    )

                    if min(vol_S, vol_T) == 0:
                        self.logger.warning(
                            f"Skipping iteration {i} due to zero volume (vol_S={vol_S}, vol_T={vol_T})."
                        )
                        continue

                    cluster_cond = nx.conductance(
                        self.G, current_cluster, weight="weight"
                    )
                    cluster_corr, corr_pvalue = self.phen_omics_corr(
                        list(current_cluster)
                    )
                    composite_score = (1 - self.k) * cluster_cond + self.k * (
                        -abs(cluster_corr)
                    )

                    self.logger.debug(
                        f"Cluster size={len(current_cluster)}, cond={cluster_cond:.3f}, "
                        f"corr={cluster_corr:.3f}, composite={composite_score:.3f}"
                    )

                    # cluster must be larger than 10 nodes
                    if len(current_cluster) >= 5 and composite_score < min_comp_score:
                        min_comp_score = composite_score
                        best_cluster = set(current_cluster)
                        best_len = len(current_cluster)
                        best_cond = cluster_cond
                        best_corr = cluster_corr
                        best_corr_pval = corr_pvalue

            if best_cluster:
                return (
                    list(best_cluster),
                    best_len,
                    round(best_cond, 3),
                    round(best_corr, 3),
                    round(min_comp_score, 3),
                    best_corr_pval,
                )
            else:
                self.logger.warning(
                    "No valid sweep cut found. Returning empty cluster."
                )
                return [], 0, 0.0, 0.0, 0.0, "0 (1.0)"

        except Exception as e:
            self.logger.error(f"Error in sweep_cut: {e}")
            raise

    def generate_weighted_personalization(self, nodes: List[Any] = []) -> Dict[Any, float]:
        """
        Generates a weighted personalization vector for PageRank.

        Args:
            nodes (List[Any]): List of node identifiers to consider.

        Returns:
            Dict[Any, float]: Personalization vector with weights for each node.
        """
        try:
            total_corr, _ = self.phen_omics_corr(nodes)
            corr_contribution = []

            for i in range(len(nodes)):
                nodes_excl = nodes[:i] + nodes[i + 1 :]
                if not nodes_excl:
                    contribution = 0.0
                else:
                    corr_excl, _ = self.phen_omics_corr(nodes_excl)
                    contribution = abs(corr_excl) - abs(total_corr)
                corr_contribution.append(contribution)

            max_contribution = max(corr_contribution, key=abs) if corr_contribution else 1
            if max_contribution == 0:
                max_contribution = 1

            weighted_personalization = {
                node: self.alpha * (corr_contribution[i] / max_contribution)
                for i, node in enumerate(nodes)
            }
            return weighted_personalization

        except Exception as e:
            self.logger.error(f"Error in generate_weighted_personalization: {e}")
            raise


    def run_pagerank_clustering(self, seed_nodes: List[Any] = []) -> Dict[str, Any]:
        """
        Executes the PageRank clustering algorithm.

        Args:
            seed_nodes (List[Any]): List of seed node identifiers for personalization.

        Returns:
            Dict[str, Any]: Dictionary containing clustering results.
        """
        if not seed_nodes:
            self.logger.error("No seed nodes provided for PageRank clustering.")
            raise ValueError("Seed nodes list cannot be empty.")

        if not set(seed_nodes).issubset(set(self.G.nodes())):
            missing = set(seed_nodes) - set(self.G.nodes())
            self.logger.error(f"Seed nodes not in graph: {missing}")
            raise ValueError(f"Seed nodes not in graph: {missing}")

        try:
            personalization = self.generate_weighted_personalization(seed_nodes)
            self.logger.info(
                f"Generated personalization vector for seed nodes: {seed_nodes}"
            )

            try:
                p = nx.pagerank(
                    self.G,
                    alpha=self.alpha,
                    personalization=personalization,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    weight="weight",
                )
            except nx.exception.PowerIterationFailedConvergence as e:
                self.logger.warning(
                    f"PageRank did not converge in {self.max_iter} iterations. Retrying with increased max_iter."
                )
                # retry with doubled iterations
                p = nx.pagerank(
                    self.G,
                    alpha=self.alpha,
                    personalization=personalization,
                    max_iter=self.max_iter * 2,
                    tol=self.tol,
                    weight="weight",
                )

            self.logger.info("PageRank computation completed.")

            nodes, n, cond, corr, min_corr, pval = self.sweep_cut(p)
            if not nodes:
                self.logger.warning("Sweep cut did not identify any cluster.")
            else:
                self.logger.info(
                    f"Sweep cut resulted in cluster of size {n} with conductance {cond} and correlation {corr}."
                )

            results = {
                "cluster_nodes": nodes,
                "cluster_size": n,
                "conductance": cond,
                "correlation": corr,
                "composite_score": min_corr,
                "correlation_pvalue": pval,
            }

            return results

        except Exception as e:
            self.logger.error(f"Error in run_pagerank_clustering: {e}")
            raise


    def run(self, seed_nodes: List[Any] = []) -> Dict[str, Any]:
        """
        Executes the correlated PageRank clustering pipeline.

        **Steps:**

        1. **Initializing Clustering**:
            - Receives a list of seed nodes to personalize the PageRank algorithm.
            - Prepares the input graph and relevant parameters for clustering.

        2. **PageRank Execution**:
            - Applies the PageRank algorithm with personalization based on the seed nodes.
            - Computes node scores and determines cluster memberships.

        3. **Result Compilation**:
            - Compiles clustering results, including cluster sizes and node memberships, into a dictionary.
            - Logs the successful completion of the clustering process.

        **Args**:
            seed_nodes (List[Any]):
                - A list of node identifiers used as seed nodes for personalized PageRank.
                - These nodes influence the clustering process by biasing the algorithm.

        **Returns**: Dict[str, Any]

            - A dictionary containing the clustering results. Keys may include:
                - `clusters`: Lists of nodes grouped into clusters.
                - `scores`: PageRank scores for each node.
                - `metadata`: Additional metrics or details about the clustering process.

        **Raises**:

            - ValueError: If the input graph is empty or seed nodes are invalid.
            - Exception: For any unexpected errors during clustering execution.

        **Notes**:

            - Seed nodes strongly influence the clustering outcome; select them carefully based on prior knowledge or experimental goals.
            - The PageRank algorithm requires a well-defined and connected graph to produce meaningful results.
            - Results are sensitive to the alpha (damping factor) and other hyperparameters.

        """
        if self.tune:
            best_config = self.run_tuning(num_samples=10)
            self.logger.info("Tuning completed successfully.")
            return {"best_config": best_config}
        try:
            results = self.run_pagerank_clustering(seed_nodes)
            self.logger.info("PageRank clustering completed successfully.")
            return results
        except Exception as e:
            self.logger.error(f"Error in run method: {e}")
            raise

    def get_quality(self) -> float:
        """
        Returns the composite score (or correlation) from the latest clustering run.
        """
        if hasattr(self, "results"):
            #  return the composite score
            return self.results.get("composite_score", 0.0)
        else:
            # run clustering on all nodes and then return the score.
            self.results = self.run_pagerank_clustering(seed_nodes=list(self.G.nodes()))
            return self.results.get("composite_score", 0.0)

    def _tune_helper(self, config):
        alpha = config["alpha"]
        max_iter = config["max_iter"]
        tol = config["tol"]
        k = config["k"]

        tuned_instance = CorrelatedPageRank(
            graph=self.G,
            omics_data=self.B,
            phenotype_data=self.Y,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            k=k,
            gpu=(self.device.type == "cuda"),
            seed=self.seed,
            tune=False,
        )
        #  tuning uses all nodes as seed nodes.
        tuned_instance.run_pagerank_clustering(seed_nodes=list(self.G.nodes()))
        quality = tuned_instance.get_quality()
        session.report({"quality": quality})

    def run_tuning(self, num_samples: int = 10) -> Dict[str, Any]:
        search_config = {
            "alpha": tune.grid_search([0.8, 0.85, 0.9, 0.95]),
            "k": tune.grid_search([0.5, 0.6, 0.7, 0.8]),
            "max_iter": tune.choice([100, 500, 1000]),
            "tol": tune.loguniform(1e-3, 1e-5),
        }
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

        analysis = tune.run(
            tune.with_parameters(self._tune_helper),
            config=search_config,
            verbose=0,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            storage_path=os.path.expanduser("~/pr"),
            trial_dirname_creator=short_dirname_creator,
            resources_per_trial=resources,
            name="l",
        )

        best_config = analysis.get_best_config(metric="quality", mode="max")
        self.logger.info(f"Best hyperparameters found: {best_config}")
        return best_config
