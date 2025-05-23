import networkx as nx
import pandas as pd
import numpy as np
from typing import Union, Optional
import torch

from bioneuralnet.clustering.correlated_pagerank import CorrelatedPageRank
from bioneuralnet.clustering.correlated_louvain import CorrelatedLouvain

from ..utils.logger import get_logger

logger = get_logger(__name__)

class HybridLouvain:
    """
    HybridLouvain Class that combines Correlated Louvain and Correlated PageRank for community detection.

    Attributes:

        G (nx.Graph): NetworkX graph object.
        B (pd.DataFrame): Omics data.
        Y (pd.DataFrame): Phenotype data.
        k3 (float): Weight for Correlated Louvain.
        k4 (float): Weight for Correlated Louvain.
        max_iter (int): Maximum number of iterations.
        weight (str): Edge weight parameter name.
        tune (bool): Flag to enable tuning of parameters
    """
    def __init__(
        self,
        G: nx.Graph,
        B: pd.DataFrame,
        Y: pd.DataFrame,
        k3: float = 0.2,
        k4: float = 0.8,
        max_iter: int = 3,
        weight: str = "weight",
        gpu: bool = False,
        seed: Optional[int] = None,
        tune: Optional[bool] = False,

    ):
        self.logger = get_logger(__name__)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.gpu = gpu
        self.seed = seed
        self.logger.info("Initializing HybridLouvain...")

        self.G = G
        graph_nodes = set(map(str, G.nodes()))

        omics_cols = set(B.columns.astype(str))
        keep_omics = B.columns[B.columns.astype(str).isin(graph_nodes)]
        dropped_omics = sorted(omics_cols - graph_nodes)
        if dropped_omics:
            self.logger.info(
                f"Dropping {len(dropped_omics)} omics columns not in graph: "
                f"{dropped_omics[:5]}{'…' if len(dropped_omics) > 5 else ''}"
            )
        self.B = B.loc[:, keep_omics]

        # pheno_idx = set(Y.index.astype(str))
        # isin_mask = Y.index.astype(str).isin(graph_nodes)
        # keep_pheno = Y.index[isin_mask]
        # dropped_pheno = sorted(pheno_idx - graph_nodes)
        # if dropped_pheno:
        #     self.logger.info(
        #         f"Dropping {len(dropped_pheno)} phenotype rows not in graph: "
        #         f"{dropped_pheno[:5]}{'…' if len(dropped_pheno) > 5 else ''}"
        #     )
        if isinstance(Y, pd.DataFrame):
            self.Y = Y.squeeze()
        elif isinstance(Y, pd.Series):
            self.Y = Y

        self.k3 = k3
        self.k4 = k4
        self.weight = weight
        self.max_iter = max_iter
        self.tune = tune

        self.logger.info(
            f"Initialized HybridLouvain with {len(self.G)} graph nodes, "
            f"{self.B.shape[1]} omics columns, {self.Y.shape[0]} phenotype rows; "
            f"max_iter={max_iter}, k3={k3}, k4={k4}, tune={tune}"
        )

    def run(self, as_dfs: bool = False) -> Union[dict, list]:
        iteration = 0
        prev_size = len(self.G.nodes())
        current_partition = None
        all_clusters = {}

        while iteration < self.max_iter:
            self.logger.info(
                f"\nIteration {iteration+1}/{self.max_iter}: Running Correlated Louvain..."
            )

            if self.tune:
                self.logger.info("Tuning Correlated Louvain for current iteration...")
                louvain_tuner = CorrelatedLouvain(
                    self.G,
                    B=self.B,
                    Y=self.Y,
                    k3=self.k3,
                    k4=self.k4,
                    weight=self.weight,
                    seed=self.seed,
                    tune=True,
                    gpu=self.gpu,
                )
                best_config_louvain = louvain_tuner.run_tuning(num_samples=5)

                tuned_k4 = best_config_louvain.get("k4", self.k4)
                tuned_k3 = 1.0 - tuned_k4
                self.logger.info(
                    f"Using tuned Louvain parameters: k3={tuned_k3}, k4={tuned_k4}"
                )
                louvain = CorrelatedLouvain(
                    self.G,
                    B=self.B,
                    Y=self.Y,
                    k3=tuned_k3,
                    k4=tuned_k4,
                    weight=self.weight,
                    tune=False,
                    gpu=self.gpu,
                    seed=self.seed,
                )
            else:
                louvain = CorrelatedLouvain(
                    self.G,
                    B=self.B,
                    Y=self.Y,
                    k3=self.k3,
                    k4=self.k4,
                    weight=self.weight,
                    tune=False,
                    gpu=self.gpu,
                    seed=self.seed,
                )

            partition = louvain.run()
            quality_val = louvain.get_quality()
            self.logger.info(
                f"Iteration {iteration+1}: Louvain Quality = {quality_val:.4f}"
            )
            current_partition = partition

            best_corr = 0
            best_seed = None

            if not isinstance(partition, dict):
                raise TypeError("Expected 'partition' to be a dict")

            for com in set(partition.values()):
                nodes = []
                for n in self.G.nodes():
                    if partition[n] == com:
                        nodes.append(n)

                if len(nodes) < 2:
                    continue

                try:
                    corr, _ = louvain._compute_community_correlation(nodes)
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_seed = nodes
                except Exception as e:
                    self.logger.info(
                        f"Error computing correlation for community {com}: {e}"
                    )

            if best_seed is None:
                self.logger.info("No valid seed community found; stopping iterations.")
                break
            self.logger.info(
                f"Selected seed community of size {len(best_seed)} with correlation {best_corr:.4f}"
            )

            if self.tune:
                self.logger.info("Tuning Correlated PageRank for current iteration...")
                pagerank_tuner = CorrelatedPageRank(
                    graph=self.G,
                    omics_data=self.B,
                    phenotype_data=self.Y,
                    alpha=0.9,
                    max_iter=100,
                    tol=1e-6,
                    k=0.5,
                    seed=self.seed,
                    gpu=self.gpu,
                    tune=True,
                )
                best_config_pr = pagerank_tuner.run_tuning(num_samples=5)
                tuned_alpha = best_config_pr.get("alpha", 0.9)
                tuned_max_iter = best_config_pr.get("max_iter", 100)
                tuned_tol = best_config_pr.get("tol", 1e-6)
                tuned_k = best_config_pr.get("k", 0.5)
                self.logger.info(
                    f"Using tuned PageRank parameters: alpha={tuned_alpha}, max_iter={tuned_max_iter}, tol={tuned_tol}, k={tuned_k}"
                )
                pagerank_instance = CorrelatedPageRank(
                    graph=self.G,
                    omics_data=self.B,
                    phenotype_data=self.Y,
                    alpha=tuned_alpha,
                    max_iter=tuned_max_iter,
                    tol=tuned_tol,
                    k=tuned_k,
                    tune=False,
                    gpu=self.gpu,
                    seed=self.seed,
                )
            else:
                pagerank_instance = CorrelatedPageRank(
                    graph=self.G, omics_data=self.B, phenotype_data=self.Y, tune=False, seed=self.seed, gpu=self.gpu,
                )

            pagerank_results = pagerank_instance.run(best_seed)
            refined_nodes = pagerank_results.get("cluster_nodes", [])
            new_size = len(refined_nodes)
            all_clusters[iteration] = refined_nodes
            self.logger.info(f"Refined subgraph size: {new_size}")
            if new_size == prev_size or new_size <= 1:
                self.logger.info(
                    "Subgraph size converged or too small. Stopping iterations."
                )
                break
            prev_size = new_size
            self.G = self.G.subgraph(refined_nodes).copy()
            iteration += 1

        self.logger.info(f"Hybrid Louvain completed after {iteration+1} iterations.")

        if as_dfs:
            dfs = []
            for nodes in all_clusters.values():
                if len(nodes) > 2:

                    dfs.append(self.B.loc[:, nodes].copy())
            return dfs
        else:
            return {"curr": current_partition, "clus": all_clusters}
