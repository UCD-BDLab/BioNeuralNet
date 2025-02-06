import networkx as nx
import pandas as pd

from bioneuralnet.clustering.correlated_pagerank import CorrelatedPageRank
from bioneuralnet.clustering.correlated_louvain import CorrelatedLouvain

from ..utils.logger import get_logger

logger = get_logger(__name__)


# review
class HybridLouvain:
    def __init__(
        self,
        G: nx.Graph,
        B: pd.DataFrame,
        Y,
        k3: float = 0.2,
        k4: float = 0.8,
        max_iter: int = 10,
        weight: str = "weight",
        tune: bool = False,
    ):
        self.logger = get_logger(__name__)
        self.G = G
        self.B = B
        self.Y = Y
        self.k3 = k3
        self.k4 = k4
        self.weight = weight
        self.max_iter = max_iter
        self.tune = tune
        self.logger.info(
            f"Initialized HybridLouvain with max_iter={max_iter}, k3={k3}, k4={k4}, tune={tune}"
        )

    def run(self) -> dict:
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
                    tune=True,
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
                )

            partition = louvain.run()
            quality_val = louvain.get_quality()
            self.logger.info(
                f"Iteration {iteration+1}: Louvain Quality = {quality_val:.4f}"
            )
            current_partition = partition

            best_corr = 0
            best_seed = None
            for com in set(partition.values()):
                nodes = [n for n in self.G.nodes() if partition[n] == com]
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
                )
            else:
                pagerank_instance = CorrelatedPageRank(
                    graph=self.G, omics_data=self.B, phenotype_data=self.Y, tune=False
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
        return {"curr": current_partition, "clus": all_clusters}


## Previous implementation before introducing hyperparameter tuning


# import networkx as nx
# import pandas as pd
# import numpy as np

# from bioneuralnet.clustering.correlated_pagerank import CorrelatedPageRank
# from bioneuralnet.clustering.correlated_louvain import CorrelatedLouvain

# from ..utils.logger import get_logger


# logger = get_logger(__name__)


# class HybridLouvain:
#     def __init__(
#         self,
#         G: nx.Graph,
#         B: pd.DataFrame,
#         Y,
#         k3: float = 0.2,
#         k4: float = 0.8,
#         max_iter: int = 10,
#         weight: str = "weight",
#         tune: bool = False,
#     ):
#         self.logger = get_logger(__name__)
#         self.G = G
#         self.B = B
#         self.Y = Y
#         self.k3 = k3
#         self.k4 = k4
#         self.weight = weight
#         self.max_iter = max_iter
#         self.tune = tune
#         self.logger.info(
#             f"Initialized HybridLouvain with max_iter={self.max_iter}, k3={self.k3}, k4={self.k4}"
#         )

#     def run(self) -> dict:
#         iteration = 0
#         prev_size = len(self.G.nodes())
#         current_partition = None

#         all_clusters = {}

#         while iteration < self.max_iter:
#             self.logger.info(
#                 f"\nIteration {iteration+1}/{self.max_iter}: Running Correlated Louvain..."
#             )
#             clouvain = CorrelatedLouvain(
#                 self.G,
#                 B=self.B,
#                 Y=self.Y,
#                 k3=self.k3,
#                 k4=self.k4,
#                 weight=self.weight,
#                 tune=self.tune,
#             )
#             partition = clouvain.run()
#             quality_val = clouvain.get_quality()

#             self.logger.info(f"Iteration {iteration+1}: Quality = {quality_val:.4f}")
#             current_partition = partition

#             best_corr = 0
#             best_seed = None
#             # loop over communities in the partition to find the best seed community.
#             for com in set(partition.values()):
#                 nodes = [n for n in self.G.nodes() if partition[n] == com]
#                 if len(nodes) < 2:
#                     continue
#                 try:
#                     corr, _ = clouvain._compute_community_correlation(nodes)
#                     if abs(corr) > abs(best_corr):
#                         best_corr = corr
#                         best_seed = nodes
#                 except Exception as e:
#                     self.logger.info(
#                         f"Error computing correlation for community {com}: {e}"
#                     )
#             if best_seed is None:
#                 self.logger.info("No valid seed community found; stopping iterations.")
#                 break
#             self.logger.info(
#                 f"Selected seed community of size {len(best_seed)} with correlation {best_corr:.4f}"
#             )

#             self.logger.info(
#                 "Running Correlated PageRank on selected seed community..."
#             )
#             pagerank_instance = CorrelatedPageRank(
#                 graph=self.G,
#                 omics_data=self.B,
#                 phenotype_data=self.Y,
#                 tune=self.tune,
#             )
#             pagerank_results = pagerank_instance.run(best_seed)
#             refined_nodes = pagerank_results.get("cluster_nodes", [])

#             new_size = len(refined_nodes)
#             all_clusters[iteration] = refined_nodes
#             self.logger.info(f"Refined subgraph size: {new_size}")
#             if new_size == prev_size:
#                 self.logger.info("Subgraph size converged. Stopping iterations.")
#                 break
#             elif new_size <= 1:
#                 self.logger.info(
#                     "Refined subgraph contains only one node. Stopping iterations."
#                 )
#                 break
#             prev_size = new_size
#             self.G = self.G.subgraph(refined_nodes).copy()
#             iteration += 1

#         self.logger.info(f"Hybrid Louvain completed after {iteration+1} iterations.")
#         return {"curr": current_partition, "clus": all_clusters}
