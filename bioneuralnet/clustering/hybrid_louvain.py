import networkx as nx
import pandas as pd
from typing import Union, Optional
from bioneuralnet.clustering.correlated_pagerank import CorrelatedPageRank
from bioneuralnet.clustering.correlated_louvain import CorrelatedLouvain
from bioneuralnet.utils import get_logger, set_seed

logger = get_logger(__name__)

class HybridLouvain:
    """
    HybridLouvain Class that combines Correlated Louvain and Correlated PageRank for community detection.

    Attributes:

        G (Union[nx.Graph, pd.DataFrame]): Input graph as a NetworkX Graph or adjacency DataFrame.
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
        G: Union[nx.Graph, pd.DataFrame],
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
        self.seed = seed
        self.gpu = gpu
        if seed is not None:
            set_seed(seed)
        self.logger.info("Initializing HybridLouvain...")

        if isinstance(G, pd.DataFrame):
            self.logger.info("Input G is a DataFrame; converting adjacency matrix to NetworkX graph.")
            G = nx.from_pandas_adjacency(G)

        if not isinstance(G, nx.Graph):
            raise TypeError("G must be a networkx.Graph or a pandas DataFrame adjacency matrix.")

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
        """Run the hybrid Louvain-PageRank refinement loop.

        Iteratively applies CorrelatedLouvain to obtain communities, selects the most phenotype-associated community as seeds, refines it with CorrelatedPageRank, and restricts the graph to the refined cluster until convergence, subgraph size < 2, or max_iter is reached. Returns either the final partition and per-iteration clusters or a list of omics subnetworks as DataFrames.

        Args:

            as_dfs (bool): If True, return a list of omics subnetworks (one DataFrame per iteration); if False, return a dict with the last partition and all clusters.

        Returns:

            Union[dict, list]: Either {"curr": partition, "clus": all_clusters} or a list of DataFrames (clusters as omics subnetworks).

        """
        iteration = 0
        prev_size = len(self.G.nodes())
        current_partition = None
        all_clusters = {}

        while iteration < self.max_iter:
            if len(self.G.nodes()) < 5:
                self.logger.info("Graph has fewer than 5 nodes; stopping iterations.")
                break

            self.logger.info(
                f"\nIteration {iteration+1}/{self.max_iter}: Running Correlated Louvain..."
            )

            if self.tune and len(self.G.nodes()) > 20:
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

            best_corr = 0.0
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

            if self.tune and len(self.G.nodes()) > 20:
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
                    graph=self.G,
                    omics_data=self.B,
                    phenotype_data=self.Y,
                    tune=False,
                    seed=self.seed,
                    gpu=self.gpu,
                )

            pagerank_results = pagerank_instance.run(best_seed)
            refined_nodes = pagerank_results.get("cluster_nodes", [])
            new_size = len(refined_nodes)
            all_clusters[iteration] = refined_nodes

            cond = pagerank_results.get("conductance", None)
            corr = pagerank_results.get("correlation", None)
            score = pagerank_results.get("composite_score", None)

            self.logger.info(
                f"Iteration {iteration+1}: cluster size={new_size}, "
                f"Conductance={cond:.3f} Correlation={corr:.3f} score={score:.3f}"
            )

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
                    valid_cols = []
                    for n in nodes:
                        if n in self.B.columns:
                            valid_cols.append(n)
                    if valid_cols:
                        dfs.append(self.B.loc[:, valid_cols].copy())
            return dfs
        else:
            return {"curr": current_partition, "clus": all_clusters}
