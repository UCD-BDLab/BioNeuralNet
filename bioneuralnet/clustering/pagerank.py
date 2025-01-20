import os
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from ..utils.logger import get_logger


class PageRank:
    """
    PageRank Class for Clustering Nodes Based on Personalized PageRank.

    This class handles the execution of the Personalized PageRank algorithm
    and identification of clusters based on sweep cuts.

    Attributes:
        alpha (float): Damping factor for PageRank.
        max_iter (int): Maximum number of iterations for PageRank convergence.
        tol (float): Tolerance for convergence.
        k (float): Weighting factor for composite correlation-conductance score.
        output_dir (str): Directory to save outputs.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(
        self,
        graph: nx.Graph,
        omics_data: pd.DataFrame,
        phenotype_data: pd.Series,
        alpha: float = 0.9,
        max_iter: int = 100,
        tol: float = 1e-6,
        k: float = 0.9,
        output_dir: Optional[str] = None,
    ):
        """
        Initializes the PageRank instance with direct data structures.

        Args:
            graph (nx.Graph): NetworkX graph object representing the network.
            omics_data (pd.DataFrame): Omics data DataFrame.
            phenotype_data (pd.Series): Phenotype data Series.
            alpha (float, optional): Damping factor for PageRank. Defaults to 0.9.
            max_iter (int, optional): Maximum iterations for PageRank. Defaults to 100.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
            k (float, optional): Weighting factor for composite score. Defaults to 0.9.
            output_dir (str, optional): Directory to save outputs. If None, creates a unique directory.
        """
        self.G = graph
        self.B = omics_data
        self.Y = phenotype_data
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.k = k
        self.output_dir = output_dir  # if output_dir else self._create_output_dir()

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
        self.logger.info(f"Output Directory: {self.output_dir}")

        self._validate_inputs()

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

    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the current PageRank run.

        Returns:
            str: Path to the created output directory.
        """
        base_dir = "pagerank_output"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def phen_omics_corr(self, nodes: List[Any]) -> Tuple[float, str]:
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
        self, p: Dict[Any, float]
    ) -> Tuple[List[Any], int, float, float, float, str]:
        """
        Performs sweep cut based on the PageRank scores.

        Args:
            p (Dict[Any, float]): Dictionary of PageRank scores.

        Returns:
            Tuple containing:
                - List of node identifiers in the cluster.
                - Cluster size.
                - Conductance.
                - Correlation.
                - Composite score.
                - Correlation with p-value.
        """
        try:
            cond_res = []
            corr_res = []
            cond_corr_res = []
            cluster = set()
            min_cut, min_cond_corr = len(p), float("inf")
            len_clus, cond, corr, cor_pval = 0, 1, 0.0, ""
            degrees = dict(self.G.degree(weight="weight"))
            vec = sorted(
                [
                    (p[node] / degrees[node] if degrees[node] > 0 else 0, node)
                    for node in p.keys()
                ],
                reverse=True,
            )

            for i, (val, node) in enumerate(vec):
                if val == 0:
                    break
                else:
                    cluster.add(node)

                if len(self.G.nodes()) > len(cluster):
                    cluster_cond = nx.conductance(self.G, cluster, weight="weight")
                    cond_res.append(round(cluster_cond, 3))

                    Nodes = list(cluster)
                    cluster_corr, corr_pvalue = self.phen_omics_corr(Nodes)
                    corr_res.append(round(cluster_corr, 3))
                    cluster_corr_neg = -abs(round(cluster_corr, 3))

                    cond_corr = round(
                        (1 - self.k) * cluster_cond + self.k * cluster_corr_neg, 3
                    )
                    cond_corr_res.append(cond_corr)

                    if cond_corr < min_cond_corr:
                        min_cond_corr, min_cut = cond_corr, i
                        len_clus = len(cluster)
                        cond = cluster_cond
                        corr = cluster_corr
                        cor_pval = corr_pvalue

            if min_cut < len(vec):
                nodes_in_cluster = [vec[i][1] for i in range(min_cut + 1)]
                return (
                    nodes_in_cluster,
                    len_clus,
                    cond,
                    corr,
                    round(min_cond_corr, 3),
                    cor_pval,
                )
            else:
                self.logger.warning(
                    "No valid sweep cut found. Returning empty cluster."
                )
                return [], 0, 0.0, 0.0, 0.0, "0 (1.0)"

        except Exception as e:
            self.logger.error(f"Error in sweep_cut: {e}")
            raise

    def generate_weighted_personalization(self, nodes: List[Any]) -> Dict[Any, float]:
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

            max_contribution = (
                max(corr_contribution, key=lambda x: abs(x)) if corr_contribution else 1
            )
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

    def run_pagerank_clustering(self, seed_nodes: List[Any]) -> Dict[str, Any]:
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

            p = nx.pagerank(
                self.G,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter,
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

            if self.output_dir is not None:
                self.save_results(results)
            return results

        except Exception as e:
            self.logger.error(f"Error in run_pagerank_clustering: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Error in run_pagerank_clustering: {e}")
            raise

    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Saves the clustering results to a CSV file.

        Args:
            results (Dict[str, Any]): Clustering results dictionary.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            filename = os.path.join(
                self.output_dir if self.output_dir is not None else "",
                f"pagerank_results_{timestamp}.csv",
            )

            df = pd.DataFrame(
                {
                    "Node": results["cluster_nodes"],
                    "Cluster Size": [results["cluster_size"]]
                    * len(results["cluster_nodes"]),
                    "Conductance": [results["conductance"]]
                    * len(results["cluster_nodes"]),
                    "Correlation": [results["correlation"]]
                    * len(results["cluster_nodes"]),
                    "Composite Score": [results["composite_score"]]
                    * len(results["cluster_nodes"]),
                    "Correlation P-Value": [results["correlation_pvalue"]]
                    * len(results["cluster_nodes"]),
                }
            )

            df.to_csv(filename, index=False)
            self.logger.info(f"Clustering results saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error in save_results: {e}")
            raise

    def run(self, seed_nodes: List[Any]) -> Dict[str, Any]:
        """
        Executes the PageRank-based clustering pipeline.

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

        Example:

        .. code-block:: python

            from bioneuralnet.clustering import PageRank

            # Initialize the PageRank clustering instance
            pagerank_clustering = PageRank(graph=my_graph, alpha=0.9, max_iter=100, tol=1e-6)

            # Define seed nodes for personalization
            seed_nodes = ['node1', 'node2', 'node3']

            # Run the clustering pipeline
            results = pagerank_clustering.run(seed_nodes=seed_nodes)

            # Output the results
            print("Clusters:", results['clusters'])
            print("Scores:", results['scores'])
        """
        try:
            results = self.run_pagerank_clustering(seed_nodes)
            self.logger.info("PageRank clustering completed successfully.")
            return results

        except Exception as e:
            self.logger.error(f"Error in run method: {e}")
            raise
