import os
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

from ..utils.logger import get_logger


class PageRankClustering:
    """
    PageRankClustering Class for Clustering Nodes Based on Personalized PageRank.

    This class handles the loading of graph data, execution of the Personalized PageRank algorithm,
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
        graph_file: str,
        omics_data_file: str,
        phenotype_data_file: str,
        alpha: float = 0.9,
        max_iter: int = 100,
        tol: float = 1e-6,
        k: float = 0.9,
        output_dir: Optional[str] = None,
    ):
        """
        Initializes the PageRankClustering instance with direct parameters.

        Args:
            graph_file (str): Path to the graph edge list file.
            omics_data_file (str): Path to the omics data file (e.g., 'X.xlsx').
            phenotype_data_file (str): Path to the phenotype data file (e.g., 'Y.xlsx').
            alpha (float, optional): Damping factor for PageRank. Defaults to 0.9.
            max_iter (int, optional): Maximum iterations for PageRank. Defaults to 100.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
            k (float, optional): Weighting factor for composite score. Defaults to 0.9.
            output_dir (str, optional): Directory to save outputs. If None, creates a unique directory.
        """
        # Assign parameters
        self.graph_file = graph_file
        self.omics_data_file = omics_data_file
        self.phenotype_data_file = phenotype_data_file
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.k = k
        self.output_dir = output_dir if output_dir else self._create_output_dir()

        # Initialize logger
        self.logger = get_logger(__name__)
        self.logger.info("Initialized PageRankClustering with the following parameters:")
        self.logger.info(f"Graph File: {self.graph_file}")
        self.logger.info(f"Omics Data File: {self.omics_data_file}")
        self.logger.info(f"Phenotype Data File: {self.phenotype_data_file}")
        self.logger.info(f"Alpha: {self.alpha}")
        self.logger.info(f"Max Iterations: {self.max_iter}")
        self.logger.info(f"Tolerance: {self.tol}")
        self.logger.info(f"K (Composite Score Weight): {self.k}")
        self.logger.info(f"Output Directory: {self.output_dir}")

        # Initialize data holders
        self.G = None  # NetworkX graph
        self.B = None  # Omics data DataFrame
        self.Y = None  # Phenotype data Series

    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the current PageRankClustering run.

        Returns:
            str: Path to the created output directory.
        """
        base_dir = "pagerank_output"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def load_data(self) -> None:
        """
        Loads the graph, omics data, and phenotype data from the provided files.
        """
        try:
            # Load the graph
            self.G = nx.read_edgelist(self.graph_file, data=(('weight', float),))
            self.logger.info(f"Loaded graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")

            # Load omics data
            self.B = pd.read_excel(self.omics_data_file)
            self.B.drop(self.B.columns[0], axis=1, inplace=True)
            self.logger.info(f"Loaded omics data with shape {self.B.shape}.")

            # Load phenotype data
            Y_df = pd.read_excel(self.phenotype_data_file)
            Y_df.drop(Y_df.columns[0], axis=1, inplace=True)
            self.Y = Y_df.iloc[:, 0]
            self.logger.info(f"Loaded phenotype data with {len(self.Y)} samples.")

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def phen_omics_corr(self, nodes: List[int]) -> Tuple[float, str]:
        """
        Calculates the Pearson correlation between the PCA of omics data and phenotype.

        Args:
            nodes (List[int]): List of node indices to include in the calculation.

        Returns:
            Tuple[float, str]: Correlation coefficient and formatted correlation with p-value.
        """
        try:
            # Subsetting the omics data
            B_sub = self.B.iloc[:, nodes]

            # Scaling the subset data
            scaler = StandardScaler()
            scaled = scaler.fit_transform(B_sub)

            # Applying PCA to the subset data
            pca = PCA(n_components=1)
            g1 = pca.fit_transform(scaled).flatten()

            # Phenotype data
            g2 = self.Y.values

            # Calculating Pearson correlation
            corr, pvalue = pearsonr(g1, g2)
            corr = round(corr, 2)
            p_value = format(pvalue, '.3g')
            corr_pvalue = f"{corr} ({p_value})"
            return corr, corr_pvalue

        except Exception as e:
            self.logger.error(f"Error in phen_omics_corr: {e}")
            raise

    def sweep_cut(self, p: Dict[str, float]) -> Tuple[List[str], int, float, float, float, str]:
        """
        Performs sweep cut based on the PageRank scores.

        Args:
            p (Dict[str, float]): Dictionary of PageRank scores.

        Returns:
            Tuple containing:
                - List of node names in the cluster.
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
            min_cut, min_cond_corr = len(p), float('inf')
            len_clus, cond, corr, cor_pval = 0, 1, 0, ''

            # Normalizing PageRank scores
            degrees = dict(self.G.degree(weight='weight'))
            vec = sorted([(p[node] / degrees[node], node) for node in p.keys()], reverse=True)

            for i, (val, node) in enumerate(vec):
                if val == 0:
                    break
                else:
                    cluster.add(node)

                if len(self.G.nodes()) > len(cluster):
                    # Calculating conductance
                    cluster_cond = nx.conductance(self.G, cluster, weight='weight')
                    cond_res.append(round(cluster_cond, 3))

                    # Calculating correlation
                    Nodes = [int(k) for k in cluster]
                    cluster_corr, corr_pvalue = self.phen_omics_corr(Nodes)
                    corr_res.append(round(cluster_corr, 3))
                    cluster_corr_neg = -abs(round(cluster_corr, 3))

                    # Composite correlation-conductance score
                    cond_corr = round((1 - self.k) * cluster_cond + self.k * cluster_corr_neg, 3)
                    cond_corr_res.append(cond_corr)

                    if cond_corr < min_cond_corr:
                        min_cond_corr, min_cut = cond_corr, i
                        len_clus = len(cluster)
                        cond = cluster_cond
                        corr = -cluster_corr_neg
                        cor_pval = corr_pvalue

            nodes_in_cluster = [vec[i][1] for i in range(min_cut + 1)]
            return nodes_in_cluster, len_clus, cond, corr, round(min_cond_corr, 3), cor_pval

        except Exception as e:
            self.logger.error(f"Error in sweep_cut: {e}")
            raise

    def generate_weighted_personalization(self, nodes: List[int]) -> Dict[str, float]:
        """
        Generates a weighted personalization vector for PageRank.

        Args:
            nodes (List[int]): List of node indices to consider.

        Returns:
            Dict[str, float]: Personalization vector with weights for each node.
        """
        try:
            total_corr = self.phen_omics_corr(nodes)[0]
            corr_contribution = []

            for i in range(len(nodes)):
                nodes_excl = nodes[:i] + nodes[i + 1:]
                corr_excl = self.phen_omics_corr(nodes_excl)[0]
                contribution = abs(corr_excl) - abs(total_corr)
                corr_contribution.append(contribution)

            max_contribution = max(corr_contribution, key=abs)
            weighted_personalization = {
                str(nodes[i]): self.alpha * corr_contribution[i] / max_contribution
                for i in range(len(nodes))
            }
            return weighted_personalization

        except Exception as e:
            self.logger.error(f"Error in generate_weighted_personalization: {e}")
            raise

    def run_pagerank_clustering(self, seed_nodes: List[int]) -> Dict[str, Any]:
        """
        Executes the PageRank clustering algorithm.

        Args:
            seed_nodes (List[int]): List of seed node indices for personalization.

        Returns:
            Dict[str, Any]: Dictionary containing clustering results.
        """
        try:
            # Generate personalization vector
            personalization = self.generate_weighted_personalization(seed_nodes)
            self.logger.info(f"Generated personalization vector for seed nodes: {seed_nodes}")

            # Run PageRank
            p = nx.pagerank(
                self.G,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=self.tol,
                weight='weight'
            )
            self.logger.info("PageRank computation completed.")

            # Perform sweep cut
            nodes, n, cond, corr, min_corr, pval = self.sweep_cut(p)
            self.logger.info(f"Sweep cut resulted in cluster of size {n} with conductance {cond} and correlation {corr}.")

            # Save results
            results = {
                'cluster_nodes': nodes,
                'cluster_size': n,
                'conductance': cond,
                'correlation': corr,
                'composite_score': min_corr,
                'correlation_pvalue': pval
            }

            # Optionally save to file
            self.save_results(results)
            return results

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
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = os.path.join(self.output_dir, f"pagerank_results_{timestamp}.csv")

            df = pd.DataFrame({
                'Node': results['cluster_nodes'],
                'Cluster Size': [results['cluster_size']] * len(results['cluster_nodes']),
                'Conductance': [results['conductance']] * len(results['cluster_nodes']),
                'Correlation': [results['correlation']] * len(results['cluster_nodes']),
                'Composite Score': [results['composite_score']] * len(results['cluster_nodes']),
                'Correlation P-Value': [results['correlation_pvalue']] * len(results['cluster_nodes']),
            })

            df.to_csv(filename, index=False)
            self.logger.info(f"Clustering results saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error in save_results: {e}")
            raise

    def run(self, seed_nodes: List[int]) -> Dict[str, Any]:
        """
        Main method to run the PageRank clustering pipeline.

        Args:
            seed_nodes (List[int]): List of seed node indices for personalization.

        Returns:
            Dict[str, Any]: Clustering results.
        """
        try:
            self.load_data()
            results = self.run_pagerank_clustering(seed_nodes)
            self.logger.info("PageRank clustering completed successfully.")
            return results

        except Exception as e:
            self.logger.error(f"Error in run method: {e}")
            raise
