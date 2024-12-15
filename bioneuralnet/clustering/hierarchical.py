import os
from typing import Optional, Dict, Any

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from datetime import datetime

from ..utils.logger import get_logger


class HierarchicalClustering:
    """
    HierarchicalClustering Class for Performing Agglomerative Hierarchical Clustering.

    This class handles the loading of adjacency matrices, execution of hierarchical clustering,
    and saving of the clustering results and cluster-specific adjacency matrices.

    Attributes:
        n_clusters (int): Number of clusters to find.
        linkage (str): Linkage criterion to use ('ward', 'complete', 'average', 'single').
        affinity (str): Metric used to compute linkage ('euclidean', 'l1', 'l2', 'manhattan', 'cosine', etc.).
        output_dir (str): Directory to save outputs.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(
        self,
        adjacency_matrix_file: str,
        n_clusters: int = 2,
        linkage: str = 'ward',
        affinity: str = 'euclidean',
        output_dir: Optional[str] = None,
    ):
        """
        Initializes the HierarchicalClustering instance with direct parameters.

        Args:
            adjacency_matrix_file (str): Path to the adjacency matrix CSV file.
            n_clusters (int, optional): Number of clusters to find. Defaults to 2.
            linkage (str, optional): Linkage criterion to use. Defaults to 'ward'.
            affinity (str, optional): Metric used to compute linkage. Defaults to 'euclidean'.
            output_dir (str, optional): Directory to save outputs. If None, creates a unique directory.
        """

        # First we assign parameters
        self.adjacency_matrix_file = adjacency_matrix_file
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity
        self.output_dir = output_dir if output_dir else self._create_output_dir()

        # Initialize logger and print parameters to make sure they are correct
        self.logger = get_logger(__name__)
        self.logger.info("Initialized HierarchicalClustering with the following parameters:")
        self.logger.info(f"Adjacency Matrix File: {self.adjacency_matrix_file}")
        self.logger.info(f"Number of Clusters: {self.n_clusters}")
        self.logger.info(f"Linkage: {self.linkage}")
        self.logger.info(f"Affinity: {self.affinity}")
        self.logger.info(f"Output Directory: {self.output_dir}")

        # Initialize data holder (this is our adjacency matrix)
        self.adjacency_matrix = None

    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the current HierarchicalClustering run.

        Returns:
            str: Path to the created output directory.
        """
        base_dir = "hierarchical_clustering_output"
        timestamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def load_data(self) -> None:
        """
        Loads the adjacency matrix from the provided CSV file.
        """
        try:
            self.adjacency_matrix = pd.read_csv(self.adjacency_matrix_file, index_col=0)
            self.logger.info(f"Loaded adjacency matrix from {self.adjacency_matrix_file} with shape {self.adjacency_matrix.shape}.")
        except Exception as e:
            self.logger.error(f"Error loading adjacency matrix: {e}")
            raise

    def run_clustering(self) -> Dict[str, Any]:
        """
        Executes the hierarchical clustering algorithm.

        Returns:
            Dict[str, Any]: Dictionary containing clustering results.
        """
        try:
            # Ensure data is loaded
            if self.adjacency_matrix is None:
                self.load_data()

            # Extract feature matrix
            feature_matrix = self.adjacency_matrix.values

            # Initialize the clustering model
            model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                affinity=self.affinity
            )

            # Fit the model and predict cluster labels
            labels = model.fit_predict(feature_matrix)
            self.logger.info("Hierarchical clustering completed.")

            # Calculate silhouette score as a measure of clustering quality (optional)
            try:
                silhouette_avg = silhouette_score(feature_matrix, labels, metric=self.affinity)
                self.logger.info(f"Silhouette Score: {silhouette_avg}")
            except Exception as e:
                self.logger.warning(f"Could not compute silhouette score: {e}")
                silhouette_avg = None

            # Create a DataFrame with cluster labels
            cluster_labels_df = pd.DataFrame({
                'node': self.adjacency_matrix.index,
                'cluster': labels
            })

            # Save cluster labels
            cluster_labels_file = os.path.join(self.output_dir, "cluster_labels.csv")
            cluster_labels_df.to_csv(cluster_labels_file, index=False)
            self.logger.info(f"Cluster labels saved to {cluster_labels_file}")

            # Save clusters as separate adjacency matrices
            for i in range(self.n_clusters):
                cluster_nodes = self.adjacency_matrix.index[labels == i]
                cluster_data = self.adjacency_matrix.loc[cluster_nodes, cluster_nodes]
                cluster_file = os.path.join(self.output_dir, f"cluster_{i + 1}.csv")
                cluster_data.to_csv(cluster_file)
                self.logger.info(f"Cluster {i + 1} data saved to {cluster_file}")

            # Prepare results
            results = {
                'cluster_labels': cluster_labels_df,
                'silhouette_score': silhouette_avg,
            }

            return results

        except Exception as e:
            self.logger.error(f"Error in run_clustering: {e}")
            raise

    def run(self) -> Dict[str, Any]:
        """
        Main method to run the hierarchical clustering pipeline.

        Returns:
            Dict[str, Any]: Clustering results.
        """
        try:
            results = self.run_clustering()
            self.logger.info("Hierarchical clustering pipeline completed successfully.")
            return results
        except Exception as e:
            self.logger.error(f"Error in run method: {e}")
            raise
