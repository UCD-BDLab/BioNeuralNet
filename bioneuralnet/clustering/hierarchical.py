from typing import Dict, Any

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging


class HierarchicalClustering:
    """
    HierarchicalClustering Class for Performing Agglomerative Hierarchical Clustering.

    This class handles the execution of hierarchical clustering on a provided adjacency matrix DataFrame
    and returns clustering results without relying on file I/O operations.

    Attributes:
        n_clusters (int): Number of clusters to find.
        linkage (str): Linkage criterion to use ('ward', 'complete', 'average', 'single').
        affinity (str): Metric used to compute linkage ('euclidean', 'l1', 'l2', 'manhattan', 'cosine', etc.).
        scaler (Optional[StandardScaler]): Scaler for data normalization.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        n_clusters: int = 2,
        linkage: str = 'ward',
        affinity: str = 'euclidean',
        scale_data: bool = True,
    ):
        """
        Initializes the HierarchicalClustering instance.

        Args:
            adjacency_matrix (pd.DataFrame): Adjacency matrix representing feature relationships.
            n_clusters (int, optional): Number of clusters to find. Defaults to 2.
            linkage (str, optional): Linkage criterion to use. Defaults to 'ward'.
            affinity (str, optional): Metric used to compute linkage. Defaults to 'euclidean'.
            scale_data (bool, optional): Whether to standardize the data. Defaults to True.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity
        self.scale_data = scale_data

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Prevent adding multiple handlers if already added
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info("Initialized HierarchicalClustering with the following parameters:")
        self.logger.info(f"Number of Clusters: {self.n_clusters}")
        self.logger.info(f"Linkage: {self.linkage}")
        self.logger.info(f"Affinity: {self.affinity}")
        self.logger.info(f"Scale Data: {self.scale_data}")

        self.adjacency_matrix = adjacency_matrix
        self.scaler = StandardScaler() if self.scale_data else None
        self.scaled_feature_matrix = None
        self.labels = None
        self.silhouette_avg = None
        self.cluster_labels_df = None

    def preprocess_data(self) -> None:
        """
        Preprocesses the adjacency matrix by scaling the data if required.
        """
        try:
            feature_matrix = self.adjacency_matrix.values
            if self.scale_data and self.scaler is not None:
                self.scaled_feature_matrix = self.scaler.fit_transform(feature_matrix)
                self.logger.info("Data has been scaled using StandardScaler.")
            else:
                self.scaled_feature_matrix = feature_matrix
                self.logger.info("Data scaling skipped.")
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {e}")
            raise

    def run_clustering(self) -> None:
        """
        Executes the hierarchical clustering algorithm.
        """
        try:
            model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                affinity=self.affinity if self.linkage != 'ward' else 'euclidean'  # 'ward' only supports 'euclidean'
            )

            self.labels = model.fit_predict(self.scaled_feature_matrix)
            self.logger.info("Hierarchical clustering completed.")

            # Compute silhouette score if applicable
            if self.affinity in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']:
                try:
                    self.silhouette_avg = silhouette_score(self.scaled_feature_matrix, self.labels, metric=self.affinity)
                    self.logger.info(f"Silhouette Score: {self.silhouette_avg:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not compute silhouette score: {e}")
                    self.silhouette_avg = None
            else:
                self.logger.warning(f"Silhouette score not computed for affinity '{self.affinity}'.")
                self.silhouette_avg = None

            # Create a DataFrame with cluster labels
            self.cluster_labels_df = pd.DataFrame({
                'node': self.adjacency_matrix.index,
                'cluster': self.labels
            })
            self.logger.info("Cluster labels DataFrame created.")

        except Exception as e:
            self.logger.error(f"Error during clustering: {e}")
            raise

    def get_results(self) -> Dict[str, Any]:
        """
        Retrieves the clustering results.

        Returns:
            Dict[str, Any]: Dictionary containing cluster labels and silhouette score.
        """
        if self.cluster_labels_df is None:
            self.logger.error("Clustering has not been run yet.")
            raise ValueError("Clustering has not been run yet.")

        results = {
            'cluster_labels': self.cluster_labels_df,
            'silhouette_score': self.silhouette_avg,
        }
        return results

    def run(self) -> Dict[str, Any]:
        """
        Runs the entire hierarchical clustering pipeline.

        Returns:
            Dict[str, Any]: Clustering results.
        """
        self.preprocess_data()
        self.run_clustering()
        self.logger.info("Hierarchical clustering pipeline completed successfully.")
        return self.get_results()
