import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional
from ..utils.logger import get_logger
from ..network_embedding import GNNEmbedding


class GraphEmbedding:
    """
    GraphEmbedding Class for Integrating Network Embeddings into Omics Data..
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        omics_data: pd.DataFrame,
        phenotype_data: pd.DataFrame,
        clinical_data: Optional[pd.DataFrame] = None,
        embeddings: Optional[pd.DataFrame] = None,
        reduce_method: str = "PCA",
    ):
        """
        Initializes the GraphEmbedding instance.

        Parameters:
            adjacency_matrix : pd.DataFrame
            omics_data : pd.DataFrame
            phenotype_data : pd.DataFrame
            clinical_data : Optional[pd.DataFrame], default=None
            embeddings : Optional[pd.DataFrame], default=None
            reduce_method : str, optional
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing GraphEmbedding with provided data inputs.")

        if adjacency_matrix is None or adjacency_matrix.empty:
            raise ValueError("Adjacency matrix is required and cannot be empty.")
        if omics_data is None or omics_data.empty:
            raise ValueError("Omics data must be non-empty.")
        if phenotype_data is None or phenotype_data.empty:
            raise ValueError("Phenotype data is required and cannot be empty.")

        if clinical_data is not None and clinical_data.empty:
            self.logger.warning(
                "Clinical data provided is empty. Node features will be initialized randomly."
            )
            clinical_data = None

        if embeddings is None or embeddings.empty:
            self.embedding_method = "GNNs"
            self.logger.info(
                "No precomputed embeddings provided. Defaulting to GNN-based embedding generation."
            )
        else:
            self.embedding_method = "precomputed"
            if not isinstance(embeddings, pd.DataFrame):
                raise ValueError("Embeddings must be provided as a pandas DataFrame.")
            missing_nodes = set(adjacency_matrix.index) - set(embeddings.index)
            if missing_nodes:
                raise ValueError(
                    f"Provided embeddings are missing nodes: {missing_nodes}"
                )
            self.logger.info(
                "Precomputed embeddings provided. Skipping GNN-based embedding generation."
            )

        self.adjacency_matrix = adjacency_matrix
        self.omics_data = omics_data
        self.phenotype_data = phenotype_data
        self.clinical_data = clinical_data
        self.embeddings = embeddings
        self.reduce_method = reduce_method

    def run(self) -> pd.DataFrame:
        """
        Main pipeline:
            - Generate (or load) node embeddings.
            - Reduce them to 1D with the specified reduction method.
            - Integrate each node's reduced embedding value into the subject-level omics data.

        Returns:
            pd.DataFrame

        Raises:
            ValueError
            Exception
        """
        self.logger.info("Starting Subject Representation workflow.")
        try:
            embeddings_df = self.generate_embeddings()
            node_embedding_values = self.reduce_embeddings(embeddings_df)
            enhanced_omics_data = self.integrate_embeddings(node_embedding_values)

            self.logger.info("Subject Representation workflow completed successfully.")
            return enhanced_omics_data

        except Exception as e:
            self.logger.error(f"Error in Subject Representation workflow: {e}")
            raise

    def generate_embeddings(self) -> pd.DataFrame:
        """
        Generate or retrieve node embeddings.

        Returns:
            pd.DataFrame
        Raises:
            ValueError
        """
        self.logger.info(
            f"Generating embeddings using method='{self.embedding_method}'."
        )

        if self.embedding_method == "precomputed":
            self.logger.info("Using precomputed embeddings.")
            # making sure embeddings are not None, or precommit will fail
            if self.embeddings is None:
                raise ValueError("Embeddings are None.")
            return self.embeddings.copy()

        else:
            self.logger.info("Generating embeddings using GNNEmbedding.")
            # for the other parameters it will just use default values
            gnn_embedder = GNNEmbedding(
                adjacency_matrix=self.adjacency_matrix,
                omics_data=self.omics_data,
                phenotype_data=self.phenotype_data,
                clinical_data=self.clinical_data,
            )

            gnn_embedder.fit()
            embeddings_tensor = gnn_embedder.embed()

            node_names = self.adjacency_matrix.index.tolist()
            embeddings_df = pd.DataFrame(
                embeddings_tensor.numpy(),
                index=node_names,
                columns=[f"Embed_{i+1}" for i in range(embeddings_tensor.shape[1])],
            )
            self.logger.info(
                f"Generated GNN-based embeddings with shape {embeddings_df.shape}."
            )
            return embeddings_df

    def reduce_embeddings(self, embeddings: pd.DataFrame) -> pd.Series:
        """
        Reduce embeddings to a single dimension per node using the specified method.

        Parameters:
            embeddings : pd.DataFrame

        Returns:
            pd.Series

        Raises:
            ValueError
        """
        self.logger.info(
            f"Reducing embeddings to 1D using method='{self.reduce_method}'."
        )

        if embeddings.empty:
            raise ValueError("Embeddings DataFrame is empty.")

        if self.reduce_method.upper() == "PCA":
            self.logger.info("Applying PCA to reduce embeddings to 1D.")
            pca = PCA(n_components=1, random_state=42)
            principal_components = pca.fit_transform(embeddings)
            reduced_embedding = pd.Series(
                principal_components.flatten(), index=embeddings.index, name="PC1"
            )
            self.logger.info("PCA reduction completed.")
        elif self.reduce_method.upper() == "AVG":
            self.logger.info("Calculating average of embedding dimensions.")
            reduced_embedding = embeddings.mean(axis=1)
            reduced_embedding.name = "Avg_Embed"
            self.logger.info("Average reduction completed.")
        elif self.reduce_method.upper() == "MAX":
            self.logger.info("Calculating maximum of embedding dimensions.")
            reduced_embedding = embeddings.max(axis=1)
            reduced_embedding.name = "Max_Embed"
            self.logger.info("Maximum reduction completed.")
        else:
            self.logger.error(f"Unsupported reduction method: {self.reduce_method}")
            raise ValueError(f"Unsupported reduction method: {self.reduce_method}")

        return reduced_embedding

    def integrate_embeddings(self, node_embedding_values: pd.Series) -> pd.DataFrame:
        """
        Integrate the reduced node embeddings into the subject-level omics data.
        """
        self.logger.info(
            "Integrating reduced node embeddings into subject-level omics data."
        )

        try:
            feature_cols = self.omics_data.columns
            missing_nodes = set(feature_cols) - set(node_embedding_values.index)
            if missing_nodes:
                self.logger.warning(
                    f"Some omics features have no corresponding embeddings: {missing_nodes}"
                )

            enhanced_omics = self.omics_data.copy()
            for node in feature_cols:
                if node in node_embedding_values.index:
                    enhanced_omics[node] = (
                        enhanced_omics[node] * node_embedding_values[node]
                    )
                else:
                    self.logger.warning(
                        f"No embedding found for feature '{node}'. Skipping integration for this feature."
                    )

            self.logger.info("Integration of embeddings completed successfully.")
            return enhanced_omics

        except Exception as e:
            self.logger.error(f"Error during integration of embeddings: {e}")
            raise
