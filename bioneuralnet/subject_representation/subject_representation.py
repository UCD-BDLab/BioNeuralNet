import pandas as pd
from sklearn.decomposition import PCA
from ..utils.logger import get_logger
from ..network_embedding import GNNEmbedding


class GraphEmbedding:
    """
    GraphEmbedding Class for Integrating Network Embeddings into Omics Data.

    This class takes already loaded data structures and applies network embeddings
    to enhance subject representations. It can either use precomputed embeddings or
    train a new GNN (regression-based or otherwise) through ``GNNEmbedding``.

    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        omics_data: pd.DataFrame,
        phenotype_data: pd.DataFrame,
        clinical_data: pd.DataFrame,
        embeddings: pd.DataFrame = None,
        reduce_method: str = "PCA",
    ):
        """
        Parameters
        ----------
        adjacency_matrix : pd.DataFrame
            NxN adjacency matrix for omics features (nodes).
        omics_data : pd.DataFrame
            (samples x features) table of omics data.
        phenotype_data : pd.DataFrame
            (samples x some_phenotype) table, must contain `phenotype_col`.
        clinical_data : pd.DataFrame
            (samples x clinical_vars) table.
        embeddings : pd.DataFrame, optional
            Precomputed node embeddings to skip GNN if not None or empty.
        """
        if adjacency_matrix is None or adjacency_matrix.empty:
            raise ValueError("Adjacency matrix is required and cannot be empty.")
        if omics_data is None or omics_data.empty:
            raise ValueError("Omics data must be non-empty.")
        if clinical_data is None or clinical_data.empty:
            raise ValueError("Clinical data is required and cannot be empty.")
        if phenotype_data is None or phenotype_data.empty:
            raise ValueError("Phenotype data is required and cannot be empty.")

        if embeddings is None or embeddings.empty:
            self.logger = get_logger(__name__)
            self.logger.info(
                "No precomputed embeddings -> defaulting to GNN-based approach."
            )
            self.embedding_method = "GNNs"
        else:
            self.embedding_method = "precomputed"

        self.adjacency_matrix = adjacency_matrix
        self.omics_data = omics_data
        self.phenotype_data = phenotype_data
        self.clinical_data = clinical_data
        self.embeddings = embeddings
        self.reduce_method = reduce_method

        self.logger = get_logger(__name__)
        self.logger.info("Initialized GraphEmbedding with direct data inputs.")

    def run(self) -> pd.DataFrame:
        """
        Main pipeline:
          1) Generate (or load) node embeddings
          2) Reduce them to 1D with PCA
          3) Integrate each node's PCA value into the subject-level omics data

        **Returns**:
            pd.DataFrame
                Enhanced omics data, weighted by the node embeddings or PCA of embeddings.

        **Raises**:
            - **ValueError**: If embeddings are empty or an error occurs in PCA/integration.
            - **Exception**: For any unforeseen errors encountered.

        **Notes**:
            - If ``self.embeddings`` is provided, we skip training.
            - Otherwise, we instantiate a ``GNNEmbedding`` object which trains a node-level
              regression model if ``model_type`` is one of {"GCN", "GAT", "SAGE", "GIN"}.
        """
        self.logger.info("Running Subject Representation workflow.")
        try:
            embeddings_df = self.generate_embeddings()
            node_embedding_values = self.reduce_embeddings(embeddings_df)
            enhanced_omics_data = self.integrate_embeddings(node_embedding_values)
            return enhanced_omics_data

        except Exception as e:
            self.logger.error(f"Error in Subject Representation: {e}")
            raise

    def generate_embeddings(self) -> pd.DataFrame:
        """
        Generate or retrieve node embeddings.

        If embeddings are provided (precomputed), return them directly.
        Otherwise, create a GNNEmbedding to do correlation-based node feature/label
        and train the GNN for MSE regression.

        **Returns**:
            pd.DataFrame
                Node embeddings of shape [num_nodes, embedding_dim].
        """
        self.logger.info(f"Generating embeddings with method='{self.embedding_method}'")

        if self.embedding_method == "precomputed":
            if not isinstance(self.embeddings, pd.DataFrame):
                raise ValueError("Embeddings must be a pandas DataFrame.")
            if self.embeddings.empty:
                raise ValueError("Provided embeddings are empty.")
            return self.embeddings

        else:
            gnn_embedder = GNNEmbedding(
                adjacency_matrix=self.adjacency_matrix,
                omics_data=self.omics_data,
                phenotype_data=self.phenotype_data,
                clinical_data=self.clinical_data,
            )

        embeddings_dict = gnn_embedder.run()
        embeddings_tensor = embeddings_dict["graph"]

        node_names = self.adjacency_matrix.index
        embeddings_df = pd.DataFrame(embeddings_tensor.numpy(), index=node_names)
        return embeddings_df

    def reduce_embeddings(self, embeddings: pd.DataFrame) -> pd.Series:
        """
        Reduce embeddings to a single dimension per node using specified method.

        **Parameters**:
            embeddings: pd.DataFrame
                A DataFrame containing the embeddings to be reduced.
            method: str, optional
                The dimensionality reduction method to use. Options:
                - "pca" (default): Reduce using the first principal component via PCA.
                - "average": Compute the average of all features.
                - "maximum": Compute the maximum value of all features.

        **Returns**:
            pd.Series
                A Series indexed by node, containing the 1D embedding.

        **Raises**:
            ValueError:
                If the embeddings DataFrame is empty or if an unsupported method is provided.
        """
        if embeddings.empty:
            raise ValueError("Embeddings DataFrame is empty.")

        if self.reduce_method == "PCA":
            self.logger.info("Reducing node embeddings to 1D via PCA.")
            pca = PCA(n_components=1)
            principal_components = pca.fit_transform(embeddings)
            reduced_embedding = pd.Series(
                principal_components.flatten(), index=embeddings.index, name="PC1"
            )
        elif self.reduce_method == "AVG":
            self.logger.info("Reducing node embeddings to 1D via averaging.")
            reduced_embedding = embeddings.mean(axis=1)
        elif self.reduce_method == "MAX":
            self.logger.info("Reducing node embeddings to 1D via maximum.")
            reduced_embedding = embeddings.max(axis=1)
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduce_method}")

        return reduced_embedding

    def integrate_embeddings(self, node_embedding_values: pd.Series) -> pd.DataFrame:
        """
        Multiply each omics feature in self.omics_data by the PCA scalar for that node.

        **Returns**:
            pd.DataFrame
                Enhanced omics data with integrated embeddings.
        """
        self.logger.info("Integrating node embeddings into subject-level omics.")
        modified_omics = self.omics_data.copy()

        feature_cols = modified_omics.columns
        missing_nodes = set(feature_cols) - set(node_embedding_values.index)
        if missing_nodes:
            self.logger.warning(f"Some features have no embeddings: {missing_nodes}")

        for node in feature_cols:
            if node in node_embedding_values.index:
                modified_omics[node] = (
                    modified_omics[node] * node_embedding_values[node]
                )

        return modified_omics
