import os
from datetime import datetime
import pandas as pd
from sklearn.decomposition import PCA
from ..utils.logger import get_logger
from ..network_embedding import GnnEmbedding
from ..network_embedding import Node2VecEmbedding


class GraphEmbedding:
    """
    GraphEmbedding Class for Integrating Network Embeddings into Omics Data.

    This class takes already loaded data structures and applies network embeddings to enhance subject representations.

    Args:
        adjacency_matrix (pd.DataFrame): The adjacency matrix of the graph representing feature interactions.
        omics_data (pd.DataFrame): Combined omics data with samples as rows and features as columns.
                                   Must include the phenotype column 'finalgold_visit'.
        clinical_data (pd.DataFrame): Clinical data for the same samples. Index must align with omics_data.
        embedding_method (str, optional): The method to use for generating embeddings ('GNNs' or 'Node2Vec').
                                          Defaults to 'GNNs'.

    Attributes:
        adjacency_matrix (pd.DataFrame)
        omics_data (pd.DataFrame)
        clinical_data (pd.DataFrame)
        embedding_method (str)

    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        omics_data: pd.DataFrame,
        clinical_data: pd.DataFrame,
        embedding_method: str = 'GNNs',
    ):
        # Basic checks
        if adjacency_matrix is None or adjacency_matrix.empty:
            raise ValueError("Adjacency matrix is required and cannot be empty.")
        if omics_data is None or omics_data.empty or 'finalgold_visit' not in omics_data.columns:
            raise ValueError("Omics data must be non-empty and contain 'finalgold_visit' column.")
        if clinical_data is None or clinical_data.empty:
            raise ValueError("Clinical data is required and cannot be empty.")

        self.adjacency_matrix = adjacency_matrix
        self.omics_data = omics_data
        self.clinical_data = clinical_data
        self.embedding_method = embedding_method
        self.logger = get_logger(__name__)
        self.logger.info("Initialized GraphEmbedding with direct data inputs.")

    def _create_output_dir(self) -> str:
        base_dir = "subject_representation_output"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def run(self) -> pd.DataFrame:
        """
        Generate subject representations by integrating network embeddings into omics data.

        Steps:
        1. **Embedding Generation**: Runs GNN or Node2Vec-based methods to produce node embeddings.
        2. **Dimensionality Reduction**: Applies PCA to condense embeddings into a single principal component.
        3. **Integration**: Multiplies original omics features by the reduced embeddings to create enhanced omics data.

        Returns:
            pd.DataFrame:

                - A DataFrame of enhanced omics data where each feature (node) has been weighted by its embedding-derived principal component.

        Raises:

            - ValueError: If embeddings are empty or omics data cannot be integrated.
            - Exception: For any unexpected issues during the integration process.

        Notes:

            - The enhanced omics data can be used downstream for clustering, classification, or regression tasks.
            - Ensure that the PCA step is appropriate for your analysis and consider adjusting the dimensionality reduction strategy if needed.
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
        Generate node embeddings using the selected embedding method.

        Returns:
            pd.DataFrame: Node embeddings (nodes as index, embedding dimensions as columns).
        """
        self.logger.info(f"Generating embeddings using {self.embedding_method}")

        if self.embedding_method == 'GNNs':
            gnn_embedding = GnnEmbedding(
                adjacency_matrix=self.adjacency_matrix,
                omics_data=self.omics_data,
                clinical_data=self.clinical_data,
                model_type='GCN',
                gnn_hidden_dim=64,
                gnn_layer_num=2,
                dropout=True,
            )
            embeddings_dict = gnn_embedding.run()
            embeddings_tensor = embeddings_dict['graph']
            embeddings_df = pd.DataFrame(
                embeddings_tensor.numpy(),
                index=self.adjacency_matrix.index
            )

        elif self.embedding_method == 'Node2Vec':
            node2vec_embedding = Node2VecEmbedding(
                adjacency_matrix=self.adjacency_matrix,
                embedding_dim=128,
                walk_length=80,
                num_walks=10,
                window_size=10,
                workers=4,
                seed=42,
            )
            embeddings_df = node2vec_embedding.run()
            embeddings_df.set_index('node', inplace=True)

        else:
            raise ValueError(f"Unsupported embedding method: {self.embedding_method}")

        return embeddings_df

    def reduce_embeddings(self, embeddings: pd.DataFrame) -> pd.Series:
        """
        Reduce embeddings to a single principal component per node using PCA.

        Args:
            embeddings (pd.DataFrame): Node embeddings.

        Returns:
            pd.Series: Reduced embedding values indexed by node names.
        """
        if embeddings.empty:
            raise ValueError("Embeddings DataFrame is empty.")
        if embeddings.shape[1] < 1:
            raise ValueError("Embeddings DataFrame must have at least one dimension.")

        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(embeddings)
        node_embedding_values = pd.Series(
            principal_components.flatten(),
            index=embeddings.index,
            name='embedding_pca'
        )
        self.logger.debug("Reduced embeddings using PCA.")
        return node_embedding_values

    def integrate_embeddings(self, node_embedding_values: pd.Series) -> pd.DataFrame:
        """
        Integrate reduced embeddings into omics data by weighting each feature by the embedding.

        Args:
            node_embedding_values (pd.Series): Embedding values per node.

        Returns:
            pd.DataFrame: Enhanced omics data with integrated embeddings.
        """
        self.logger.info("Integrating embeddings into omics data.")

        modified_omics_data = self.omics_data.copy()
        feature_cols = [col for col in modified_omics_data.columns if col != 'finalgold_visit']

        missing_nodes = set(feature_cols) - set(node_embedding_values.index)
        if missing_nodes:
            self.logger.warning(f"These nodes are missing embeddings and will be skipped: {missing_nodes}")

        for node in feature_cols:
            if node in node_embedding_values.index:
                modified_omics_data[node] = modified_omics_data[node] * node_embedding_values[node]

        self.logger.debug("Integrated embeddings into omics data.")
        return modified_omics_data
