import os
from datetime import datetime
from typing import Optional, List

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from ..utils.logger import get_logger
from ..network_embedding.gnns import GNNEmbedding
from ..network_embedding.node2vec import Node2VecEmbedding
from ..utils.data_utils import combine_omics_data


class SubjectRepresentationEmbedding:
    """
    SubjectRepresentationEmbedding Class for Integrating Network Embeddings into Omics Data.

    This class handles the generation of embeddings using selected embedding methods,
    reduces node embeddings using PCA (if needed), and integrates these embeddings into
    the original omics data to enhance subject representations.

    Attributes:
        adjacency_matrix (pd.DataFrame): The adjacency matrix of the graph representing feature interactions.
        omics_list (List[str]): List of paths to omics data CSV files.
        phenotype_file (str): Path to the phenotype CSV file.
        clinical_data_file (str): Path to the clinical data CSV file.
        embedding_method (str): The method to use for generating embeddings ('GNNs' or 'Node2Vec').
        output_dir (str): Directory to save enhanced omics data.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        omics_list: List[str],
        phenotype_file: str,
        clinical_data_file: str,
        embedding_method: str = 'GNNs',
        output_dir: Optional[str] = None,
    ):
        """
        Initializes the SubjectRepresentationEmbedding instance.

        Args:
            adjacency_matrix (pd.DataFrame): The adjacency matrix of the graph.
            omics_list (List[str]): List of paths to omics data CSV files.
            phenotype_file (str): Path to the phenotype CSV file.
            clinical_data_file (str): Path to the clinical data CSV file.
            embedding_method (str, optional): Embedding method to use ('GNNs' or 'Node2Vec'). Defaults to 'GNNs'.
            output_dir (str, optional): Directory to save outputs. If None, creates a unique directory.
        """
        self.adjacency_matrix = adjacency_matrix
        self.omics_list = omics_list
        self.phenotype_file = phenotype_file
        self.clinical_data_file = clinical_data_file
        self.embedding_method = embedding_method
        self.output_dir = output_dir if output_dir else self._create_output_dir()
        self.logger = get_logger(__name__)
        self.logger.info("Initialized SubjectRepresentationEmbedding.")

    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the current SubjectRepresentationEmbedding run.

        Returns:
            str: Path to the created output directory.
        """
        base_dir = "subject_representation_output"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def run(self) -> pd.DataFrame:
        """
        Generate subject representations by integrating network embeddings.

        This method generates embeddings using the selected embedding method,
        reduces node embeddings using PCA if specified, and integrates these embeddings
        into the original omics data.

        Returns:
            pd.DataFrame:
                Enhanced omics data with integrated network embeddings.

        Raises:
            Exception: If any error occurs during the subject representation process.
        """
        self.logger.info("Running Subject Representation")

        try:
            # Step 1: Generate embeddings using the selected method
            self.logger.info(f"Generating embeddings using {self.embedding_method}")
            embeddings_df = self.generate_embeddings()

            # Step 2: Reduce embeddings with PCA
            self.logger.info("Reducing node embeddings with PCA")
            node_embedding_values = self.reduce_embeddings(embeddings_df)

            # Step 3: Integrate embeddings into omics data
            self.logger.info("Integrating embeddings into omics data")
            enhanced_omics_data = self.integrate_embeddings(node_embedding_values)

            # Save the enhanced omics data
            output_file = os.path.join(self.output_dir, "enhanced_omics_data.csv")
            enhanced_omics_data.to_csv(output_file)
            self.logger.info(f"Enhanced omics data saved to {output_file}")

            return enhanced_omics_data

        except Exception as e:
            self.logger.error(f"Error in Subject Representation: {e}")
            raise

    def generate_embeddings(self) -> pd.DataFrame:
        """
        Generate node embeddings using the selected embedding method.

        Returns:
            pd.DataFrame: Node embeddings with nodes as index.

        Raises:
            ValueError: If the embedding method is not recognized.
        """
        if self.embedding_method == 'GNNs':
            self.logger.info("Using GNNEmbedding to generate embeddings")
            gnn_embedding = GNNEmbedding(
                omics_list=self.omics_list,
                phenotype_file=self.phenotype_file,
                clinical_data_file=self.clinical_data_file,
                adjacency_matrix=self.adjacency_matrix,
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
            self.logger.info("Using Node2VecEmbedding to generate embeddings")
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
            self.logger.error(f"Embedding method '{self.embedding_method}' is not recognized.")
            raise ValueError(f"Embedding method '{self.embedding_method}' is not recognized. Choose 'GNNs' or 'Node2Vec'.")

        return embeddings_df

    def reduce_embeddings(self, embeddings: pd.DataFrame) -> pd.Series:
        """
        Reduce embeddings to a single value per node using PCA.

        Args:
            embeddings (pd.DataFrame):
                Node embeddings with nodes as index and embedding dimensions as columns.

        Returns:
            pd.Series:
                Reduced embedding values indexed by node names.

        Raises:
            ValueError: If embeddings DataFrame is empty or has invalid dimensions.
        """
        if embeddings.empty:
            self.logger.error("Embeddings DataFrame is empty.")
            raise ValueError("Embeddings DataFrame is empty.")

        if embeddings.shape[1] < 1:
            self.logger.error("Embeddings DataFrame must have at least one dimension.")
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

    def integrate_embeddings(
        self,
        node_embedding_values: pd.Series
    ) -> pd.DataFrame:
        """
        Integrate reduced embeddings into omics data.

        Args:
            node_embedding_values (pd.Series):
                Reduced embedding values per node.

        Returns:
            pd.DataFrame:
                Enhanced omics data with integrated embeddings.

        Raises:
            KeyError: If a node in omics_data is not found in node_embedding_values.
        """
        # Load and combine omics data
        self.logger.info("Loading and combining omics data for integration.")
        omics_data = combine_omics_data(self.omics_list)

        # Clean column names to match nodes
        omics_data.columns = self._clean_column_names(omics_data.columns)

        # Ensure that node_embedding_values has the same index as omics_data columns
        missing_nodes = set(omics_data.columns) - set(node_embedding_values.index)
        if missing_nodes:
            self.logger.warning(
                f"The following nodes are missing in embedding values and will be skipped: {missing_nodes}"
            )

        # Modify the omics data by weighting features with embedding values
        modified_omics_data = omics_data.copy()
        for node in omics_data.columns:
            if node in node_embedding_values.index:
                modified_omics_data[node] = omics_data[node] * node_embedding_values[node]
                self.logger.debug(f"Integrated embedding for node '{node}': {node_embedding_values[node]}")
            else:
                self.logger.warning(f"Node '{node}' not found in embedding values. Skipping integration.")

        self.logger.debug("Integrated embeddings into omics data.")
        return modified_omics_data

    def _clean_column_names(self, columns: pd.Index) -> pd.Index:
        """
        Clean column names to match node names in the network.

        Args:
            columns (pd.Index): Original column names.

        Returns:
            pd.Index: Cleaned column names.
        """
        import re
        clean_columns = []
        for col in columns:
            col_clean = re.sub(r'[^0-9a-zA-Z_]', '.', col)
            if not col_clean[0].isalpha():
                col_clean = 'X' + col_clean
            clean_columns.append(col_clean)
        return pd.Index(clean_columns)
