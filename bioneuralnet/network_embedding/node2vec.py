import os
from typing import Dict, Optional, List

import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from datetime import datetime
import glob

from ..utils.logger import get_logger


def find_files(directory: str, pattern: str) -> List[str]:
    """
    Find files in a directory matching a given pattern.

    Args:
        directory (str): Directory to search in.
        pattern (str): Pattern to match, e.g., "*.csv"

    Returns:
        List[str]: List of file paths matching the pattern.
    """
    return glob.glob(os.path.join(directory, pattern))


class Node2VecEmbedding:
    """
    Node2VecEmbedding Class for Generating Node2Vec-Based Embeddings.

    This class handles the loading of graph data, execution of the Node2Vec algorithm,
    and saving of the resulting node embeddings.
    """

    def __init__(
        self,
        input_dir: str,
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        window_size: int = 10,
        workers: int = 4,
        seed: int = 42,
        output_dir: Optional[str] = None,
    ):
        """
        Initializes the Node2VecEmbedding instance with direct parameters.

        Args:
            input_dir (str): Directory where input graphs (CSV files) are located.
            embedding_dim (int, optional): Dimension of the embeddings. Defaults to 128.
            walk_length (int, optional): Length of each walk. Defaults to 80.
            num_walks (int, optional): Number of walks per node. Defaults to 10.
            window_size (int, optional): Window size for Word2Vec. Defaults to 10.
            workers (int, optional): Number of worker threads. Defaults to 4.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            output_dir (str, optional): Directory to save outputs. If None, creates a unique directory.
        """
        # Assign parameters
        self.input_dir = input_dir
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.workers = workers
        self.seed = seed
        self.output_dir = output_dir if output_dir else self._create_output_dir()

        # Initialize logger
        self.logger = get_logger(__name__)
        self.logger.info("Initialized Node2VecEmbedding with the following parameters:")
        self.logger.info(f"Input Directory: {self.input_dir}")
        self.logger.info(f"Embedding Dimension: {self.embedding_dim}")
        self.logger.info(f"Walk Length: {self.walk_length}")
        self.logger.info(f"Number of Walks: {self.num_walks}")
        self.logger.info(f"Window Size: {self.window_size}")
        self.logger.info(f"Workers: {self.workers}")
        self.logger.info(f"Seed: {self.seed}")
        self.logger.info(f"Output Directory: {self.output_dir}")

    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the current Node2VecEmbedding run.

        The directory is named 'node2vec_embedding_output_timestamp' and is created in the current working directory.

        Returns:
            str: Path to the created output directory.
        """
        base_dir = "node2vec_embedding_output"
        timestamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def run(self, graphs: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
        """
        Perform Node2Vec embedding on the provided graphs.

        This method orchestrates the loading of graphs, execution of the Node2Vec algorithm,
        and saving of the resulting embeddings.

        Args:
            graphs (dict, optional):
                A dictionary where keys are graph names and values are adjacency matrices (pd.DataFrame).
                If None, graphs will be loaded from CSV files in the input directory.

        Returns:
            dict:
                A dictionary where keys are graph names and values are embeddings as DataFrames.

        Raises:
            FileNotFoundError: If no CSV files are found in the input directory.
            Exception: For any other unforeseen errors during execution.

        Notes:
            - Ensure that the Node2Vec parameters (e.g., walk_length, num_walks) are set appropriately 
            for your specific dataset and analysis goals.
            - The resulting embeddings can be used for downstream tasks such as clustering or 
            visualization.
    """
        self.logger.info("Running Node2Vec Embedding")

        try:
            embeddings: Dict[str, pd.DataFrame] = {}

            if graphs is None:
                # Load graphs from CSV files in the input directory
                graphs = self.load_graphs_from_directory()
                self.logger.info(f"Loaded {len(graphs)} graphs from {self.input_dir}")
            else:
                self.logger.info("Using provided graphs for embedding")

            for graph_name, adjacency_matrix in graphs.items():
                self.logger.info(f"Processing graph: {graph_name}")

                # Create a graph from the adjacency matrix
                G = nx.from_pandas_adjacency(adjacency_matrix)
                self.logger.debug(f"Converted adjacency matrix to NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

                # Initialize Node2Vec model
                node2vec = Node2Vec(
                    G,
                    dimensions=self.embedding_dim,
                    walk_length=self.walk_length,
                    num_walks=self.num_walks,
                    workers=self.workers,
                    seed=self.seed
                )
                self.logger.debug("Initialized Node2Vec model with parameters: "
                                  f"dimensions={self.embedding_dim}, walk_length={self.walk_length}, "
                                  f"num_walks={self.num_walks}, workers={self.workers}, seed={self.seed}")

                # Fit the model to generate walks and train the embeddings
                model = node2vec.fit(window=self.window_size, min_count=1, batch_words=4)
                self.logger.info("Node2Vec model fitted successfully.")

                # Create embeddings DataFrame
                embeddings_df = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)
                embeddings_df.index.name = 'node'
                embeddings_df.reset_index(inplace=True)
                self.logger.debug("Created embeddings DataFrame.")

                # Save embeddings to a CSV file
                embeddings_file = os.path.join(self.output_dir, f"{graph_name}_embeddings.csv")
                embeddings_df.to_csv(embeddings_file, index=False)
                self.logger.info(f"Node2Vec embeddings saved to {embeddings_file}")

                # Store embeddings in the dictionary
                embeddings[graph_name] = embeddings_df

            self.logger.info("Node2Vec Embedding completed successfully.")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error in Node2Vec Embedding: {e}")
            raise

    def load_graphs_from_directory(self) -> Dict[str, pd.DataFrame]:
        """
        Load graphs from CSV files in the input directory.

        Scans the input directory for CSV files, reads each as an adjacency matrix,
        and stores them in a dictionary.

        Returns:
            dict:
                A dictionary where keys are graph names and values are adjacency matrices (pd.DataFrame).

        Raises:
            FileNotFoundError: If no CSV files are found in the input directory.
        """
        graph_files = find_files(self.input_dir, "*.csv")
        graphs: Dict[str, pd.DataFrame] = {}

        for graph_file in graph_files:
            graph_name = os.path.splitext(os.path.basename(graph_file))[0]
            adjacency_matrix = pd.read_csv(graph_file, index_col=0)
            graphs[graph_name] = adjacency_matrix
            self.logger.debug(f"Loaded graph '{graph_name}' with shape {adjacency_matrix.shape}")

        if not graphs:
            self.logger.error(f"No CSV graph files found in the input directory: {self.input_dir}")
            raise FileNotFoundError(f"No CSV graph files found in the input directory: {self.input_dir}")

        return graphs
