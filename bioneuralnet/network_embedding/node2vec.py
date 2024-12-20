from typing import Optional
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from ..utils.logger import get_logger


class Node2VecEmbedding:
    """
    Node2VecEmbedding Class for Generating Node2Vec-Based Embeddings.

    This class handles the execution of the Node2Vec algorithm on a provided graph adjacency matrix
    and returns the resulting node embeddings without relying on file I/O operations.

    Attributes:
        embedding_dim (int): Dimension of the embeddings.
        walk_length (int): Length of each walk.
        num_walks (int): Number of walks per node.
        window_size (int): Window size for Word2Vec.
        workers (int): Number of worker threads.
        seed (int): Random seed for reproducibility.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        window_size: int = 10,
        workers: int = 4,
        seed: int = 42,
    ):
        """
        Initializes the Node2VecEmbedding instance.

        Args:
            adjacency_matrix (pd.DataFrame): Adjacency matrix representing the graph.
            embedding_dim (int, optional): Dimension of the embeddings. Defaults to 128.
            walk_length (int, optional): Length of each walk. Defaults to 80.
            num_walks (int, optional): Number of walks per node. Defaults to 10.
            window_size (int, optional): Window size for Word2Vec. Defaults to 10.
            workers (int, optional): Number of worker threads. Defaults to 4.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.adjacency_matrix = adjacency_matrix
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.workers = workers
        self.seed = seed
        self.logger = get_logger(__name__)

        self.logger.info("Initialized Node2VecEmbedding with the following parameters:")
        self.logger.info(f"Embedding Dimension: {self.embedding_dim}")
        self.logger.info(f"Walk Length: {self.walk_length}")
        self.logger.info(f"Number of Walks: {self.num_walks}")
        self.logger.info(f"Window Size: {self.window_size}")
        self.logger.info(f"Workers: {self.workers}")
        self.logger.info(f"Seed: {self.seed}")

        self.embeddings: Optional[pd.DataFrame] = None

    def _convert_to_networkx(self) -> nx.Graph:
        """
        Converts the adjacency matrix to a NetworkX graph.

        Returns:
            nx.Graph: The converted NetworkX graph.
        """
        try:
            G = nx.from_pandas_adjacency(self.adjacency_matrix)
            self.logger.info(
                f"Converted adjacency matrix to NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
            )
            return G
        except Exception as e:
            self.logger.error(f"Error converting adjacency matrix to NetworkX graph: {e}")
            raise

    def _generate_embeddings(self, G: nx.Graph) -> pd.DataFrame:
        """
        Generates node embeddings using Node2Vec.

        Args:
            G (nx.Graph): The NetworkX graph.

        Returns:
            pd.DataFrame: DataFrame containing node embeddings.
        """
        try:
            self.logger.info("Initializing Node2Vec model.")
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                workers=self.workers,
                seed=self.seed,
                quiet=True 
            )

            self.logger.info("Fitting Node2Vec model to generate embeddings.")
            model = node2vec.fit(window=self.window_size, min_count=1, batch_words=4)

            self.logger.info("Generating embeddings DataFrame.")
            embeddings_df = pd.DataFrame(
                model.wv.vectors,
                index=model.wv.index_to_key,
                columns=[str(i) for i in range(self.embedding_dim)] 
            )
            embeddings_df.index.name = 'node'
            embeddings_df.reset_index(inplace=True)

            self.logger.info("Embeddings generated successfully.")
            return embeddings_df

        except Exception as e:
            self.logger.error(f"Error generating embeddings with Node2Vec: {e}")
            raise


    def run(self) -> pd.DataFrame:
        """
        Runs the Node2Vec embedding process.

        Returns:
            pd.DataFrame: DataFrame containing node embeddings.
        """
        try:
            self.logger.info("Starting Node2Vec embedding process.")
            G = self._convert_to_networkx()
            self.embeddings = self._generate_embeddings(G)
            self.logger.info("Node2Vec embedding process completed successfully.")
            return self.embeddings
        except Exception as e:
            self.logger.error(f"Error in Node2VecEmbedding run method: {e}")
            raise

    def get_embeddings(self) -> pd.DataFrame:
        """
        Retrieves the generated node embeddings.

        Returns:
            pd.DataFrame: DataFrame containing node embeddings.

        Raises:
            ValueError: If embeddings have not been generated yet.
        """
        if self.embeddings is None:
            self.logger.error("Embeddings have not been generated yet. Call the run() method first.")
            raise ValueError("Embeddings have not been generated yet. Call the run() method first.")
        return self.embeddings

    def save_embeddings(self, filepath: str) -> None:
        """
        Saves the generated embeddings to a CSV file.

        Args:
            filepath (str): Path to save the embeddings CSV file.

        Raises:
            ValueError: If embeddings have not been generated yet.
        """
        if self.embeddings is None:
            self.logger.error("Embeddings have not been generated yet. Call the run() method first.")
            raise ValueError("Embeddings have not been generated yet. Call the run() method first.")

        try:
            self.embeddings.to_csv(filepath, index=False)
            self.logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving embeddings to {filepath}: {e}")
            raise
