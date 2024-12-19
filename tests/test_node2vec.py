import os
import unittest
import pandas as pd
from bioneuralnet.network_embedding import Node2VecEmbedding


class TestNode2VecEmbedding(unittest.TestCase):

    def setUp(self):
        # Sample adjacency matrix
        self.adjacency_matrix = pd.DataFrame({
            'GeneA': [1.0, 1.0, 0.0],
            'GeneB': [1.0, 1.0, 1.0],
            'GeneC': [0.0, 1.0, 1.0]
        }, index=['GeneA', 'GeneB', 'GeneC'])

    def test_embedding_output(self):
        node2vec = Node2VecEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            embedding_dim=64,
            walk_length=30,
            num_walks=200,
            window_size=10,
            workers=2,
            seed=123
        )
        embeddings = node2vec.run()

        # Check if embeddings is a DataFrame
        self.assertIsInstance(embeddings, pd.DataFrame, f"Expected DataFrame, got {type(embeddings)} instead.")

        # Check if 'node' column exists
        self.assertIn('node', embeddings.columns, "'node' column is missing in the embeddings DataFrame.")

        # Check if embedding dimensions are correct
        expected_columns = ['node'] + [str(i) for i in range(64)]
        self.assertTrue(
            all(col in embeddings.columns for col in expected_columns),
            f"Embeddings DataFrame is missing expected columns. Found: {embeddings.columns}"
        )


    def test_get_embeddings_before_run(self):
        node2vec = Node2VecEmbedding(adjacency_matrix=self.adjacency_matrix)
        with self.assertRaises(ValueError):
            node2vec.get_embeddings()

    def test_save_embeddings_before_run(self):
        node2vec = Node2VecEmbedding(adjacency_matrix=self.adjacency_matrix)
        with self.assertRaises(ValueError):
            node2vec.save_embeddings('embeddings.csv')

    def test_save_embeddings_after_run(self):
        node2vec = Node2VecEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            embedding_dim=64,
            walk_length=30,
            num_walks=200,
            window_size=10,
            workers=2,
            seed=123
        )
        embeddings = node2vec.run()
        node2vec.save_embeddings('test_embeddings.csv')

        # Check if file exists
        self.assertTrue(os.path.exists('test_embeddings.csv'))

        # Clean up
        os.remove('test_embeddings.csv')


if __name__ == '__main__':
    unittest.main()
