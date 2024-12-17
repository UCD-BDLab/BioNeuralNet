import unittest
from unittest.mock import patch
import pandas as pd
import torch
from bioneuralnet.network_embedding import GnnEmbedding

class TestGnnEmbedding(unittest.TestCase):

    def setUp(self):
        # Define a simple adjacency matrix
        self.adjacency_matrix = pd.DataFrame(
            [[0,1],[1,0]],
            index=['gene1','gene2'],
            columns=['gene1','gene2']
        )

        # Mock omics data as a single DataFrame with required genes
        self.omics_data = pd.DataFrame({
            'gene1': [1,2],
            'gene2': [3,4]
        }, index=['sample1', 'sample2'])

        # Mock clinical data with relevant clinical variables
        self.clinical_data = pd.DataFrame({
            'age': [30, 45],
            'bmi': [22.5, 28.0]
        }, index=['sample1', 'sample2'])

    @patch.object(GnnEmbedding, '_generate_embeddings', return_value=torch.tensor([[0.1, 0.2],[0.3, 0.4]]))
    def test_run(self, mock_gen_emb):
        # Initialize GnnEmbedding with correct parameters
        gnn = GnnEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            clinical_data=self.clinical_data,
            model_type='GCN'
        )

        # Run the embedding process
        embeddings_dict = gnn.run()

        # Assertions to verify the output
        self.assertIn('graph', embeddings_dict, "Embeddings dictionary should contain 'graph' key.")
        embeddings = embeddings_dict['graph']
        self.assertIsInstance(embeddings, torch.Tensor, "'graph' value should be a torch.Tensor.")
        self.assertEqual(embeddings.shape, (2,2), "Embeddings tensor should have shape (2,2).")
        mock_gen_emb.assert_called_once()

if __name__ == '__main__':
    unittest.main()
