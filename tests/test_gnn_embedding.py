import unittest
from unittest.mock import patch
import pandas as pd
import torch
from bioneuralnet.network_embedding import GNNEmbedding

class TestGNNEmbedding(unittest.TestCase):

    def setUp(self):
        self.adjacency_matrix = pd.DataFrame(
            [[0,1],[1,0]],
            index=['gene1','gene2'],
            columns=['gene1','gene2']
        )

        self.omics_data = pd.DataFrame({
            'gene1': [1,2],
            'gene2': [3,4]
        }, index=['sample1', 'sample2'])

        self.clinical_data = pd.DataFrame({
            'age': [30, 45],
            'bmi': [22.5, 28.0]
        }, index=['sample1', 'sample2'])

    @patch.object(GNNEmbedding, '_generate_embeddings', return_value=torch.tensor([[0.1, 0.2],[0.3, 0.4]]))
    def test_run(self, mock_gen_emb):
        gnn = GNNEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            clinical_data=self.clinical_data,
            model_type='GCN'
        )

        embeddings_dict = gnn.run()
        self.assertIn('graph', embeddings_dict, "Embeddings dictionary should contain 'graph' key.")
        embeddings = embeddings_dict['graph']
        self.assertIsInstance(embeddings, torch.Tensor, "'graph' value should be a torch.Tensor.")
        self.assertEqual(embeddings.shape, (2,2), "Embeddings tensor should have shape (2,2).")
        mock_gen_emb.assert_called_once()

if __name__ == '__main__':
    unittest.main()
