import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from bioneuralnet.network_embedding import GnnEmbedding

class TestGnnEmbedding(unittest.TestCase):

    def setUp(self):
        self.adjacency_matrix = pd.DataFrame(
            [[0,1],[1,0]],
            index=['gene1','gene2'],
            columns=['gene1','gene2']
        )

        # Mock omics data as DataFrames
        self.omics_data1 = pd.DataFrame({'gene1':[1,2],'gene2':[3,4]}, index=['sample1','sample2'])
        self.omics_data2 = pd.DataFrame({'gene3':[5,6],'gene4':[7,8]}, index=['sample1','sample2'])

        # Create node features
        self.node_features = pd.DataFrame({'feat1':[0.1,0.2]}, index=['gene1','gene2'])

    @patch.object(GnnEmbedding, '_generate_embeddings', return_value={'graph': torch.tensor([[0.1, 0.2],[0.3, 0.4]])})
    def test_run(self, mock_gen_emb):
        gnn = GnnEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            node_features=self.node_features,
            model_type='GCN'
        )
        embeddings_dict = gnn.run()
        self.assertIn('graph', embeddings_dict)
        embeddings = embeddings_dict['graph']
        self.assertEqual(embeddings.shape, (2,2))
        mock_gen_emb.assert_called_once()

if __name__ == '__main__':
    unittest.main()
