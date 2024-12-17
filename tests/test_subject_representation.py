import unittest
from unittest.mock import patch
import pandas as pd
from bioneuralnet.subject_representation import GraphEmbedding

class TestGraphEmbedding(unittest.TestCase):

    def setUp(self):
        self.adjacency_matrix = pd.DataFrame([[0,1,0],[1,0,1],[0,1,0]], 
                                             index=['gene1','gene2','gene3'], 
                                             columns=['gene1','gene2','gene3'])

        self.omics_data1 = pd.DataFrame({'gene1': [1, 2, 3], 'gene2': [4, 5, 6], 'gene3': [7, 8, 9]},
                                        index=['sample1', 'sample2', 'sample3'])
        self.omics_data2 = pd.DataFrame({'gene1': [2, 1, 0], 'gene2': [1, 3, 5], 'gene3': [9, 8, 7]},
                                        index=['sample1', 'sample2', 'sample3'])

        self.phenotype_df = pd.DataFrame({'finalgold_visit': [0, 1, 2]}, index=['sample1', 'sample2', 'sample3'])
        self.clinical_data_df = pd.DataFrame({'age': [30, 40, 50]}, index=['sample1', 'sample2', 'sample3'])

    def test_run(self):
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_list=[self.omics_data1, self.omics_data2],
            phenotype_df=self.phenotype_df,
            clinical_data_df=self.clinical_data_df,
            embedding_method='GNNs'
        )

        # Mock some internal methods if needed
        with patch.object(GraphEmbedding, 'generate_embeddings', return_value=pd.DataFrame({'dim1': [0.1, 0.2, 0.3]}, index=['gene1', 'gene2', 'gene3'])), \
             patch.object(GraphEmbedding, 'reduce_embeddings', return_value=pd.Series([0.1, 0.2, 0.3], index=['gene1', 'gene2', 'gene3'])), \
             patch.object(GraphEmbedding, 'integrate_embeddings', return_value=self.omics_data1):
            enhanced_omics_data = graph_embed.run()
            self.assertTrue(isinstance(enhanced_omics_data, pd.DataFrame))
            self.assertEqual(enhanced_omics_data.shape, self.omics_data1.shape)

if __name__ == '__main__':
    unittest.main()
