import unittest
from unittest.mock import patch
import pandas as pd
import torch
from bioneuralnet.subject_representation import GraphEmbedding

class TestGraphEmbedding(unittest.TestCase):

    def setUp(self):
        # simple adjacency matrix
        self.adjacency_matrix = pd.DataFrame(
            [[0,1,0],
             [1,0,1],
             [0,1,0]], 
            index=['gene1','gene2','gene3'], 
            columns=['gene1','gene2','gene3']
        )

        # Mock omics data as a single dataframe with required genes and 'finalgold_visit'
        self.omics_data1 = pd.DataFrame({
            'gene1': [1, 2, 3],
            'gene2': [4, 5, 6],
            'gene3': [7, 8, 9]
        }, index=['sample1', 'sample2', 'sample3'])
        
        self.omics_data2 = pd.DataFrame({
            'gene4': [2, 1, 0],
            'gene5': [1, 3, 5],
            'gene6': [9, 8, 7]
        }, index=['sample1', 'sample2', 'sample3'])

        # Combine omics_data1 and omics_data2 into a single omics_data DataFrame with unique column names
        self.omics_data = pd.concat([self.omics_data1, self.omics_data2], axis=1)

        # Add finalgold_visit (target class) to omics_data as per GraphEmbedding's requirement
        self.omics_data['finalgold_visit'] = [0, 1, 2]

        # Mock clinical data with relevant clinical variables
        self.clinical_data_df = pd.DataFrame({
            'age': [30, 40, 50]
        }, index=['sample1', 'sample2', 'sample3'])


    @patch.object(GraphEmbedding, 'generate_embeddings', return_value=pd.DataFrame({
        'dim1': [0.1, 0.2, 0.3]
    }, index=['gene1', 'gene2', 'gene3']))
    @patch.object(GraphEmbedding, 'reduce_embeddings', return_value=pd.Series({
        'gene1': 0.1,
        'gene2': 0.2,
        'gene3': 0.3
    }))
    @patch.object(GraphEmbedding, 'integrate_embeddings', return_value=pd.DataFrame({
        'gene1': [1.1, 2.2, 3.3],
        'gene2': [4.4, 5.5, 6.6],
        'gene3': [7.7, 8.8, 9.9],
        'gene4': [2.2, 1.1, 0.0],
        'gene5': [1.1, 3.3, 5.5],
        'gene6': [9.9, 8.8, 7.7],
        'finalgold_visit': [0, 1, 2]
    }, index=['sample1', 'sample2', 'sample3']))
    def test_run(self, mock_integrate, mock_reduce, mock_generate):
        """
        Test the run method of GraphEmbedding to ensure it returns the expected enhanced omics data.
        """
        # Initialize GraphEmbedding with correct parameters
        graph_embed = GraphEmbedding(
            adjacency_matrix=self.adjacency_matrix,
            omics_data=self.omics_data,
            clinical_data=self.clinical_data_df,
            embedding_method='GNNs'
        )

        # Run the embedding process
        enhanced_omics_data = graph_embed.run()

        # Assertions to verify the output
        self.assertTrue(isinstance(enhanced_omics_data, pd.DataFrame), "Output should be a pandas DataFrame.")
        self.assertEqual(enhanced_omics_data.shape, self.omics_data.shape, "Output shape should match input omics_data shape.")
        self.assertListEqual(list(enhanced_omics_data.columns), list(self.omics_data.columns), "Columns should match input omics_data columns.")
        self.assertListEqual(list(enhanced_omics_data.index), list(self.omics_data.index), "Indices should match input omics_data indices.")

        # Ensure that the mocked methods were called
        mock_generate.assert_called_once()
        mock_reduce.assert_called_once()
        mock_integrate.assert_called_once()

if __name__ == '__main__':
    unittest.main()
