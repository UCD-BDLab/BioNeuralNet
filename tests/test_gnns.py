import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from bioneuralnet.network_embedding.gnns import GNNEmbedding

class TestGNNEmbedding(unittest.TestCase):

    def setUp(self):
        # Create sample adjacency matrix
        self.adjacency_matrix = pd.DataFrame(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            index=['gene1', 'gene2', 'gene3'],
            columns=['gene1', 'gene2', 'gene3']
        )

        # Create sample omics data
        self.omics_data1 = pd.DataFrame({
            'gene1': [1, 2, 3],
            'gene2': [4, 5, 6]
        }, index=['sample1', 'sample2', 'sample3'])

        self.omics_data2 = pd.DataFrame({
            'gene3': [7, 8, 9],
            'gene4': [10, 11, 12]
        }, index=['sample1', 'sample2', 'sample3'])

        # Save sample omics data to CSV files
        self.omics_file1 = 'tests/omics_data1.csv'
        self.omics_file2 = 'tests/omics_data2.csv'
        self.omics_data1.to_csv(self.omics_file1)
        self.omics_data2.to_csv(self.omics_file2)

        # Create sample phenotype data
        self.phenotype_data = pd.DataFrame({
            'phenotype': [0.1, 0.2, 0.3]
        }, index=['sample1', 'sample2', 'sample3'])
        self.phenotype_file = 'tests/phenotype_data.csv'
        self.phenotype_data.to_csv(self.phenotype_file)

        # Create sample clinical data
        self.clinical_data = pd.DataFrame({
            'age': [30, 40, 50],
            'bmi': [22.5, 27.8, 31.4]
        }, index=['sample1', 'sample2', 'sample3'])
        self.clinical_data_file = 'tests/clinical_data.csv'
        self.clinical_data.to_csv(self.clinical_data_file)

    def test_gnn_embedding(self):
        # Initialize GNNEmbedding
        gnn_embedding = GNNEmbedding(
            omics_list=[self.omics_file1, self.omics_file2],
            phenotype_file=self.phenotype_file,
            clinical_data_file=self.clinical_data_file,
            adjacency_matrix=self.adjacency_matrix,
            model_type='GCN'
        )

        # Run GNN embedding
        embeddings_dict = gnn_embedding.run()

        # Assertions
        self.assertIn('graph', embeddings_dict)
        embeddings = embeddings_dict['graph']
        self.assertEqual(embeddings.shape[0], self.adjacency_matrix.shape[0])

    def tearDown(self):
        # Remove temporary files
        import os
        os.remove(self.omics_file1)
        os.remove(self.omics_file2)
        os.remove(self.phenotype_file)
        os.remove(self.clinical_data_file)

if __name__ == '__main__':
    unittest.main()
