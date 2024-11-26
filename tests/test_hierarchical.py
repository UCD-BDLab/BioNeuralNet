import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from bioneuralnet.clustering.hierarchical import HierarchicalClustering

class TestHierarchicalClustering(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        # Mock the pandas.read_csv function
        mock_adjacency_matrix = pd.DataFrame([[0,1],[1,0]], index=['node1','node2'], columns=['node1','node2'])
        mock_read_csv.return_value = mock_adjacency_matrix

        hierarchical_cluster = HierarchicalClustering(adjacency_matrix_file='input/global_network.csv')

        hierarchical_cluster.load_data()

        # Assertions
        self.assertTrue(hierarchical_cluster.adjacency_matrix.equals(mock_adjacency_matrix))

    @patch('bioneuralnet.clustering.hierarchical.AgglomerativeClustering')
    @patch('bioneuralnet.clustering.hierarchical.HierarchicalClustering.load_data')
    @patch('pandas.DataFrame.to_csv')
    def test_run_clustering(self, mock_to_csv, mock_load_data, mock_agglomerative_clustering):
        # Mock the load_data method
        mock_load_data.return_value = None

        # Mock the adjacency matrix
        hierarchical_cluster = HierarchicalClustering(adjacency_matrix_file='input/global_network.csv')
        hierarchical_cluster.adjacency_matrix = pd.DataFrame([[0,1],[1,0]], index=['node1','node2'], columns=['node1','node2'])

        # Mock the AgglomerativeClustering
        mock_model = MagicMock()
        mock_model.fit_predict.return_value = [0,1]
        mock_agglomerative_clustering.return_value = mock_model

        # Run the clustering
        results = hierarchical_cluster.run_clustering()

        # Assertions
        mock_agglomerative_clustering.assert_called_once()
        mock_model.fit_predict.assert_called_once()
        mock_to_csv.assert_any_call(f"{hierarchical_cluster.output_dir}/cluster_labels.csv", index=False)
        self.assertIn('cluster_labels', results)
        self.assertEqual(len(results['cluster_labels']), 2)

if __name__ == '__main__':
    unittest.main()
